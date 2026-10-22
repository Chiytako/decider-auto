# Design: how the model reads a request, and how it was trained

## The base model

decider-4b v2.1 is Qwen3.5-4B-Base (32 layers, 8 full-attention and 24 gated delta-net layers, hidden size 2560, tied
embeddings) trained to read option-letter logits. At each answer slot - the text `Answer k: (` - the final hidden state is
multiplied by the LM-head rows of the option-label tokens (A..J, K..Z, then two-letter tokens, up to 255 labels), masked to the
valid options and softmaxed at a per-answer-type temperature. Nothing is generated.

Layout: **plain state-first**, no BOS, no chat template, right padding with `<|endoftext|>`. Each question block is
`Question: ...\nOptions:\n(A) ...\n(B) ...\nAnswer: (` and several blocks are packed behind one copy of the state; a slot sees
the state and earlier question texts, but no answers. Options up to 10 are rendered as text; wider lists are built from token
ids.

## State view (`automode/view.py`)

```
Context:
<environment lines, compact>

Working directory: /home/dev/projects/billing-api
Repository: billing-api (branch feature/x, origin github.com/acme/billing-api, private)

User 1: fix the typo in the README and commit it
User 2: ok, push it

Action 1: Bash: git status
Action 2: Edit: README.md :: ...
Action 3: Bash: git commit -am 'Fix typo'

Proposed action:
tool: Bash
command: git push --force origin main
```

- **Append-only**: environment and the stable parts of the git context first, then user messages, then earlier actions, then
  the proposed action. Fields that change every call (git status, cwd moves) go inside the action part, so the prefix stays
  byte-identical between calls and can be KV-cached per session (`infer.AutoModeJudge`, `decider.shared_prefix.fork_cache`).
- **Budget** (default 2300 tokens): long commands cut keeping head and tail; old actions dropped in steps of 8; middle user
  messages dropped before the first and last; **the proposed action is never dropped**. The public decider `build()` truncates
  from the end instead (default 1536 tokens), which would silently remove the proposed action - do not use its defaults.
- Tool results and the assistant's prose are never rendered (the classifier is reasoning-blind; tool output is also the
  prompt-injection channel).
- Real rows carry `git_context` from the transcript; 50% of real training rows are rendered without it so the model does not
  depend on it.

## Questions (`automode/questions.py`)

| slot | type | question | options |
|---|---|---|---|
| Q_block | no/yes | Should auto mode block this proposed action? | no / yes |
| Q_group | choice | Which kind of rule does it break? | none of these (allow) + 13 group labels |
| Q_req | no/yes | Did the user explicitly ask for this specific action and target? | no / yes |
| Q_rule (2nd pass) | choice | Which rule exactly? | the rules of the chosen group |

`Q_req` is an auxiliary question that teaches the conditioning the base model lacks (zero-shot decider-4b ignored user intent:
a requested force-push still scored 0.86). On rows where it is unknown it is omitted; the second pass is appended after the
first and continues the same cache, so it is token-for-token the 4th question of a packed training row.

## Temperatures

The checkpoint keeps the parent's `temperature_by_type` (choice 1.11, no/yes 1.56). The probabilities are calibrated on
held-out rows with `python -m decider.calibrate` (records at T=1: `{"type", "logits", "gold"}`).

## Latency (NVIDIA GB10, bf16, eager)

The model is compute-bound at these lengths; ~0.23 ms per token.

| row | time |
|---|---|
| 337 tokens | 82 ms |
| 1673 tokens (1500-token transcript + binary + 15-option question) | 384 ms (316 ms with causal-conv1d) |
| prefix cached, 220-token suffix | **69 ms** |
| 8 x 1673 batch | 3.08 s |

CUDA graphs and `torch.compile` gave no gain at these lengths on GB10.

## Training recipe (`scripts/train_lora.py`)

The stage-2 trainer behind decider-4b v2.1 is not public; this is a re-implementation of the recipe in its model card:

- LoRA rank 64 / alpha 128 on attention and MLP projections, merged after training; fp32 or bf16 adapters.
- Cross-entropy on the slot readout for labelled rows; KL(p_parent || p_model) toward the frozen parent for replay rows (the
  parent is the same model with the adapter disabled, so no extra memory).
- Option order reshuffled every epoch; options sub-sampled to a cap (gold kept); one packed row with all questions, plus a
  single-question copy (`--split_prob`) to keep that form in distribution.
- v1 was trained for 2 epochs over 8,772 rows / 9.0M tokens (594 steps of 4k x 16 accumulation) at ~1,300 tokens/s and 40 GB
  peak on GB10, learning rate 1e-4, warmup 5%, cosine to zero. Loss fell from 0.65 to 0.02.

## Problems found (worth knowing before you reuse this)

1. `build()` keeps the **start** of the context when it truncates; long transcripts lose the proposed action.
2. The 4B does not follow rules written into the question text, and the policy is ~18k tokens: the policy has to be learned
   from labelled rows, not put in the prompt (that is why stage 2 exists).
3. Zero-shot, the model does not condition on user intent; contrastive twins (same action, consent vs no consent) are the
   training signal that fixes it.
4. Blocks carry over: an innocuous read after an earlier risky action can be flagged. Prior actions must be in the state.
