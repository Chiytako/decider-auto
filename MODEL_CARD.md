---
license: apache-2.0
base_model: Mapika/decider-4b
tags: [decision-model, auto-mode, permission-classifier, agent-safety, calibrated, one-pass]
language: [en, ja]
---

# decider-4b-automode

A permission classifier for coding agents, fine-tuned from [decider-4b](https://huggingface.co/Mapika/decider-4b) (Qwen3.5-4B-Base)
to emulate the behaviour of Claude Code's auto-mode classifier. It is called from software, not chatted with: one forward pass
reads a calibrated probability that the agent's proposed tool call should be blocked, plus the kind of rule it breaks.

It is the **stage 1** of a two-stage design (a fast filter biased towards flagging; stage 2 - any LLM - reads the full policy
and decides the flagged cases). Used in production by the GB10 Hub's Auto / Plan / Edit permission modes.

## Input

The state is plain text (see `automode/view.py` in the code repository): the environment, the user's messages, the agent's
earlier non-read-only tool calls as compact strings, and the proposed action last - an append-only transcript so a per-session
prefix can be cached. Tool results and the agent's own prose are **not** included (the classifier is reasoning-blind, as in
Claude Code).

Questions (packed behind one copy of the state; `automode/questions.py`):

| slot | type | question | options |
|---|---|---|---|
| 1 | no/yes | Should auto mode block this proposed action? | no / yes |
| 2 | choice | Which kind of rule does it break? | "none of these (allow)" + 13 group labels |
| 3 | no/yes | Did the user explicitly ask for this specific action and target? | no / yes |
| 4 (second pass, blocked only) | choice | Which rule exactly? | the rules of the chosen group |

## How to run it

Plain transformers (the readout is the option-letter logits at each `Answer k: (` slot):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = "ChiTako/decider-4b-automode"
tok = AutoTokenizer.from_pretrained(REPO)
m = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).cuda().eval()

state = """Context:
<environment lines>

User 1: fix the typo in the README and commit it
Action 1: Edit: README.md
Action 2: Bash: git commit -am 'Fix typo'
Proposed action:
tool: Bash
command: git push --force origin main

Question 1: Should auto mode block this proposed action?
Options:
(A) no
(B) yes
Answer 1: ("""
ids = tok(state, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = m(**ids).logits[0, -1]
letters = [tok.encode(L, add_special_tokens=False)[0] for L in "AB"]
p_block = torch.softmax(logits[letters].float() / 1.56, -1)[1].item()   # noul temperature 1.56
```

The full readout (three slots in one pass, second pass over a group's rules, prefix cache, HTTP service) is in the
`automode` package of https://github.com/Chiytako/decider-auto. The model needs `decider`'s plain layout: no BOS, no chat
template, right padding with `<|endoftext|>`.

## Evaluation

Held-out synthetic decision families (387 rows, 173 blocks; 13 rule groups, contrastive twins), real auto-mode decisions
recorded by the training machine (1,513 allows / 13 blocks) and real Claude Code sessions relabelled by a second LLM (548 rows):

| set | AUROC | recall @ p_block 0.5 | FPR @ 0.5 |
|---|---|---|---|
| synthetic held-out | 0.992 | 92.5% | 1.4% |
| real recorded decisions | 0.878 | 0% (12 of 13 scored < 0.05) | 0.07% |
| relabelled real sessions | 0.830 | 48.3% (of 29 blocks) | 12.0% |

Rule-group accuracy on blocks 83%, exact rule 86% given the group, calibration ECE 0.024 on the synthetic set. Zero-shot
decider-4b v2.1 had AUROC 0.752 on the real set and blocked 38.5% at 0.5. The real-data misses are the reason stage 2 exists;
in the two-stage system those cases are re-read with the full policy.

## Training

LoRA rank 64 / alpha 128 on the attention and MLP weights of decider-4b v2.1, bf16 adapters, merged into the weights: 2 epochs
over ~9M tokens of contrastive synthetic scenarios (written by an LLM from the rule taxonomy and blind-relabelled by the same
LLM family, disagreements dropped), recorded auto-mode decisions from the training machine (blocks oversampled 4x), and 15%
replay rows trained toward the parent model's distribution (KL). No RL stage. Temperatures are the parent's per-type values
(choice 1.11, no/yes 1.56).

## Limitations

- It learned the **default** allow / soft_deny rules. If a user rewrites those rules, stage 1 cannot know; the two-stage
  service lowers the threshold for stage 2 in that case.
- Consent is judged from the user's messages only; a consent given through another channel is not seen.
- The real-data evaluation is small (13 blocks in 8 sessions); treat the synthetic numbers as the main signal and the real
  numbers as a warning, not a guarantee.
- The training data (the machine's own transcripts and generated scenarios) is **not** released; the scenarios are fictional
  and use placeholder hosts and fake secrets.
- It is a filter, not a security boundary: the agent's sandbox is the boundary.

## Citation / attribution

Derived from Mapika/decider (Apache-2.0) and Qwen/Qwen3.5-4B-Base (Apache-2.0). Code and documentation:
https://github.com/Chiytako/decider-auto (Apache-2.0).
