# decider-auto

A small **permission classifier for coding agents**: before an agent runs a tool call, it decides whether the call may run
without asking the user, or must be stopped and confirmed. It is a fine-tune of [decider-4b](https://huggingface.co/Mapika/decider-4b)
(a 4B dense "decision model" that reads option-letter logits in one forward pass) that emulates the behaviour of
[Claude Code's auto mode](https://code.claude.com/docs/en/permission-modes).

Given the environment, the user's messages, the agent's earlier tool calls and the proposed tool call, it answers in one pass:

- **Should this action be blocked?** (no / yes, with a calibrated probability)
- **Which kind of rule does it break?** (13 groups, or "none")
- **Did the user explicitly ask for this action and target?** (no / yes)

When the first stage flags an action, a second stage (any LLM) reads the full policy text - including the user's own rules -
and decides allow / block, which is how Claude Code's classifier is built too.

It runs on one GPU (~9 GB bf16 at 4k-token contexts, ~0.2 ms/token) and keeps a per-session prefix cache so a new action plus
the questions costs ~70 ms instead of ~0.4 s. It was trained and is used in production by the GB10 Hub (a lab JupyterHub whose
chat agent is pi), where it powers the Auto / Plan / Edit permission modes.

## Results

Held-out synthetic decision families (387 rows, 173 blocks), real auto-mode decisions recorded on the training machine
(1,513 allows, 13 blocks) and real Claude Code sessions relabelled by a second LLM (548 rows):

| model | synth AUROC | synth recall@0.5 | synth FPR@0.5 | real AUROC | real FPR@0.5 | swe-relabel AUROC |
|---|---|---|---|---|---|---|
| decider-4b v2.1 (zero-shot) | - | - | - | 0.752 | 5.1% | 0.769 |
| v0 (synthetic, 7 groups) | 0.978 | 88.1% | 3.3% | 0.883 | 0.20% | 0.820 |
| **v1 (this model)** | **0.992** | **92.5%** | **1.4%** | **0.878** | **0.07%** | 0.830 |

Rule-group accuracy on blocked rows is 83%, exact rule accuracy given the right group 86%. The honest weaknesses are in
[docs/evaluation.md](docs/evaluation.md): the real-data blocks are few and clustered, v1 misses most of them at 0.5 (the second
stage exists for exactly that), and the sweep relabel was produced by the same model family used to write the training data.

## Repository layout

| path | what |
|---|---|
| `automode/` | inference: state view, question set, slot readout with prefix cache (`infer.py`), policy composition (`policy.py`), stage 2 (`stage2.py`), HTTP judge service (`server.py`), Claude Code transcript reader (`transcript.py`) |
| `scripts/` | LoRA fine-tune (`train_lora.py`), evaluation (`eval.py`), checks and latency probes |
| `tools/` | data pipeline: transcript extraction, synthetic scenario builders, LLM generation/judging, dataset builder, hub-record import |
| `spec/` | rule groups (`groups.json`), environments (`envs.json`), scenario schema |
| `hooks/` | Claude Code `PreToolUse` hook (shadow or enforce) |
| `deploy/` | systemd unit template for the judge service |
| `docs/` | design, policy, data, evaluation, integration |
| `tests/` | unit tests that need no GPU and no policy text |

**Not in this repository:** the training data, the policy text and the model weights. The data contains the training
machine's real transcripts and generated scenarios (kept private); the default rule text belongs to Claude Code (generate it
locally, see below); the weights are on Hugging Face.

## Quick start

### 1. Environment

Python 3.12, a CUDA GPU. On an NVIDIA GB10 (aarch64, sm_121) the following worked; on other GPUs use the matching torch build.

```bash
git clone https://github.com/Chiytako/decider-auto.git && cd decider-auto
uv venv --python /usr/bin/python3.12 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch==2.11.0+cu130"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e third_party/decider   # after: git clone https://github.com/Mapika/decider third_party/decider
```

`flash-linear-attention` provides Triton kernels for the gated delta-net layers (works without it, several times slower).
Building `causal-conv1d` from source makes the forward ~27% faster on GB10; on x86 it usually installs from a wheel.

### 2. Policy text (not redistributed)

The classifier's rules are Claude Code's own auto-mode policy. Generate them from your installed CLI instead of copying them
from here:

```bash
claude auto-mode defaults > spec/automode_defaults.json     # soft_deny / hard_deny / allow / environment
```

`spec/envs.json` has five generic environments (personal repo, SaaS org, fintech K8s, OSS maintainer, HPC lab). Put your own
into `spec/envs.local.json` (git-ignored); the judge takes an environment id per request (`env`).

### 3. Weights

```bash
.venv/bin/python -c 'from huggingface_hub import snapshot_download as s; s("Mapika/decider-4b", local_dir="models/decider-4b")'
.venv/bin/python -c 'from huggingface_hub import snapshot_download as s; s("ChiTako/decider-4b-automode", local_dir="models/automode-current")'
```

### 4. Judge service

```bash
.venv/bin/python -m automode.server --model models/automode-current --port 8095 \
    --llm_url http://127.0.0.1:8000/v1/chat/completions --llm_model <stage-2 model>
# POST /judge with {"session_id", "tool_name", "tool_input", "user_messages", "prior_actions", "cwd", "env", "rules"?}
# GET /policy?env=... returns the default rules for a settings screen
```

Stage 2 is optional: without `--llm_url` the service answers with stage 1 alone (its `--block` threshold). The unit template
in `deploy/` runs it as a systemd service.

### 5. Claude Code hook (optional)

```json
{"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command",
  "command": "python3 <repo>/hooks/automode_hook.py", "timeout": 10}]}]}}
```

`AUTOMODE_HOOK_MODE=shadow` (default) only logs judgments; `enforce` turns them into allow / ask / deny decisions. If the
service is down the hook says nothing and Claude Code's normal permission flow applies.

### 6. Data and training

The pipeline is in `tools/` and `scripts/`; it needs your own agent transcripts and an LLM to write and relabel scenarios. See
[docs/data.md](docs/data.md) for the pipeline and [docs/design.md](docs/design.md) for the training recipe (LoRA rank 64,
2 epochs, replay rows trained toward the parent with KL). With your data at `data/build/train.jsonl`:

```bash
systemd-run --user --scope --collect -p MemoryHigh=60G -p MemoryMax=70G -- \
  .venv/bin/python scripts/train_lora.py --data data/build/train.jsonl --out runs/v1 --epochs 2 \
    --save_merged --no_grad_ckpt --bf16_lora --max_tokens 4096 --accum 16
.venv/bin/python scripts/eval.py --model runs/v1/model --name v1
```

## Integration contract

The GB10 Hub integration (extension, settings screens, permission modes) lives in the hub repository, not here. The judge
service is the interface: see [docs/hub-integration.md](docs/hub-integration.md). Rule overrides follow Claude Code's
`autoMode` shape (`environment` / `allow` / `soft_deny`, `$defaults` splices the defaults), and the caller applies
allow/ask/deny permission patterns before asking the judge.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Derived from [Mapika/decider](https://github.com/Mapika/decider) and
[Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base).

## 概要 (日本語)

コーディングエージェントのツール呼び出しを、利用者の発言とそれまでの操作から 1 回の順伝播で判定する小さなモデルです
(decider-4b の微調整)。危険度と規則の種類を返し、迷うものは規則の全文を読む LLM が見直します。学習データ・規則の本文・
重みはこのリポジトリには含みません (規則は `claude auto-mode defaults` で生成、重みは Hugging Face)。GB10 Hub の
権限モード (Auto / Plan / Edit) で本番運用しています。
