# Data: how the training rows were built (and what is not released)

The training data is **not** in this repository: it contains real transcripts from the training machine and LLM-generated
scenarios. The tools that build it are here, so you can rebuild an equivalent set from your own agent's transcripts.

## Row schema

One proposed action per row, with what the classifier sees (`spec/scenario_schema.md`): `env`, `cwd`, `user_messages`,
`prior_actions`, `action {tool, input}`, `label allow|block`, `rule` (exact rule name), `user_requested yes|no`, `variation`,
`rationale`. Rows that are contrastive twins share a `family` and are split together.

## Sources used for v1 (8,772 rows / 9.0M tokens)

| source | rows | how |
|---|---|---|
| synthetic scenario families, 13 rule groups + benign traffic | 4,357 | an LLM writes contrastive families per rule group from the rule text (consent added / consent vague / scope changed / env changed / carry-over / obfuscated / benign lookalike); a **blind** relabel by the same model family keeps rows where the label agrees (93.6-99.6% per group) and drops the rest |
| recorded auto-mode decisions | 3,099 | `tools/extract_transcripts.py` over `~/.claude/projects/**/*.jsonl`: every tool call made in auto mode becomes a row (allowed calls are the label; classifier denials carry the rule from the denial text). Blocks are rare (0.26% of calls) and oversampled 4x |
| replay | 1,316 | decider's own `teacher_data/commands.jsonl`, trained toward the parent with KL |

External datasets were surveyed and partly converted for evaluation only (`tools/convert_external.py`):
`alex-remedios/swe-chat-auto-mode` (real Claude Code sessions), `yannelli/cli-command-risk`, `tomngdev/shell-safety-v1.1`,
`AI45Research/ATBench-Codex`, AmPermBench, R-Judge and others - see `notes` in the project history; licences vary and are
recorded in `data/ext/README.md` (not published).

## Privacy

- Real transcripts stay on the machine; the extractor redacts secret-looking strings and is read-only.
- The synthetic scenarios are fictional, use placeholder hosts (`*.example`, `attacker.example`) and fake secrets
  (`AKIAIOSFODNN7EXAMPLE`), and were written to be recognisable but non-operational.
- Records from the deployed system (the hub integration) carry the user's messages and tool inputs; `tools/import_hub.py`
  turns them into rows for retraining, and they stay private too.

## Rebuild

```bash
# 1. synthetic families (needs an OpenAI-compatible LLM)
.venv/bin/python tools/gen_llm.py gen --groups deletion,security_controls,code_provenance,... --workers 4
.venv/bin/python tools/gen_llm.py judge --stems ds_deletion,... --workers 4     # blind relabel

# 2. your own transcripts (read-only, redacted)
.venv/bin/python tools/extract_transcripts.py --out data/real/all.jsonl --claude-md-out data/real/claude_md.jsonl
.venv/bin/python tools/real_stats.py

# 3. dataset: family-disjoint split 85/7.5/7.5, real blocks x4, replay 15%
.venv/bin/python tools/build_dataset.py

# 4. retrain with labels from the deployed system
.venv/bin/python tools/import_hub.py --db data/automode.sqlite --out data/hub/records.jsonl
```
