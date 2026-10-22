# Evaluation

## Sets

| set | rows | blocks | what it is |
|---|---|---|---|
| synthetic held-out | 387 | 173 | held-out contrastive families, family-disjoint from training; all 13 groups |
| real recorded | 1,513 | 13 | auto-mode decisions recorded by the training machine, sessions after the split date; every block kept, allows sampled |
| swe relabel | 548 | 29 | real Claude Code sessions from `alex-remedios/swe-chat-auto-mode`; 300 random rows plus the 300 with the highest v0 scores, relabelled by a second LLM (top-300: 27 blocks) |

## Results (p_block threshold 0.5)

| model | synth AUROC / recall / FPR | real AUROC / recall / FPR | swe-relabel AUROC / recall@top / FPR@top |
|---|---|---|---|
| decider-4b v2.1 zero-shot | - | 0.752 / 38.5% / 5.1% | 0.769 / 22.2% / 13.1% |
| v0 (7 synthetic groups + real) | 0.978 / 88.1% / 3.3% | 0.883 / 0% / 0.20% | 0.820 / 59.3% / 32.6% |
| **v1** | **0.992 / 92.5% / 1.4%** | **0.878 / 0% / 0.07%** | **0.830 / 51.9% / 28.1%** |

v1: rule-group accuracy on blocks 83%, exact rule 86% given the group, ECE 0.024 (synthetic). Allows answer "none of these" with
p >= 0.5 in 98.6% of cases; user-request accuracy 91.3% where known.

## What the numbers mean

- **Synthetic performance is strong but partly self-referential**: the generator and the blind relabel are the same model
  family, so agreement rates (93.6-99.6%) measure consistency, not truth.
- **The real set is the warning**: at 0.5 v1 blocks none of the 13 real blocks (12 score below 0.05). They are clustered in 2-3
  sessions, mostly workflow subagents reading other users' data or touching a live service - exactly the context the training
  rows underrepresent. This is why the deployed system uses stage 2 on anything not near-zero.
- **False blocks**: 0.07% of real allows and 0% of the random swe rows; the flagged swe subset keeps a 28% rate, but that
  subset was chosen by the previous model's scores (selection bias).
- Thresholds for the asymmetric cost are set on validation: stage 1 pass 0.02 (0.002 with custom rules), block 0.5, and stage
  2 decides the rest.

## Reproduce

```bash
.venv/bin/python scripts/eval.py --model runs/v1/model --name v1
.venv/bin/python scripts/eval.py --model runs/v1/model --name v1_swerelabel \
    --data data/build/test_swe_relabel.jsonl --latency_n 0
```

`scripts/eval.py` re-asks every row with the inference question set in canonical option order and writes `metrics.json`,
`metrics.md`, `predictions.jsonl` and calibration records (`python -m decider.calibrate`).
