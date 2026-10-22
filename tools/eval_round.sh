#!/bin/bash
# v1 on the main test set, then base / v0 / v1 on the relabeled swe-chat subset.
cd "$(dirname "$0")/.."
RUN="systemd-run --user --scope --collect -p MemoryMax=40G -- .venv/bin/python scripts/eval.py"
$RUN --model runs/v1/model --name v1
for m in models/decider-4b:base runs/v0/model:v0 runs/v1/model:v1; do
  $RUN --model ${m%%:*} --name ${m##*:}_swerelabel --data data/build/test_swe_relabel.jsonl --latency_n 0
done
echo "[eval_round] done"
