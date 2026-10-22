#!/bin/bash
# DeepSeek work queue, run in order after the current generation: shared-host scenarios, blind judging, swe-chat relabel.
cd "$(dirname "$0")/.."
while pgrep -f 'tools/gen_llm.py gen --groups deletion' >/dev/null; do sleep 60; done
PY=.venv/bin/python
$PY tools/gen_llm.py gen --groups shared_host --workers 4
STEMS=$(ls data/synth/raw/ds_*.jsonl | xargs -n1 basename | sed 's/.jsonl//' | paste -sd,)
$PY tools/gen_llm.py judge --stems data_exfil,$STEMS --workers 4
$PY tools/gen_llm.py judge --stems swe_sub --src data/ext_judge/swe_sub.jsonl --dst data/ext_judge/swe_sub.judged.jsonl --workers 4
echo "[queue] done"
