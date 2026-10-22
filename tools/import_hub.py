"""Import GB10 Hub auto-mode records (data/automode.sqlite, gb10/automode.py) as labelled rows for retraining.

Labels come from the people who used it:
  feedback  correct -> the judged decision; should_allow -> allow; should_block -> block; should_ask -> block (unsure = flag)
  outcome   an ask the user approved -> allow-with-consent context is unknown, so kept as "ask_approved" (eval only by default);
            denied -> block
Records without a label are written with label null (useful for shadow-mode statistics, not for training).

    python3 tools/import_hub.py --db data/automode.sqlite --out data/hub/records.jsonl
"""
import argparse
import json
import os
import sqlite3


def label_of(r):
    fb, oc, dec = r["feedback"], r["outcome"], r["decision"]
    if fb == "correct" and dec in ("allow", "block"):
        return dec, "feedback"
    if fb == "should_allow":
        return "allow", "feedback"
    if fb in ("should_block", "should_ask"):
        return "block", "feedback"
    if oc == "denied":
        return "block", "outcome"
    if oc == "approved":
        return "ask_approved", "outcome"
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True, help="the automode.sqlite written by the hub integration")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hub", "records.jsonl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    n, counts = 0, {}
    with open(a.out, "w") as f:
        for r in c.execute("SELECT * FROM records ORDER BY id"):
            st = json.loads(r["state"] or "{}")
            label, src = label_of(r)
            row = {"id": f"hub-{r['id']}", "family": f"hub-{r['user']}-{r['session']}", "source": "hub", "env": "default",
                   "cwd": st.get("cwd"), "user_messages": st.get("user_messages") or [], "prior_actions": st.get("prior_actions") or [],
                   "action": {"tool": r["tool"], "input": json.loads(r["input"] or "{}")}, "label": label, "label_source": src,
                   "judged": {"decision": r["decision"], "p_block": r["p_block"], "group": r["grp"], "rule": r["rule"]},
                   "rule": r["rule"] if label == "block" and r["feedback"] in (None, "correct") else None,
                   "mode": r["mode"], "outcome": r["outcome"], "feedback": r["feedback"], "feedback_note": r["feedback_note"], "ts": r["ts"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            counts[label] = counts.get(label, 0) + 1
    print(f"{n} records -> {a.out}; labels {counts}")


if __name__ == "__main__":
    main()
