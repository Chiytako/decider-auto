"""Build data/build/{train,val,test}.jsonl for scripts/train_lora.py from the verified synthetic rows, the real auto-mode
records, optional external rows and decider replay rows. Also writes data/build/stats.md.

    .venv/bin/python tools/build_dataset.py                       # defaults below
    .venv/bin/python tools/build_dataset.py --real_allows 3000 --replay_frac 0.15 --budget 1800

Sources
  synthetic  data/synth/raw/<g>.jsonl joined by id with the auditor's data/synth/judged/<g>.jsonl. A row is kept when the
             auditor's label equals the generator's label. For blocks the rule label is kept when the two rules fall in the
             same group (--rule_exact: only when they are identical); when the groups differ the row keeps its block label but
             Q_group is left unanswered and Q_rule is dropped. Q_req is answered with the generator's user_requested unless
             the auditor gave a different one. A group whose judged file is missing is skipped (reported). Rows the
             auditor has not judged yet are dropped. Split by FAMILY (contrastive twins never cross splits): 85 / 7.5 / 7.5,
             per group file, deterministic in the family id.
  real       data/real/all.jsonl, train_eligible only, sessions from data/real/split.json. Train side -> train: every block
             plus a stratified sample of allows (--real_allows, strata = tool x main/subagent, exact duplicates skipped).
             Eval side -> test only: every block plus a uniform sample of allows (--real_test_allows; 0 = all).
             Git context is dropped from a share of the real train rows (--real_git_dropout) so its presence does not become a
             "this is real data" shortcut (synthetic rows have none).
  external   data/ext/*.jsonl when present (source = file stem). Rows in the scenario schema (user_messages, prior_actions,
             action, label, rule?, env?) or already in the train row format (state + questions). Their user_requested is
             not used (not annotated for external sources), so they carry no Q_req.
             Sources named swe_chat* or *_eval / *_test go to test only; the others go to train, capped per source
             (--ext_cap, blocks first up to half of the cap).
  replay     data/ext/replay_*.jsonl (rows with `replay: true`, from tools/convert_external.py), else
             data/external/decider_commands/commands.jsonl, as `replay: true` rows (KL toward the frozen parent in
             train_lora.py), sized to --replay_frac of the final train set. Replay files are never used as ext train rows.

Row format (scripts/train_lora.py): {"state": str, "questions": [...], "replay": bool, "task": str, "meta": {...}}.
`meta` is ignored by the trainer and used by scripts/eval.py (source, id, label, group, rule, user_requested, n_tokens).
Q_req is present only where the label is known (synthetic rows).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from automode import view as V                                  # noqa: E402
from automode import questions as QS                            # noqa: E402

SYNTH = os.path.join(ROOT, "data", "synth")
REAL = os.path.join(ROOT, "data", "real")
EXT = os.path.join(ROOT, "data", "ext")
REPLAY = os.path.join(ROOT, "data", "external", "decider_commands", "commands.jsonl")
OUT = os.path.join(ROOT, "data", "build")


def h01(s, salt="split"):
    """Deterministic hash of a string in [0, 1)."""
    return int(hashlib.sha1(f"{salt}:{s}".encode()).hexdigest()[:12], 16) / 16 ** 12


def read_jsonl(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass                           # a file being appended concurrently may end in a partial line
    return out


def yn(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    v = str(v).strip().lower()
    return "yes" if v in ("yes", "true", "1") else "no" if v in ("no", "false", "0") else None


def make_row(scn, label, group, rule, user_requested, meta, tok, budget, include_git=True, task="automode"):
    """Scenario + labels -> train_lora row (state text, packed questions with answers, meta)."""
    v = V.render(scn, tok=tok, budget=budget, include_git=include_git)
    qs = QS.questions_for(label=label, group=group, rule=rule, user_requested=user_requested)
    m = dict(meta, label=label, group=group if label == "block" else None, rule=rule if label == "block" else None,
             user_requested=user_requested, n_tokens=v.n_tokens, q=[k for k, _ in qs])
    return {"state": v.text, "questions": [q for _, q in qs], "replay": False, "task": task, "meta": m}


# ---------------------------------------------------------------------------------------------------------------------
# synthetic

def load_synth(a, tok, log, stats):
    out = {"train": [], "val": [], "test": []}
    raws = sorted(glob.glob(os.path.join(SYNTH, "raw", "*.jsonl")))
    for path in raws:
        g = os.path.splitext(os.path.basename(path))[0]
        jpath = os.path.join(SYNTH, "judged", f"{g}.jsonl")
        raw = read_jsonl(path)
        if not raw:
            log(f"[synth] {g}: raw file empty, skipped"); stats["synth_skipped_groups"].append(f"{g} (empty)"); continue
        if not os.path.exists(jpath):
            log(f"[synth] {g}: no judged file, group SKIPPED ({len(raw)} raw rows)")
            stats["synth_skipped_groups"].append(f"{g} (no judged file, {len(raw)} raw rows)"); continue
        judged = {j.get("id"): j for j in read_jsonl(jpath)}
        kept, drops = [], collections.Counter()
        for r in raw:
            j = judged.get(r.get("id"))
            if j is None:
                drops["not judged yet"] += 1; continue
            if r.get("label") not in ("allow", "block") or "action" not in r or not r.get("family"):
                drops["malformed"] += 1; continue
            if j.get("label") != r["label"]:
                drops["label disagrees"] += 1; continue
            if a.min_confidence and j.get("confidence") == "low":
                drops["auditor confidence low"] += 1; continue
            group = rule = None
            if r["label"] == "block":
                rg, ra = QS.canon_rule(r.get("rule")), QS.canon_rule(j.get("rule"))
                gg, ga = QS.RULE_GROUP.get(rg), QS.RULE_GROUP.get(ra)
                if rg is None:
                    drops["block without a valid rule"] += 1; continue
                if rg == ra:
                    group, rule = gg, rg
                elif gg is not None and gg == ga and not a.rule_exact:
                    group, rule = gg, rg; stats["synth_rule_same_group_only"] += 1
                elif gg is not None and gg == ga:
                    group = gg; stats["synth_rule_same_group_only"] += 1
                else:
                    stats["synth_group_disagrees"] += 1          # keep the block label, leave group/rule unanswered
            ur = yn(r.get("user_requested"))
            ja = yn(j.get("user_requested"))
            if ur is not None and ja is not None and ja != ur:
                ur = None; stats["synth_req_disagrees"] += 1
            kept.append((r, group, rule, ur, j.get("confidence")))
        for k, n in drops.items():
            stats["synth_drops"][k] += n
        # family split per group file: order families by hash, cut by count
        fams = sorted({r["family"] for r, *_ in kept}, key=lambda f: h01(f))
        n = len(fams); n_tr = round(n * a.synth_train); n_va = round(n * (1 - a.synth_train) / 2)
        if n >= 3 and n_va == 0:
            n_va = 1
        if n >= 3 and n - n_tr - n_va <= 0:
            n_tr -= 1
        split_of = {f: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, f in enumerate(fams)}
        for r, group, rule, ur, conf in kept:
            sp = split_of[r["family"]]
            meta = dict(source="synth", id=r["id"], family=r["family"], file_group=g, env=r.get("env"),
                        variation=r.get("variation"), confidence=conf, split=sp)
            out[sp].append(make_row(r, r["label"], group, rule, ur, meta, tok, a.budget, task=f"automode/synth/{g}"))
        log(f"[synth] {g}: {len(raw)} raw, {len(judged)} judged, kept {len(kept)} rows in {n} families "
            f"(train {n_tr}, val {n_va}, test {n - n_tr - n_va}); dropped {dict(drops)}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# real

def real_meta(r, side):
    return dict(source="real", id=r["id"], family=r["session_id"], env=V.REAL_ENV, tool=r["tool_name"],
                subagent=bool(r.get("is_subagent")), block_source=r.get("block_source"), split=side)


def load_real(a, tok, log, stats):
    split = json.load(open(os.path.join(REAL, "split.json")))
    tr_s, ev_s = set(split["train_sessions"]), set(split["eval_sessions"])
    rows = [r for r in read_jsonl(os.path.join(REAL, "all.jsonl")) if r.get("train_eligible") and r.get("label") in ("allow", "block")]
    rng = random.Random(a.seed)
    out = {"train": [], "test": []}

    def emit(r, side, include_git=True):
        rule = QS.canon_rule(r.get("block_rule")) if r["label"] == "block" else None
        group = QS.RULE_GROUP.get(rule) if rule else None
        meta = real_meta(r, side)
        meta["git_shown"] = include_git and bool(r.get("git_context"))
        return make_row(r, r["label"], group, rule, None, meta, tok, a.budget, include_git=include_git, task="automode/real")

    # train side: all blocks + stratified allows
    tr = [r for r in rows if r["session_id"] in tr_s]
    blocks = [r for r in tr if r["label"] == "block"]
    allows = [r for r in tr if r["label"] == "allow" and not r.get("dup_of")]
    strata = collections.defaultdict(list)
    for r in allows:
        tool = r["tool_name"] if r["tool_name"] in ("Bash", "WebFetch", "Write", "Edit", "WebSearch") else \
            ("mcp" if r["tool_name"].startswith("mcp__") else "other")
        strata[(tool, bool(r.get("is_subagent")))].append(r)
    want = min(a.real_allows, len(allows)); picked = []
    for key in sorted(strata):                      # proportional allocation, at least 5 per stratum when available
        pool = strata[key]; k = min(len(pool), max(5, round(want * len(pool) / max(1, len(allows)))))
        picked += rng.sample(pool, k)
    rng.shuffle(picked); picked = picked[:want] if len(picked) > want else picked
    for r in blocks * a.real_block_repeat + picked:   # repeats differ in git dropout and option order
        out["train"].append(emit(r, "train", include_git=rng.random() >= a.real_git_dropout))
    log(f"[real] train side: {len(tr)} eligible ({len(blocks)} blocks); kept {len(blocks)} blocks + {len(picked)} allows")
    # eval side: test only
    ev = [r for r in rows if r["session_id"] in ev_s]
    eb = [r for r in ev if r["label"] == "block"]; ea = [r for r in ev if r["label"] == "allow"]
    if a.real_test_allows and len(ea) > a.real_test_allows:
        ea = rng.sample(ea, a.real_test_allows)
    for r in eb + ea:
        out["test"].append(emit(r, "test"))
    log(f"[real] eval side: {len(ev)} eligible ({len(eb)} blocks); test gets {len(eb)} blocks + {len(ea)} allows")
    stats["real_eval_allow_sample_rate"] = len(ea) / max(1, sum(r["label"] == "allow" for r in ev))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# external

def is_test_only(src):
    return src.startswith("swe_chat") or src.endswith("_eval") or src.endswith("_test")


def load_ext(a, tok, log, stats):
    out = {"train": [], "test": []}
    files = sorted(f for f in glob.glob(os.path.join(EXT, "*.jsonl")) if not os.path.basename(f).startswith("replay"))
    if not files:
        log(f"[ext] no files under {EXT}"); return out
    rng = random.Random(a.seed + 1)
    for path in files:
        src = os.path.splitext(os.path.basename(path))[0]
        rows = read_jsonl(path)
        side = "test" if is_test_only(src) else "train"
        conv, bad = [], 0
        for i, r in enumerate(rows):
            rid = str(r.get("id", f"{src}-{i}"))
            if "state" in r and "questions" in r:                   # already a train row
                rr = dict(r); rr.setdefault("replay", False); rr.setdefault("task", f"automode/ext/{src}")
                rr["meta"] = dict(r.get("meta") or {}, source=f"ext/{src}", id=rid, split=side)
                conv.append(rr); continue
            label = r.get("label")
            if isinstance(label, bool):
                label = "block" if label else "allow"
            if label not in ("allow", "block") or not (r.get("action") or r.get("tool_name")):
                bad += 1; continue
            rule = QS.canon_rule(r.get("rule") or r.get("block_rule"))
            group = QS.RULE_GROUP.get(rule) if rule else (r.get("group") if r.get("group") in QS.GROUP_LABEL else None)
            meta = dict(source=f"ext/{src}", id=rid, family=r.get("family") or r.get("session_id") or rid,
                        env=r.get("env") if isinstance(r.get("env"), str) else None, split=side)
            conv.append(make_row(r, label, group, rule, None, meta, tok, a.budget,
                                 task=f"automode/ext/{src}"))
        if side == "train" and a.ext_cap and len(conv) > a.ext_cap:
            bl = [x for x in conv if x["meta"].get("label") == "block"]; al = [x for x in conv if x["meta"].get("label") != "block"]
            nb = min(len(bl), a.ext_cap // 2)
            conv = rng.sample(bl, nb) + rng.sample(al, min(len(al), a.ext_cap - nb))
        out[side] += conv
        log(f"[ext] {src}: {len(rows)} rows -> {side} {len(conv)} (unusable {bad})")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# replay

def load_replay(n, seed, log):
    files = sorted(glob.glob(os.path.join(EXT, "replay*.jsonl"))) or ([REPLAY] if os.path.exists(REPLAY) else [])
    if n <= 0 or not files:
        return []
    rows = [r for f in files for r in read_jsonl(f) if r.get("state") and r.get("questions")]
    rng = random.Random(seed + 2)
    pick = rng.sample(rows, min(n, len(rows)))
    out = []
    for r in pick:
        qs = [{k: v for k, v in q.items() if k in ("type", "instructions", "criteria", "answer")} for q in r["questions"]]
        out.append({"state": r["state"], "questions": qs, "replay": True, "task": r.get("task") or "replay/decider_commands",
                    "meta": dict(source="replay", group=r.get("group") or (r.get("source_meta") or {}).get("group"))})
    log(f"[replay] {len(out)} rows from {len(rows)} ({', '.join(os.path.relpath(f, ROOT) for f in files)})")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# stats

def pct(xs, qs=(50, 90, 95, 99, 100)):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {q: None for q in qs}
    return {q: xs[min(len(xs) - 1, int(len(xs) * q / 100))] for q in qs}


def write_stats(splits, stats, log_lines, tok, a):
    L = ["# data/build statistics", "", f"Built by `tools/build_dataset.py` (budget {a.budget} tokens, seed {a.seed}).", ""]
    L += ["## Rows by split / source / label", "", "| split | source | allow | block | replay | total |", "|---|---|--:|--:|--:|--:|"]
    for sp, rows in splits.items():
        c = collections.Counter((r["meta"].get("source"), "replay" if r["replay"] else r["meta"].get("label")) for r in rows)
        for src in sorted({s for s, _ in c}):
            al, bl, rp = c[(src, "allow")], c[(src, "block")], c[(src, "replay")]
            L.append(f"| {sp} | {src} | {al} | {bl} | {rp} | {al + bl + rp} |")
        L.append(f"| **{sp}** | all | {sum(v for (s, l), v in c.items() if l == 'allow')} | "
                 f"{sum(v for (s, l), v in c.items() if l == 'block')} | {sum(v for (s, l), v in c.items() if l == 'replay')} | **{len(rows)}** |")
    L += ["", "## Blocks by group (answered Q_group)", "", "| group | " + " | ".join(splits) + " |", "|---|" + "--:|" * len(splits)]
    for g in QS.GROUP_IDS + [None]:
        cells = [sum(1 for r in rows if not r["replay"] and r["meta"].get("label") == "block" and r["meta"].get("group") == g)
                 for rows in splits.values()]
        L.append(f"| {g or '(unknown group)'} | " + " | ".join(map(str, cells)) + " |")
    L += ["", "## Blocks by rule (train)", ""]
    rc = collections.Counter(r["meta"].get("rule") for r in splits["train"] if not r["replay"] and r["meta"].get("label") == "block")
    L += [", ".join(f"{k or '(none)'} {v}" for k, v in rc.most_common())]
    L += ["", "## Questions answered", "", "| split | Q_block | Q_group | Q_req (yes / no) | Q_rule |", "|---|--:|--:|--:|--:|"]
    for sp, rows in splits.items():
        nr = [r for r in rows if not r["replay"]]
        def answered(k):
            return sum(1 for r in nr for kk, q in zip(r["meta"].get("q", []), r["questions"]) if kk == k and "answer" in q)
        ry = sum(1 for r in nr if r["meta"].get("user_requested") == "yes" and "req" in r["meta"].get("q", []))
        rn = sum(1 for r in nr if r["meta"].get("user_requested") == "no" and "req" in r["meta"].get("q", []))
        L.append(f"| {sp} | {answered('block')} | {answered('group')} | {answered('req')} ({ry} / {rn}) | {answered('rule')} |")
    L += ["", "## State token length (decider-4b tokenizer)", "",
          "| split | source | n | p50 | p90 | p95 | p99 | max |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for sp, rows in splits.items():
        by = collections.defaultdict(list)
        for r in rows:
            n = r["meta"].get("n_tokens")
            if n is None:
                n = len(tok.encode(r["state"], add_special_tokens=False)); r["meta"]["n_tokens"] = n
            by[r["meta"].get("source")].append(n)
        for src in sorted(by):
            p = pct(by[src]); L.append(f"| {sp} | {src} | {len(by[src])} | {p[50]} | {p[90]} | {p[95]} | {p[99]} | {p[100]} |")
    syn = [r for rows in splits.values() for r in rows if r["meta"].get("source") == "synth"]
    if syn:
        L += ["", "## Synthetic rows by variation", ""]
        vc = collections.Counter((r["meta"].get("variation"), r["meta"]["label"]) for r in syn)
        L += [", ".join(f"{v}/{l} {n}" for (v, l), n in sorted(vc.items(), key=lambda x: -x[1]))]
    L += ["", "## Verification and filtering", ""]
    L.append(f"- synthetic groups skipped: {', '.join(stats['synth_skipped_groups']) or 'none'}")
    L.append(f"- synthetic rows dropped: {dict(stats['synth_drops']) or 'none'}")
    L.append(f"- synthetic blocks whose auditor rule differs but is in the same group: {stats['synth_rule_same_group_only']}"
             f" ({'rule label dropped' if a.rule_exact else 'generator rule kept'})")
    L.append(f"- synthetic blocks whose auditor group differs (group/rule left unanswered): {stats['synth_group_disagrees']}")
    L.append(f"- synthetic Q_req answers dropped because the auditor disagreed: {stats['synth_req_disagrees']}")
    if "real_eval_allow_sample_rate" in stats:
        L.append(f"- real eval-side allows in test are a uniform {stats['real_eval_allow_sample_rate']:.1%} sample "
                 "(FPR on real data is unbiased; block prevalence is not the natural one)")
    L += ["", "## Build log", "", "```"] + log_lines + ["```", ""]
    open(os.path.join(OUT, "stats.md"), "w").write("\n".join(L))


# ---------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=V.DEFAULT_BUDGET, help="state token budget (automode.view)")
    ap.add_argument("--synth_train", type=float, default=0.85, help="train share of synthetic families (val = test = rest / 2)")
    ap.add_argument("--rule_exact", action="store_true", help="keep a synthetic rule label only when the auditor names the same rule")
    ap.add_argument("--min_confidence", action="store_true", help="drop synthetic rows the auditor marked low-confidence")
    ap.add_argument("--real_allows", type=int, default=3000, help="real train-side allows (stratified sample)")
    ap.add_argument("--real_block_repeat", type=int, default=4, help="copies of each real train-side block (they are rare)")
    ap.add_argument("--real_test_allows", type=int, default=1500, help="real eval-side allows in test (0 = all)")
    ap.add_argument("--real_git_dropout", type=float, default=0.5, help="share of real train rows rendered without git context")
    ap.add_argument("--ext_cap", type=int, default=2000, help="max rows per external train source")
    ap.add_argument("--replay_frac", type=float, default=0.15, help="replay share of the final train set")
    ap.add_argument("--no_real", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    gi = os.path.join(OUT, ".gitignore")              # the built rows contain real transcript text: keep them out of git
    if not os.path.exists(gi):
        open(gi, "w").write("*.jsonl\n")
    tok = V.get_tokenizer()
    log_lines = []

    def log(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); log_lines.append(msg)

    stats = dict(synth_skipped_groups=[], synth_drops=collections.Counter(), synth_rule_same_group_only=0,
                 synth_group_disagrees=0, synth_req_disagrees=0)
    splits = {"train": [], "val": [], "test": []}
    for sp, rows in load_synth(a, tok, log, stats).items():
        splits[sp] += rows
    if not a.no_real:
        for sp, rows in load_real(a, tok, log, stats).items():
            splits[sp] += rows
    for sp, rows in load_ext(a, tok, log, stats).items():
        splits[sp] += rows
    n_lab = len(splits["train"])
    n_rep = round(a.replay_frac / max(1e-9, 1 - a.replay_frac) * n_lab) if a.replay_frac > 0 else 0
    splits["train"] += load_replay(n_rep, a.seed, log)
    rng = random.Random(a.seed)
    for sp in splits:
        rng.shuffle(splits[sp])
    # a family must never appear in two splits
    fam = {sp: {r["meta"].get("family") for r in rows if r["meta"].get("source") == "synth"} for sp, rows in splits.items()}
    assert not (fam["train"] & fam["val"] or fam["train"] & fam["test"] or fam["val"] & fam["test"]), "family leak"
    for sp, rows in splits.items():
        with open(os.path.join(OUT, f"{sp}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"[write] {sp}: {len(rows)} rows -> data/build/{sp}.jsonl")
    write_stats(splits, stats, log_lines, tok, a)
    print("[write] data/build/stats.md")


if __name__ == "__main__":
    main()
