"""Smoke test + latency for decider-4b on this machine.

    systemd-run --user --scope --collect -p MemoryHigh=60G -p MemoryMax=70G -- \
        .venv/bin/python scripts/check_infer.py [--model models/decider-4b] [--skip-graphs]

1. README example through decider.infer.Decider (eager) and through plain transformers (same numbers expected).
2. Latency of a ~1500-token auto-mode-like context with 2 questions (binary + 15 options): eager vs CUDA graphs
   (decider.engine.Engine, with and without torch.compile).
3. Zero-shot sanity check on hand-written auto-mode cases.
"""
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from automode_cases import long_context, RULE_OPTIONS, ZERO_SHOT_CASES   # noqa: E402


def timeit(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    return f"p50 {np.median(ts):7.1f} ms  p90 {np.percentile(ts, 90):7.1f} ms  min {ts.min():7.1f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/decider-4b")
    ap.add_argument("--skip-graphs", action="store_true")
    ap.add_argument("--skip-compile", action="store_true")
    a = ap.parse_args()
    from decider.infer import Decider
    from decider.prompt import build
    from decider.infer import Example, Q, _NoShuffle
    out = {}

    # ---- 1. README example, eager -----------------------------------------------------------------------------------------
    t0 = time.time(); d = Decider(a.model, use_graphs=False); print(f"[load] eager Decider {time.time()-t0:.1f}s; T={d.T} by_type={d.T_by_type}")
    qs = [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}]
    r = d.decide("My card was charged twice for the same purchase.", qs)
    print("[readme decide]", json.dumps([{k: v for k, v in x.items() if k != "probs_list"} for x in r]))
    out["readme_decide"] = [x["probs"] for x in r]
    so = d.system_one(
        {"ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
         "refund_policy": "Duplicate charges are eligible for a refund."},
        {"department": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "billing": {"what": "Charges, invoices", "not_for": "delivery"}, "other": None}},
         "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
         "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
    print("[readme system_one]", json.dumps(so["answers"]))

    # ---- 1b. plain transformers path (README of the model card) --------------------------------------------------------------
    m = d.m.lm; tok = d.m.tok
    prompt = ("Context:\nMy card was charged twice for the same purchase.\n\n"
              "Question: Which department should handle this?\nOptions:\n(A) billing\n(B) technical support\n(C) sales\nAnswer: (")
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = m(**ids).logits[0, -1]
    letters = [tok.encode(L, add_special_tokens=False)[0] for L in "ABC"]
    p_plain = torch.softmax(logits[letters].float() / 1.110, -1).tolist()
    r1 = d.decide("My card was charged twice for the same purchase.", qs[:1])[0]["probs_list"]
    print(f"[plain transformers] {p_plain}  vs Decider single-question {r1}  max|diff|={max(abs(x-y) for x, y in zip(p_plain, r1)):.2e}")
    out["plain_vs_decider_maxdiff"] = max(abs(x - y) for x, y in zip(p_plain, r1))

    # ---- 2. latency: ~1500-token context, binary + 15-option question ---------------------------------------------------------
    ctx = long_context(tok, target_tokens=1500)
    lat_qs = [{"question": "Should the proposed action be blocked?", "options": ["no", "yes"]},
              {"question": "Which rule does the proposed action violate?", "options": RULE_OPTIONS[:15]}]
    exs = Example(ctx, [Q(q["question"], q["options"], 0) for q in lat_qs])
    item = build(exs, tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)
    item["types"] = ["noul", "choice"]
    n_tok = len(item["ids"]); print(f"[latency] row length {n_tok} tokens, {len(item['slots'])} slots")
    out["latency_row_tokens"] = n_tok
    from decider.model import collate
    b = collate([item], tok.pad_token_id); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    with torch.no_grad():
        f_eager = lambda: d.m.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"])
        out["eager_masked"] = timeit(f_eager); print("[latency eager, masked forward (Decider use_graphs=False)]", out["eager_masked"])
        p_eager = torch.softmax(f_eager(), -1).cpu()
        # 8 rows at once (batch throughput)
        b8 = collate([item] * 8, tok.pad_token_id); b8 = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b8.items()}
        out["eager_masked_b8"] = timeit(lambda: d.m.slot_logits(b8["input_ids"], b8["attention_mask"], b8["slot_idx"], b8["slot_batch"], b8["nopts"]), n=10, warm=2)
        print("[latency eager, batch of 8 rows]", out["eager_masked_b8"])
    print(f"[mem] eager peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")

    # ---- 3. zero-shot auto-mode cases -----------------------------------------------------------------------------------------
    zs = []
    for c in ZERO_SHOT_CASES:
        res = d.system_one(c["state"], c["questions"])["answers"]
        zs.append(dict(name=c["name"], expect=c["expect"], answers=res))
        brief = {k: (v.get("noul") if v["type"] == "noul" else {kk: vv for kk, vv in sorted(v["probabilities"].items(), key=lambda x: -x[1])[:3]}) for k, v in res.items()}
        print(f"[zero-shot] {c['name']:42s} expect={c['expect']:5s} {json.dumps(brief)}")
    out["zero_shot"] = zs
    # the same cases as one packed row (all questions behind one copy of the state)
    for c in ZERO_SHOT_CASES[:3]:
        res = d.system_one(c["state"], c["questions"], independent=False)["answers"]
        print(f"[zero-shot packed] {c['name']:35s} " + json.dumps({k: (v.get("noul") if v["type"] == "noul" else v["choice"]) for k, v in res.items()}))
    del d, m; torch.cuda.empty_cache()

    # ---- 2b. CUDA graphs ----------------------------------------------------------------------------------------------------
    if not a.skip_graphs:
        from decider.engine import Engine
        for comp in ([False] if a.skip_compile else [False, True]):
            t0 = time.time(); eng = Engine(a.model, compile=comp); print(f"[load] Engine compile={comp} {time.time()-t0:.1f}s")
            t0 = time.time(); p_g = torch.cat(eng.score_items([item])); print(f"[graph] first call (capture{' + compile' if comp else ''}) {time.time()-t0:.1f}s")
            key = f"graphs_compile={comp}"
            out[key] = timeit(lambda: eng.score_items([item])); print(f"[latency CUDA graph, compile={comp}]", out[key])
            out[key + "_b8"] = timeit(lambda: eng.score_items([item] * 8), n=10, warm=2); print(f"[latency CUDA graph, compile={comp}, batch 8]", out[key + "_b8"])
            print(f"[graph vs eager] max |dp| = {(p_g[:, :15] - p_eager[:, :15]).abs().max():.4f}")
            del eng; torch.cuda.empty_cache()
    os.makedirs("logs", exist_ok=True)
    json.dump(out, open("logs/check_infer.json", "w"), indent=1, default=str)
    print("[done] logs/check_infer.json")


if __name__ == "__main__":
    main()
