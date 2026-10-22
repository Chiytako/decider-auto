"""Latency of one decision row on this GPU, uncontended runs only.

    systemd-run --user --scope --collect -p MemoryHigh=60G -p MemoryMax=70G -- .venv/bin/python scripts/bench_latency.py

eager (DecisionModel.slot_logits, masked) at several lengths; CUDA graphs (decider.engine.Engine, no torch.compile) with and
without the conv patch; and the incremental case an auto-mode classifier sees: the transcript prefix is cached once
(attention KV + delta-net states) and each new proposed action only runs its suffix (action + questions)."""
import json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from automode_cases import long_context, RULE_OPTIONS                                    # noqa: E402
from decider.infer import Example, Q, _NoShuffle
from decider.prompt import build
from decider.model import collate


def timeit(fn, n=20, warm=4):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def row(tok, target):
    ctx = long_context(tok, target)
    qs = [Q("Should the proposed action be blocked?", ["no", "yes"], 0), Q("Which rule does the proposed action violate?", RULE_OPTIONS[:15], 0)]
    return build(Example(ctx, qs), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "models/decider-4b"
    res = {}
    from decider.engine import Engine
    from decider import shared_prefix as SP
    eng = Engine(path, compile=False, conv_patch=False); tok = eng.tok; m = eng.m
    items = {L: row(tok, L) for L in (150, 400, 900, 1500, 3800)}
    with torch.no_grad():
        for L, it in items.items():
            n = len(it["ids"]); b = collate([it], tok.pad_token_id); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            res[f"eager_{n}"] = timeit(lambda: m.slot_logits(b["input_ids"], b["attention_mask"], b["slot_idx"], b["slot_batch"], b["nopts"]))
            if n <= 2048:
                eng.score_items([it]); res[f"graph_noconvpatch_{n}"] = timeit(lambda: eng.score_items([it]))
            print({k: round(v, 1) for k, v in res.items() if k.endswith(f"_{n}")}, flush=True)
        # incremental: cache the transcript prefix once, then score only the suffix for a new action
        it = items[1500]; ids = it["ids"]
        suffix_len = 220                                                                              # new action (~60 tok) + 2 question blocks
        pre = torch.tensor(ids[:-suffix_len], device="cuda")[None]; suf = torch.tensor(ids[-suffix_len:], device="cuda")[None]
        cache = eng.core(input_ids=pre, use_cache=True).past_key_values
        def step():
            fork = SP.fork_cache(cache, 1)
            h = eng.core(input_ids=suf, past_key_values=fork, use_cache=True).last_hidden_state
            return h[0, -1] @ eng.W.T
        res[f"incremental_prefix{pre.shape[1]}_suffix{suffix_len}"] = timeit(step)
        res[f"prefix_only_{pre.shape[1]}"] = timeit(lambda: eng.core(input_ids=pre, use_cache=True).past_key_values)
        full = eng.core(input_ids=torch.tensor(ids, device="cuda")[None]).last_hidden_state[0, -1] @ eng.W.T
        diff = (torch.softmax(step().float()[:15], -1) - torch.softmax(full.float()[:15], -1)).abs().max().item()
        res["incremental_vs_full_maxdp"] = diff
        print({k: v for k, v in res.items() if "incremental" in k or "prefix_only" in k}, flush=True)
    del eng; torch.cuda.empty_cache()
    eng = Engine(path, compile=False, conv_patch=True)
    with torch.no_grad():
        for L in (400, 1500):
            it = items[L]; n = len(it["ids"]); eng.score_items([it]); res[f"graph_convpatch_{n}"] = timeit(lambda: eng.score_items([it]))
    print({k: round(v, 1) if isinstance(v, float) else v for k, v in res.items()})
    json.dump(res, open("logs/bench_latency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
