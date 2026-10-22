"""Stage-2-style LoRA fine-tune of a decider checkpoint (the stage-2 trainer of decider-4b v2.1 is not public; this is our
re-implementation from the model card: LoRA r64 / alpha 128 on attention + MLP, CE on the slot readout for labelled rows,
KL(p_ref || p_model) toward the frozen parent for replay rows, step loss = sum over answers / number of answers).

The frozen reference costs no extra memory: it is the same PEFT model with the adapter disabled (`disable_adapter()`), run under
no_grad on the same rendered prompt, so the option permutation of the reference and the student is identical.

Row format (JSONL, one decision context per line):
    {"state": str | dict | list,                       # rendered with decider.systemone.render_state (JSON compact, _index arrays)
     "questions": [{"type": "noul" | "choice" | "score", "instructions": str, "criteria": ...,   # /v1/systemone question spec
                    "answer": <option name> | true/false | <level index>,                        # gold (omitted on replay rows)
                    "soft": {<option name>: p, ...}}],                                            # optional soft target (distill)
     "replay": false,                                   # true: loss is KL toward the frozen parent, the answer is ignored
     "task": "automode/..."}
Several questions in one row are packed behind one copy of the state (one slot each, all read from one pass), exactly like
decider.prompt.build; `--split_prob` also emits one answered question alone so the single-question form stays in
distribution. A question without "answer" (and not on a replay row) is rendered but gets no loss, which keeps the question
numbering of packed rows stable. Extra row keys (e.g. "meta" from tools/build_dataset.py) are ignored. Option order is
re-shuffled every epoch (`--no_reshuffle` keeps the first epoch's order).

    systemd-run --user --scope --collect -p MemoryHigh=60G -p MemoryMax=70G -- \
      .venv/bin/python scripts/train_lora.py --data data/train.jsonl --out runs/x --epochs 2
    .venv/bin/python scripts/train_lora.py --smoke --steps 5 --max_tokens 4096 --out runs/smoke     # synthetic 4k-token rows
"""
import argparse, json, math, os, random, sys, time
import torch, torch.nn.functional as F

from decider.model import DecisionModel, collate
from decider.prompt import build
from decider import systemone as S1
from decider.data.core import Example, Q

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",                          # 8 full-attention layers
                "in_proj_qkv", "in_proj_z", "out_proj",                          # 24 gated delta-net layers (in_proj_a/b are 32-wide)
                "gate_proj", "up_proj", "down_proj"]                             # MLP


def row_to_example(row):
    """JSONL row -> (Example with gold indices, per-question soft-target vectors in option order, answer types)."""
    qs, soft, types = [], [], []
    for q in row["questions"]:
        r = S1.render_question(q)
        if q.get("type") == "score":
            gold = int(q["answer"]) if "answer" in q else -1
        elif "answer" in q:
            gold = r["names"].index(q["answer"]) if q.get("type", "choice") != "noul" else int(bool(q["answer"]))
        else:
            gold = -1
        s = None
        if "soft" in q:
            if r["type"] == "noul":                     # noul names are [False, True]; accept yes/no or true/false keys
                m = {{"yes": "true", "no": "false"}.get(str(k).lower(), str(k).lower()): float(v) for k, v in q["soft"].items()}
                s = [m.get(str(n).lower(), 0.0) for n in r["names"]]
            else:
                m = {str(k): float(v) for k, v in q["soft"].items()}
                s = [m.get(str(n), 0.0) for n in r["names"]]
        qs.append(Q(r["question"], r["options"], gold)); soft.append(s); types.append(r["type"])
    return Example(S1.render_state(row["state"]), qs, row.get("task", "row")), soft, types


def make_items(rows, tok, rng, max_options, max_ctx, split_prob):
    items = []
    for i, row in enumerate(rows):
        ex, soft, types = row_to_example(row)
        variants = [(ex, soft)]
        # the single-question variant: only questions that carry a loss (a gold, a soft target, or any on a replay row)
        useful = [k for k, q in enumerate(ex.qs) if row.get("replay") or q.gold >= 0 or soft[k] is not None]
        if len(ex.qs) > 1 and useful and rng.random() < split_prob:
            k = rng.choice(useful); variants.append((Example(ex.context, [ex.qs[k]], ex.task), [soft[k]]))
        for e, s in variants:
            it = build(e, tok, rng, max_options=max_options, max_ctx_tokens=max_ctx)
            # soft targets follow the rendered (shuffled / sub-sampled) option order
            it["soft"] = [None if sv is None else [sv[j] for j in perm] for sv, perm in zip(s, it["perms"])]
            it["replay"] = bool(row.get("replay", False)); it["task"] = e.task; it["ex_id"] = i
            items.append(it)
    return items


def batches_by_tokens(items, max_tokens, rng):
    """Length-sorted buckets (pad to 64), B = max_tokens // T rows per batch."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out, cur, curT = [], [], 0
    for i in order:
        T = -(-len(items[i]["ids"]) // 64) * 64
        if cur and max(curT, T) * (len(cur) + 1) > max_tokens:
            out.append(cur); cur, curT = [], 0
        cur.append(i); curT = max(curT, T)
    if cur: out.append(cur)
    rng.shuffle(out); return out


def loss_terms(logits, ref_logits, items, golds, nopts):
    """Per-answer loss: CE for labelled answers, KL(p_ref || p) for replay rows or answers with a soft target."""
    losses, n_ce, n_kl, ce_sum, kl_sum = [], 0, 0, 0.0, 0.0
    k = 0
    for it in items:
        for j in range(len(it["slots"])):
            lg = logits[k, :nopts[k]]
            soft = it["soft"][j]
            if it["replay"] and ref_logits is not None:
                p_ref = torch.softmax(ref_logits[k, :nopts[k]], -1)
                l = F.kl_div(torch.log_softmax(lg, -1), p_ref, reduction="sum"); n_kl += 1; kl_sum += l.item()
            elif soft is not None:
                p = torch.tensor(soft, device=lg.device, dtype=torch.float32); p = p / p.sum().clamp_min(1e-9)
                l = F.kl_div(torch.log_softmax(lg, -1), p, reduction="sum"); n_kl += 1; kl_sum += l.item()
            elif golds[k] >= 0:
                l = F.cross_entropy(lg[None], golds[k:k + 1]); n_ce += 1; ce_sum += l.item()
            else:
                k += 1; continue
            losses.append(l); k += 1
    if not losses:
        return None, {}
    return torch.stack(losses).mean(), dict(ce=ce_sum / max(n_ce, 1), kl=kl_sum / max(n_kl, 1), n_ce=n_ce, n_kl=n_kl)


def smoke_rows(tok, n, target_tokens, seed=0):
    rng = random.Random(seed); words = "git push rm read edit file test build deploy config secret token branch main origin".split()
    rows = []
    for i in range(n):
        s = " ".join(rng.choice(words) for _ in range(target_tokens))      # ~1 token per word
        ids = tok.encode(s, add_special_tokens=False)[:target_tokens - 120]
        rows.append(dict(state=tok.decode(ids), task="smoke", replay=(i % 4 == 3), questions=[
            dict(type="noul", instructions="Should the proposed action be blocked?", answer=bool(i % 2)),
            dict(type="choice", instructions="Which rule does the proposed action violate?",
                 criteria={f"rule_{j}": None for j in range(15)}, answer=f"rule_{i % 15}")]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/decider-4b")
    ap.add_argument("--data", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=0, help="stop after this many optimizer steps (0: epochs)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=8192, help="tokens per micro-batch (B*T)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_ctx", type=int, default=16384, help="cap on Context tokens (build() keeps the START of the state)")
    ap.add_argument("--max_options", type=int, default=32, help="sub-sample options to this many per question (gold kept); "
                    "keep >= 14 so the auto-mode group question is never sub-sampled")
    ap.add_argument("--no_reshuffle", action="store_true", help="render the items once (option order fixed across epochs)")
    ap.add_argument("--split_prob", type=float, default=0.5)
    ap.add_argument("--no_grad_ckpt", action="store_true", help="faster (1350 tok/s at 4k) but 38 GB peak at 4k tokens per micro-batch")
    ap.add_argument("--bf16_lora", action="store_true", help="bf16 adapter weights and optimizer states (faster, no fp32 master)")
    ap.add_argument("--smoke", action="store_true", help="synthetic rows of --smoke_tokens tokens")
    ap.add_argument("--smoke_tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_merged", action="store_true", help="also write merged bf16 weights + decider_config.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/train.log", "a")
    def log(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log("[args]", json.dumps(vars(a)))
    torch.manual_seed(a.seed); rng = random.Random(a.seed)

    from peft import LoraConfig, get_peft_model
    dm = DecisionModel(a.model, grad_ckpt=not a.no_grad_ckpt).cuda(); tok = dm.tok
    if not a.no_grad_ckpt:
        dm.lm.config.use_cache = False
        dm.lm.enable_input_require_grads()            # checkpointed blocks need a grad-requiring input when the base is frozen
    # fp32 LoRA weights (optimizer master copy) with bf16 compute under autocast: 851 tok/s at 4k tokens on GB10, against 731 with
    # fp32 matmuls in the adapters (PEFT's default) and 981 with bf16 adapters (--bf16_lora, no fp32 master copy)
    dm.lm = get_peft_model(dm.lm, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, target_modules=LORA_TARGETS, bias="none"),
                           autocast_adapter_dtype=not a.bf16_lora)
    ntr = sum(p.numel() for p in dm.lm.parameters() if p.requires_grad); log(f"[lora] trainable {ntr/1e6:.1f}M params")
    # DecisionModel.slot_logits uses self.lm.model / self.lm.lm_head: point them through the PEFT wrapper
    base = dm.lm.base_model.model
    dm_core, dm_head = base.model, base.lm_head

    def slot_logits(b):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = dm_core(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            hs = h[b["slot_batch"], b["slot_idx"]]
            lg = F.linear(hs, dm_head.weight[dm.letters]).float()
        ar = torch.arange(lg.shape[1], device=lg.device)[None, :]
        return lg.masked_fill(ar >= b["nopts"][:, None], float("-inf"))

    rows = smoke_rows(tok, 64, a.smoke_tokens) if a.smoke else [json.loads(l) for l in open(a.data)]
    t0 = time.time(); items = make_items(rows, tok, rng, a.max_options, a.max_ctx, a.split_prob)
    ntok = sum(len(it["ids"]) for it in items); log(f"[data] {len(rows)} rows -> {len(items)} items, {sum(len(it['slots']) for it in items)} answers, {ntok/1e6:.2f}M tokens ({time.time()-t0:.0f}s)")
    params = [p for p in dm.lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    spe = math.ceil(len(batches_by_tokens(items, a.max_tokens, random.Random(0))) / a.accum)
    total = a.steps or int(spe * a.epochs); warm = max(1, int(total * a.warmup_frac))
    log(f"[sched] {spe} steps/epoch, {total} total, warmup {warm}")
    lr_at = lambda s: a.lr * s / warm if s < warm else a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))

    step = micro = 0; dm.lm.train(); t_last = time.time(); tok_acc = 0; stats = []
    torch.cuda.reset_peak_memory_stats()
    epoch = 0
    while step < total:
        if epoch > 0 and not a.no_reshuffle:              # new option permutations / split variants each epoch
            items = make_items(rows, tok, rng, a.max_options, a.max_ctx, a.split_prob)
        epoch += 1
        for bidx in batches_by_tokens(items, a.max_tokens, rng):
            if step >= total: break
            its = [items[i] for i in bidx]
            b = collate(its, tok.pad_token_id); b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            ref = None
            if any(it["replay"] for it in its):
                with torch.no_grad(), dm.lm.disable_adapter():
                    ref = slot_logits(b)
            logits = slot_logits(b)
            loss, st = loss_terms(logits, ref, its, b["golds"], b["nopts"])
            if loss is None: continue
            (loss / a.accum).backward(); micro += 1; tok_acc += b["input_ids"].numel(); stats.append(st)
            if micro % a.accum == 0:
                for g in opt.param_groups: g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                now = time.time(); dt = now - t_last
                if step % 10 == 0 or step <= 5 or step == total:
                    ce = sum(s["ce"] * s["n_ce"] for s in stats) / max(1, sum(s["n_ce"] for s in stats))
                    kl = sum(s["kl"] * s["n_kl"] for s in stats) / max(1, sum(s["n_kl"] for s in stats))
                    log(f"[train] step {step}/{total} ce {ce:.4f} kl {kl:.4f} gn {gn:.2f} lr {lr_at(step):.2e} {dt:.1f}s/step "
                        f"{tok_acc/dt:.0f} tok/s (padded) peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                    stats = []
                t_last = now; tok_acc = 0
    dm.lm.save_pretrained(f"{a.out}/adapter"); log("[save]", f"{a.out}/adapter")
    if a.save_merged:
        merged = dm.lm.merge_and_unload(); merged.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model")
        cfg_src = os.path.join(a.model, "decider_config.json")
        if os.path.exists(cfg_src):
            cfg = json.load(open(cfg_src)); cfg["version"] = "automode-dev"; cfg["base"] = f"{a.model} + LoRA r{a.rank}"
            json.dump(cfg, open(f"{a.out}/model/decider_config.json", "w"), indent=1)
        log("[save]", f"{a.out}/model (merged)")


if __name__ == "__main__":
    main()
