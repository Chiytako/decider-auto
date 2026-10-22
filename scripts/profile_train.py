"""Time forward / backward of one LoRA training micro-batch.  python scripts/profile_train.py [T] [--prof]"""
import sys, time, torch, torch.nn.functional as F
from decider.model import DecisionModel
from peft import LoraConfig, get_peft_model
sys.path.insert(0, "scripts"); from train_lora import LORA_TARGETS
T = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4096
ckpt = "--no_ckpt" not in sys.argv
dm = DecisionModel("models/decider-4b", grad_ckpt=ckpt).cuda()
if ckpt: dm.lm.config.use_cache = False; dm.lm.enable_input_require_grads()
if "--tf32" in sys.argv: torch.backends.cuda.matmul.allow_tf32 = True
lm = get_peft_model(dm.lm, LoraConfig(r=64, lora_alpha=128, target_modules=LORA_TARGETS), autocast_adapter_dtype="--bf16_lora" not in sys.argv); lm.train()
print("[lora dtype]", next(p for n, p in lm.named_parameters() if "lora_A" in n).dtype)
core = lm.base_model.model.model; W = lm.base_model.model.lm_head.weight
ids = torch.randint(1000, 50000, (1, T), device="cuda"); am = torch.ones_like(ids)
def step():
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled="--autocast" in sys.argv):
        h = core(input_ids=ids, attention_mask=am).last_hidden_state
        lg = F.linear(h[0, [-1, -2]], W[dm.letters[:10]]).float()
    loss = F.cross_entropy(lg, torch.tensor([0, 1], device="cuda"))
    torch.cuda.synchronize(); t1 = time.perf_counter(); loss.backward(); torch.cuda.synchronize(); t2 = time.perf_counter()
    return t1 - t0, t2 - t1
for _ in range(2): step()
r = [step() for _ in range(3)]
f = sum(x[0] for x in r) / 3; b = sum(x[1] for x in r) / 3
print(f"[train-step] T={T} ckpt={ckpt} flags={[x for x in sys.argv[2:]]}: fwd {f*1000:.0f} ms, bwd {b*1000:.0f} ms, {T/(f+b):.0f} tok/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
if "--prof" in sys.argv:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25, max_name_column_width=80))
