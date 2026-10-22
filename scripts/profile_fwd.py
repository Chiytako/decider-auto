"""Where does the eager forward time go?  python scripts/profile_fwd.py [T]"""
import sys, time, torch
from decider.model import DecisionModel
T = int(sys.argv[1]) if len(sys.argv) > 1 else 1664
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3): a @ b
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(10): a @ b
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 10; print(f"[matmul] bf16 8192^3: {2*8192**3/dt/1e12:.1f} TFLOP/s"); del a, b
m = DecisionModel("models/decider-4b", grad_ckpt=False).cuda().eval()
ids = torch.randint(1000, 50000, (1, T), device="cuda"); am = torch.ones_like(ids)
with torch.no_grad():
    for mask in (am, None):
        for _ in range(3): m.lm.model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5): m.lm.model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 5
        print(f"[fwd] T={T} mask={'yes' if mask is not None else 'no'}: {dt*1000:.1f} ms, {2*4.2e9*T/dt/1e12:.1f} TFLOP/s (2*N*T)")
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        m.lm.model(input_ids=ids, attention_mask=am); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
