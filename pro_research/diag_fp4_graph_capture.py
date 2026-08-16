"""The gate that decides whether native FP4 can enter the record path at all.

Our 51.0 tok/s exists because the whole token is captured into ONE CUDA graph.
C2c measured native FP4 beating our ERVF kernel 2.52x on lm_head and 1.68x on
shared_down, and C3A-v2 confirmed it is representationally correct on the real
checkpoint with M=8 free. FP4_CUPY_INTEROP.json showed CuPy and Torch share
device pointers zero-copy in one process.

None of that helps if `F.scaled_mm` cannot be captured. If it cannot, adopting
FP4 means breaking the token graph at every FP4 call -- 23 shared_down plus
lm_head = 24 breaks per token -- and at a measured 3.53 us per launch plus the
loss of graph residency, that would cost more than the 1.275 ms the kernel wins.

Two specific reasons it might fail, both worth distinguishing rather than
lumping into "it errored":

  1. **Stream.** Torch has its own current stream. Unless it is told to use the
     stream CuPy is capturing on, its work goes elsewhere and is simply not in
     the graph -- which can look like success while capturing nothing.
  2. **Allocation.** `F.scaled_mm` has no `out=` parameter, so it allocates its
     result. A cudaMalloc during capture is illegal. Torch's caching allocator
     may serve the block from cache without calling cudaMalloc, which would make
     this work -- but only if the block is already cached, i.e. after a warmup
     of exactly the same shape.

Arms:
  A  CuPy-only capture/replay                     control, must pass
  B  scaled_mm in a CuPy capture, no warmup       tests the allocator path cold
  C  scaled_mm in a CuPy capture, after warmup    the realistic integration
  D  replay C's graph twice and check the output  proves the graph really
                                                  contains the GEMM, not nothing

D is the one that matters: a capture that silently contains no work replays
fine and produces stale data, which would look like a pass.

Run in .venv-fp4-c2b (cupy + torch 2.12.1+cu132).
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "FP4_GRAPH_CAPTURE.json"

E4M3_ONE = 0x38          # NOT 0x3C, which is 1.5
M, N, K = 2, 4096, 2688


def _ceil(x, q):
    return ((x + q - 1) // q) * q


def main() -> int:
    import cupy as cp
    import torch
    import torch.nn.functional as F

    rec: dict = {"kind": "diag_fp4_graph_capture", "arms": {},
                 "torch": torch.__version__, "cupy": cp.__version__}
    ScalingType = getattr(F, "ScalingType")
    SwizzleType = getattr(F, "SwizzleType")

    # ---- A: CuPy-only capture/replay, the control ------------------------
    try:
        s = cp.cuda.Stream(non_blocking=True)
        x = cp.zeros(1024, dtype=cp.float32)
        with s:
            s.begin_capture()
            x += 1.0
            g = s.end_capture()
        for _ in range(3):
            g.launch(s)
        s.synchronize()
        rec["arms"]["A_cupy_only"] = {"pass": bool(float(x[0]) == 3.0),
                                      "value": float(x[0]), "expected": 3.0}
    except Exception as exc:
        rec["arms"]["A_cupy_only"] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}

    # operands, CuPy-owned, Torch borrows pointers
    sfp = _ceil(K // 16, 4)
    a_cp = cp.full((M, K // 2), 0x22, dtype=cp.uint8)
    b_cp = cp.full((N, K // 2), 0x22, dtype=cp.uint8)
    sa_cp = cp.full((_ceil(M, 128), sfp), E4M3_ONE, dtype=cp.uint8)
    sb_cp = cp.full((sfp, _ceil(N, 128)), E4M3_ONE, dtype=cp.uint8)
    a = torch.as_tensor(a_cp, device="cuda").view(torch.float4_e2m1fn_x2)
    b = torch.as_tensor(b_cp, device="cuda").view(torch.float4_e2m1fn_x2).t()
    sa = torch.as_tensor(sa_cp, device="cuda").view(torch.float8_e4m3fn)
    sb = torch.as_tensor(sb_cp, device="cuda").view(torch.float8_e4m3fn)

    def gemm():
        return F.scaled_mm(a, b,
                           scale_a=sa, scale_recipe_a=ScalingType.BlockWise1x16,
                           scale_b=sb, scale_recipe_b=ScalingType.BlockWise1x16,
                           swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                           swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                           output_dtype=torch.bfloat16, use_fast_accum=False)

    expected = float(K)   # all +1 codes, unit scales

    def capture(warm: bool):
        st = cp.cuda.Stream(non_blocking=True)
        ext = torch.cuda.ExternalStream(st.ptr)
        if warm:
            with torch.cuda.stream(ext):
                for _ in range(3):
                    gemm()
            st.synchronize()
        holder = {}
        with st:
            st.begin_capture()
            with torch.cuda.stream(ext):
                holder["out"] = gemm()
            gr = st.end_capture()
        return st, gr, holder

    # ---- B: no warmup ----------------------------------------------------
    try:
        st, gr, h = capture(warm=False)
        gr.launch(st)
        st.synchronize()
        v = float(h["out"].flatten()[0].item())
        rec["arms"]["B_capture_cold"] = {"pass": v == expected, "value": v,
                                         "expected": expected}
    except Exception as exc:
        rec["arms"]["B_capture_cold"] = {
            "pass": False, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-900:]}

    # ---- C + D: warmed capture, and proof the graph really holds the GEMM -
    try:
        st, gr, h = capture(warm=True)
        out = h["out"]
        # Poison the output. If the graph contains no real work, the replay
        # leaves the poison in place and a naive value check would still see a
        # "correct" number from the capture-time execution.
        out.zero_()
        torch.cuda.synchronize()
        gr.launch(st)
        st.synchronize()
        v1 = float(out.flatten()[0].item())
        out.zero_()
        torch.cuda.synchronize()
        gr.launch(st)
        st.synchronize()
        v2 = float(out.flatten()[0].item())
        allsame = bool(torch.all(out == out.flatten()[0]).item())
        rec["arms"]["C_capture_warm"] = {"pass": v1 == expected, "value": v1,
                                         "expected": expected}
        rec["arms"]["D_replay_recomputes"] = {
            "pass": bool(v1 == expected and v2 == expected and allsame),
            "replay1": v1, "replay2": v2, "all_elements_equal": allsame,
            "note": "output zeroed before each replay, so a matching value proves the graph re-executes the GEMM rather than replaying an empty capture",
        }
    except Exception as exc:
        rec["arms"]["C_capture_warm"] = {
            "pass": False, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-900:]}

    passed = {k: v.get("pass") for k, v in rec["arms"].items()}
    decisive = bool(passed.get("A_cupy_only") and passed.get("D_replay_recomputes"))
    rec["gates"] = passed
    rec["verdict"] = "fp4_is_graph_capturable" if decisive else "fp4_not_graph_capturable"
    rec["what_this_means"] = (
        "native FP4 can live inside the captured token graph, so the measured "
        "-1.275 ms/token (51.0 -> 54.6) is reachable without giving up graph residency"
        if decisive else
        "adopting FP4 would break the token graph at every FP4 call (23 shared_down "
        "+ lm_head = 24 breaks/token at ~3.53 us each plus loss of residency), which "
        "plausibly costs more than the 1.275 ms the kernel wins")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0 if decisive else 2


if __name__ == "__main__":
    raise SystemExit(main())
