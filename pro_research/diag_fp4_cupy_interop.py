"""Gating feasibility test for putting native FP4 into the production runtime.

C2c measured the only remaining format-preserving win: native Blackwell FP4
beats our ERVF kernel by 2.52x on lm_head and 1.68x on shared_down, worth
-1.275 ms/token (51.0 -> 54.6 tok/s). Nothing else measured this session is
both positive and free of a quantisation change.

But it is measured in a *different environment*. The runtime is CuPy on Torch
2.9.1+cu128 (.venv-nemotron); the FP4 API only exists in Torch 2.12.1+cu132
(.venv-fp4-c2b), which has no CuPy. Integration therefore needs both libraries
alive in ONE process, sharing device memory without copying:

    CuPy owns the weights, the LRU cache, the graph, everything.
    Torch would own only the FP4 GEMM call.
    The activation and the output must pass between them by POINTER.

If that handshake does not work, native FP4 cannot reach production without
migrating the whole runtime to a new CUDA toolchain -- which risks the working
50 tok/s stack. So this is the question to answer before writing any
integration code, and it is cheap to answer.

Three things are checked, in order, and any failure stops the rest:

  T1  CuPy and Torch 2.12.1+cu132 both initialise in one process on this GPU
  T2  a CuPy array is visible to Torch with no copy (same device pointer)
      via __cuda_array_interface__ / DLPack, and vice versa
  T3  a real FP4 scaled_mm consuming a CuPy-owned buffer produces the exact
      value the pure-Torch C2b path produced (256.0 for the all-ones case)

T3 is the one that matters: pointer sharing that type-checks but silently
copies, or hands over a stale view, would look fine and be wrong.

Run inside the venv under test:
    .venv-fp4-c2b/Scripts/python.exe pro_research/diag_fp4_cupy_interop.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "FP4_CUPY_INTEROP.json"


def main() -> int:
    rec: dict = {"kind": "diag_fp4_cupy_interop", "tests": {}}

    # ---- T1: both libraries alive in one process ------------------------
    try:
        import torch
        rec["torch_version"] = torch.__version__
        rec["torch_cuda"] = torch.version.cuda
        import cupy as cp
        rec["cupy_version"] = cp.__version__
        rec["cupy_cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        a = cp.arange(16, dtype=cp.float32)
        t = torch.arange(16, device="cuda", dtype=torch.float32)
        cp.cuda.Device(0).synchronize()
        torch.cuda.synchronize()
        rec["tests"]["T1_both_initialise"] = {
            "pass": True,
            "cupy_sum": float(a.sum()),
            "torch_sum": float(t.sum().item()),
            "device": torch.cuda.get_device_name(0),
        }
    except Exception as exc:
        rec["tests"]["T1_both_initialise"] = {
            "pass": False, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()}
        rec["verdict"] = "T1_failed_libraries_cannot_coexist"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps(rec, indent=2))
        return 2

    # ---- T2: zero-copy pointer handover both directions ------------------
    try:
        cup = cp.arange(256, dtype=cp.uint8)
        cup_ptr = int(cup.data.ptr)
        # CuPy -> Torch
        t_view = torch.as_tensor(cup, device="cuda")
        t_ptr = int(t_view.data_ptr())
        # Torch -> CuPy
        tt = torch.arange(256, device="cuda", dtype=torch.uint8)
        c_view = cp.asarray(tt)
        same_fwd = (cup_ptr == t_ptr)
        same_bwd = (int(tt.data_ptr()) == int(c_view.data.ptr))
        # mutation must be visible through the other view if it is truly shared
        cup[0] = 99
        cp.cuda.Device(0).synchronize()
        sees_mutation = int(t_view[0].item()) == 99
        rec["tests"]["T2_zero_copy"] = {
            "pass": bool(same_fwd and same_bwd and sees_mutation),
            "cupy_to_torch_same_ptr": same_fwd,
            "torch_to_cupy_same_ptr": same_bwd,
            "torch_view_sees_cupy_mutation": sees_mutation,
        }
        if not rec["tests"]["T2_zero_copy"]["pass"]:
            raise RuntimeError("pointer handover is not zero-copy")
    except Exception as exc:
        rec["tests"].setdefault("T2_zero_copy", {})
        rec["tests"]["T2_zero_copy"].update({
            "pass": False, "error": f"{type(exc).__name__}: {exc}"})
        rec["verdict"] = "T2_failed_no_zero_copy_handover"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps(rec, indent=2))
        return 2

    # ---- T3: a real FP4 GEMM over CuPy-owned memory -----------------------
    try:
        import torch.nn.functional as F
        ScalingType = getattr(F, "ScalingType")
        SwizzleType = getattr(F, "SwizzleType")
        M, N, K = 1, 128, 256

        def ceil(x, q):
            return ((x + q - 1) // q) * q
        sfp = ceil(K // 16, 4)

        # CuPy owns every buffer; Torch only borrows pointers.
        a_cp = cp.full((M, K // 2), 0x22, dtype=cp.uint8)
        b_cp = cp.full((N, K // 2), 0x22, dtype=cp.uint8)
        # e4m3 1.0 is 0x38, NOT 0x3C. 0x3C is s=0 E=0111 m=100 = (1+4/8)*2^0 =
        # 1.5, which is why the first run of this test returned 576.0 instead of
        # 256.0: 256 * 1.5 * 1.5 = 576 exactly. The arithmetic was right and the
        # constant was wrong. C2b/C2c did not have this problem because they use
        # torch.ones(dtype=float8_e4m3fn) rather than a hand-written byte.
        E4M3_ONE = 0x38
        sa_cp = cp.full((ceil(M, 128), sfp), E4M3_ONE, dtype=cp.uint8)
        sb_cp = cp.full((sfp, ceil(N, 128)), E4M3_ONE, dtype=cp.uint8)

        a = torch.as_tensor(a_cp, device="cuda").view(torch.float4_e2m1fn_x2)
        b = torch.as_tensor(b_cp, device="cuda").view(torch.float4_e2m1fn_x2).t()
        sa = torch.as_tensor(sa_cp, device="cuda").view(torch.float8_e4m3fn)
        sb = torch.as_tensor(sb_cp, device="cuda").view(torch.float8_e4m3fn)

        out = F.scaled_mm(a, b,
                          scale_a=sa, scale_recipe_a=ScalingType.BlockWise1x16,
                          scale_b=sb, scale_recipe_b=ScalingType.BlockWise1x16,
                          swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                          swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                          output_dtype=torch.bfloat16, use_fast_accum=False)
        torch.cuda.synchronize()
        val = float(out.flatten()[0].item())
        allsame = bool(torch.all(out == out.flatten()[0]).item())
        # C2b's pure-Torch known-value case produced exactly 256.0 for K=256
        rec["tests"]["T3_fp4_gemm_over_cupy_memory"] = {
            "pass": bool(allsame and val == 256.0),
            "value": val, "expected": 256.0, "all_elements_equal": allsame,
            "note": "every operand allocated by CuPy, borrowed by Torch as a pointer view; matches C2b's pure-Torch known-value result",
        }
    except Exception as exc:
        rec["tests"]["T3_fp4_gemm_over_cupy_memory"] = {
            "pass": False, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()}

    ok = all(t.get("pass") for t in rec["tests"].values())
    rec["verdict"] = ("integration_feasible" if ok else "integration_blocked")
    rec["what_this_means"] = (
        "feasible: CuPy can keep owning the runtime while Torch executes only the "
        "FP4 GEMM on borrowed pointers, so native FP4 can be integrated without "
        "migrating the whole stack"
        if ok else
        "blocked: native FP4 cannot reach production without migrating the runtime "
        "to the newer CUDA toolchain, which risks the working 50 tok/s stack")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
