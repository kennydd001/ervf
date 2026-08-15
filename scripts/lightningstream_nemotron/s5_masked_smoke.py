"""S5 smoke: transpose exactness + masked down GEMV vs fused reference.

Component-level check on ONE MoE layer (layer 1), before any full run:
  T1  down_panel_major is a pure permutation: independent inverse mapping
      reproduces the original nibble matrix and scale matrix exactly.
  T2  masked GEMV (mapped-host panel-major) vs the existing fused row-major
      GEMV on a real routed call: rel_l2 reported, plus max |diff|.
  T3  zero-copy sanity: the kernel reading a mapped host pointer produces
      finite values and matches the device-resident copy bit-for-bit.

No performance claims here; a per-call latency print is informational only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import (  # noqa: E402
    DOWN_PANEL_BYTES, LightningRuntime, down_panel_major)

LAYER = 1
EXPERT = 3


def main() -> int:
    import cupy as cp

    rt = LightningRuntime(REPO_ROOT / "models" / "nemotron_3_5_lightning",
                          contexts_max=512, verbose=False)
    idx = rt.index
    pre = f"backbone.layers.{LAYER}.mixer.experts.{EXPERT}"

    # ------------------------------------------------------------- T1
    codes = idx.read_raw(f"{pre}.down_proj.weight")
    scales = idx.read_raw(f"{pre}.down_proj.weight_scale")
    block = down_panel_major(codes, scales)                    # (116, 24192)

    # Independent inverse: rebuild the (2688, 1856) nibble matrix and the
    # (2688, 116) scale matrix from the panel-major block WITHOUT reusing
    # down_panel_major's internals.
    sc_back = block[:, :2688].T                                # (2688, 116)
    packed = block[:, 2688:].reshape(116, 16, 1344)
    nib_back = np.empty((2688, 1856), dtype=np.uint8)
    cols = np.empty((116, 16, 2688), dtype=np.uint8)
    cols[..., 0::2] = packed & 15
    cols[..., 1::2] = packed >> 4
    nib_back = cols.transpose(2, 0, 1).reshape(2688, 1856)
    nib_orig = np.empty((2688, 1856), dtype=np.uint8)
    dc = codes.reshape(2688, 928)
    nib_orig[:, 0::2] = dc & 15
    nib_orig[:, 1::2] = dc >> 4
    t1_codes = bool(np.array_equal(nib_back, nib_orig))
    t1_scales = bool(np.array_equal(sc_back, scales.reshape(2688, 116)))
    print(f"T1 transpose pure permutation: codes={t1_codes} scales={t1_scales}")
    if not (t1_codes and t1_scales):
        return 3

    # ------------------------------------------------- T2/T3 on the GPU
    rt.load_routed_bank(layers=[LAYER])
    bank = rt.bank[LAYER]

    # Component-level smoke: a synthetic normed input suffices here; the
    # preregistered C3 gate uses REAL routed calls in the full-run verifier.
    rng = np.random.default_rng(7)
    rt.normed = rt.cp.asarray(rng.standard_normal(2688).astype(np.float32))

    up_codes = cp.asarray(bank["up_codes"][EXPERT * 2494464:(EXPERT + 1) * 2494464])
    up_scales = cp.asarray(bank["up_scales"][EXPERT * 311808:(EXPERT + 1) * 311808])
    g_up = float(bank["globals"][EXPERT, 1])
    g_dn = float(bank["globals"][EXPERT, 0])
    rt.fused.gemv_into(rt.act[:1856], up_codes, up_scales, rt.normed, g_up,
                       1856, 2688, apply_relu2=True)

    # reference: existing fused row-major down GEMV, codes uploaded to device
    dn_codes = cp.asarray(codes)
    dn_scales = cp.asarray(scales)
    ref = cp.zeros(2688, dtype=cp.float32)
    rt.fused.gemv_into(ref, dn_codes, dn_scales, rt.act[:1856], g_dn,
                       2688, 1856, apply_relu2=False)

    # masked from mapped host bank
    out_h = cp.zeros(2688, dtype=cp.float32)
    ptr = bank["down_base_ptr"] + EXPERT * DOWN_PANEL_BYTES
    rt.fused.down_masked_into(out_h, ptr, rt.act[:1856], rt.mstate,
                              g_dn, 2688, 1856)
    # masked from a DEVICE copy of the same panel-major block (T3 bis)
    block_dev = cp.asarray(
        bank["down_pm"][EXPERT * DOWN_PANEL_BYTES:(EXPERT + 1) * DOWN_PANEL_BYTES])
    out_d = cp.zeros(2688, dtype=cp.float32)
    rt.fused.down_masked_into(out_d, block_dev.data.ptr, rt.act[:1856],
                              rt.mstate, g_dn, 2688, 1856,
                              gather_from_host=False)

    # A2: gather sparse bytes from mapped host into the mirror, then compute
    out_g = cp.zeros(2688, dtype=cp.float32)
    rt.fused.down_masked_into(out_g, ptr, rt.act[:1856], rt.mstate,
                              g_dn, 2688, 1856, gather_from_host=True)

    cp.cuda.Device(0).synchronize()
    ref_np, h_np, d_np, g_np = map(cp.asnumpy, (ref, out_h, out_d, out_g))
    rel = float(np.linalg.norm(h_np - ref_np) / (np.linalg.norm(ref_np) + 1e-30))
    bit = bool(np.array_equal(h_np.view(np.uint32), d_np.view(np.uint32)))
    bit_g = bool(np.array_equal(h_np.view(np.uint32), g_np.view(np.uint32)))
    z = float((cp.asnumpy(rt.act[:1856]) == 0).mean())
    print(f"T2 masked vs fused reference: rel_l2 {rel:.3e} "
          f"max|d| {float(np.abs(h_np-ref_np).max()):.3e} (act zeros {z:.3f})")
    print(f"T3 mapped-host == device-resident bitwise: {bit}")
    print(f"T3b A2 gather-mirror == direct bitwise: {bit_g}")

    # informational latency: A2 path vs direct-host, 200 reps
    for name, kw in (("A2 gather+device", dict(gather_from_host=True)),
                     ("A direct host", dict(gather_from_host=False))):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(200):
            rt.fused.down_masked_into(out_h, ptr, rt.act[:1856], rt.mstate,
                                      g_dn, 2688, 1856, **kw)
        cp.cuda.Device(0).synchronize()
        print(f"    {name}: {(time.perf_counter_ns()-t0)/200/1e3:.1f} us/call")

    ok = t1_codes and t1_scales and rel < 1e-6 and bit and bit_g
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
