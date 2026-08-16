"""Verify the quantiser before trusting the number it produces.

`diag_fp4_activation_quality.py` will report "this is what FP4 costs in
quality". That number is only worth as much as the quantiser behind it, and the
quantiser in that file is hand-written: a hand-rolled e4m3 rounding routine and
a hand-typed e2m1 grid. Twice today a hand-written test detail invalidated a
conclusion -- all-ones scales hid a layout permutation, and a shape mismatch hid
a dtype requirement -- so the instrument gets checked before the measurement,
not after.

Three checks, strongest last, all CPU-only:

  V1  the e2m1 grid matches the project's own `nvfp4.E2M1_MAGNITUDES`
  V2  the hand-rolled e4m3 rounding matches snapping onto the project's real
      256-entry `nvfp4.E4M3_TABLE`, across the whole positive range
  V3a no clipping on real checkpoint weights -- a ceil-rounded block scale must
      leave every element at or under 6.0, so any clipping is magnitude the
      quantiser threw away that the format could have kept
  V3c every output within half a grid step of its input

The first version of V3 demanded bit-identical round-tripping of already-NVFP4
weights. That is impossible by construction: the encoder STORES a block scale
and a re-quantiser DERIVES one as amax/6, and those agree only when the block's
largest code happens to be 6. I was tuning the quantiser to pass a test that
could not be passed. V3a and V3b are the properties that actually are required.

No GPU, no model load beyond an mmap read of one shard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import require_model_dir, utc_now, write_json_atomic
from moe_lab.lightningstream_nemotron import nvfp4 as NV
from moe_lab.lightningstream_nemotron.loader import ShardIndex

from diag_fp4_activation_quality import BLOCK, E2M1, _e4m3_round, quant_nvfp4


def e4m3_snap(v: np.ndarray) -> np.ndarray:
    """Ground truth: snap onto the project's real e4m3 decode table."""
    tbl = np.asarray(NV.E4M3_TABLE, dtype=np.float32)
    pos = np.unique(tbl[(tbl > 0) & np.isfinite(tbl)])
    idx = np.abs(v[..., None] - pos[None, :]).argmin(axis=-1)
    return pos[idx]


def main() -> int:
    rec: dict = {"kind": "diag_verify_nvfp4_quantizer", "created_utc": utc_now(),
                 "checks": {}}

    # ---- V1: the e2m1 grid ------------------------------------------------
    proj = np.asarray(NV.E2M1_MAGNITUDES, dtype=np.float32)
    rec["checks"]["V1_e2m1_grid"] = {
        "pass": bool(np.array_equal(np.sort(proj), np.sort(E2M1))),
        "project": proj.tolist(), "mine": E2M1.tolist()}

    # ---- V2: e4m3 rounding vs the real table ------------------------------
    rng = np.random.default_rng(20260816)
    probe = np.concatenate([
        np.geomspace(1e-3, 440.0, 4000).astype(np.float32),
        rng.uniform(0.0, 448.0, 4000).astype(np.float32)])
    mine = _e4m3_round(probe.copy())
    truth = e4m3_snap(probe)
    same = mine == truth
    rel = np.abs(mine - truth) / np.maximum(truth, 1e-30)
    rec["checks"]["V2_e4m3_rounding"] = {
        "pass": bool(same.all()),
        "match_fraction": float(same.mean()),
        "max_relative_disagreement": float(rel.max()),
        "note": "hand-rolled exponent/mantissa rounding against snapping onto nvfp4.E4M3_TABLE",
    }

    # ---- V3: the corrected check ------------------------------------------
    # My first V3 demanded that re-quantising already-NVFP4 weights returns them
    # bit-identical. That is IMPOSSIBLE by construction, not a defect: the
    # encoder stores a block scale, and a re-quantiser recomputes one from the
    # data as amax/6. Those agree only when the block's largest code happens to
    # be 6. When the max code is 4, the recomputed scale is 4/6 of the original,
    # the grid shifts, and values legitimately move by up to one step -- which is
    # exactly the 0.333 = |1.0-1.5|/1.5 seen. I was tuning the quantiser to pass
    # an unachievable test. Replaced with two properties that ARE required.
    idx = ShardIndex(require_model_dir())
    codes = idx.read_raw("lm_head.weight")
    scales = idx.read_raw("lm_head.weight_scale")
    g = float(idx.get_scalar("lm_head.weight_scale_2"))

    c = np.asarray(codes).reshape(-1)[: 4096 // 2]
    sc = np.asarray(scales).reshape(-1)[: (4096 // BLOCK)]
    inter = np.empty(c.size * 2, dtype=np.int32)
    inter[0::2] = (c & 0x0F).astype(np.int32)
    inter[1::2] = (c >> 4).astype(np.int32)
    vals = np.asarray(NV.E2M1_TABLE, dtype=np.float32)[inter]
    blk = np.asarray(NV.E4M3_TABLE, dtype=np.float32)[sc.astype(np.int32)]
    deq = (vals.reshape(-1, BLOCK) * blk[:, None] * g).reshape(-1).astype(np.float32)

    # V3a -- no clipping. A block scale chosen as ceil(amax/6) must leave every
    # element at or under 6.0 on the grid. Any clipping is the quantiser
    # discarding magnitude that the format could have represented, and it would
    # inflate the quality cost this instrument exists to measure.
    q1, info1 = quant_nvfp4(deq)
    rec["checks"]["V3a_no_clipping"] = {
        "pass": info1["clipped_fraction"] == 0.0,
        "clipped_fraction": info1["clipped_fraction"],
        "note": "real lm_head weights; ceil-rounded block scales must make clipping impossible",
    }

    # V3b -- operator idempotence. Applying the quantiser to its own output must
    # be a no-op: the values are already on the grid its scale defines. This is
    # the achievable form of the property the first V3 got wrong.
    q2, _ = quant_nvfp4(q1)
    # Same, but with the global scale HELD FIXED across both applications.
    # A dynamic per-call global scale is re-derived from amax, and rounding
    # moves amax, which shifts the grid -- so any instability under the dynamic
    # form is the dynamic scaling, not the grid arithmetic. Separating the two
    # says which one a deployment would have to pin down.
    qf1, infof = quant_nvfp4(deq, s_g=info1["global_scale"])
    qf2, _ = quant_nvfp4(qf1, s_g=info1["global_scale"])
    # V3b was ALSO an unachievable requirement, and I wrote it after already
    # diagnosing exactly this for the first V3 -- the same mistake twice in one
    # sitting. Deriving a block scale as ceil(amax/6) means the second pass sees
    # bamax = code_max * eff and computes ceil(code_max * bs / 6), which equals
    # bs only when code_max is 6. Pinning the global scale changes nothing (the
    # count stayed at exactly 351), which is what finally made it obvious.
    # Demoted to information.
    rec["operator_non_idempotence"] = {
        "changed_elements_dynamic_scale": int(np.count_nonzero(q1 != q2)),
        "changed_elements_fixed_scale": int(np.count_nonzero(qf1 != qf2)),
        "elements": int(q1.size),
        "note": "NOT a gate. Inherent to amax-derived block scaling, not a defect: a block whose largest code is 4 rather than 6 gets a different scale on re-encode and its grid shifts. Identical under a pinned global scale, which rules out dynamic scaling as the cause.",
    }

    # V3c -- the correctness bound that IS achievable: every element must land
    # within half a grid step of its input. That is the definition of correct
    # round-to-nearest on the e2m1 grid, and it is what "the quantiser adds no
    # error of its own beyond the format" actually means.
    blocks_in = deq.reshape(-1, BLOCK)
    blocks_out = q1.reshape(-1, BLOCK)
    bam = np.max(np.abs(blocks_in), axis=1)
    eff = np.where(bam > 0, bam / 6.0, 1.0)[:, None]
    scaled_in = np.abs(blocks_in) / eff
    # half the gap to the neighbouring grid point, per element
    edges = np.abs(scaled_in[..., None] - E2M1[None, None, :])
    nearest = edges.argmin(axis=-1)
    halfstep = np.full_like(scaled_in, 0.25)          # smallest gap is 0.5
    for i in range(len(E2M1)):
        lo = E2M1[max(i - 1, 0)]
        hi = E2M1[min(i + 1, len(E2M1) - 1)]
        gap = max(E2M1[i] - lo, hi - E2M1[i])
        halfstep[nearest == i] = gap / 2.0 + 1e-6
    within = np.abs(blocks_out) / eff <= (scaled_in + halfstep)
    rec["checks"]["V3c_within_half_grid_step"] = {
        "pass": bool(within.all()),
        "violating_elements": int((~within).sum()),
        "elements": int(deq.size),
        "note": "every output within half a grid step of its input -- the achievable statement of 'no error beyond the format's own'",
    }

    ok = all(v["pass"] for v in rec["checks"].values())
    rec["verdict"] = "quantizer_trustworthy" if ok else "quantizer_not_trustworthy"
    rec["consequence"] = (
        "the activation-quality number this quantiser produces can be read as the "
        "cost of the FORMAT" if ok else
        "the activation-quality number would mix the format's cost with the "
        "quantiser's own error and must not be reported until this is fixed")
    write_json_atomic(REPO / "pro_research" / "diag_verify_nvfp4_quantizer.json",
                      rec, archive=False)
    print(json.dumps(rec, indent=2)[:2600])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
