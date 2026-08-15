"""Correctness smoke test for the fused NVFP4 kernels, before any timing.

Compares one real routed expert against the N3-validated numpy reference.
No timing figure is produced here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron import reference as ref  # noqa: E402
from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4  # noqa: E402
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
HIDDEN = 2688
MOE_INTERMEDIATE = 1856
SEED = 20260814


def main() -> int:
    import cupy as cp

    print("cupy", cp.__version__, "cc", cp.cuda.Device(0).compute_capability)

    index = ShardIndex(MODEL_DIR)
    capture = json.loads((OUT_DIR / "n3_official_route_capture.json").read_text(encoding="utf-8"))
    expert = int(capture["indices"][0][0])
    prefix = f"backbone.layers.1.mixer.experts.{expert}"

    rng = np.random.default_rng(SEED)
    hidden = (rng.standard_normal((1, HIDDEN)) * 0.5)
    norm_w = index.get_float32("backbone.layers.1.norm.weight")
    x_np = ref.rms_norm(hidden, norm_w, index.config["layer_norm_epsilon"])[0]

    up_w = index.dequantize_linear(f"{prefix}.up_proj")
    down_w = index.dequantize_linear(f"{prefix}.down_proj")
    expected = ref.mlp_relu2(x_np[None, :], up_w, down_w)[0]

    fused = FusedNVFP4()
    up_codes = cp.asarray(index.read_raw(f"{prefix}.up_proj.weight"))
    up_scales = cp.asarray(index.read_raw(f"{prefix}.up_proj.weight_scale"))
    up_g = index.get_scalar(f"{prefix}.up_proj.weight_scale_2")
    down_codes = cp.asarray(index.read_raw(f"{prefix}.down_proj.weight"))
    down_scales = cp.asarray(index.read_raw(f"{prefix}.down_proj.weight_scale"))
    down_g = index.get_scalar(f"{prefix}.down_proj.weight_scale_2")

    x = cp.asarray(x_np, dtype=cp.float32)
    act = cp.zeros(MOE_INTERMEDIATE, dtype=cp.float32)
    out = cp.zeros(HIDDEN, dtype=cp.float32)

    fused.expert(up_codes, up_scales, up_g, down_codes, down_scales, down_g,
                 x, act, out, HIDDEN, MOE_INTERMEDIATE)
    cp.cuda.Device(0).synchronize()
    got = cp.asnumpy(out).astype(np.float64)

    # intermediate check too
    act_expected = np.maximum(x_np @ up_w.T, 0.0) ** 2
    act_got = cp.asnumpy(act).astype(np.float64)

    rel_act = float(np.linalg.norm(act_got - act_expected) / np.linalg.norm(act_expected))
    rel_out = float(np.linalg.norm(got - expected) / np.linalg.norm(expected))

    print(f"activation rel_l2 : {rel_act:.6e}")
    print(f"expert out rel_l2 : {rel_out:.6e}")
    print(f"finite            : {bool(np.isfinite(got).all())}")
    print(f"PASS              : {rel_out <= 1e-5 and rel_act <= 1e-5}")
    return 0 if (rel_out <= 1e-5 and rel_act <= 1e-5) else 3


if __name__ == "__main__":
    sys.exit(main())
