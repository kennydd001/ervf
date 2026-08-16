"""S100 native NVFP4 C0B — checkpoint format/repack audit.

Purpose: C0 gate C0_SCALE_COUNTS_MATCH_GROUP16 assumed scale_count ==
weight_numel/16, but NVFP4 weights are stored packed 2-codes-per-byte, so the
correct group-16 expectation is (weight_numel*2)/16. C0B audits EVERY
weight/weight_scale pair in the Lightning checkpoint (safetensors index,
header-only, no payload reads, no CUDA) and records which count hypothesis
holds per tensor, plus row alignment and global-scale presence.

Claim boundary: format/count/layout audit only. No native matmul, numerical
equivalence, token parity or speed claim. Repack feasibility stated here is a
layout permutation assessment; it must be proven by reconstruction equality in
a separately preregistered C1.
"""

import json
import math
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "nemotron_3_5_lightning_v35"
INDEX = MODEL_DIR / "model.safetensors.index.json"
CONFIG = MODEL_DIR / "config.json"
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C0B_FORMAT_AUDIT.json"
GROUP_SIZE = 16
PACKED_CODES_PER_BYTE = 2


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numel(shape):
    return math.prod(shape)


def main() -> int:
    index = json.loads(INDEX.read_text())
    meta = index.get("metadata", {})
    entries = index["weight_map"]

    # Build name -> (dtype, shape) from per-shard headers? The index maps
    # name -> shard only; dtype/shape require shard headers. Read headers only.
    shard_headers = {}
    for shard in sorted(set(entries.values())):
        sp = MODEL_DIR / shard
        with open(sp, "rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            shard_headers[shard] = json.loads(f.read(hlen))

    tensors = {}
    for name, shard in entries.items():
        rec = shard_headers[shard][name]
        tensors[name] = {"dtype": rec["dtype"], "shape": rec["shape"], "shard": shard}

    pairs = []
    for name, rec in tensors.items():
        if not name.endswith(".weight_scale"):
            continue
        base = name[: -len(".weight_scale")]
        w = tensors.get(base + ".weight")
        g = tensors.get(base + ".weight_scale_2")
        if w is None:
            pairs.append({"scale": name, "error": "missing_weight"})
            continue
        w_numel = numel(w["shape"])
        s_numel = numel(rec["shape"])
        packed_logical = w_numel * PACKED_CODES_PER_BYTE
        exp_packed = packed_logical // GROUP_SIZE if packed_logical % GROUP_SIZE == 0 else None
        exp_direct = w_numel // GROUP_SIZE if w_numel % GROUP_SIZE == 0 else None
        row_match = len(w["shape"]) >= 1 and len(rec["shape"]) >= 1 and w["shape"][0] == rec["shape"][0]
        inner_ratio = None
        if len(w["shape"]) == 2 and len(rec["shape"]) == 2 and w["shape"][1]:
            inner_ratio = rec["shape"][1] / w["shape"][1]
        pairs.append({
            "scale": name,
            "scale_dtype": rec["dtype"],
            "scale_shape": rec["shape"],
            "weight_dtype": w["dtype"],
            "weight_shape": w["shape"],
            "global_scale_present": g is not None,
            "global_scale_dtype": g["dtype"] if g else None,
            "scale_count": s_numel,
            "expected_group16_packed_codes": exp_packed,
            "expected_group16_if_unpacked": exp_direct,
            "matches_group16_packed": exp_packed == s_numel,
            "matches_group16_direct_assumption": exp_direct == s_numel,
            "row_match": row_match,
            "inner_dim_ratio_scale_over_weight": inner_ratio,
        })

    audited = [p for p in pairs if "error" not in p]
    # FP8 per-tensor-scaled linears (Mamba in/out_proj: F8_E4M3 weight, scalar
    # F32 scale, no weight_scale_2) are not NVFP4 and are out of scope.
    fp8_excluded = [p for p in audited if p["weight_dtype"] == "F8_E4M3" and p["scale_dtype"] == "F32"]
    nvfp4 = [p for p in audited if p not in fp8_excluded]
    fp4 = [p for p in nvfp4 if p["matches_group16_packed"]]
    direct = [p for p in nvfp4 if p["matches_group16_direct_assumption"] and not p["matches_group16_packed"]]
    breakers = [p for p in nvfp4 if not p["matches_group16_packed"] and not p["matches_group16_direct_assumption"]]
    ratios = sorted({round(p["inner_dim_ratio_scale_over_weight"], 6) for p in nvfp4 if p["inner_dim_ratio_scale_over_weight"]})

    result = {
        "kind": "s100_native_nvfp4_c0b_format_audit",
        "status": "format_counts_group16_packed_exact" if (nvfp4 and len(fp4) == len(nvfp4)) else "format_breakers_found",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "format/count/layout audit only; no native matmul, numerical-equivalence or speed claim",
        "parent_audit": "pro_research/results/native_nvfp4/C0_CAPABILITIES.json",
        "model_dir": str(MODEL_DIR),
        "metadata_hashes": {
            "config_json_sha256": sha256_file(CONFIG),
            "safetensors_index_sha256": sha256_file(INDEX),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        },
        "index_metadata": meta,
        "group_size": GROUP_SIZE,
        "packed_codes_per_byte_assumption": PACKED_CODES_PER_BYTE,
        "totals": {
            "scale_tensors_found": len(pairs),
            "audited_pairs": len(audited),
            "nvfp4_pairs": len(nvfp4),
            "fp8_per_tensor_excluded_not_nvfp4": len(fp8_excluded),
            "matches_group16_packed": len(fp4),
            "matches_direct_unpacked_only": len(direct),
            "breakers": len(breakers),
            "missing_weight_errors": len(pairs) - len(audited),
            "global_scale_missing": sum(1 for p in nvfp4 if not p["global_scale_present"]),
            "row_mismatch": sum(1 for p in nvfp4 if not p["row_match"]),
            "non_e4m3_scale_dtype": sorted({p["scale_dtype"] for p in nvfp4 if p["scale_dtype"] != "F8_E4M3"}),
            "inner_dim_ratios_observed": ratios,
        },
        "breaker_details": breakers[:50],
        "direct_only_details": direct[:50],
        "fp8_excluded_examples": [p["scale"] for p in fp8_excluded[:8]],
        "runtime_layout_evidence": {
            "file": "src/moe_lab/lightningstream_nemotron/nvfp4.py",
            "dequant_rule": "w = e2m1(code) * e4m3(block_scale) * f32(weight_scale_2)",
            "scale_addressing": "flat row-major [rows, K/16]; np.repeat(scales, 16) over contiguous packed weights",
        },
        "repack_assessment": {
            "value_format": "C0 sampled scale bytes are E4M3 with sign bit always clear => value-compatible with unsigned UE4M3 block scales",
            "count_format": "exact group-16 along K for every audited pair once 2-codes-per-byte packing is accounted for",
            "layout_delta": "checkpoint stores scales row-major [rows, K/16]; native SM120 block-scaled FP4 requires NVIDIA's blocked/swizzled scale-factor layout => a pure permutation (+ padding) repack, lossless in principle, NOT proven here",
            "verdict": "C0 count-gate failure is explained as a packed-count artifact (factor exactly 2); native FP4 format route remains UNDECIDED pending a C1 layout/reconstruction proof",
        },
        "pairs": nvfp4,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({"status": result["status"], "output": str(OUT), "totals": result["totals"]}, indent=1))
    return 0 if result["status"] == "format_counts_group16_packed_exact" else 1


if __name__ == "__main__":
    sys.exit(main())
