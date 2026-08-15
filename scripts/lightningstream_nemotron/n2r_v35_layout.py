"""N2R: layout adjudication for the 3.5 Lightning checkpoint.

The decisive question before anything else: is this NVFP4 like Nemotron 3 Nano,
or FP8 as the inline quantization_config suggests? Every downstream component --
decoder, bank builder, fused GEMV, cache record size -- depends on the answer,
and assuming it would be exactly the failure mode this line exists to avoid.
"""

from __future__ import annotations

import collections
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "nemotron_3_5_lightning_v35"
OUT = REPO / "reports" / "lightningstream_nemotron" / "n2r_v35_layout.json"

# Frozen Nemotron 3 Nano facts for comparison.
NANO = {"routed_record_bytes": 5_612_560, "routed_records": 2944,
        "code_bytes": 4_988_928, "scale_bytes": 623_616,
        "quant": "NVFP4 group16 + FP8 block scales + 2x FP32 globals"}


def read_header(p: Path):
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    return json.loads(raw.decode("utf-8")), n


def main() -> int:
    idx_path = MODEL / "model.safetensors.index.json"
    weight_map = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
    shards = sorted(set(weight_map.values()))

    tensors = {}
    for s in shards:
        hdr, _ = read_header(MODEL / s)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            tensors[k] = {**v, "shard": s}

    def nbytes(v):
        a, b = v["data_offsets"]
        return b - a

    dt_bytes = collections.defaultdict(int)
    dt_count = collections.defaultdict(int)
    for v in tensors.values():
        dt_bytes[v["dtype"]] += nbytes(v)
        dt_count[v["dtype"]] += 1

    # --- one routed expert, field by field
    pre = "backbone.layers.1.mixer.experts.0"
    fields = {k[len(pre) + 1:]: tensors[k] for k in tensors if k.startswith(pre + ".")}
    expert_bytes = sum(nbytes(v) for v in fields.values())

    # --- MTP block
    mtp = {k: v for k, v in tensors.items() if k.startswith("mtp.")}
    mtp_bytes = sum(nbytes(v) for v in mtp.values())
    mtp_pre = "mtp.layers.1.mixer.experts.0"
    mtp_fields = {k[len(mtp_pre) + 1:]: mtp[k] for k in mtp if k.startswith(mtp_pre + ".")}

    # --- routed / shared / trunk partition
    routed = shared = trunk = 0
    routed_recs = collections.defaultdict(int)
    for k, v in tensors.items():
        b = nbytes(v)
        if ".mixer.experts." in k and k.startswith("backbone."):
            routed += b
            lay = k.split(".")[2]
            eid = k.split(".mixer.experts.")[1].split(".")[0]
            routed_recs[(lay, eid)] += b
        elif ".shared_experts." in k and k.startswith("backbone."):
            shared += b
        elif k.startswith("backbone.") or k == "lm_head.weight":
            trunk += b

    rec_sizes = collections.Counter(routed_recs.values())

    result = {
        "kind": "lightningstream_nemotron_n2r_v35_layout",
        "phase": "N2R_V35_LAYOUT",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(MODEL.relative_to(REPO)).replace("\\", "/"),
        "shards": len(shards),
        "tensors": len(tensors),
        "dtype_bytes": dict(sorted(dt_bytes.items())),
        "dtype_counts": dict(sorted(dt_count.items())),
        "routed_expert_fields": {
            k: {"dtype": v["dtype"], "shape": v["shape"], "bytes": nbytes(v)}
            for k, v in sorted(fields.items())},
        "routed_expert_record_bytes": expert_bytes,
        "routed_record_size_distribution": {str(k): v for k, v in rec_sizes.items()},
        "routed_record_count": len(routed_recs),
        "partition": {"routed": routed, "shared": shared, "trunk_other": trunk},
        "mtp": {
            "tensor_count": len(mtp),
            "bytes": mtp_bytes,
            "expert0_fields": {
                k: {"dtype": v["dtype"], "shape": v["shape"], "bytes": nbytes(v)}
                for k, v in sorted(mtp_fields.items())},
        },
        "nano_comparison": NANO,
        "verdicts": {},
    }

    # --- decisive verdicts
    wq = fields.get("up_proj.weight")
    ws = fields.get("up_proj.weight_scale")
    v = result["verdicts"]
    v["same_record_bytes_as_nano"] = expert_bytes == NANO["routed_record_bytes"]
    v["weight_dtype"] = wq["dtype"] if wq else None
    v["scale_dtype"] = ws["dtype"] if ws else None
    v["weight_shape"] = wq["shape"] if wq else None
    v["scale_shape"] = ws["shape"] if ws else None
    if wq and ws:
        rows, cols = wq["shape"]
        # NVFP4: cols == hidden/2 (2 codes per byte) and scales == hidden/16
        v["looks_like_nvfp4_group16"] = (cols * 2 == 2688) and (ws["shape"][1] == 2688 // 16)
        # FP8: one byte per weight -> cols == hidden
        v["looks_like_fp8_per_weight"] = (cols == 2688)
    v["mtp_present"] = len(mtp) > 0

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"shards {len(shards)}  tensors {len(tensors):,}")
    print("dtype bytes:")
    for k, b in sorted(dt_bytes.items()):
        print(f"  {k:<10} {b:>16,}  n={dt_count[k]:,}")
    print(f"\nrouted expert 0 of layer 1 -- {expert_bytes:,} B "
          f"(Nano {NANO['routed_record_bytes']:,})")
    for k, f in sorted(fields.items()):
        print(f"  {f['dtype']:<8} {str(f['shape']):<18} {nbytes(f):>10,}  {k}")
    print(f"\nrouted records {len(routed_recs):,}  sizes {dict(rec_sizes)}")
    print(f"partition routed {routed:,} shared {shared:,} trunk {trunk:,}")
    print(f"\nMTP tensors {len(mtp)}  bytes {mtp_bytes:,}")
    for k, f in sorted(mtp_fields.items()):
        print(f"  {f['dtype']:<8} {str(f['shape']):<18} {nbytes(f):>10,}  {k}")
    print("\nverdicts:")
    for k, val in v.items():
        print(f"  {k}: {val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
