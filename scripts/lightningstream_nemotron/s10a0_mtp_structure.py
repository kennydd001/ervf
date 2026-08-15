"""S10-A0: what is actually in the MTP block?

Before an MTP forward can be written, its wiring has to be known rather than
assumed. This enumerates every mtp.* tensor with dtype and shape, groups them by
role, and checks which pieces an acceptance measurement would need: an input
projection or fusion of the previous hidden state, the attention block, the MoE
block, a norm, and whether it reuses the backbone embedding and lm_head.

Reads headers only. No payload decode, no GPU.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL = REPO / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning_v35")
OUT = REPO / "reports" / "lightningstream_nemotron" / "s10a0_mtp_structure.json"


def main() -> int:
    idx = ShardIndex(MODEL)
    mtp = {k: v for k, v in idx.entries.items() if k.startswith("mtp.")}

    # Everything that is not a routed expert -- the structural skeleton.
    expert_re = re.compile(r"^mtp\.layers\.\d+\.mixer\.experts\.\d+\.")
    skeleton = {k: v for k, v in mtp.items() if not expert_re.match(k)}
    experts = {k: v for k, v in mtp.items() if expert_re.match(k)}

    by_layer = collections.defaultdict(list)
    for k in skeleton:
        m = re.match(r"^mtp\.layers\.(\d+)\.(.*)$", k)
        by_layer[m.group(1) if m else "root"].append(k)

    expert_ids = collections.defaultdict(set)
    for k in experts:
        m = re.match(r"^mtp\.layers\.(\d+)\.mixer\.experts\.(\d+)\.", k)
        if m:
            expert_ids[m.group(1)].add(int(m.group(2)))

    def row(k):
        e = idx.entries[k]
        return {"dtype": e.dtype, "shape": list(e.shape), "bytes": e.nbytes}

    result = {
        "kind": "lightningstream_nemotron_s10a0_mtp_structure",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL.name,
        "config": {
            "num_nextn_predict_layers": idx.config.get("num_nextn_predict_layers"),
            "mtp_layers_block_type": idx.config.get("mtp_layers_block_type"),
        },
        "mtp_tensor_count": len(mtp),
        "mtp_bytes": sum(v.nbytes for v in mtp.values()),
        "expert_tensor_count": len(experts),
        "expert_bytes": sum(idx.entries[k].nbytes for k in experts),
        "experts_per_layer": {k: len(v) for k, v in sorted(expert_ids.items())},
        "skeleton": {k: row(k) for k in sorted(skeleton)},
        "reuses_backbone_embedding": "backbone.embeddings.weight" in idx.entries,
        "has_own_embedding": any("embed" in k for k in mtp),
        "has_own_lm_head": any("lm_head" in k for k in mtp),
        "quant_kind_expert0": (
            idx.quant_kind("mtp.layers.1.mixer.experts.0.up_proj")
            if "mtp.layers.1.mixer.experts.0.up_proj.weight" in idx.entries else None),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"mtp tensors {len(mtp)}  bytes {result['mtp_bytes']:,}")
    print(f"  experts: {len(experts)} tensors, {result['expert_bytes']:,} B, "
          f"per layer {result['experts_per_layer']}")
    print(f"  expert quant kind: {result['quant_kind_expert0']}")
    print(f"  own embedding {result['has_own_embedding']}  "
          f"own lm_head {result['has_own_lm_head']}")
    print(f"\nskeleton ({len(skeleton)} tensors):")
    for k in sorted(skeleton):
        r = row(k)
        print(f"  {r['dtype']:<8} {str(r['shape']):<18} {r['bytes']:>12,}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
