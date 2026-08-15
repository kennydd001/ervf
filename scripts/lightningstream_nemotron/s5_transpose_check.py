"""S5 G-S5-C2: transpose exactness over ALL 2,944 routed records.

For every routed expert record: rebuild the (2688, 1856) nibble matrix and the
(2688, 116) scale matrix from the panel-major block via an INDEPENDENT inverse
mapping (written here, not reusing down_panel_major's internals) and require
exact equality with the unpack of the shard-original bytes.

CPU-only; reads the shards directly. No GPU, no timing.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402
from moe_lab.lightningstream_nemotron.runtime import down_panel_major  # noqa: E402

ROWS, INTER = 2688, 1856
NPANEL = INTER // 16


def inverse_unpack(block: np.ndarray):
    """panel-major (116, 24192) -> (nibble matrix (2688,1856), scales (2688,116)).

    Independent of down_panel_major: explicit index arithmetic.
    """
    scales_back = block[:, :ROWS].T.copy()                       # (2688, 116)
    packed = block[:, ROWS:].reshape(NPANEL, 16, ROWS // 2)
    cols = np.empty((NPANEL, 16, ROWS), dtype=np.uint8)
    cols[..., 0::2] = packed & 15
    cols[..., 1::2] = packed >> 4
    nib = cols.transpose(2, 0, 1).reshape(ROWS, INTER)
    return nib, scales_back


def main() -> int:
    idx = ShardIndex(REPO_ROOT / "models" / "nemotron_3_5_lightning")
    pattern = idx.config["hybrid_override_pattern"]
    moe_layers = [i for i, ch in enumerate(pattern) if ch == "E"]

    checked, bad = 0, []
    t0 = datetime.now(timezone.utc)
    for layer in moe_layers:
        for e in range(128):
            pre = f"backbone.layers.{layer}.mixer.experts.{e}"
            codes = idx.read_raw(f"{pre}.down_proj.weight")
            scales = idx.read_raw(f"{pre}.down_proj.weight_scale")
            block = down_panel_major(codes, scales)
            nib, sc = inverse_unpack(block)
            dc = codes.reshape(ROWS, INTER // 2)
            nib_orig = np.empty((ROWS, INTER), dtype=np.uint8)
            nib_orig[:, 0::2] = dc & 15
            nib_orig[:, 1::2] = dc >> 4
            if not (np.array_equal(nib, nib_orig)
                    and np.array_equal(sc, scales.reshape(ROWS, NPANEL))):
                bad.append((layer, e))
            checked += 1
        print(f"layer {layer:>2}: 128 records ok so far, bad={len(bad)}",
              flush=True)

    out = {
        "kind": "lightningstream_nemotron_s5_transpose_check",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "records_checked": checked, "records_bad": bad,
        "gate_G_S5_C2": len(bad) == 0,
    }
    (REPO_ROOT / "reports/lightningstream_nemotron/s5_transpose_check.json"
     ).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"C2: {checked} records checked, {len(bad)} bad -> "
          f"{'PASS' if not bad else 'FAIL'}")
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
