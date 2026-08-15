"""N7-A: measure natural route locality and size an expert cache from it.

N6-A found all 128 experts used with only an 8.7x popularity spread, which says
a *static* prior is weak.  It says nothing about *temporal* locality across
consecutive tokens, which is what an LRU cache actually exploits.  This measures
that directly on a real generation, then simulates per-layer caches.

Hit rates are measured.  The tok/s figures derived from them are labelled
projections and are NOT measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
RECORD_BYTES = 5_612_560
MEASURED_TRANSFER_GB_S = 26.03      # N4 measured
CAPACITIES = [0, 8, 16, 24, 32, 48, 64, 96, 128]


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def simulate_lru(seq_per_layer: dict, capacity: int) -> dict:
    """Per-layer LRU over the natural route sequence."""
    hits = misses = 0
    for layer, steps in seq_per_layer.items():
        cache: OrderedDict[int, None] = OrderedDict()
        for experts in steps:
            for e in experts:
                if e in cache:
                    cache.move_to_end(e)
                    hits += 1
                else:
                    misses += 1
                    cache[e] = None
                    if len(cache) > capacity:
                        cache.popitem(last=False)
    total = hits + misses
    return {"capacity_per_layer": capacity, "hits": hits, "misses": misses,
            "requests": total, "hit_rate": hits / total if total else 0.0}


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()

    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        foreign = [l for l in o.stdout.strip().splitlines()
                   if l.strip() and int(l.split(",")[0]) != os.getpid()]
    except Exception:
        foreign = ["query failed"]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    rt = LightningRuntime(MODEL_DIR, contexts_max=args.max_ctx)
    rt.load_routed_bank()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    prompt = "The history of computing began when"
    ids = tok.encode(prompt, add_special_tokens=False)
    rt.reset()

    captured: dict[str, list] = {}
    cur = ids[0]
    for i in range(len(ids) + args.tokens):
        nxt = rt.step(cur, capture_routes=captured)
        cur = nxt if i >= len(ids) - 1 else ids[i + 1]
    cp.cuda.Device(0).synchronize()

    seq = {k: [row for row in v] for k, v in captured.items()}
    steps = len(next(iter(seq.values())))
    print(f"captured {steps} steps x {len(seq)} MoE layers", flush=True)

    # ---- consecutive-step overlap: the direct locality signal -------------
    overlaps = []
    for layer, rows in seq.items():
        for a, b in zip(rows, rows[1:]):
            overlaps.append(len(set(a) & set(b)))
    overlap_mean = float(np.mean(overlaps))
    print(f"mean experts shared between consecutive tokens: "
          f"{overlap_mean:.3f} of {rt.top_k}", flush=True)

    sims = [simulate_lru(seq, c) for c in CAPACITIES]

    # ---- project the effect on the measured token -------------------------
    # Measured at ctx 0 after the N6-C optimisations.
    token_ms = 62.95
    moe_ms = 48.432
    routed_transfer_floor_ms = (
        rt.top_k * len(seq) * RECORD_BYTES / (MEASURED_TRANSFER_GB_S * 1e9) * 1e3)
    non_transfer_ms = token_ms - routed_transfer_floor_ms

    for s in sims:
        saved = routed_transfer_floor_ms * s["hit_rate"]
        s["projected_token_ms"] = token_ms - saved
        s["projected_tok_s"] = 1000.0 / s["projected_token_ms"]
        s["cache_bytes"] = s["capacity_per_layer"] * len(seq) * RECORD_BYTES
        s["cache_gib"] = s["cache_bytes"] / (1024 ** 3)
        print(f"  cap {s['capacity_per_layer']:>3}/layer  hit {s['hit_rate']*100:5.1f}%  "
              f"cache {s['cache_gib']:5.2f} GiB  -> projected {s['projected_tok_s']:6.2f} tok/s")

    result = {
        "kind": "lightningstream_nemotron_n7a_route_locality",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N7_A_ROUTE_LOCALITY",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_path(Path(__file__)),
        "prompt": prompt,
        "steps_captured": steps,
        "moe_layers": len(seq),
        "top_k": rt.top_k,
        "n_routed_experts": rt.n_experts,
        "consecutive_overlap_mean": overlap_mean,
        "consecutive_overlap_fraction": overlap_mean / rt.top_k,
        "measured_token_ms_ctx0": token_ms,
        "measured_moe_ms": moe_ms,
        "routed_transfer_floor_ms": routed_transfer_floor_ms,
        "non_transfer_ms": non_transfer_ms,
        "simulations": sims,
        "measurement_vs_projection": (
            "Hit rates and consecutive-overlap are MEASURED on real natural "
            "routes. The projected_tok_s column is a PROJECTION assuming a hit "
            "removes its record's transfer time entirely and nothing else "
            "changes. It is not a measurement and must not be quoted as one."),
        "claim_boundary": (
            "Route locality measured on one prompt and one generation of this "
            "length. Not a general routing statistic, not a quality result, and "
            "the projections are not throughput measurements."),
    }
    (OUT_DIR / "n7a_route_locality.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nrouted transfer floor {routed_transfer_floor_ms:.2f} ms of a "
          f"{token_ms:.2f} ms token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
