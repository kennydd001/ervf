"""Independent verification of X1. Imports nothing from the runner or kernels.

Recomputes the bracketed sequential baseline, every ratio, the preregistered
threshold from the measured acceptance, and the per-emitted-token cost, and
re-applies all four gates.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
RUNTIME = REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    path = OUT_DIR / "x1_sweepspec_moe_oracle.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    checks = []

    # ---- the acceptance the threshold rests on is S10-A's, unchanged
    s10 = json.loads((OUT_DIR / "s10a_mtp_acceptance.json").read_text(encoding="utf-8"))
    emitted = 1.0 + s10["pooled"]["mean_A"]
    checks.append({"check": "emitted-per-round equals 1 + the S10-A pooled acceptance",
                   "ok": abs(emitted - res["config"]["emitted_per_round"]) < 1e-4,
                   "recomputed": emitted})

    b_max = max(int(b) for b in res["timing"])
    threshold = emitted / b_max
    checks.append({"check": "gate threshold is emitted/B_max, not an invented number",
                   "ok": abs(threshold - res["gates"]["G_X1_P1_ratio"]["required_ratio_below"]) < 1e-4,
                   "recomputed": threshold})

    # ---- exactness
    for b, e in res["exactness"].items():
        checks.append({"check": f"B={b}: zero mismatches over >=80 layer-blocks",
                       "ok": e["mismatches"] == 0 and e["layer_blocks_checked"] >= 80,
                       "checked": e["layer_blocks_checked"]})
        checks.append({"check": f"B={b}: bit_identical flag consistent with the count",
                       "ok": e["bit_identical"] == (e["mismatches"] == 0)})
    checks.append({"check": "G-X1-C1 verdict agrees with the runner",
                   "ok": res["exactness"]["1"]["bit_identical"]
                   == res["gates"]["G_X1_C1_batched_equals_gemv_at_B1"]["passed"]})
    checks.append({"check": "G-X1-C2 verdict agrees with the runner",
                   "ok": all(v["bit_identical"] for v in res["exactness"].values())
                   == res["gates"]["G_X1_C2_sweep_equals_sequential"]["passed"]})

    # ---- timings
    rec = {}
    for b, row in res["timing"].items():
        base = 0.5 * (row["seq_ms"]["p50"] + row["seq_repeat_ms"]["p50"])
        drift = abs(row["seq_repeat_ms"]["p50"] - row["seq_ms"]["p50"])
        ratio = row["sweep_ms"]["p50"] / base
        rec[b] = {"seq_bracketed_p50_ms": base, "local_drift_ms": drift,
                  "ratio": ratio,
                  "conclusive": abs(row["sweep_ms"]["p50"] - base) > drift,
                  "seq_scaling_vs_B1": None}
        checks.append({"check": f"B={b}: bracketed sequential baseline reproduces",
                       "ok": abs(base - row["seq_bracketed_p50_ms"]) < 1e-9})
        checks.append({"check": f"B={b}: ratio reproduces",
                       "ok": abs(ratio - row["ratio_sweep_over_seq"]) < 1e-9,
                       "recomputed": ratio})
        checks.append({"check": f"B={b}: samples present in all three arms",
                       "ok": all(row[a]["n"] > 0 for a in
                                 ("seq_ms", "sweep_ms", "seq_repeat_ms"))})

    b1 = rec["1"]["seq_bracketed_p50_ms"]
    for b in rec:
        rec[b]["seq_scaling_vs_B1"] = rec[b]["seq_bracketed_p50_ms"] / b1
        rec[b]["sweep_ms_per_emitted"] = (res["timing"][b]["sweep_ms"]["p50"] / emitted)

    r5 = rec[str(b_max)]
    passed = r5["ratio"] < threshold
    checks.append({"check": "G-X1-P1 verdict agrees with the runner",
                   "ok": passed == res["gates"]["G_X1_P1_ratio"]["passed"],
                   "recomputed_ratio": r5["ratio"], "threshold": threshold})
    checks.append({"check": "the sequential arm scales close to linearly in B",
                   "ok": abs(rec[str(b_max)]["seq_scaling_vs_B1"] - b_max) / b_max < 0.10,
                   "scaling": {b: rec[b]["seq_scaling_vs_B1"] for b in rec}})

    checks.append({"check": "runtime.py was not modified for this phase",
                   "ok": sha256_path(RUNTIME) == res["runtime_sha256"]})
    checks.append({"check": "sweepspec.py on disk hashes to the measured value",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/sweepspec.py")
                   == res["sweepspec_sha256"]})
    checks.append({"check": "no tokens-per-second figure anywhere in the result",
                   "ok": not re.search(r"tok_s|tokens_per_second", json.dumps(res))})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_x1_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"emitted_per_round": emitted, "threshold": threshold,
                       "per_B": rec},
        "gates": {"G_X1_C1": res["exactness"]["1"]["bit_identical"],
                  "G_X1_C2": all(v["bit_identical"] for v in res["exactness"].values()),
                  "G_X1_P1": passed},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "x1_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}  {c}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)\n")
    print(f"{'B':>2} {'seq p50':>9} {'sweep p50':>10} {'ratio':>7} {'seq/B=1':>8} {'union':>6}")
    for b in sorted(rec, key=int):
        print(f"{b:>2} {rec[b]['seq_bracketed_p50_ms']:9.3f} "
              f"{res['timing'][b]['sweep_ms']['p50']:10.3f} {rec[b]['ratio']:7.4f} "
              f"{rec[b]['seq_scaling_vs_B1']:8.3f} "
              f"{res['timing'][b]['mean_union_per_layer']:6.2f}")
    print(f"\nthreshold {threshold:.4f}, measured {r5['ratio']:.4f} -> "
          f"G-X1-P1 {'PASS' if passed else 'FAIL'}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
