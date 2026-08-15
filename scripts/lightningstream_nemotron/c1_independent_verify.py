"""Independent verification of C1. Imports nothing from the runner.

Recomputes the aggregate rates from the per-pair records, re-applies the three
gates, and checks the two things the oracle's meaning rests on: that the bound
never certified a neuron that was actually non-zero, and that the certification
rate is reported against the pack's own thresholds and not a softened pair.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GATE_TAIL = 0.30
GATE_ZERO_SHARE = 0.30


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    path = OUT_DIR / "c1_certiplane_oracle.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    pairs = res["per_pair"]
    inter = res["config"]["inter"]
    checks = []

    checks.append({"check": "at least 200 (expert, activation) pairs",
                   "ok": len(pairs) >= 200, "pairs": len(pairs)})
    checks.append({"check": "pairs span more than one MoE layer",
                   "ok": len({p["layer"] for p in pairs}) > 1,
                   "layers": len({p["layer"] for p in pairs})})
    checks.append({"check": "pairs span many distinct experts",
                   "ok": len({p["expert"] for p in pairs}) >= 20,
                   "experts": len({p["expert"] for p in pairs})})

    rec = {}
    for c, s in res["summary"].items():
        cert = sum(p[f"c{c}"]["certified_fraction"] for p in pairs) / len(pairs)
        false_total = sum(p[f"c{c}"]["false_cert"] for p in pairs)
        rows = len(pairs) * inter
        rec[c] = {"certified_fraction": cert, "false_certificates": false_total,
                  "rows": rows,
                  "mean_bound": sum(p[f"c{c}"]["mean_bound"] for p in pairs) / len(pairs),
                  "mean_abs_y0": sum(p[f"c{c}"]["mean_abs_y0"] for p in pairs) / len(pairs)}
        rec[c]["bound_over_y0"] = rec[c]["mean_bound"] / rec[c]["mean_abs_y0"]
        checks.append({"check": f"core {c}: certified fraction reproduces from per-pair records",
                       "ok": abs(cert - s["certified_fraction"]) < 5e-4,
                       "recomputed": cert, "stored": s["certified_fraction"]})
        checks.append({"check": f"core {c}: false-certificate total reproduces",
                       "ok": false_total == s["false_certificates"]})
        checks.append({"check": f"core {c}: row count reproduces",
                       "ok": rows == s["rows"]})
        checks.append({"check": f"core {c}: the bound is SOUND (zero false certificates)",
                       "ok": false_total == 0})

    true_zero = sum(p["true_zero_fraction"] for p in pairs) / len(pairs)
    checks.append({"check": "true-zero fraction is consistent with S5's ~91% ReLU2 sparsity",
                   "ok": 0.85 <= true_zero <= 0.95, "measured": true_zero})

    best = max(rec, key=lambda k: rec[k]["certified_fraction"])
    r1 = rec[best]["certified_fraction"] >= GATE_TAIL
    b1 = res["summary"][best]["certified_share_of_true_zero"] >= GATE_ZERO_SHARE
    sound = all(v["false_certificates"] == 0 for v in rec.values())

    checks.append({"check": "G-C1-S1 verdict agrees with the runner",
                   "ok": sound == res["gates"]["G_C1_S1_soundness"]["passed"]})
    checks.append({"check": "G-C1-R1 verdict agrees with the runner",
                   "ok": r1 == res["gates"]["G_C1_R1_tail_yield"]["passed"]})
    checks.append({"check": "G-C1-B1 verdict agrees with the runner",
                   "ok": b1 == res["gates"]["G_C1_B1_bound_useful"]["passed"]})
    checks.append({"check": "gates are still the pack's own 30% thresholds",
                   "ok": abs(res["gates"]["G_C1_R1_tail_yield"]["required_fraction"]
                             - GATE_TAIL) < 1e-12
                   and abs(res["gates"]["G_C1_B1_bound_useful"]["required_share_of_true_zeros"]
                           - GATE_ZERO_SHARE) < 1e-12})
    checks.append({"check": "claim boundary names the upper-bound nature of the bound",
                   "ok": "UPPER" in res.get("claim_boundary", "").upper()})
    checks.append({"check": "runtime.py hashes to the measured value",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py")
                   == res["runtime_sha256"]})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_c1_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"per_core": rec, "true_zero_fraction": true_zero,
                       "best_core": best},
        "gates": {"G_C1_S1": sound, "G_C1_R1": r1, "G_C1_B1": b1},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "c1_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}  {c}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    for c, v in rec.items():
        print(f"  core {c}: certified {v['certified_fraction'] * 100:.2f}%  "
              f"false {v['false_certificates']}  "
              f"bound/|y0| {v['bound_over_y0']:.2f}")
    print(f"  true zeros {true_zero * 100:.2f}% | G-C1-S1 {sound} "
          f"G-C1-R1 {r1} G-C1-B1 {b1}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
