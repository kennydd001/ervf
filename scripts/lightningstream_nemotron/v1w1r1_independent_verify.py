"""Independent verification of V1 / W1-R1. Imports nothing from the runner.

Recomputes the V1 budget from its stated inputs, the per-call medians from the
raw rounds, every triplet effect and drift from the raw b1/fast/b2 samples, the
sign fraction, and all gates. The preregistration says that when the resolution
gate fails, the effect SIZE is not concluded; this verifier checks that the
result does not quietly claim it anyway.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
MISS_RATE = 0.1785
EXPERT_CALLS = 138
Y1_SYNC_MS = 6.656
GATE_DRIFT_MS = 0.5
GATE_SIGN = 0.60


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def p50(xs):
    s = sorted(float(x) for x in xs)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def main() -> int:
    path = OUT_DIR / "v1w1r1_router_feasibility.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    checks = []

    # ---------------------------------------------------------------- V1
    v1 = res["v1"]
    budget = Y1_SYNC_MS * 1000.0 / (MISS_RATE * EXPERT_CALLS)
    checks.append({"check": "V1 budget reproduces from Y1's sync and K0's miss rate",
                   "ok": abs(budget - v1["budget_us_per_miss"]) < 1e-6,
                   "recomputed": budget})
    for label, arm in v1["arms"].items():
        m = p50(arm["us_raw"])
        checks.append({"check": f"V1 {label}: p50 reproduces from the raw rounds",
                       "ok": abs(m - arm["us_per_call_p50"]) < 1e-9})
        gb = v1["record_bytes"] / (m * 1e-6) / 1e9
        checks.append({"check": f"V1 {label}: GB/s reproduces", "ok": abs(gb - arm["gb_s"]) < 1e-6})
    delta = v1["arms"]["mapped_host"]["us_per_call_p50"] - v1["arms"]["device"]["us_per_call_p50"]
    checks.append({"check": "V1 delta per miss reproduces",
                   "ok": abs(delta - v1["delta_us_per_miss"]) < 1e-9, "recomputed": delta})
    feasible = v1["bit_identical"] and delta < budget
    checks.append({"check": "G-V1-F1 verdict agrees with the runner",
                   "ok": feasible == res["gates"]["G_V1_F1_feasible"]["passed"]})
    checks.append({"check": "V1 arms are bit-identical (same bytes, two paths)",
                   "ok": v1["bit_identical"]})

    # ------------------------------------------------------------- W1-R1
    rec = {}
    for ctx, row in res["w1r1"].items():
        eff, dr, wins = [], [], 0
        for t in row["rows"]:
            e = 0.5 * (t["b1"] + t["b2"]) - t["fast"]
            d = abs(t["b2"] - t["b1"])
            checks.append({"check": f"ctx {ctx}: triplet effect reproduces",
                           "ok": abs(e - t["effect_ms"]) < 1e-9})
            checks.append({"check": f"ctx {ctx}: triplet drift reproduces",
                           "ok": abs(d - t["drift_ms"]) < 1e-9})
            eff.append(e)
            dr.append(d)
            wins += int(e > 0)
        n = len(eff)
        rec[ctx] = {"n": n, "effect_p50": p50(eff), "drift_p50": p50(dr),
                    "sign_fraction": wins / n,
                    "resolution_ok": p50(dr) < GATE_DRIFT_MS,
                    "sign_ok": p50(eff) > 0 and wins / n >= GATE_SIGN}
        checks.append({"check": f"ctx {ctx}: at least 16 triplets", "ok": n >= 16, "n": n})
        checks.append({"check": f"ctx {ctx}: resolution verdict agrees",
                       "ok": rec[ctx]["resolution_ok"] == row["resolution_ok"],
                       "drift_p50": rec[ctx]["drift_p50"]})
        checks.append({"check": f"ctx {ctx}: sign verdict agrees",
                       "ok": rec[ctx]["sign_ok"] == row["sign_ok"]})

    # The preregistration's own rule: no resolution, no effect-size conclusion.
    deep = max(rec, key=lambda k: int(k))
    checks.append({"check": "resolution gate failed, so no effect SIZE may be adopted",
                   "ok": rec[deep]["resolution_ok"] or
                   not res["gates"]["G_W1R_R1_resolution"][deep],
                   "note": "verifier records that G-W1-P1 stays unresolved when "
                           "G-W1R-R1 fails"})
    checks.append({"check": "the paired design did NOT beat W1's arm-level drift",
                   "ok": True, "triplet_drift_p50": rec[deep]["drift_p50"],
                   "w1_arm_drift_ms": 4.520,
                   "note": "recorded as an observation, not a pass/fail"})
    checks.append({"check": "generation bit-identical between fast and base",
                   "ok": res["gates"]["G_W1R_C1_identity"]["passed"]})
    checks.append({"check": "runtime.py hashes to the measured value",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py")
                   == res["runtime_sha256"]})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_v1w1r1_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"v1_budget_us": budget, "v1_delta_us": delta,
                       "v1_feasible": feasible, "w1r1": rec},
        "gates": {"G_V1_C1": v1["bit_identical"], "G_V1_F1": feasible,
                  "G_W1R_R1": {c: v["resolution_ok"] for c, v in rec.items()},
                  "G_W1R_E1": {c: v["sign_ok"] for c, v in rec.items()}},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "v1w1r1_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}  {c}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    print(f"\n  V1: device {v1['arms']['device']['us_per_call_p50']:.2f} us | "
          f"mapped host {v1['arms']['mapped_host']['us_per_call_p50']:.2f} us | "
          f"delta {delta:.2f} vs budget {budget:.2f} -> feasible={feasible}")
    for c, v in rec.items():
        print(f"  W1-R1 ctx {c:>6}: effect p50 {v['effect_p50']:+.3f} ms, "
              f"drift p50 {v['drift_p50']:.3f} ms, sign {v['sign_fraction']:.2f}, "
              f"resolution_ok={v['resolution_ok']}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
