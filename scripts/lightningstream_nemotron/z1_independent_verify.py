"""Independent verification of Z1. Imports nothing from the runner.

Refits the line from the raw per-N rounds, recomputes R2, the scaling ratio, the
ceiling, and all three gates, and cross-checks the per-position cost against X1's
independent measurement of the same quantity.
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
GATE_P2C = 250.0
GATE_R2 = 0.99
X1_T1_MS = 22.454


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


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    c = sxy / sxx
    d = my - c * mx
    ss_res = sum((y - (c * x + d)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return c, d, (1 - ss_res / ss_tot if ss_tot else 1.0)


def main() -> int:
    path = OUT_DIR / "z1_tree_verifier_ceiling.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    rows = res["rows"]
    checks = []

    xs, ys = [], []
    for k, row in sorted(rows.items(), key=lambda kv: int(kv[0])):
        m = p50(row["raw"]["run"])
        checks.append({"check": f"N={k}: p50 reproduces from the raw rounds",
                       "ok": abs(m - row["ms_p50"]) < 1e-9, "recomputed": m})
        checks.append({"check": f"N={k}: ms per position reproduces",
                       "ok": abs(m / int(k) - row["ms_per_position"]) < 1e-9})
        checks.append({"check": f"N={k}: all three bracket arms have samples",
                       "ok": all(len(row["raw"][a]) > 0 for a in ("a1", "run", "a2"))})
        xs.append(float(int(k)))
        ys.append(m)

    c, d, r2 = fit(xs, ys)
    checks.append({"check": "linear fit slope reproduces",
                   "ok": abs(c - res["linear_fit"]["ms_per_position"]) < 1e-6,
                   "recomputed": c})
    checks.append({"check": "R2 reproduces", "ok": abs(r2 - res["linear_fit"]["r2"]) < 1e-6,
                   "recomputed": r2})
    checks.append({"check": f"R2 >= {GATE_R2} (the cost really is linear in positions)",
                   "ok": r2 >= GATE_R2})

    nmax = max(int(k) for k in rows)
    scaling = rows[str(nmax)]["ms_p50"] / rows["1"]["ms_p50"]
    checks.append({"check": f"T({nmax})/T(1) is within 10% of {nmax}",
                   "ok": abs(scaling - nmax) / nmax <= 0.10, "scaling": scaling})

    ceiling = 1000.0 / c
    checks.append({"check": "ceiling 1/c reproduces",
                   "ok": abs(ceiling - res["ceiling_tok_s"]) < 1e-3,
                   "recomputed": ceiling})
    p2c = ceiling >= GATE_P2C
    checks.append({"check": "G-Z1-P2C verdict agrees with the runner",
                   "ok": p2c == res["gates"]["G_Z1_P2C"]["passed"]})
    checks.append({"check": "pack gate is still the preregistered 250 tok/s",
                   "ok": abs(res["gates"]["G_Z1_P2C"]["required_tok_s"] - GATE_P2C) < 1e-9})

    # cross-check against X1, which measured the same quantity independently
    x1 = json.loads((OUT_DIR / "x1_sweepspec_moe_oracle.json").read_text(encoding="utf-8"))
    x1_slope = x1["timing"]["5"]["seq_bracketed_p50_ms"] / 5.0
    sanity = abs(rows["1"]["ms_p50"] - X1_T1_MS) / X1_T1_MS <= 0.10
    checks.append({"check": "G-Z1-S1 sanity verdict agrees with the runner",
                   "ok": sanity == res["gates"]["G_Z1_S1_sanity"]["passed"]})
    checks.append({"check": "both independent slopes give a ceiling far below the pack gate",
                   "ok": (1000.0 / c < GATE_P2C) and (1000.0 / x1_slope < GATE_P2C),
                   "z1_ceiling": 1000.0 / c, "x1_ceiling": 1000.0 / x1_slope})

    checks.append({"check": "runtime.py was not modified for this phase",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py")
                   == res["runtime_sha256"]})
    checks.append({"check": "claim boundary states the ceiling is an upper bound",
                   "ok": "UPPER BOUND" in res.get("claim_boundary", "")})
    checks.append({"check": "no measured tokens-per-second is claimed",
                   "ok": not re.search(r"measured_tok_s|achieved", json.dumps(res))})

    failed = [ch for ch in checks if not ch["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_z1_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"slope_ms_per_position": c, "intercept_ms": d, "r2": r2,
                       "scaling": scaling, "ceiling_tok_s": ceiling,
                       "x1_slope_ms_per_position": x1_slope,
                       "x1_ceiling_tok_s": 1000.0 / x1_slope},
        "gates": {"G_Z1_L1": r2 >= GATE_R2 and abs(scaling - nmax) / nmax <= 0.10,
                  "G_Z1_P2C": p2c, "G_Z1_S1": sanity},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "z1_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for ch in checks:
        if not ch["ok"]:
            print(f"  [FAIL] {ch['check']}  {ch}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    print(f"\n  slope {c:.4f} ms/position, R2 {r2:.5f}, scaling {scaling:.3f}/{nmax}")
    print(f"  Z1 ceiling {ceiling:.2f} tok/s | X1 ceiling {1000.0 / x1_slope:.2f} tok/s "
          f"| pack gate {GATE_P2C}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
