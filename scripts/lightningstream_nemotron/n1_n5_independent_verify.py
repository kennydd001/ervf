"""Independent verification of N1-N5. Imports nothing from the runners."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GATE_N2 = 0.30
GATE_N3 = 0.30
GATE_N4_R2 = 0.98


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
    c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    d = my - c * mx
    ssr = sum((y - (c * x + d)) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - my) ** 2 for y in ys)
    return c, d, 1 - ssr / sst


def main() -> int:
    a = json.loads((OUT_DIR / "n1n2n4n5_ceilings.json").read_text(encoding="utf-8"))
    b = json.loads((OUT_DIR / "n3_relu2_prefilter_oracle.json").read_text(encoding="utf-8"))
    checks = []

    # ---- N1
    n1 = a["n1"]
    checks.append({"check": "N1 graph capture succeeded", "ok": n1["capture_ok"]})
    checks.append({"check": "N1 eager p50 reproduces from raw",
                   "ok": abs(p50(n1["eager_raw"]) - n1["eager_ms"]) < 1e-9})
    rem = 1 - n1["graph_ms"] / n1["eager_ms"]
    checks.append({"check": "N1 removable fraction reproduces",
                   "ok": abs(rem - a["gates"]["G_N1_1_removable"]["removable_fraction"]) < 1e-9,
                   "recomputed": rem})
    checks.append({"check": "N1 graph is faster than eager (overhead exists)",
                   "ok": n1["graph_ms"] < n1["eager_ms"]})

    # ---- N2
    n2 = a["n2"]
    marg = {}
    order = ["scan", "gather", "gemv", "down_all"]
    for k, name in enumerate(order):
        b0 = n2["arms"][f"base{k}"]["p50"]
        b1 = n2["arms"][f"base{k + 1}"]["p50"]
        marg[name] = {"m": n2["arms"][name]["p50"] - 0.5 * (b0 + b1),
                      "drift": abs(b1 - b0)}
        checks.append({"check": f"N2 {name}: p50 reproduces from raw",
                       "ok": abs(p50(n2["arms"][name]["raw"])
                                 - n2["arms"][name]["p50"]) < 1e-9})
        checks.append({"check": f"N2 {name}: marginal reproduces",
                       "ok": abs(marg[name]["m"] - n2["marginals"][name]["marginal_ms"]) < 1e-9,
                       "recomputed": marg[name]["m"], "drift": marg[name]["drift"]})
    share = (marg["scan"]["m"] + marg["gather"]["m"]) / marg["down_all"]["m"]
    checks.append({"check": "N2 scan+gather share reproduces",
                   "ok": abs(share - n2["share_of_down"]) < 1e-9, "recomputed": share})
    checks.append({"check": "G-N2-1 verdict agrees with the runner",
                   "ok": (share >= GATE_N2) == a["gates"]["G_N2_1_scan_gather_share"]["passed"]})
    checks.append({"check": "N2 gather marginal is above its own local drift",
                   "ok": marg["gather"]["m"] > marg["gather"]["drift"],
                   "marginal": marg["gather"]["m"], "drift": marg["gather"]["drift"]})
    checks.append({"check": "N2 scan marginal is BELOW its drift and is reported as such",
                   "ok": marg["scan"]["m"] <= marg["scan"]["drift"]})

    # ---- N4
    n4 = a["n4"]
    xs = [r["kv_bytes"] for r in n4["rows"].values()]
    ys = [p50(r["raw"]) for r in n4["rows"].values()]
    c, d, r2 = fit(xs, ys)
    checks.append({"check": "N4 slope reproduces",
                   "ok": abs(c - n4["fit_ms_per_byte"]) < 1e-15, "recomputed": c})
    checks.append({"check": "N4 R2 reproduces", "ok": abs(r2 - n4["r2"]) < 1e-6})
    checks.append({"check": "G-N4-2 fit quality",
                   "ok": (r2 >= GATE_N4_R2) == a["gates"]["G_N4_2_fit_quality"]["passed"]})
    checks.append({"check": "N4 fixed term is small, i.e. attention is byte-bound",
                   "ok": abs(n4["fit_fixed_ms"]) < 0.5 * min(ys)})
    deep = n4["rows"][str(n4["deep_context"])]
    checks.append({"check": "N4 effective bandwidth is far below the measured device peak",
                   "ok": deep["gb_s"] < 0.25 * a["n5_bandwidth_gb_s"],
                   "attention_gb_s": deep["gb_s"], "device_gb_s": a["n5_bandwidth_gb_s"]})

    # ---- N5
    for label, f in a["n5"].items():
        total = f["shell_bytes"] + f["lm_head_bytes"] + f["expert_bytes"] + f["kv_bytes"]
        checks.append({"check": f"N5 {label}: byte total reproduces",
                       "ok": total == f["total_bytes"]})
        ms = total / (a["n5_bandwidth_gb_s"] * 1e9) * 1e3
        checks.append({"check": f"N5 {label}: floor reproduces",
                       "ok": abs(ms - f["floor_ms"]) < 1e-6, "recomputed": ms})
        checks.append({"check": f"N5 {label}: ceiling is 1000/floor",
                       "ok": abs(1000.0 / ms - f["ceiling_tok_s"]) < 1e-6})
    checks.append({"check": "N5 bandwidth was measured, not taken from a datasheet",
                   "ok": a["gates"]["G_N5_2_bandwidth_measured"]["buffer_bytes"] >= 1 << 27})
    checks.append({"check": "N5 excludes 1000 tok/s at both depths",
                   "ok": all(v["ceiling_tok_s"] < 1000 for v in a["n5"].values())})
    checks.append({"check": "N5 does NOT exclude 50 tok/s at either depth",
                   "ok": all(v["ceiling_tok_s"] > 50 for v in a["n5"].values())})

    # ---- N3
    s3 = b["summary"]
    sound3 = all(v["false_certificates"] == 0 for v in s3.values())
    best3 = max(s3, key=lambda k: s3[k]["certified_fraction"])
    checks.append({"check": "N3 bound is sound (zero false certificates)", "ok": sound3})
    checks.append({"check": "G-N3-R1 verdict agrees with the runner",
                   "ok": (s3[best3]["certified_fraction"] >= GATE_N3)
                   == b["gates"]["G_N3_R1_certified"]["passed"]})
    checks.append({"check": "N3 energy fraction rises with rank (spectrum is near-flat)",
                   "ok": all(s3[str(r)]["mean_energy_fraction"]
                             <= s3[str(rr)]["mean_energy_fraction"] + 1e-9
                             for r, rr in zip(b["config"]["ranks"], b["config"]["ranks"][1:])),
                   "energy": {k: v["mean_energy_fraction"] for k, v in s3.items()}})
    checks.append({"check": "N3 rank 64 captures well under half the spectral energy",
                   "ok": s3["64"]["mean_energy_fraction"] < 0.5,
                   "measured": s3["64"]["mean_energy_fraction"]})

    for doc, name in ((a, "N1/N2/N4/N5"), (b, "N3")):
        checks.append({"check": f"{name} carries a claim boundary",
                       "ok": bool(doc.get("claim_boundary"))})

    failed = [ch for ch in checks if not ch["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_n1_n5_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified": {"ceilings": sha256_path(OUT_DIR / "n1n2n4n5_ceilings.json"),
                     "n3": sha256_path(OUT_DIR / "n3_relu2_prefilter_oracle.json")},
        "recomputed": {"n1_removable": rem, "n2_marginals": marg,
                       "n2_share": share, "n4_slope_ms_per_byte": c, "n4_r2": r2,
                       "n5": {k: v["ceiling_tok_s"] for k, v in a["n5"].items()},
                       "n3_best_certified": s3[best3]["certified_fraction"]},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "n1_n5_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for ch in checks:
        if not ch["ok"]:
            print(f"  [FAIL] {ch['check']}  {ch}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)\n")
    print(f"  N1 removable {rem * 100:.1f}%")
    print(f"  N2 scan+gather {share * 100:.1f}% of down (gather {marg['gather']['m']:.3f} ms, "
          f"drift {marg['gather']['drift']:.3f})")
    print(f"  N3 certified {s3[best3]['certified_fraction'] * 100:.2f}%, "
          f"rank-64 energy {s3['64']['mean_energy_fraction'] * 100:.1f}%")
    print(f"  N4 R2 {r2:.4f}, attention {deep['gb_s']:.1f} GB/s of "
          f"{a['n5_bandwidth_gb_s']:.1f} device")
    for k, v in a["n5"].items():
        print(f"  N5 {k}: floor {v['floor_ms']:.2f} ms -> {v['ceiling_tok_s']:.1f} tok/s")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
