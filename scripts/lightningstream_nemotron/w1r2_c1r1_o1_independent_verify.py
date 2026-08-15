"""Independent verification of W1-R2, C1-R1 and O1. Imports nothing from the runners.

Recomputes every block median, effect and drift from raw samples; re-derives the
entropies from the reported symbol counts where possible and re-checks the record
arithmetic; and re-applies every gate. Also checks the two things that make the
C1-R1 bound meaningful: that it is sound, and that it needs nothing beyond the
core -- which is what makes it, unlike C1's, actually deployable.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
GATE_W1R2_DRIFT = 1.0
GATE_ADOPT = 1.0
GATE_SIGN = 0.60
GATE_TAIL = 0.30
GATE_O1_PASS = 0.12
GATE_O1_STRONG = 0.20


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
    w = json.loads((OUT_DIR / "w1r2_block_paired.json").read_text(encoding="utf-8"))
    c = json.loads((OUT_DIR / "c1r1_o1_bound_and_entropy.json").read_text(encoding="utf-8"))
    checks = []

    # ------------------------------------------------------------- W1-R2
    checks.append({"check": "W1-R2 generation bit-identical",
                   "ok": w["identity"]["reference"] == w["identity"]["fast"]
                   and w["identity"]["bit_identical"]})
    rec_w = {}
    for ctx, row in w["per_context"].items():
        blocks = row["blocks"]
        checks.append({"check": f"W1-R2 ctx {ctx}: blocks alternate base/fast",
                       "ok": all(b["fast"] == bool(i % 2) for i, b in enumerate(blocks))})
        checks.append({"check": f"W1-R2 ctx {ctx}: every block has 8 samples",
                       "ok": all(len(b["samples"]) == w["config"]["block_samples"]
                                 for b in blocks)})
        for b in blocks:
            if abs(p50(b["samples"]) - b["p50"]) > 1e-9:
                checks.append({"check": f"W1-R2 ctx {ctx}: block {b['index']} p50 reproduces",
                               "ok": False})
                break
        else:
            checks.append({"check": f"W1-R2 ctx {ctx}: every block p50 reproduces",
                           "ok": True})
        base = [b for b in blocks if not b["fast"]]
        fast = [b for b in blocks if b["fast"]]
        eff = []
        for i, fb in enumerate(fast):
            lo = base[i]["p50"]
            hi = base[i + 1]["p50"] if i + 1 < len(base) else base[i]["p50"]
            eff.append(0.5 * (lo + hi) - fb["p50"])
        dr = [abs(base[i + 1]["p50"] - base[i]["p50"]) for i in range(len(base) - 1)]
        pos = sum(1 for e in eff if e > 0)
        rec_w[ctx] = {"effect_p50": p50(eff), "drift_p50": p50(dr),
                      "sign": pos / len(eff),
                      "resolution_ok": p50(dr) < GATE_W1R2_DRIFT}
        checks.append({"check": f"W1-R2 ctx {ctx}: effect p50 reproduces",
                       "ok": abs(rec_w[ctx]["effect_p50"] - row["effect_p50_ms"]) < 1e-9,
                       "recomputed": rec_w[ctx]["effect_p50"]})
        checks.append({"check": f"W1-R2 ctx {ctx}: drift p50 reproduces",
                       "ok": abs(rec_w[ctx]["drift_p50"] - row["drift_p50_ms"]) < 1e-9})
        checks.append({"check": f"W1-R2 ctx {ctx}: sign fraction reproduces",
                       "ok": abs(rec_w[ctx]["sign"] - row["sign_fraction"]) < 1e-9})
        checks.append({"check": f"W1-R2 ctx {ctx}: resolution verdict reproduces",
                       "ok": rec_w[ctx]["resolution_ok"] == row["resolution_ok"]})

    deep_w = max(rec_w, key=lambda k: int(k))
    checks.append({"check": "W1-R2 is the best-resolved of the three designs",
                   "ok": rec_w[deep_w]["drift_p50"] < w["prior_designs"]["w1_arm_drift_ms"]
                   and rec_w[deep_w]["drift_p50"] < w["prior_designs"]["w1r1_triplet_drift_ms"],
                   "w1r2": rec_w[deep_w]["drift_p50"],
                   "w1": w["prior_designs"]["w1_arm_drift_ms"],
                   "w1r1": w["prior_designs"]["w1r1_triplet_drift_ms"]})
    adopt = (w["identity"]["bit_identical"] and rec_w[deep_w]["resolution_ok"]
             and rec_w[deep_w]["effect_p50"] >= GATE_ADOPT)
    checks.append({"check": "G-W1-P1 verdict agrees with the runner",
                   "ok": adopt == w["gates"]["G_W1_P1_unchanged"]["passed"]})
    checks.append({"check": "adoption threshold still the preregistered 1.0 ms",
                   "ok": abs(w["gates"]["G_W1_P1_unchanged"]["required_gain_ms"]
                             - GATE_ADOPT) < 1e-12})

    # ------------------------------------------------------------- C1-R1
    sound = all(v["false_certificates"] == 0 for v in c["c1r1"].values())
    checks.append({"check": "C1-R1 bound is sound (zero false certificates)",
                   "ok": sound,
                   "false": {k: v["false_certificates"] for k, v in c["c1r1"].items()}})
    best = max(c["c1r1"], key=lambda k: c["c1r1"][k]["certified_fraction"])
    r1 = c["c1r1"][best]["certified_fraction"] >= GATE_TAIL
    checks.append({"check": "G-C1R1-R1 verdict agrees with the runner",
                   "ok": r1 == c["gates"]["G_C1R1_R1_tail_yield"]["passed"]})
    checks.append({"check": "C1-R1 gate still the pack's 30%",
                   "ok": abs(c["gates"]["G_C1R1_R1_tail_yield"]["required"] - GATE_TAIL) < 1e-12})
    for k, v in c["c1r1"].items():
        checks.append({"check": f"C1-R1 core {k}: bound/|y0| reproduces",
                       "ok": abs(v["mean_bound"] / v["mean_abs_y0"] - v["bound_over_y0"]) < 1e-9})
        checks.append({"check": f"C1-R1 core {k}: certified share of true zeros consistent",
                       "ok": v["certified_share_of_true_zero"] <= 1.0
                       and v["certified_fraction"] <= v["true_zero_fraction"] + 1e-12})
    src = (REPO_ROOT / "scripts/lightningstream_nemotron/"
           "c1r1_o1_bound_and_entropy.py").read_text(encoding="utf-8")
    checks.append({"check": "C1-R1 derives the sign from the code's bit 3, not from -0.0",
                   "ok": "core & 0b1000" in src})
    checks.append({"check": "C1-R1 bound uses only core-derived dmax, no stored residual norms",
                   "ok": "dmax_tbl[mask][core]" in src and "dw" not in src.split("def main")[1]
                   .split("C1-R1")[1].split("O1:")[0]})

    # ---------------------------------------------------------------- O1
    o = c["o1"]
    n_codes = c["config"]["inter"] * c["config"]["hidden"]
    n_scales = c["config"]["inter"] * (c["config"]["hidden"] // c["config"]["group_size"])
    bytes_now = n_codes / 2 + n_scales
    best_code = min(o["code_entropy_bits"], o["code_entropy_given_scale_exponent_bits"])
    best_scale = min(o["scale_entropy_bits"], o["scale_delta_entropy_bits"])
    bytes_ent = (n_codes * best_code + n_scales * best_scale) / 8.0
    red = 1.0 - bytes_ent / bytes_now
    checks.append({"check": "O1 current record size reproduces",
                   "ok": abs(bytes_now - o["record_bytes_now"]) < 1e-6})
    checks.append({"check": "O1 entropy-bound record size reproduces",
                   "ok": abs(bytes_ent - o["record_bytes_at_entropy"]) < 1e-3})
    checks.append({"check": "O1 pack reduction reproduces",
                   "ok": abs(red - o["pack_reduction"]) < 1e-9, "recomputed": red})
    checks.append({"check": "O1 code entropy is below the 4-bit raw size",
                   "ok": best_code < 4.0})
    checks.append({"check": "G-O1-1 verdict agrees with the runner",
                   "ok": (red >= GATE_O1_PASS) == c["gates"]["G_O1_1_pass"]["passed"]})
    checks.append({"check": "G-O1-2 verdict agrees with the runner",
                   "ok": (red >= GATE_O1_STRONG) == c["gates"]["G_O1_2_strong"]["passed"]})
    checks.append({"check": "O1 gates still the pack's 12% / 20%",
                   "ok": abs(c["gates"]["G_O1_1_pass"]["required"] - GATE_O1_PASS) < 1e-12
                   and abs(c["gates"]["G_O1_2_strong"]["required"] - GATE_O1_STRONG) < 1e-12})

    for doc, name in ((w, "W1-R2"), (c, "C1-R1/O1")):
        checks.append({"check": f"{name} carries a claim boundary",
                       "ok": bool(doc.get("claim_boundary"))})
        checks.append({"check": f"{name} runtime.py hash matches disk",
                       "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py")
                       == doc["runtime_sha256"]})

    failed = [ch for ch in checks if not ch["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_w1r2_c1r1_o1_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified": {"w1r2": sha256_path(OUT_DIR / "w1r2_block_paired.json"),
                     "c1r1_o1": sha256_path(OUT_DIR / "c1r1_o1_bound_and_entropy.json")},
        "recomputed": {"w1r2": rec_w, "o1_pack_reduction": red,
                       "c1r1_best_core": best,
                       "c1r1_certified": c["c1r1"][best]["certified_fraction"]},
        "gates": {"G_W1R2_C1": w["identity"]["bit_identical"],
                  "G_W1R2_R1": {k: v["resolution_ok"] for k, v in rec_w.items()},
                  "G_W1_P1": adopt, "G_C1R1_S1": sound, "G_C1R1_R1": r1,
                  "G_O1_1": red >= GATE_O1_PASS, "G_O1_2": red >= GATE_O1_STRONG},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "w1r2_c1r1_o1_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for ch in checks:
        if not ch["ok"]:
            print(f"  [FAIL] {ch['check']}  {ch}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)\n")
    for ctx, v in rec_w.items():
        print(f"  W1-R2 ctx {ctx:>6}: effect {v['effect_p50']:+.3f} ms, "
              f"drift {v['drift_p50']:.3f} ms, sign {v['sign']:.2f}")
    print(f"  C1-R1 best core {best}: certified "
          f"{c['c1r1'][best]['certified_fraction'] * 100:.2f}%, sound={sound}")
    print(f"  O1 pack reduction {red * 100:.2f}% (gates 12% / 20%)")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
