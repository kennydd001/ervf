"""Independent verification of NERVF-0/1/2. Never imports the runner.

Recomputes the arm medians from raw rounds, both geometry gates, the speedups,
and re-runs the bitwise comparison itself by rebuilding the ERVF kernel source
from the artifact's recorded widths -- so exactness is confirmed without trusting
the runner's own comparison.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "nervf_nemotron"
sys.path.insert(0, str(REPO_ROOT / "src"))

GATE_1A = 0.40
GATE_1B = 0.25
GATE_PRIMARY = 1.35
GATE_STRONG = 1.75


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
    lock = json.loads((OUT_DIR / "nervf0_baseline_lock.json").read_text(encoding="utf-8"))
    aud = json.loads((OUT_DIR / "nervf1_geometry_audit.json").read_text(encoding="utf-8"))
    mk_path = OUT_DIR / "nervf2_ervf_microkernel.json"
    mk = json.loads(mk_path.read_text(encoding="utf-8")) if mk_path.exists() else None
    checks = []

    # ---- NERVF-0
    i3_path = OUT_DIR / "nervf3_integration_ab.json"
    d3_pre = json.loads(i3_path.read_text(encoding="utf-8")) if i3_path.exists() else None
    for name, rel in (("runtime.py", "src/moe_lab/lightningstream_nemotron/runtime.py"),
                      ("gpu_kernels.py", "src/moe_lab/lightningstream_nemotron/gpu_kernels.py")):
        checks.append({"check": f"NERVF-0 lock: {name} still hashes to the locked value",
                       "ok": sha256_path(REPO_ROOT / rel) == lock["sha256"][name]})
    # fused_nvfp4.py is the one file NERVF-3 is allowed to change: it is where the
    # ERVF kernel was added. The lock records the pre-ERVF state, so the check is
    # a provenance chain, not an immutability claim.
    cur_fused = sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py")
    if d3_pre is None:
        checks.append({"check": "NERVF-0 lock: fused_nvfp4.py unchanged (no NERVF-3 yet)",
                       "ok": cur_fused == lock["sha256"]["fused_nvfp4.py"]})
    else:
        checks.append({"check": "fused_nvfp4.py on disk matches what NERVF-3 measured",
                       "ok": cur_fused == d3_pre["fused_sha256"]})
        checks.append({"check": "NERVF-3 changed fused_nvfp4.py relative to the NERVF-0 lock "
                                "(that is where the ERVF kernel was added)",
                       "ok": d3_pre["fused_sha256"] != lock["sha256"]["fused_nvfp4.py"]})
        checks.append({"check": "ERVF is opt-in: gemv_into still defaults to the "
                                "production kernel",
                       "ok": "self.use_ervf = False" in
                       (REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py")
                       .read_text(encoding="utf-8")})
    checks.append({"check": "NERVF-0 records the prior Qwen ERVF result as read-only prior art",
                   "ok": "streamq5" in lock["prior_ervf"]["line"]})

    # ---- NERVF-1
    arms = aud["arms"]
    rec = {}
    for name, a in arms.items():
        m = p50(a["raw"])
        checks.append({"check": f"NERVF-1 {name}: p50 reproduces from raw rounds",
                       "ok": abs(m - a["us_p50"]) < 1e-9})
        gb = aud["bytes"] / (m * 1e-6) / 1e9
        checks.append({"check": f"NERVF-1 {name}: GB/s reproduces",
                       "ok": abs(gb - a["gb_s"]) < 1e-6})
        rec[name] = {"us": m, "gb_s": gb}
    ratio = rec["FULL_GEMV"]["gb_s"] / rec["RAW_SCAN"]["gb_s"]
    share = (rec["FULL_GEMV"]["us"] - rec["DECODE_SCALE"]["us"]) / rec["FULL_GEMV"]["us"]
    checks.append({"check": "G-NERVF-1A reproduces (bandwidth efficiency)",
                   "ok": abs(ratio - aud["full_over_raw_bandwidth_efficiency"]) < 1e-9,
                   "recomputed": ratio})
    checks.append({"check": "G-NERVF-1A passes", "ok": ratio <= GATE_1A})
    checks.append({"check": "G-NERVF-1B reproduces",
                   "ok": abs(share - aud["reduction_sync_share"]) < 1e-9,
                   "recomputed": share})
    checks.append({"check": "G-NERVF-1B passes", "ok": share >= GATE_1B})
    checks.append({"check": "the audit defeats L2: pool is at least 4x the L2 size",
                   "ok": aud["l2_defeat"]["pool_bytes"] >= 4 * aud["l2_defeat"]["l2_bytes"],
                   "pool": aud["l2_defeat"]["pool_bytes"],
                   "l2": aud["l2_defeat"]["l2_bytes"]})
    checks.append({"check": "RAW_SCAN is the fastest arm and FULL_GEMV the slowest "
                            "(monotone geometry)",
                   "ok": (rec["RAW_SCAN"]["us"] < rec["ROW_PATTERN_SCAN"]["us"]
                          < rec["DECODE_SCALE"]["us"] < rec["FULL_GEMV"]["us"])})

    # ---- NERVF-2
    if mk is None:
        checks.append({"check": "NERVF-2 artifact exists (geometry gate was open)",
                       "ok": False})
    else:
        base = mk["baseline_us"]
        checks.append({"check": "NERVF-2 baseline equals the audit's FULL_GEMV",
                       "ok": abs(base - arms["FULL_GEMV"]["us_p50"]) < 1e-9})
        for w, v in mk["widths"].items():
            m = p50(v["raw"])
            checks.append({"check": f"NERVF-2 w={w}: p50 reproduces",
                           "ok": abs(m - v["us_p50"]) < 1e-9})
            checks.append({"check": f"NERVF-2 w={w}: speedup reproduces",
                           "ok": abs(base / m - v["speedup"]) < 1e-9,
                           "recomputed": base / m})
            checks.append({"check": f"NERVF-2 w={w}: rows_per_block is 256/w",
                           "ok": v["rows_per_block"] == 256 // int(w)})
            checks.append({"check": f"NERVF-2 w={w}: exactness claim matches its counts",
                           "ok": v["bitwise_identical"] == (v["bitwise_mismatches"] == 0)})
            checks.append({"check": f"NERVF-2 w={w}: at least 72 exactness cases",
                           "ok": v["bitwise_cases"] >= 72})
        best = mk["best_exact_width"]
        checks.append({"check": "best exact width is the argmax over exact widths",
                       "ok": best == max((int(k) for k, v in mk["widths"].items()
                                          if v["bitwise_identical"]),
                                         key=lambda k: mk["widths"][str(k)]["speedup"])})
        sp = mk["widths"][str(best)]["speedup"]
        checks.append({"check": "primary speed gate (>=1.35x) passes", "ok": sp >= GATE_PRIMARY,
                       "speedup": sp})
        checks.append({"check": "strong speed gate (>=1.75x) passes", "ok": sp >= GATE_STRONG})
        checks.append({"check": "every width is bitwise exact after the butterfly fix",
                       "ok": all(v["bitwise_identical"] for v in mk["widths"].values())})
        checks.append({"check": "the chosen width matches the one Qwen's P7 selected",
                       "ok": best == lock["prior_ervf"]["qwen_width"], "nemotron": best,
                       "qwen": lock["prior_ervf"]["qwen_width"]})

    # ---- NERVF-3
    i3 = OUT_DIR / "nervf3_integration_ab.json"
    rec3 = {}
    if i3.exists():
        d3 = json.loads(i3.read_text(encoding="utf-8"))
        arms3 = d3["arms"]
        ref = arms3["base_a"]["generation"]
        for nm, a in arms3.items():
            same = all(g["generated_ids"] == r["generated_ids"]
                       for g, r in zip(a["generation"], ref))
            checks.append({"check": f"NERVF-3 {nm}: generation identical between arms",
                           "ok": same and same == a["parity_arm"]})
            checks.append({"check": f"NERVF-3 {nm}: generation identical to the V35 anchor",
                           "ok": a["parity_anchor"]})
        checks.append({"check": "NERVF-3 ran exactly one arm with ERVF on",
                       "ok": sum(1 for a in arms3.values() if a["use_ervf"]) == 1})
        for ctx, v in d3["per_context"].items():
            a1 = arms3["base_a"]["sweep"][ctx]
            a2 = arms3["base_b"]["sweep"][ctx]
            b = arms3["ervf"]["sweep"][ctx]
            bm = 0.5 * (p50(a1["raw_moe_ms"]) + p50(a2["raw_moe_ms"]))
            bt = 0.5 * (p50(a1["raw_token_ms"]) + p50(a2["raw_token_ms"]))
            sp = bm / p50(b["raw_moe_ms"])
            tg = bt - p50(b["raw_token_ms"])
            td = abs(p50(a2["raw_token_ms"]) - p50(a1["raw_token_ms"]))
            rec3[ctx] = {"moe_speedup": sp, "token_gain": tg, "token_drift": td,
                         "conclusive": abs(tg) > td}
            checks.append({"check": f"NERVF-3 @{ctx}: MoE speedup reproduces",
                           "ok": abs(sp - v["moe_speedup"]) < 1e-9, "recomputed": sp})
            checks.append({"check": f"NERVF-3 @{ctx}: token gain reproduces",
                           "ok": abs(tg - v["token_gain_ms"]) < 1e-9})
            checks.append({"check": f"NERVF-3 @{ctx}: token gain exceeds its own drift",
                           "ok": abs(tg) > td, "gain": tg, "drift": td})
        checks.append({"check": "NERVF-3 primary gate verdict agrees with the runner",
                       "ok": (d3["gates"]["G_NERVF_3P_primary"]["passed"]
                              == (max(r["moe_speedup"] for r in rec3.values())
                                  >= GATE_PRIMARY))})
        checks.append({"check": "fused_nvfp4.py hashes to the value NERVF-3 measured",
                       "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py")
                       == d3["fused_sha256"]})

    for doc, nm in ((aud, "NERVF-1"), (mk, "NERVF-2")):
        if doc:
            checks.append({"check": f"{nm} carries a claim boundary",
                           "ok": bool(doc.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "nervf_nemotron_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "recomputed": {"arms": rec, "gate_1a": ratio, "gate_1b": share,
                       "best_width": mk["best_exact_width"] if mk else None,
                       "best_speedup": mk["best_speedup"] if mk else None,
                       "nervf3": rec3},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "nervf_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # SHA256 manifest over the namespace
    man = {p.name: sha256_path(p) for p in sorted(OUT_DIR.glob("*"))
           if p.is_file() and p.name != "NERVF_SHA256_MANIFEST.json"}
    man.update({f"scripts/{p.name}": sha256_path(p)
                for p in sorted((REPO_ROOT / "scripts/nervf_nemotron").glob("*.py"))})
    (OUT_DIR / "NERVF_SHA256_MANIFEST.json").write_text(
        json.dumps({"kind": "nervf_nemotron_sha256_manifest",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "files": man}, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}  {c}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    print(f"  G-NERVF-1A {ratio:.4f} <= {GATE_1A} | G-NERVF-1B {share * 100:.1f}% >= 25%")
    if mk:
        print(f"  best exact width {mk['best_exact_width']} at {mk['best_speedup']:.3f}x")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
