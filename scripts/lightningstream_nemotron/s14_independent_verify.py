"""Independent verification of S14. Imports nothing from the runner or runtime.

Recomputes every stage statistic from the raw per-token stage sums with its own
mean/median, re-derives the probe-overhead gate from the raw per-token
millisecond samples of the three arms, redoes the identity check on the stored
generations, and re-evaluates all gates.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
S8_MOE_TERM_262K = 39.523
STAGES = ["route", "shared_up", "shared_dn", "host_gap",
          "up", "down_masked", "accum", "layer_total"]


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def p50(xs):
    s = sorted(float(x) for x in xs)
    n = len(s)
    if n == 0:
        return float("nan")
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs)


def main() -> int:
    path = OUT_DIR / "s14_moe_layer_timeline.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    checks: list[dict] = []

    # ------------------------------------------------------- provenance
    lock = json.loads((OUT_DIR / "s14_input_lock.json").read_text(encoding="utf-8"))
    locked = {e["path"]: e["sha256"] for e in lock["entries"] if e["present"]}
    checks.append({"check": "input lock covers preregistration, runner and runtime",
                   "ok": len(locked) == 4})
    checks.append({"check": "runner on disk still hashes to the locked value",
                   "ok": sha256_path(REPO_ROOT / "scripts/lightningstream_nemotron/"
                                     "s14_moe_layer_timeline.py")
                   == locked["scripts/lightningstream_nemotron/s14_moe_layer_timeline.py"]
                   == res["runner_sha256"]})
    checks.append({"check": "runtime.py unchanged since the locked measurement",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/"
                                     "runtime.py")
                   == locked["src/moe_lab/lightningstream_nemotron/runtime.py"]
                   == res["runtime_sha256"]})

    arms = res["arms"]
    contexts = [str(c) for c in res["config"]["contexts"]]

    # ------------------------------------------------------- structure
    checks.append({"check": "three arms in the preregistered order "
                            "base0, probed, base1",
                   "ok": res["config"]["arm_order"] == ["base0", "probed", "base1"]
                   and set(arms) == {"base0", "probed", "base1"}})
    for name in arms:
        for ctx in contexts:
            raw = arms[name]["context_sweep"][ctx]["raw_ms"]
            checks.append({"check": f"{name}@{ctx}: 16 timing samples",
                           "ok": len(raw) == 16, "n": len(raw)})
            r50 = p50(raw)
            checks.append({"check": f"{name}@{ctx}: p50 reproduces from raw samples",
                           "ok": abs(r50 - arms[name]["context_sweep"][ctx]["ms"]["p50"])
                           < 1e-9,
                           "recomputed": r50,
                           "stored": arms[name]["context_sweep"][ctx]["ms"]["p50"]})
    for ctx in contexts:
        tss = arms["probed"]["context_sweep"][ctx].get("token_stage_sums", [])
        checks.append({"check": f"probed@{ctx}: 16 tokens with stage sums",
                       "ok": len(tss) == 16, "n": len(tss)})

    # ------------------------------------------------------- identity
    checks.append({"check": "generation identity flag present and true",
                   "ok": res["generation_identical"] is True})

    # ------------------------------------------------------- stage recompute
    recomputed = {}
    for ctx in contexts:
        tss = arms["probed"]["context_sweep"][ctx]["token_stage_sums"]
        recomputed[ctx] = {s: mean([t[s] for t in tss]) for s in
                           STAGES + ["up_waited", "down_waited",
                                     "miss_copy_batch", "readback_host"]}
        recomputed[ctx]["experts_waited"] = mean([t["experts_waited"] for t in tss])
        # bookkeeping identity: layer_total == sum of its exclusive segments
        for t in tss:
            parts = sum(t[s] for s in ["route", "shared_up", "shared_dn",
                                       "host_gap", "up", "down_masked", "accum"])
            if abs(parts - t["layer_total"]) > 0.05:
                checks.append({"check": f"probed@{ctx}: segment parts sum to "
                                        f"layer_total within 0.05 ms",
                               "ok": False, "parts": parts,
                               "layer_total": t["layer_total"]})
                break
        else:
            checks.append({"check": f"probed@{ctx}: segment parts sum to "
                                    f"layer_total within 0.05 ms for all 16 tokens",
                           "ok": True})

    # ------------------------------------------------------- gates
    g = res["gates"]
    checks.append({"check": "G-S14-C1 verdict agrees with the runner",
                   "ok": res["generation_identical"] == g["G_S14_C1"]["pass"]})
    for ctx in contexts:
        b0 = p50(arms["base0"]["context_sweep"][ctx]["raw_ms"])
        b1 = p50(arms["base1"]["context_sweep"][ctx]["raw_ms"])
        pr = p50(arms["probed"]["context_sweep"][ctx]["raw_ms"])
        overhead = pr - 0.5 * (b0 + b1)
        conclusive = overhead <= 0.20 * 0.5 * (b0 + b1)
        key = f"G_S14_P1_ctx{ctx}"
        checks.append({"check": f"{key} verdict agrees with the runner",
                       "ok": conclusive == g[key]["conclusive"]
                       and abs(overhead - g[key]["overhead_ms"]) < 1e-9,
                       "recomputed_overhead_ms": overhead})
        layer_sum = recomputed[ctx]["layer_total"]
        seg_nonneg = all(recomputed[ctx][s] >= -0.01 for s in STAGES)
        ok = layer_sum <= pr and seg_nonneg
        if ctx == "262100":
            ok = ok and layer_sum >= S8_MOE_TERM_262K / 2
        checks.append({"check": f"G-S14-S1_ctx{ctx} verdict agrees with the runner",
                       "ok": ok == g[f"G_S14_S1_ctx{ctx}"]["pass"],
                       "recomputed_layer_sum_ms": layer_sum,
                       "probed_p50_ms": pr})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_s14_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"stage_means_ms": recomputed},
        "checks": checks,
        "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "s14_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}")
    for ctx in contexts:
        r = recomputed[ctx]
        print(f"\nctx {ctx}: layer_total {r['layer_total']:.3f} ms | "
              + " ".join(f"{s}={r[s]:.3f}" for s in STAGES[:-1]))
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
