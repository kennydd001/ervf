"""Independent verification of S12-R1. Imports nothing from the runner or runtime.

Rebuilds the bracket schedule from scratch, recomputes every p50 from raw
samples, re-derives each marginal as probe minus the mean of its two bracketing
baselines, recomputes local and global drift, and re-applies all four gates.
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
PROBES = ["up", "down", "router", "shared", "accum"]
EXPECTED_SAMPLES = 16
S8_MOE_TERM_MS = 39.523


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
    path = OUT_DIR / "s12r1_bracketed_attribution.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    arms = res["arms"]
    contexts = [str(c) for c in res["config"]["contexts"]]
    checks: list[dict] = []

    expected_schedule = []
    for k, probe in enumerate(PROBES):
        expected_schedule += [f"base{k}", probe]
    expected_schedule.append(f"base{len(PROBES)}")
    checks.append({"check": "schedule is base/probe alternating with a closing baseline",
                   "ok": res["config"]["schedule"] == expected_schedule,
                   "expected": expected_schedule})
    checks.append({"check": "every scheduled arm produced results",
                   "ok": sorted(arms) == sorted(expected_schedule)})

    checks.append({"check": "runtime.py contains no probe hook",
                   "ok": not re.search(r"\bprobe\b", RUNTIME.read_text(encoding="utf-8"))})
    checks.append({"check": "runtime.py on disk still hashes to the measured value",
                   "ok": sha256_path(RUNTIME) == res["runtime_sha256"]})

    ref = arms["base0"]["generation_token_ids"]
    checks.append({"check": "reference generation is 2 prompts x 32 tokens",
                   "ok": len(ref) == 2 and all(len(g) == 32 for g in ref)})
    identity_ok = True
    for name, arm in arms.items():
        same = arm["generation_token_ids"] == ref
        identity_ok &= same
        checks.append({"check": f"{name}: generation identical to base0",
                       "ok": same and same == arm["identical_to_base0"]})

    rec = {}
    for name, arm in arms.items():
        rec[name] = {}
        for ctx in contexts:
            row = arm["context_sweep"][ctx]
            checks.append({"check": f"{name}@{ctx}: {EXPECTED_SAMPLES} timing samples",
                           "ok": len(row["raw_ms"]) == EXPECTED_SAMPLES})
            v = p50(row["raw_ms"])
            checks.append({"check": f"{name}@{ctx}: p50 reproduces from raw samples",
                           "ok": abs(v - row["ms"]["p50"]) < 1e-9})
            rec[name][ctx] = v

    marginal, local_drift, reported, global_drift = {}, {}, {}, {}
    for ctx in contexts:
        marginal[ctx], local_drift[ctx], reported[ctx] = {}, {}, {}
        for k, probe in enumerate(PROBES):
            b0, b1 = rec[f"base{k}"][ctx], rec[f"base{k + 1}"][ctx]
            m = rec[probe][ctx] - 0.5 * (b0 + b1)
            marginal[ctx][probe] = m
            local_drift[ctx][probe] = abs(b1 - b0)
            reported[ctx][probe] = abs(m) > abs(b1 - b0)
            checks.append({"check": f"marginal[{probe}]@{ctx} reproduces",
                           "ok": abs(m - res["marginal_ms_per_token"][ctx][probe]) < 1e-9,
                           "recomputed": m})
            checks.append({"check": f"local drift[{probe}]@{ctx} reproduces",
                           "ok": abs(local_drift[ctx][probe]
                                     - res["local_drift_ms"][ctx][probe]) < 1e-9})
            checks.append({"check": f"noise-floor flag[{probe}]@{ctx} reproduces",
                           "ok": reported[ctx][probe] == res["reported_above_noise"][ctx][probe]})
        global_drift[ctx] = abs(rec[f"base{len(PROBES)}"][ctx] - rec["base0"][ctx])
        checks.append({"check": f"global drift@{ctx} reproduces",
                       "ok": abs(global_drift[ctx] - res["global_drift_ms"][ctx]) < 1e-9})

    deep = contexts[-1]
    reported_sum = {c: sum(v for p, v in marginal[c].items() if reported[c][p])
                    for c in contexts}
    largest = {c: max(marginal[c].values()) for c in contexts}
    sanity_ok = reported_sum[deep] <= S8_MOE_TERM_MS
    conclusive = {c: global_drift[c] < largest[c] for c in contexts}

    g = res["gates"]
    checks.append({"check": "G-S12R-C1 verdict agrees with the runner",
                   "ok": identity_ok == g["G_S12R_C1_identity"]["passed"]})
    checks.append({"check": "G-S12R-S1 verdict agrees with the runner",
                   "ok": sanity_ok == g["G_S12R_S1_sanity"]["passed"],
                   "sum_ms": reported_sum})
    checks.append({"check": "G-S12R-T1 conclusiveness agrees with the runner",
                   "ok": conclusive == {c: bool(v) for c, v
                                        in g["G_S12R_T1_thermal"]["conclusive"].items()},
                   "recomputed": conclusive, "global_drift_ms": global_drift})
    checks.append({"check": "S8 MoE term used by the sanity gate is unchanged",
                   "ok": abs(g["G_S12R_S1_sanity"]["s8_moe_term_ms"] - S8_MOE_TERM_MS) < 1e-9})

    def all_keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from all_keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from all_keys(v)

    offenders = [k for k in all_keys(res)
                 if re.search(r"\bshares?\b|percent|\bpct\b|\bfraction", k, re.I)]
    checks.append({"check": "no result field is named as a share or percentage",
                   "ok": not offenders, "offending_keys": offenders})
    blob = json.dumps(res)
    checks.append({"check": "no tokens-per-second figure anywhere in the result",
                   "ok": "tok_s" not in blob and "tokens_per_second" not in blob})
    checks.append({"check": "claim boundary names the lower bound",
                   "ok": "LOWER BOUND" in res.get("claim_boundary", "")})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_s12r1_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"p50_ms": rec, "marginal_ms": marginal,
                       "local_drift_ms": local_drift, "global_drift_ms": global_drift,
                       "reported_above_noise": reported,
                       "reported_sum_ms": reported_sum},
        "gates": {"G_S12R_C1_identity": identity_ok,
                  "G_S12R_S1_sanity": sanity_ok,
                  "G_S12R_T1_conclusive": conclusive},
        "checks": checks,
        "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "s12r1_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    for ctx in contexts:
        print(f"\nctx {ctx}: base0 {rec['base0'][ctx]:.3f} ms, "
              f"global drift {global_drift[ctx]:.3f} ms")
        for p, v in sorted(marginal[ctx].items(), key=lambda kv: -kv[1]):
            flag = "" if reported[ctx][p] else "  (below its local drift)"
            print(f"   {p:<8} {v:+7.3f} ms  local drift {local_drift[ctx][p]:5.3f}{flag}")
        print(f"   {'sum':<8} {reported_sum[ctx]:+7.3f} ms (reported only)")
    print(f"\nverdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
