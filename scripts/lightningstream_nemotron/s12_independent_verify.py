"""Independent verification of S12. Imports nothing from the runner or runtime.

Recomputes every p50 from the raw millisecond samples with its own median, the
marginals as arm minus base1, the drift, and all three gates. Also checks the
claim the method rests on: that `runtime.py` contains no probe hook at all, so
the loop the marginals were measured against is the shipped loop.
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
EXPECTED_SAMPLES = 16
S8_MOE_TERM_MS = 39.523
BASE_ARMS = ("base1", "base2")


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
    path = OUT_DIR / "s12_in_loop_attribution.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    arms = res["arms"]
    contexts = [str(c) for c in res["config"]["contexts"]]
    checks: list[dict] = []

    # ---- the method's own premise: the measured loop is the shipped loop
    src = RUNTIME.read_text(encoding="utf-8")
    checks.append({"check": "runtime.py contains no probe hook",
                   "ok": not re.search(r"\bprobe\b", src)})
    checks.append({"check": "runtime.py on disk still hashes to the measured value",
                   "ok": sha256_path(RUNTIME) == res["runtime_sha256"]})
    checks.append({"check": "probe lives in a subclass in the runner",
                   "ok": "class ProbedRuntime(LightningRuntime)" in
                         (REPO_ROOT / "scripts/lightningstream_nemotron/"
                          "s12_in_loop_attribution.py").read_text(encoding="utf-8")})

    # ---- identity
    ref = arms["base1"]["generation_token_ids"]
    checks.append({"check": "reference generation is 2 prompts x 32 tokens",
                   "ok": len(ref) == 2 and all(len(g) == 32 for g in ref)})
    for name, arm in arms.items():
        same = arm["generation_token_ids"] == ref
        checks.append({"check": f"{name}: generation identical to base1",
                       "ok": same and same == arm["identical_to_base1"]})

    # ---- timings and marginals
    rec_p50, marginal, drift = {}, {}, {}
    for name, arm in arms.items():
        rec_p50[name] = {}
        for ctx in contexts:
            row = arm["context_sweep"][ctx]
            checks.append({"check": f"{name}@{ctx}: {EXPECTED_SAMPLES} timing samples",
                           "ok": len(row["raw_ms"]) == EXPECTED_SAMPLES})
            v = p50(row["raw_ms"])
            checks.append({"check": f"{name}@{ctx}: p50 reproduces from raw samples",
                           "ok": abs(v - row["ms"]["p50"]) < 1e-9})
            rec_p50[name][ctx] = v

    for ctx in contexts:
        drift[ctx] = abs(rec_p50["base2"][ctx] - rec_p50["base1"][ctx])
        marginal[ctx] = {a: rec_p50[a][ctx] - rec_p50["base1"][ctx]
                         for a in arms if a not in BASE_ARMS}
        for a, v in marginal[ctx].items():
            checks.append({"check": f"marginal[{a}]@{ctx} reproduces",
                           "ok": abs(v - res["marginal_ms_per_token"][ctx][a]) < 1e-9,
                           "recomputed": v})
        checks.append({"check": f"drift@{ctx} reproduces",
                       "ok": abs(drift[ctx] - res["baseline_drift_ms"][ctx]) < 1e-9})

    deep = contexts[-1]
    smallest = min(abs(v) for v in marginal[deep].values())
    marg_sum = {c: sum(marginal[c].values()) for c in contexts}
    identity_ok = all(a["generation_token_ids"] == ref for a in arms.values())
    drift_ok = drift[deep] < smallest
    sanity_ok = marg_sum[deep] <= S8_MOE_TERM_MS

    g = res["gates"]
    checks.append({"check": "G-S12-C1 verdict agrees with the runner",
                   "ok": identity_ok == g["G_S12_C1_identity"]["passed"]})
    checks.append({"check": "G-S12-D1 verdict agrees with the runner",
                   "ok": drift_ok == g["G_S12_D1_drift"]["passed_at_deep"],
                   "drift_ms": drift[deep], "smallest_marginal_ms": smallest})
    checks.append({"check": "G-S12-S1 verdict agrees with the runner",
                   "ok": sanity_ok == g["G_S12_S1_sanity"]["passed"],
                   "sum_ms": marg_sum[deep]})
    checks.append({"check": "S8 MoE term used by the sanity gate is unchanged",
                   "ok": abs(g["G_S12_S1_sanity"]["s8_moe_term_ms"] - S8_MOE_TERM_MS) < 1e-9})

    # ---- hygiene the preregistration promised
    blob = json.dumps(res)
    checks.append({"check": "no tokens-per-second figure anywhere in the result",
                   "ok": "tok_s" not in blob and "tokens_per_second" not in blob})
    # Field NAMES, not prose: an earlier version of this check matched the claim
    # boundary's own sentence saying the marginals are not shares, which is the
    # opposite of a violation. Same class of defect as the S9 block-size probe's
    # over-strict equality check.
    def all_keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from all_keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from all_keys(v)

    # Word-bounded: `shared` is the name of the shared-expert arm, not a share.
    offenders = [k for k in all_keys(res)
                 if re.search(r"\bshares?\b|percent|\bpct\b|\bfraction", k, re.I)]
    checks.append({"check": "no result field is named as a share or percentage",
                   "ok": not offenders, "offending_keys": offenders})
    checks.append({"check": "result carries a claim boundary naming the lower bound",
                   "ok": "LOWER BOUND" in res.get("claim_boundary", "")})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_s12_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"p50_ms": rec_p50, "marginal_ms": marginal,
                       "drift_ms": drift, "marginal_sum_ms": marg_sum,
                       "below_noise_floor": {c: [a for a, v in marginal[c].items()
                                                 if abs(v) <= drift[c]]
                                             for c in contexts}},
        "gates": {"G_S12_C1_identity": identity_ok,
                  "G_S12_D1_drift": drift_ok,
                  "G_S12_S1_sanity": sanity_ok},
        "checks": checks,
        "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "s12_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    for ctx in contexts:
        print(f"\nctx {ctx}: base {rec_p50['base1'][ctx]:.3f} ms, drift {drift[ctx]:.3f} ms")
        for a, v in sorted(marginal[ctx].items(), key=lambda kv: -kv[1]):
            print(f"   {a:<8} {v:+7.3f} ms" + ("  (below noise floor)"
                                               if abs(v) <= drift[ctx] else ""))
        print(f"   {'sum':<8} {marg_sum[ctx]:+7.3f} ms")
    print(f"\nverdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
