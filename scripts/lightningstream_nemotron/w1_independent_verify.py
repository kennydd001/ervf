"""Independent verification of W1. Imports nothing from the runner or runtime.

Recomputes every p50 from the raw samples, the bracketed baselines, the gains,
the drift, and re-applies the four gates. Also checks the thing the identity gate
rests on: that the fast path really is opt-in and off by default.
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
ADOPT_GAIN_MS = 1.0
S14_HOST_GAP_MS = {"0": 5.058, "262100": 4.672}


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
    path = OUT_DIR / "w1_host_path_ab.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    arms = res["arms"]
    checks = []

    src = RUNTIME.read_text(encoding="utf-8")
    checks.append({"check": "fast_host is declared and defaults to False",
                   "ok": bool(re.search(r"fast_host\s*=\s*False", src))})
    checks.append({"check": "the fast path is a separate method, not an edit of _moe_cached",
                   "ok": "_moe_cached_fast" in src and "def _moe_cached(" in src})
    checks.append({"check": "runtime.py on disk hashes to the measured value",
                   "ok": sha256_path(RUNTIME) == res["runtime_sha256"]})

    ref = arms["base1"]["generation_token_ids"]
    checks.append({"check": "reference generation is 2 prompts x 64 tokens",
                   "ok": len(ref) == 2 and all(len(g) == 64 for g in ref)})
    identity = True
    for name, arm in arms.items():
        same = arm["generation_token_ids"] == ref
        identity &= same
        checks.append({"check": f"{name}: generation identical to base1",
                       "ok": same and same == arm["identical_to_base1"]})
    checks.append({"check": "exactly one arm ran with fast_host on",
                   "ok": sum(1 for a in arms.values() if a["fast_host"]) == 1})

    rec = {}
    contexts = [str(c) for c in res["config"]["contexts"]]
    for ctx in contexts:
        vals = {}
        for name, arm in arms.items():
            raw = arm["context_sweep"][ctx]["raw_ms"]
            m = p50(raw)
            checks.append({"check": f"{name}@{ctx}: p50 reproduces from raw samples",
                           "ok": abs(m - arm["context_sweep"][ctx]["ms"]["p50"]) < 1e-9})
            checks.append({"check": f"{name}@{ctx}: at least 32 samples",
                           "ok": len(raw) >= 32, "n": len(raw)})
            vals[name] = m
        base = 0.5 * (vals["base1"] + vals["base2"])
        drift = abs(vals["base2"] - vals["base1"])
        gain = base - vals["fast"]
        rec[ctx] = {"base": base, "fast": vals["fast"], "drift": drift, "gain": gain,
                    "relative": gain / base, "conclusive": abs(gain) > drift,
                    "within_s14": gain <= S14_HOST_GAP_MS.get(ctx, 1e9) + drift}
        stored = res["per_context"][ctx]
        checks.append({"check": f"gain@{ctx} reproduces",
                       "ok": abs(gain - stored["gain_ms"]) < 1e-9, "recomputed": gain})
        checks.append({"check": f"conclusiveness@{ctx} reproduces",
                       "ok": rec[ctx]["conclusive"] == stored["conclusive"]})
        checks.append({"check": f"gain@{ctx} within the S14 host_gap bound",
                       "ok": rec[ctx]["within_s14"]})

    deep, shallow = contexts[-1], contexts[0]
    adopt = (identity and rec[deep]["gain"] >= ADOPT_GAIN_MS
             and rec[shallow]["gain"] >= 0.0)
    checks.append({"check": "G-W1-C1 verdict agrees with the runner",
                   "ok": arms["fast"]["identical_to_base1"]
                   == res["gates"]["G_W1_C1_identity"]["passed"]})
    checks.append({"check": "G-W1-P1 verdict agrees with the runner",
                   "ok": adopt == res["gates"]["G_W1_P1_adopt"]["passed"],
                   "recomputed_gain_at_deep": rec[deep]["gain"]})
    checks.append({"check": "adoption threshold is still the preregistered 1.0 ms",
                   "ok": abs(res["gates"]["G_W1_P1_adopt"]["required_gain_ms_at_deep"]
                             - ADOPT_GAIN_MS) < 1e-12})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_w1_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": rec,
        "gates": {"G_W1_C1": identity, "G_W1_P1": adopt,
                  "G_W1_D1": {c: rec[c]["conclusive"] for c in contexts},
                  "G_W1_S1": {c: rec[c]["within_s14"] for c in contexts}},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "w1_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}  {c}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    for ctx in contexts:
        v = rec[ctx]
        print(f"  ctx {ctx:>6}: base {v['base']:7.3f} fast {v['fast']:7.3f} "
              f"gain {v['gain']:+.3f} ms ({v['relative'] * 100:+.1f}%) "
              f"drift {v['drift']:.3f} conclusive={v['conclusive']}")
    print(f"  G-W1-C1 {identity} | G-W1-P1 {adopt}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
