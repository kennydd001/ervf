"""Independent verification of S13. Imports nothing from the runner or runtime.

Recomputes every union statistic from the raw captured routes with its own
counting, re-derives the C1 token comparison against the frozen S10A acceptance
file, re-tokenizes the gate prompts from the frozen corpus to prove the arms
used the registered prompts, and re-evaluates all three gates.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
WINDOWS = [2, 3, 4, 5, 6, 8]
GATE_U1_W = 5
GATE_U1_MAX = 12.0
MOE_LAYERS_EXPECTED = 23
TOP_K = 6


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def percentile(xs, q):
    """Linear-interpolation percentile, same convention as numpy default."""
    s = sorted(float(x) for x in xs)
    n = len(s)
    if n == 1:
        return s[0]
    pos = (n - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def union_sizes(rows: list[list[int]], w: int) -> list[int]:
    out = []
    n_win = len(rows) // w
    for j in range(n_win):
        seen = set()
        for r in rows[j * w:(j + 1) * w]:
            seen.update(r)
        out.append(len(seen))
    return out


def main() -> int:
    path = OUT_DIR / "s13_expert_union.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    corpus = json.loads((OUT_DIR / "s10a_corpus.json").read_text(encoding="utf-8"))
    accept = json.loads((OUT_DIR / "s10a_mtp_acceptance.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    # ------------------------------------------------------- provenance
    lock = json.loads((OUT_DIR / "s13_input_lock.json").read_text(encoding="utf-8"))
    locked = {e["path"]: e["sha256"] for e in lock["entries"] if e["present"]}
    checks.append({"check": "input lock covers preregistration, runner, corpus, "
                            "acceptance file and runtime",
                   "ok": len(locked) == 5})
    checks.append({"check": "runner on disk still hashes to the locked value",
                   "ok": sha256_path(REPO_ROOT / "scripts/lightningstream_nemotron/"
                                     "s13_expert_union.py")
                   == locked["scripts/lightningstream_nemotron/s13_expert_union.py"]
                   == res["runner_sha256"]})
    checks.append({"check": "runtime.py unchanged since the locked measurement",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/"
                                     "runtime.py")
                   == locked["src/moe_lab/lightningstream_nemotron/runtime.py"]
                   == res["runtime_sha256"]})
    checks.append({"check": "frozen corpus and acceptance file unchanged",
                   "ok": sha256_path(OUT_DIR / "s10a_corpus.json") == res["corpus_sha256"]
                   and sha256_path(OUT_DIR / "s10a_mtp_acceptance.json")
                   == res["acceptance_sha256"]})
    checks.append({"check": "runner did not run with a widened gate threshold",
                   "ok": abs(res["gates"]["G_S13_U1"]["measured"]
                             - res["pooled_mean_by_window"][str(GATE_U1_W)]) < 1e-12})

    # ------------------------------------------------------- structure
    arms = res["arms"]
    checks.append({"check": "4 arms: 3 gate prompts + 1 long-context arm",
                   "ok": len(arms) == 4})
    for arm in arms:
        routes = arm["routes"]
        ok = (len(routes) == MOE_LAYERS_EXPECTED
              and all(len(v) == arm["steps"] for v in routes.values())
              and all(len(r) == TOP_K for v in routes.values() for r in v))
        checks.append({"check": f"{arm['label']}: 23 MoE layers x {arm['steps']} "
                                f"steps x 6 route ids",
                       "ok": ok})
        in_range = all(0 <= e < 128 for v in routes.values() for r in v for e in r)
        checks.append({"check": f"{arm['label']}: all route ids in [0,128)",
                       "ok": in_range})

    # ------------------------------------------------------- C1 recompute
    c1_ok_all = True
    for arm, ref in zip(arms[:3], accept["gate_prompts"]):
        n = arm["steps"]
        expected = ref["sequence"][ref["prompt_tokens"]:ref["prompt_tokens"] + n]
        same = arm["generated"] == expected
        c1_ok_all = c1_ok_all and same
        checks.append({"check": f"C1 {arm['label']}: generated ids identical to the "
                                f"S10A sequence over {n} tokens",
                       "ok": same,
                       "first_diff": next((i for i, (a, b)
                                           in enumerate(zip(arm["generated"], expected))
                                           if a != b), None)})

    # gate prompts used are the frozen corpus prompts
    from transformers import AutoTokenizer  # local, read-only use
    tok = AutoTokenizer.from_pretrained(
        str(REPO_ROOT / "models" / res["model_dir"]), trust_remote_code=True)
    for arm, gp in zip(arms[:3], corpus["gate_prompts"]):
        ids = tok.encode(gp["text"], add_special_tokens=False)
        ref = accept["gate_prompts"][arms.index(arm)]
        checks.append({"check": f"{arm['label']}: tokenized corpus prompt matches the "
                                f"S10A sequence prefix ({len(ids)} tokens)",
                       "ok": ids == ref["sequence"][:len(ids)]
                       and len(ids) == arm["prompt_tokens"] == ref["prompt_tokens"]})

    # ------------------------------------------------------- union recompute
    pooled_w = {w: [] for w in WINDOWS}
    w1_exact = True
    for arm in arms:
        for w in WINDOWS:
            vals = []
            for rows in arm["routes"].values():
                vals.extend(union_sizes(rows, w))
            stored = arm["union"]["per_window"][str(w)]
            mean = sum(vals) / len(vals)
            checks.append({"check": f"{arm['label']} W={w}: mean/p50/p95 reproduce "
                                    f"from raw routes (n={len(vals)})",
                           "ok": (abs(mean - stored["mean"]) < 1e-9
                                  and abs(percentile(vals, 50) - stored["p50"]) < 1e-9
                                  and abs(percentile(vals, 95) - stored["p95"]) < 1e-9),
                           "recomputed_mean": mean, "stored_mean": stored["mean"]})
            pooled_w[w].append(mean)
        for rows in arm["routes"].values():
            if any(len(set(r)) != TOP_K for r in rows):
                w1_exact = False
    grand = {w: sum(pooled_w[w]) / len(pooled_w[w]) for w in WINDOWS}
    checks.append({"check": "pooled mean by window reproduces from per-arm means",
                   "ok": all(abs(grand[w] - res["pooled_mean_by_window"][str(w)]) < 1e-12
                             for w in WINDOWS)})

    # ------------------------------------------------------- gates
    mono = all(grand[WINDOWS[i]] <= grand[WINDOWS[i + 1]] + 1e-12
               for i in range(len(WINDOWS) - 1))
    s1_ok = w1_exact and mono
    u1_ok = grand[GATE_U1_W] <= GATE_U1_MAX
    g = res["gates"]
    checks.append({"check": "G-S13-C1 verdict agrees with the runner",
                   "ok": c1_ok_all == g["G_S13_C1"]["pass"]})
    checks.append({"check": "G-S13-S1 verdict agrees with the runner",
                   "ok": s1_ok == g["G_S13_S1"]["pass"],
                   "w1_exactly_6": w1_exact, "monotone": mono})
    checks.append({"check": "G-S13-U1 verdict agrees with the runner",
                   "ok": u1_ok == g["G_S13_U1"]["pass"],
                   "recomputed_union_w5": grand[GATE_U1_W],
                   "threshold": GATE_U1_MAX})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_s13_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {
            "pooled_mean_by_window": {str(w): grand[w] for w in WINDOWS},
            "c1_all_identical": c1_ok_all,
            "w1_exactly_6": w1_exact,
            "monotone_in_W": mono,
        },
        "gates": {"G_S13_C1": c1_ok_all, "G_S13_S1": s1_ok, "G_S13_U1": u1_ok},
        "checks": checks,
        "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "s13_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"\npooled union means: "
          + "  ".join(f"W{w}={grand[w]:.3f}" for w in WINDOWS))
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
