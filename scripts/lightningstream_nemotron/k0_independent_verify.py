"""Independent verification of K0/K1/K2. Imports nothing from the runner.

Recomputes the route unions and the LRU replays from the raw captured routes
with its own implementations, re-derives the emitted-token curve from the S10-A
histogram (and checks that histogram against the S10-A artifact), recounts every
acceptance from the raw draft/truth token lists, re-derives the bracketed chain
marginals, and re-applies every gate.

Also checks the reproduction that matters most: the full-vocabulary arm must
reproduce the S10-A pooled acceptance exactly, because it is the same wiring on
the same frozen prompts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
D_DRAFT = 4
GATE_RECALL = 0.995
GATE_A_DROP = 0.05
GATE_CHAIN_CUT = 0.30
CHAIN_BASE_MS = 19.10
MIN_WINDOWS = 300


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def count_leading_equal(a, b):
    n = 0
    while n < min(len(a), len(b)) and int(a[n]) == int(b[n]):
        n += 1
    return n


def union_mean(routes, B):
    vals = []
    for steps in routes.values():
        for s in range(len(steps) - B + 1):
            u = set()
            for t in range(B):
                u.update(int(e) for e in steps[s + t])
            vals.append(len(u))
    return mean(vals), len(vals)


def replay(routes, capacity, B):
    hits = misses = lookups = 0
    for steps in routes.values():
        cache = OrderedDict()

        def touch(e):
            nonlocal hits, misses, lookups
            lookups += 1
            if e in cache:
                cache.move_to_end(e)
                hits += 1
                return
            misses += 1
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[e] = 1

        if B is None:
            for st in steps:
                for e in st:
                    touch(int(e))
        else:
            for s in range(0, len(steps) - B + 1, B):
                for e in sorted({int(e) for t in range(B) for e in steps[s + t]}):
                    touch(e)
    return {"hits": hits, "misses": misses, "lookups": lookups,
            "hit_rate": hits / max(1, lookups)}


def main() -> int:
    path = OUT_DIR / "k0_route_union_census.json"
    if not path.exists():
        print(f"MISSING: {path.name}")
        return 2
    res = json.loads(path.read_text(encoding="utf-8"))
    checks = []
    routes = res["k0_raw_routes"]
    top_k = res["config"]["top_k"]

    # ---- the reference histogram really is S10-A's
    s10 = json.loads((OUT_DIR / "s10a_mtp_acceptance.json").read_text(encoding="utf-8"))
    hist_ref = {int(k): int(v) for k, v in s10["pooled"]["histogram_A"].items()}
    checks.append({"check": "S10-A histogram used as reference matches the S10-A artifact",
                   "ok": {int(k): int(v) for k, v in res["s10a_reference"]["histogram"].items()}
                   == hist_ref})

    total = sum(hist_ref.values())
    emitted, acc = [], 1.0
    for k in range(1, D_DRAFT + 1):
        acc += sum(c for a, c in hist_ref.items() if a >= k) / total
        emitted.append(acc)
    checks.append({"check": "emitted-token curve reproduces from the histogram",
                   "ok": all(abs(a - b) < 1e-9
                             for a, b in zip(emitted, res["s10a_reference"]["emitted_curve_D1_D4"])),
                   "recomputed": emitted})

    # ---- routes are the official ones: top_k ids per layer per step
    shapes_ok = all(len(st) == top_k for steps in routes.values() for st in steps)
    checks.append({"check": f"every captured route has exactly top_k={top_k} ids",
                   "ok": shapes_ok})
    checks.append({"check": "routes captured for every MoE layer of all three prompts",
                   "ok": len(routes) == 3 * res["config"]["moe_layers"],
                   "series": len(routes)})

    # ---- union census
    rec_union = {}
    for B_str, row in res["k0_census_short_context"].items():
        m, n = union_mean(routes, int(B_str))
        rec_union[B_str] = m
        checks.append({"check": f"union mean at B={B_str} reproduces",
                       "ok": abs(m - row["unique_experts"]["mean"]) < 1e-9,
                       "recomputed": m})
        checks.append({"check": f"window count at B={B_str} >= {MIN_WINDOWS}",
                       "ok": n >= MIN_WINDOWS, "windows": n})

    # ---- records per emitted token, Kimi's parity logic
    u5 = rec_union["5"]
    e5 = emitted[D_DRAFT - 1]
    rpe = u5 / e5
    parity_union = top_k * e5
    checks.append({"check": "records/emitted at B=5 reproduces",
                   "ok": abs(rpe - res["k0_records_per_emitted"]["5"]["records_per_emitted"]) < 1e-9,
                   "recomputed": rpe})
    checks.append({"check": "Kimi parity union U* = top_k x (A+1) reproduces",
                   "ok": abs(parity_union
                             - res["gates"]["G_K0_1_records_per_emitted_below_AR"]["kimi_parity_union"]) < 1e-9,
                   "recomputed": parity_union})
    g1 = rpe < top_k
    checks.append({"check": "G-K0-1 verdict agrees with the runner",
                   "ok": g1 == res["gates"]["G_K0_1_records_per_emitted_below_AR"]["passed"]})

    # ---- LRU replay
    for cap_str, row in res["k0_lru_replay"].items():
        ar = replay(routes, int(cap_str), None)
        rnd = replay(routes, int(cap_str), 5)
        checks.append({"check": f"AR replay at capacity {cap_str} reproduces",
                       "ok": ar == row["ar"], "recomputed": ar})
        checks.append({"check": f"round-B5 replay at capacity {cap_str} reproduces",
                       "ok": rnd == row["round_B5"], "recomputed": rnd})
    r72 = res["k0_lru_replay"]["72"]
    g2 = r72["round_misses_per_emitted"] < r72["ar_misses_per_emitted"]
    checks.append({"check": "G-K0-2 verdict agrees with the runner",
                   "ok": g2 == res["gates"]["G_K0_2_round_misses_below_AR"]["passed"]})

    # The runner divided total misses by the mean SERIES length, so its
    # *_per_emitted fields are inflated by the number of prompts. The gate is a
    # ratio and is unaffected, but the absolute numbers are not usable as
    # published. Recompute them properly: misses per model token, summed over
    # all MoE layers.
    n_prompts = len({k.split("|")[0] for k in routes})
    n_layers = res["config"]["moe_layers"]
    steps = mean(len(v) for v in routes.values())
    corrected = {}
    for cap_str, row in res["k0_lru_replay"].items():
        n_rounds = (int(steps) - 5 + 1 + 4) // 5
        corrected[cap_str] = {
            "ar_misses_per_token": row["ar"]["misses"] / (steps * n_prompts),
            "round_B5_misses_per_emitted": row["round_B5"]["misses"]
            / (n_rounds * e5 * n_prompts),
            "ratio_round_over_ar": None}
        c = corrected[cap_str]
        c["ratio_round_over_ar"] = c["round_B5_misses_per_emitted"] / c["ar_misses_per_token"]
    checks.append({"check": "AR misses/token is consistent with the AR hit rate over 23x6 lookups",
                   "ok": abs(corrected["72"]["ar_misses_per_token"]
                             - n_layers * top_k * (1 - r72["ar"]["hit_rate"])) < 0.5,
                   "recomputed": corrected["72"]["ar_misses_per_token"],
                   "from_hit_rate": n_layers * top_k * (1 - r72["ar"]["hit_rate"])})
    checks.append({"check": "runner per_emitted fields are inflated by exactly n_prompts",
                   "ok": abs(r72["ar_misses_per_emitted"]
                             / corrected["72"]["ar_misses_per_token"] - n_prompts) < 1e-6,
                   "n_prompts": n_prompts})

    # ---- acceptance per vocabulary arm, recounted from raw tokens
    arms = res["k2_vocab_arms"]
    rec_A = {}
    for label, arm in arms.items():
        pooled = []
        for pr in arm["prompts"]:
            seq = pr["sequence"]
            for st in pr["per_step"]:
                w = seq[st["i"] + 2:st["i"] + 2 + D_DRAFT]
                if [int(v) for v in w] != [int(v) for v in st["truth"]]:
                    checks.append({"check": f"{label}/{pr['label']}: truth window matches sequence",
                                   "ok": False})
                pooled.append(count_leading_equal(st["drafts"], w))
        rec_A[label] = mean(pooled)
        checks.append({"check": f"vocab arm {label}: pooled mean A reproduces",
                       "ok": abs(rec_A[label] - arm["pooled_mean_A"]) < 1e-9,
                       "recomputed": rec_A[label], "steps": len(pooled)})
        checks.append({"check": f"vocab arm {label}: at least 360 measured steps",
                       "ok": len(pooled) >= 360})

    checks.append({"check": "full-vocabulary arm reproduces the S10-A pooled acceptance",
                   "ok": abs(rec_A["full"] - s10["pooled"]["mean_A"]) < 1e-9,
                   "recomputed": rec_A["full"], "s10a": s10["pooled"]["mean_A"]})

    # ---- the K2 gates
    full_A = rec_A["full"]
    best = None
    for n in res["config"]["vocab_n"]:
        if n is None:
            continue
        a = arms[str(n)]
        hit = a["recall_counts"]["hit"] / a["recall_counts"]["total"]
        checks.append({"check": f"vocab {n}: recall reproduces from its counts",
                       "ok": abs(hit - a["recall"]) < 1e-9})
        ok = (hit >= GATE_RECALL and rec_A[str(n)] >= full_A - GATE_A_DROP
              and a["chain_p50_mean_ms"] <= CHAIN_BASE_MS * (1 - GATE_CHAIN_CUT))
        if ok and best is None:
            best = n
    checks.append({"check": "smallest vocabulary passing all three K2 gates agrees",
                   "ok": best == res["gates"]["G_K2_combined_pass_smallest_n"],
                   "recomputed": best})

    # ---- K1 bracketed marginals
    d = res["k1_chain_decomposition"]
    probes = [p for p in d["schedule"] if not p.startswith("base")]
    rec_marg, rec_drift, rec_rep = {}, {}, {}
    for k, p in enumerate(probes):
        b0 = d["arms"][f"base{k}"]["p50"]
        b1 = d["arms"][f"base{k + 1}"]["p50"]
        rec_marg[p] = d["arms"][p]["p50"] - 0.5 * (b0 + b1)
        rec_drift[p] = abs(b1 - b0)
        rec_rep[p] = abs(rec_marg[p]) > rec_drift[p]
        checks.append({"check": f"chain marginal[{p}] reproduces",
                       "ok": abs(rec_marg[p] - d["marginal_ms_per_chain"][p]) < 1e-9,
                       "recomputed": rec_marg[p]})
        checks.append({"check": f"chain noise-floor flag[{p}] reproduces",
                       "ok": rec_rep[p] == d["reported_above_noise"][p]})

    checks.append({"check": "runtime.py on disk still hashes to the measured value",
                   "ok": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py")
                   == res["runtime_sha256"]})
    checks.append({"check": "corpus sha256 matches the frozen corpus lock",
                   "ok": sha256_path(OUT_DIR / "s10a_corpus.json") == res["corpus_sha256"]})
    checks.append({"check": "no tokens-per-second figure at top level",
                   "ok": not any("tok_s" in k for k in res)})
    checks.append({"check": "records-per-emitted is undefined for B>5 (A not measured there)",
                   "ok": all(res["k0_records_per_emitted"][b]["records_per_emitted"] is None
                             for b in res["k0_records_per_emitted"] if int(b) > 5)})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_k0_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(path),
        "recomputed": {"union_mean": rec_union, "emitted_curve": emitted,
                       "records_per_emitted_B5": rpe, "parity_union": parity_union,
                       "pooled_mean_A": rec_A, "chain_marginal_ms": rec_marg,
                       "chain_local_drift_ms": rec_drift,
                       "smallest_vocab_passing": best,
                       "lru_corrected_per_token": corrected},
        "gates": {"G_K0_1": g1, "G_K0_2": g2, "G_K2_smallest_n": best},
        "checks": checks, "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "k0_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        if not c["ok"]:
            print(f"  [FAIL] {c['check']}  {c}")
    print(f"  ({len(checks) - len(failed)}/{len(checks)} checks ok)")
    print(f"\nB=5 union {rec_union['5']:.3f} -> {rpe:.3f} records/emitted "
          f"(AR {top_k}, parity union {parity_union:.3f})")
    print(f"full-vocab A {rec_A['full']:.4f}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
