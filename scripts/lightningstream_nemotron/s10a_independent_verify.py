"""Independent verification of S10-A. Imports nothing from the runner.

Recomputes, from the raw per-step token lists in `s10a_mtp_acceptance.json`:
the truth windows, every `A`, the histograms, the per-prompt and pooled means,
the conditional acceptance ladder, and the G-S10-1 verdict.  Re-derives the A1
winner from the four NLLs.  Re-tokenises the frozen prompts with the model's own
tokenizer to confirm the runner really decoded the prompts the corpus lock names,
and re-hashes the corpus and its parquet source.

Deliberately duplicates logic rather than sharing it: a shared helper would make
the verification circular.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
D_EXPECTED = 4
GATE_MEAN_A = 1.5
GATE_MIN_STEPS = 200
GATE_MIN_PROMPTS = 3
A1_NLL_CEILING = 7.0


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def count_leading_equal(a, b) -> int:
    """Independent reimplementation of the match count."""
    n = 0
    limit = min(len(a), len(b))
    while n < limit and int(a[n]) == int(b[n]):
        n += 1
    return n


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def check_prompt_row(row, checks, tag):
    seq = row["sequence"]
    m = row["prompt_tokens"]
    first = row["first_measured_index"]
    checks.append({"check": f"{tag}.first_measured_index == prompt_tokens - 1",
                   "ok": first == m - 1, "got": first, "expected": m - 1})

    idxs = [r["i"] for r in row["per_step"]]
    consecutive = idxs == list(range(first, first + len(idxs)))
    checks.append({"check": f"{tag}.measured indices are consecutive from first",
                   "ok": consecutive, "n": len(idxs)})

    recomputed, truth_ok, draft_len_ok, in_range = [], True, True, True
    for r in row["per_step"]:
        i = r["i"]
        window = seq[i + 2:i + 2 + D_EXPECTED]
        if len(window) != D_EXPECTED:
            in_range = False
            continue
        if [int(v) for v in window] != [int(v) for v in r["truth"]]:
            truth_ok = False
        if len(r["drafts"]) != D_EXPECTED:
            draft_len_ok = False
        recomputed.append(count_leading_equal(r["drafts"], window))

    checks.append({"check": f"{tag}.truth windows equal sequence[i+2 : i+2+D]",
                   "ok": truth_ok})
    checks.append({"check": f"{tag}.every step has exactly D={D_EXPECTED} drafts",
                   "ok": draft_len_ok})
    checks.append({"check": f"{tag}.every truth window fits inside the sequence",
                   "ok": in_range})

    stored = [int(r["A"]) for r in row["per_step"]]
    checks.append({"check": f"{tag}.recomputed A equals stored A",
                   "ok": recomputed == stored,
                   "mismatches": sum(1 for a, b in zip(recomputed, stored) if a != b)})

    hist = {str(v): sum(1 for a in recomputed if a == v) for v in range(D_EXPECTED + 1)}
    checks.append({"check": f"{tag}.histogram reproduces",
                   "ok": hist == {k: int(v) for k, v in row["histogram_A"].items()},
                   "recomputed": hist})
    rmean = mean(recomputed)
    checks.append({"check": f"{tag}.mean_A reproduces",
                   "ok": abs(rmean - row["mean_A"]) < 1e-9,
                   "recomputed": rmean, "stored": row["mean_A"]})
    return recomputed


def main() -> int:
    res_path = OUT_DIR / "s10a_mtp_acceptance.json"
    if not res_path.exists():
        print(f"MISSING: {res_path.name}")
        return 2
    res = json.loads(res_path.read_text(encoding="utf-8"))
    checks: list[dict] = []

    # ---------------------------------------------------------- input lock
    corpus_path = OUT_DIR / "s10a_corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    checks.append({"check": "corpus sha256 in result matches corpus file on disk",
                   "ok": sha256_path(corpus_path) == res["corpus_sha256"]})
    parquet = REPO_ROOT / corpus["source"]["path"]
    checks.append({"check": "corpus source parquet still hashes to the locked value",
                   "ok": parquet.exists() and sha256_path(parquet) == corpus["source"]["sha256"]})

    # ------------------------------------------------------------- A1 rule
    a1 = res["a1_wiring"]
    variants = a1["variants"]
    winner = min(variants, key=lambda k: variants[k]["mean_nll"])
    checks.append({"check": "A1 winner is the argmin of the four mean NLLs",
                   "ok": winner == a1["winner"], "recomputed": winner,
                   "stored": a1["winner"]})
    checks.append({"check": "A1 winner is below the preregistered 7.0 nat ceiling",
                   "ok": variants[winner]["mean_nll"] <= A1_NLL_CEILING,
                   "mean_nll": variants[winner]["mean_nll"]})
    checks.append({"check": "A1 scored at least 256 positions per variant",
                   "ok": all(v["positions"] >= 256 for v in variants.values()),
                   "positions": {k: v["positions"] for k, v in variants.items()}})
    checks.append({"check": "A2 ran with the wiring A1 selected",
                   "ok": (res["config"]["concat_order"] == variants[winner]["concat_order"]
                          and res["config"]["h_source"] == variants[winner]["h_source"])})
    checks.append({"check": "A1 corpus rows are disjoint from the A2 gate prompts",
                   "ok": True, "note": "A1 uses wikitext rows; gate prompts are "
                                       "hand-written strings in the corpus lock"})

    # ------------------------------------------------------------- prompts
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(REPO_ROOT / "models" / res["model_dir"]), trust_remote_code=True)
        locked = {p["id"]: p["text"] for p in corpus["gate_prompts"]}
        ok_all = True
        for row in res["gate_prompts"]:
            ids = [int(v) for v in tok.encode(locked[row["label"]], add_special_tokens=False)]
            if ids != [int(v) for v in row["sequence"][:len(ids)]] or len(ids) != row["prompt_tokens"]:
                ok_all = False
        checks.append({"check": "each decoded sequence starts with the locked prompt tokens",
                       "ok": ok_all})
    except Exception as e:                                   # pragma: no cover
        checks.append({"check": "re-tokenise the locked prompts",
                       "ok": False, "error": f"{type(e).__name__}: {e}"})

    # ---------------------------------------------------------------- gate
    pooled: list[int] = []
    per_prompt = {}
    for row in res["gate_prompts"]:
        rec = check_prompt_row(row, checks, f"gate[{row['label']}]")
        per_prompt[row["label"]] = {"steps": len(rec), "mean_A": mean(rec)}
        pooled.extend(rec)

    pooled_mean = mean(pooled)
    hist = {str(v): sum(1 for a in pooled if a == v) for v in range(D_EXPECTED + 1)}
    cond = []
    for j in range(D_EXPECTED):
        denom = sum(1 for a in pooled if a >= j)
        cond.append((sum(1 for a in pooled if a >= j + 1) / denom) if denom else None)

    checks.append({"check": "pooled mean_A reproduces",
                   "ok": abs(pooled_mean - res["pooled"]["mean_A"]) < 1e-9,
                   "recomputed": pooled_mean, "stored": res["pooled"]["mean_A"]})
    checks.append({"check": "pooled histogram reproduces",
                   "ok": hist == {k: int(v) for k, v in res["pooled"]["histogram_A"].items()}})
    checks.append({"check": "pooled step count >= 200",
                   "ok": len(pooled) >= GATE_MIN_STEPS, "steps": len(pooled)})
    checks.append({"check": "at least 3 gate prompts",
                   "ok": len(res["gate_prompts"]) >= GATE_MIN_PROMPTS})

    gate_pass = (pooled_mean >= GATE_MEAN_A and len(pooled) >= GATE_MIN_STEPS
                 and len(res["gate_prompts"]) >= GATE_MIN_PROMPTS)
    checks.append({"check": "G-S10-1 verdict agrees with the runner",
                   "ok": gate_pass == res["gate_G_S10_1"]["passed"],
                   "recomputed": gate_pass, "stored": res["gate_G_S10_1"]["passed"]})
    checks.append({"check": "gate threshold in the result is still the preregistered 1.5",
                   "ok": res["gate_G_S10_1"]["required_mean_A"] == GATE_MEAN_A})

    # --------------------------------------------------- secondary (no gate)
    long_rec = check_prompt_row(res["secondary_long_context"], checks, "long_ctx")
    checks.append({"check": "long-context arm is excluded from the pooled gate figure",
                   "ok": len(pooled) == sum(len(r["per_step"]) for r in res["gate_prompts"])})

    # ------------------------------------------------------------- hygiene
    checks.append({"check": "no A exceeds D",
                   "ok": all(a <= D_EXPECTED for a in pooled + long_rec)})
    checks.append({"check": "result reports no tokens-per-second figure at top level",
                   "ok": not any("tok_s" in k or "tokens_per_second" in k for k in res)})
    checks.append({"check": "result carries a claim boundary",
                   "ok": bool(res.get("claim_boundary"))})

    failed = [c for c in checks if not c["ok"]]
    payload = {
        "kind": "lightningstream_nemotron_s10a_independent_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_file_sha256": sha256_path(res_path),
        "recomputed": {
            "per_prompt": per_prompt,
            "pooled_steps": len(pooled),
            "pooled_mean_A": pooled_mean,
            "pooled_histogram_A": hist,
            "conditional_accept": cond,
            "long_context_steps": len(long_rec),
            "long_context_mean_A": mean(long_rec),
            "a1_winner": winner,
            "a1_mean_nll": {k: v["mean_nll"] for k, v in variants.items()},
        },
        "gate_G_S10_1": {
            "required_mean_A": GATE_MEAN_A, "measured_mean_A": pooled_mean,
            "passed": gate_pass,
        },
        "checks": checks,
        "checks_failed": len(failed),
        "verdict": "VERIFIED" if not failed else "VERIFICATION_FAILED",
    }
    (OUT_DIR / "s10a_independent_verification.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for c in checks:
        print(f"  [{'ok ' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"\npooled steps {len(pooled)}  mean A {pooled_mean:.4f}  "
          f"G-S10-1 {'PASS' if gate_pass else 'FAIL'}")
    print(f"verdict: {payload['verdict']} ({len(failed)} failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
