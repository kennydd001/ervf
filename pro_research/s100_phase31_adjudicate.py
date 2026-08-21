from __future__ import annotations

import json

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase31_common import RESULTS


SELECTED = "attention_head_m4"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def measured_summary(name: str) -> dict:
    payload = load(name)
    if payload.get("status") != "measured":
        raise RuntimeError(f"incomplete result: {name}")
    summary = payload.get("summary") or {}
    if not summary.get("all_token_exact"):
        raise RuntimeError(f"non-exact result: {name}")
    return summary


def main() -> int:
    rounds = []
    gains = []
    for index in range(1, 5):
        parent = measured_summary(
            f"S100_PHASE31_THERMAL_R{index}_PARENT.json"
        )
        candidate = measured_summary(
            f"S100_PHASE31_THERMAL_R{index}_ATTENTION_HEAD_M4.json"
        )
        parent_ms = float(parent["median_ms"])
        candidate_ms = float(candidate["median_ms"])
        gain = (parent_ms - candidate_ms) / parent_ms
        gains.append(gain)
        rounds.append(
            {
                "round": index,
                "parent_ms_per_h4": parent_ms,
                "candidate_ms_per_h4": candidate_ms,
                "gain_pct": 100.0 * gain,
                "candidate_target_only_tok_s": float(
                    candidate["target_only_tok_s"]
                ),
                "exact": True,
            }
        )

    rng = np.random.default_rng(31)
    samples = np.median(
        rng.choice(np.asarray(gains), size=(200_000, len(gains)), replace=True),
        axis=1,
    )
    median_gain = float(np.median(gains))
    lower95 = float(np.percentile(samples, 2.5))
    upper95 = float(np.percentile(samples, 97.5))
    median_candidate_ms = float(
        np.median([row["candidate_ms_per_h4"] for row in rounds])
    )

    contexts = {}
    for context in (128, 4096):
        parent = measured_summary(f"S100_PHASE31_CTX{context}_PARENT.json")
        candidate = measured_summary(
            f"S100_PHASE31_CTX{context}_ATTENTION_HEAD_M4.json"
        )
        parent_ms = float(parent["median_ms"])
        candidate_ms = float(candidate["median_ms"])
        contexts[str(context)] = {
            "parent_ms_per_h4": parent_ms,
            "candidate_ms_per_h4": candidate_ms,
            "gain_pct": 100.0 * (parent_ms - candidate_ms) / parent_ms,
            "candidate_target_only_tok_s": float(
                candidate["target_only_tok_s"]
            ),
            "exact": True,
        }

    state = load("S100_PHASE31_STATE_CHECK.json")
    compile_result = load("S100_PHASE31_COMPILE_COMPILE.json")
    state_green = bool(state.get("PHASE31_STATE_GREEN"))
    compile_green = compile_result.get("status") == "compiled"
    resources = compile_result.get("kernel_resources") or {}
    local_sizes = [
        record.get("local_size_bytes")
        for family in resources.values()
        for record in family.values()
        if isinstance(record, dict)
    ]
    zero_spills = bool(local_sizes) and all(size == 0 for size in local_sizes)
    context_green = all(
        record["exact"] and record["gain_pct"] > 0.0
        for record in contexts.values()
    )
    adopted = bool(
        compile_green
        and zero_spills
        and state_green
        and len(rounds) == 4
        and all(gain > 0.0 for gain in gains)
        and median_gain >= 0.05
        and lower95 > 0.0
        and context_green
    )

    payload = {
        "kind": "s100_phase31_adjudication",
        "status": "measured",
        "created_utc": utc_now(),
        "frozen_parent": (
            "codex/s100-phase30e-breakthrough@"
            "f51d207914ccd32bc7c3133d8826ab70b747fca1"
        ),
        "selected_arm": SELECTED,
        "mechanisms": [
            "direct-L2 BF16 M4 for attention Q/K/V/O",
            "direct-L2 NVFP4 M4 for the LM head",
        ],
        "rounds": rounds,
        "summary": {
            "positive_rounds": sum(gain > 0.0 for gain in gains),
            "round_count": len(gains),
            "median_gain_pct": 100.0 * median_gain,
            "bootstrap_lower95_gain_pct": 100.0 * lower95,
            "bootstrap_upper95_gain_pct": 100.0 * upper95,
            "median_candidate_ms_per_h4": median_candidate_ms,
            "median_candidate_target_only_tok_s": 4000.0
            / median_candidate_ms,
        },
        "contexts": contexts,
        "gates": {
            "compile_green": compile_green,
            "zero_local_memory_spills": zero_spills,
            "state_green": state_green,
            "all_four_rounds_positive": all(gain > 0.0 for gain in gains),
            "median_gain_at_least_5pct": median_gain >= 0.05,
            "bootstrap_lower95_positive": lower95 > 0.0,
            "context_128_4096_green": context_green,
        },
        "PHASE31_ADOPTED": adopted,
        "S100_SINGLE_ACHIEVED": False,
        "NEXT_ROUTE": "REBASE_BLOCK_DRAFTER_ECONOMICS_ON_PHASE31_PARENT",
        "claim_boundary": (
            "exact target-only H4 verifier throughput; no drafter, acceptance, "
            "rollback, rejection or end-to-end speculative decoding cost"
        ),
    }
    write_json_atomic(RESULTS / "S100_PHASE31_ADJUDICATION.json", payload, archive=True)

    lines = [
        "S100 PHASE 31 — DIRECT-L2 ATTENTION + LM HEAD",
        f"Selected arm: {SELECTED}",
        f"Adopted: {adopted}",
        f"State green: {state_green}",
        f"Zero local-memory spills: {zero_spills}",
        f"Positive thermal rounds: {sum(g > 0 for g in gains)}/{len(gains)}",
        f"Median gain: {100.0 * median_gain:.3f}%",
        f"Bootstrap lower-95 gain: {100.0 * lower95:.3f}%",
        f"Median H4: {median_candidate_ms:.5f} ms",
        f"Median target-only tok/s: {4000.0 / median_candidate_ms:.3f}",
        f"Context 128 gain: {contexts['128']['gain_pct']:.3f}%",
        f"Context 4096 gain: {contexts['4096']['gain_pct']:.3f}%",
        "S100 single-stream achieved: False",
        f"Next route: {payload['NEXT_ROUTE']}",
    ]
    (RESULTS / "S100_PHASE31_SUMMARY.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0 if adopted else 2


if __name__ == "__main__":
    raise SystemExit(main())
