from __future__ import annotations

import json

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase32_common import RESULTS


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def robust_cv(values) -> float:
    values = np.asarray(values, np.float64)
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)) / median)


def main() -> int:
    state = load("S100_PHASE32_STATE_CHECK.json")
    compile_result = load("S100_PHASE32_COMPILE_COMPILE_CTX1024.json")
    screen_parent = load("S100_PHASE32_SCREEN_PARENT_CTX1024.json")
    screen_candidate = load("S100_PHASE32_SCREEN_DENSE_M8_CTX1024.json")

    rounds = []
    parent_medians = []
    candidate_medians = []
    all_paired_gains = []
    for index in range(1, 5):
        parent = load(f"S100_PHASE32_THERMAL_R{index}_PARENT_CTX1024.json")
        candidate = load(
            f"S100_PHASE32_THERMAL_R{index}_DENSE_M8_CTX1024.json"
        )
        parent_ms = float(parent["summary"]["median_ms"])
        candidate_ms = float(candidate["summary"]["median_ms"])
        parent_medians.append(parent_ms)
        candidate_medians.append(candidate_ms)
        parent_records = parent["records"]
        candidate_records = candidate["records"]
        positions_aligned = [r["pos"] for r in parent_records] == [
            r["pos"] for r in candidate_records
        ]
        paired = [
            (float(p["ms"]) - float(c["ms"])) / float(p["ms"])
            for p, c in zip(parent_records, candidate_records)
        ]
        all_paired_gains.extend(paired)
        rounds.append(
            {
                "round": index,
                "parent_median_ms": parent_ms,
                "candidate_median_ms": candidate_ms,
                "round_gain_fraction": (parent_ms - candidate_ms) / parent_ms,
                "positions_aligned": positions_aligned,
                "paired_block_gain_median": float(np.median(paired)),
                "all_tokens_exact": bool(
                    parent["summary"]["all_token_exact"]
                    and candidate["summary"]["all_token_exact"]
                ),
            }
        )

    round_gains = [row["round_gain_fraction"] for row in rounds]
    median_round_gain = float(np.median(round_gains))
    median_paired_gain = float(np.median(all_paired_gains))
    candidate_process_median = float(np.median(candidate_medians))

    resources = compile_result["kernel_resources"]
    resource_rows = [
        row
        for family in resources.values()
        for row in family.values()
    ]
    zero_local = all(int(row.get("local_size_bytes") or 0) == 0 for row in resource_rows)
    gates = {
        "complete_four_rounds": len(rounds) == 4,
        "all_correctness_green": all(r["all_tokens_exact"] for r in rounds),
        "positions_aligned": all(r["positions_aligned"] for r in rounds),
        "state_green": bool(state.get("PHASE32_STATE_GREEN")),
        "zero_local_memory_spills": zero_local,
        "median_round_gain_ge_5pct": median_round_gain >= 0.05,
        "median_paired_gain_ge_5pct": median_paired_gain >= 0.05,
        "positive_rounds_ge_3of4": sum(g > 0 for g in round_gains) >= 3,
        "parent_robust_cv_le_5pct": robust_cv(parent_medians) <= 0.05,
        "candidate_robust_cv_le_5pct": robust_cv(candidate_medians) <= 0.05,
    }
    adopted = bool(all(gates.values()))

    mtp_acceptance_plus_anchor = 3.113888888888889
    mtp_draft_ms = 19.0951
    phase31_h4_ms = 63.53125
    dflash2_empirical_tokens = 2.465625
    economics = {
        "candidate_process_median_ms_per_h8": candidate_process_median,
        "candidate_target_only_tok_s": 8000.0 / candidate_process_median,
        "perfect_draft_s100_ceiling_open": candidate_process_median <= 80.0,
        "mtp": {
            "acceptance_plus_anchor": mtp_acceptance_plus_anchor,
            "measured_four_draft_ms": mtp_draft_ms,
            "phase31_h4_proxy_tok_s": (
                1000.0 * mtp_acceptance_plus_anchor
                / (phase31_h4_ms + mtp_draft_ms)
            ),
            "claim_boundary": "H5 verifier unmeasured; H4 value is only a pessimistic/shape-mismatched scenario",
            "reopened": False,
        },
        "dflash2": {
            "empirical_proxy_tokens_per_round": dflash2_empirical_tokens,
            "zero_cost_drafter_tok_s_at_phase32_h8": (
                1000.0 * dflash2_empirical_tokens / candidate_process_median
            ),
            "perfect_acceptance_zero_cost_tok_s": 8000.0 / candidate_process_median,
            "training_build_open": False,
            "reopened": False,
        },
    }
    payload = {
        "kind": "s100_phase32_adjudication",
        "status": "measured",
        "created_utc": utc_now(),
        "screen": {
            "parent_median_ms": screen_parent["summary"]["median_ms"],
            "candidate_median_ms": screen_candidate["summary"]["median_ms"],
            "gain_fraction": (
                float(screen_parent["summary"]["median_ms"])
                - float(screen_candidate["summary"]["median_ms"])
            ) / float(screen_parent["summary"]["median_ms"]),
        },
        "rounds": rounds,
        "parent_process_medians_ms": parent_medians,
        "candidate_process_medians_ms": candidate_medians,
        "round_gain_fractions": round_gains,
        "median_round_gain_fraction": median_round_gain,
        "median_paired_block_gain_fraction": median_paired_gain,
        "positive_rounds": sum(g > 0 for g in round_gains),
        "parent_robust_cv": robust_cv(parent_medians),
        "candidate_robust_cv": robust_cv(candidate_medians),
        "gates": gates,
        "economics": economics,
        "PHASE32_H8_ADOPTED": adopted,
        "S100_SINGLE_ACHIEVED": False,
        "NEXT_ROUTE": "EXACT_NVFP4_M8_WARP32_WEIGHT_REUSE",
        "claim_boundary": "exact target-only H8; no drafter or achieved speculative throughput",
    }
    write_json_atomic(RESULTS / "S100_PHASE32_ADJUDICATION.json", payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
