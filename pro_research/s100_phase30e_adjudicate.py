from __future__ import annotations

import argparse
import json
import statistics

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase30e_common import RESULTS


def read_arm(tag: str, arm: str) -> float:
    path = RESULTS / f"S100_PHASE30E_{tag.upper()}_{arm.upper()}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "measured" or not data.get("correctness_green"):
        raise RuntimeError(f"invalid {tag}/{arm}")
    return float(data["summary"]["median_ms"])


def bootstrap_lower95(values: list[float], seed: int = 350) -> float:
    array = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(10000)
    for i in range(samples.size):
        index = rng.integers(0, array.size, array.size)
        samples[i] = np.median(array[index])
    return float(np.percentile(samples, 2.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args()

    rounds = []
    for tag in args.tags:
        parent = read_arm(tag, "parent")
        combined = read_arm(tag, "combined")
        candidate = read_arm(tag, "candidate")
        rounds.append(
            {
                "tag": tag,
                "parent_ms": parent,
                "combined_ms": combined,
                "candidate_ms": candidate,
                "combined_gain_pct": (parent - combined) / parent * 100.0,
                "candidate_gain_pct": (parent - candidate) / parent * 100.0,
                "dispatch_incremental_vs_combined_pct": (
                    (combined - candidate) / combined * 100.0
                ),
            }
        )

    gains = [row["candidate_gain_pct"] for row in rounds]
    state_path = RESULTS / "S100_PHASE30E_STATE_CHECK.json"
    state_green = state_path.exists() and bool(
        json.loads(state_path.read_text(encoding="utf-8")).get(
            "PHASE30E_STATE_GREEN"
        )
    )
    full = len(rounds) >= 4
    median_gain = float(statistics.median(gains))
    lower95 = bootstrap_lower95(gains)
    adopted = bool(
        full and state_green and all(gain > 0.0 for gain in gains)
        and median_gain >= 5.0 and lower95 > 0.0
    )
    payload = {
        "kind": "s100_phase30e_adjudication",
        "status": "measured",
        "created_utc": utc_now(),
        "rounds": rounds,
        "summary": {
            "round_count": len(rounds),
            "candidate_median_gain_pct": median_gain,
            "candidate_bootstrap_lower95_gain_pct": lower95,
            "dispatch_incremental_median_pct": float(
                statistics.median(
                    row["dispatch_incremental_vs_combined_pct"] for row in rounds
                )
            ),
            "positive_rounds": int(sum(gain > 0.0 for gain in gains)),
        },
        "PHASE30E_STATE_GREEN": state_green,
        "FULL_THERMAL_PROTOCOL": full,
        "PHASE30E_ADOPTED": adopted,
    }
    out = RESULTS / "S100_PHASE30E_ADJUDICATION.json"
    write_json_atomic(out, payload, archive=True)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
