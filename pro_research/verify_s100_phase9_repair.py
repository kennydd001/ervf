from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    directory = Path(args.dir)

    trace_json = json.loads(
        (directory / "S100_PHASE9_TRACE.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(directory / "S100_PHASE9_TRACE.npz") as trace:
        layers = [int(value) for value in trace["layers"]]
        counted_tokens = int(trace["counted"].astype(bool).sum())

    oracle = json.loads(
        (directory / "S100_PHASE9_CACHE_ORACLE.json").read_text(
            encoding="utf-8"
        )
    )
    profiles = json.loads(
        (
            directory / "S100_PHASE9_CAPACITY_PROFILES.json"
        ).read_text(encoding="utf-8")
    )["profiles"]

    failures = []
    if trace_json.get("status") != "measured":
        failures.append("trace status is not measured")
    if counted_tokens != 8192:
        failures.append(f"counted trace tokens={counted_tokens}")
    if oracle.get("status") != "measured":
        failures.append("oracle status is not measured")
    if not oracle.get("simulation_gate"):
        failures.append("oracle simulation gate failed")

    current = (
        oracle.get("test_current") or {}
    ).get("miss_fraction")
    belady = (
        oracle.get("belady_current_map_test") or {}
    ).get("miss_fraction")
    if current is None or belady is None:
        failures.append("current/Belady miss fraction missing")
    elif belady > current:
        failures.append("Belady is worse than LRU")

    expected_names = {
        "current",
        "budget_neutral",
        "plus_128",
        "plus_256",
        "plus_379",
    }
    if set(profiles) != expected_names:
        failures.append(
            f"profile names differ: {sorted(profiles)}"
        )

    for name, raw_map in profiles.items():
        mapping = {
            int(key): int(value) for key, value in raw_map.items()
        }
        if set(mapping) != set(layers):
            failures.append(f"{name}: wrong layer set")
        if any(
            value < 32 or value > 128 or value % 2
            for value in mapping.values()
        ):
            failures.append(f"{name}: invalid capacity")
        invariant = (
            oracle.get("profile_invariants") or {}
        ).get(name, {})
        if not all(
            bool(invariant.get(key))
            for key in (
                "all_layers_present",
                "all_caps_even_32_to_128",
            )
        ):
            failures.append(f"{name}: invariant record failed")

    result = {
        "kind": "verify_s100_phase9_repair",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "counted_tokens": counted_tokens,
        "layer_count": len(layers),
        "current_test_miss_fraction": current,
        "belady_test_miss_fraction": belady,
    }
    output = directory / "S100_PHASE9_REPAIR_VERIFY.json"
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
