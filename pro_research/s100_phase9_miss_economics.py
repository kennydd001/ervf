from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ACT_D2H_PER_BYTE = 0.0117 / 44_544.0
OUT_H2D_PER_BYTE = 0.0487 / 10_752.0
HIDDEN = 2688
INTER = 1856
EXPECTED_N = (1, 2, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    directory = Path(args.dir)

    samples_path = directory / "S100_PHASE9_UPMISS_SAMPLES.json"
    expected_layers = []
    if samples_path.exists():
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        expected_layers = [
            int(value) for value in samples.get("target_layers", [])
        ]

    rows = []
    missing = []
    for layer in expected_layers:
        rtx_path = directory / f"RTX_UPMISS_LAYER_{layer}.json"
        arc_path = directory / f"ARC_UPMISS_LAYER_{layer}.json"

        if not rtx_path.exists():
            missing.append(f"RTX layer {layer}: missing file")
            continue
        if not arc_path.exists():
            missing.append(f"Arc layer {layer}: missing file")
            continue

        rtx = json.loads(rtx_path.read_text(encoding="utf-8"))
        arc = json.loads(arc_path.read_text(encoding="utf-8"))
        if rtx.get("status") != "measured":
            missing.append(
                f"RTX layer {layer}: status={rtx.get('status')}"
            )
            continue
        if arc.get("status") != "measured":
            missing.append(
                f"Arc layer {layer}: status={arc.get('status')}"
            )
            continue

        for rtx_row in rtx["rows"]:
            n_experts = int(rtx_row["nexperts"])
            arc_row = next(
                (
                    row for row in arc["rows"]
                    if int(row["nexperts"]) == n_experts
                ),
                None,
            )
            if arc_row is None:
                missing.append(
                    f"Arc layer {layer}: missing N={n_experts}"
                )
                continue

            bridge = (
                HIDDEN * 4 * ACT_D2H_PER_BYTE
                + n_experts * INTER * 4 * OUT_H2D_PER_BYTE
            )
            staged = float(
                rtx_row["staged_fetch_plus_up"]["median_ms"]
            )
            direct = float(rtx_row["direct_host_up"]["median_ms"])
            arc_total = (
                float(arc_row["best"]["wall_median_ms"]) + bridge
            )
            correctness = arc_row["correctness"]
            arc_correct = bool(
                correctness["cosine"] >= 0.999
                and correctness["nrmse"] <= 0.02
                and correctness["finite"]
            )

            rows.append(
                {
                    "layer": layer,
                    "nexperts": n_experts,
                    "staged_rtx_ms": staged,
                    "direct_host_rtx_ms": direct,
                    "direct_bitexact": bool(
                        rtx_row["direct_bitexact"]
                    ),
                    "arc_wall_plus_bridge_ms": arc_total,
                    "arc_kernel_wall_ms": float(
                        arc_row["best"]["wall_median_ms"]
                    ),
                    "bridge_estimate_ms": bridge,
                    "arc_correct": arc_correct,
                }
            )

    expected_row_count = len(expected_layers) * len(EXPECTED_N)
    instrumentation_complete = (
        bool(expected_layers)
        and len(rows) == expected_row_count
        and not missing
    )

    def median_for(n_experts, key):
        values = [
            row[key]
            for row in rows
            if row["nexperts"] == n_experts
            and isinstance(row.get(key), (int, float))
        ]
        return statistics.median(values) if values else None

    summary = {}
    for n_experts in EXPECTED_N:
        staged = median_for(n_experts, "staged_rtx_ms")
        direct = median_for(n_experts, "direct_host_rtx_ms")
        arc_total = median_for(
            n_experts, "arc_wall_plus_bridge_ms"
        )
        summary[str(n_experts)] = {
            "staged_rtx_ms": staged,
            "direct_host_rtx_ms": direct,
            "arc_wall_plus_bridge_ms": arc_total,
            "direct_gain_fraction": (
                (staged - direct) / staged
                if staged is not None and direct is not None
                else None
            ),
            "arc_gain_fraction": (
                (staged - arc_total) / staged
                if staged is not None and arc_total is not None
                else None
            ),
        }

    promote_direct = bool(
        instrumentation_complete
        and all(row["direct_bitexact"] for row in rows)
        and all(
            summary[str(n)]["direct_gain_fraction"] is not None
            and summary[str(n)]["direct_gain_fraction"] >= 0.10
            for n in (1, 2)
        )
    )
    promote_arc = bool(
        instrumentation_complete
        and all(row["arc_correct"] for row in rows)
        and all(
            summary[str(n)]["arc_gain_fraction"] is not None
            and summary[str(n)]["arc_gain_fraction"] >= 0.10
            for n in (1, 2)
        )
    )

    output = {
        "kind": "s100_phase9_miss_economics",
        "status": (
            "measured" if instrumentation_complete else "incomplete"
        ),
        "instrumentation_complete": instrumentation_complete,
        "expected_layers": expected_layers,
        "expected_row_count": expected_row_count,
        "observed_row_count": len(rows),
        "missing_evidence": missing,
        "rows": rows,
        "median_by_n": summary,
        "DIRECTHOST_PROMOTE": promote_direct,
        "ARC_MISS_PROMOTE": promote_arc,
    }
    path = directory / "S100_PHASE9_MISS_ECONOMICS.json"
    path.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0 if instrumentation_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
