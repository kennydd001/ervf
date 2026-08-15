from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from moe_lab.ergv_compiler import (  # noqa: E402
    FINAL_CAST,
    FMA_POLICY,
    REDUCTION_STRIDES,
    WIDTHS,
    audit_manual_n1c_source,
    audit_manual_p7_source,
    build_exact_reduction_ir,
    evaluate_physical_schedule,
    evaluate_reference_tree,
    generate_cuda_source,
    mutate_schedule_node,
    n1c_frozen_choices,
    schedule_exact_reduction,
    source_sha256,
    verify_graph_isomorphism,
)


REPORTS = ROOT / "reports" / "streamq5_moe"
PREREG = REPORTS / "ERGV_COMPILER_PREREGISTRATION.md"
OUTPUT = REPORTS / "ergv_compiler_cpu_tests.json"
P6B_SOURCE = ROOT / "scripts" / "streamq5_moe" / "run_p6a_end_to_end_decode.py"
P7_SOURCE = ROOT / "scripts" / "streamq5_moe" / "run_p7b_ervf_kernel.py"
N1C_SOURCE = ROOT / "scripts" / "streamq5_moe" / "run_n1c_generalized_exact_reduction_autotuner.py"
SEED = 120842


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float_bits(value: np.float32) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def assert_source_partition(ir) -> None:
    scalar_columns: list[int] = []
    for accumulator in ir.accumulators:
        work_indices = [load.work_index for load in accumulator.source_order]
        assert work_indices == sorted(work_indices)
        for load in accumulator.source_order:
            scalar_columns.extend(load.scalar_columns)
            if ir.family == "q5":
                assert len(load.scalar_columns) == 8
                assert load.scalar_columns == tuple(
                    range(load.work_index * 8, load.work_index * 8 + 8)
                )
            else:
                assert load.scalar_columns == (load.work_index,)
    assert sorted(scalar_columns) == list(range(ir.columns))
    assert len(scalar_columns) == len(set(scalar_columns)) == ir.columns


def adversarial_inputs(rng: np.random.Generator) -> list[np.ndarray]:
    cases: list[np.ndarray] = []
    cases.append(np.zeros(256, dtype=np.float32))
    signed_zero = np.zeros(256, dtype=np.float32)
    signed_zero[1::2] = np.float32(-0.0)
    cases.append(signed_zero)
    cancellation = np.resize(
        np.asarray([1.0e20, 1.0, -1.0e20, -1.0, 3.0, -3.0], dtype=np.float32),
        256,
    )
    cases.append(cancellation)
    powers = np.asarray(
        [np.ldexp(-1.0 if index & 1 else 1.0, (index % 240) - 120) for index in range(256)],
        dtype=np.float32,
    )
    cases.append(powers)
    subnormal = np.full(256, np.nextafter(np.float32(0), np.float32(1)), dtype=np.float32)
    subnormal[1::2] *= np.float32(-1)
    cases.append(subnormal)
    near_max = np.full(256, np.finfo(np.float32).max / np.float32(512), dtype=np.float32)
    near_max[1::2] *= np.float32(-1)
    cases.append(near_max)
    for _ in range(128):
        exponent = rng.integers(-60, 60, size=256)
        mantissa = rng.uniform(-1.0, 1.0, size=256)
        cases.append(np.ldexp(mantissa, exponent).astype(np.float32))
    return cases


def main() -> None:
    started = time.perf_counter()
    tests: list[dict] = []

    def run(name: str, action) -> None:
        test_started = time.perf_counter()
        try:
            detail = action()
            tests.append(
                {
                    "name": name,
                    "passed": True,
                    "detail": detail,
                    "wall_ms": (time.perf_counter() - test_started) * 1000,
                }
            )
        except Exception as error:  # preserve every failure in the output
            tests.append(
                {
                    "name": name,
                    "passed": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "wall_ms": (time.perf_counter() - test_started) * 1000,
                }
            )

    irs = {
        "q8_2048": build_exact_reduction_ir("q8", 2048),
        "q8_4096": build_exact_reduction_ir("q8", 4096),
        "q5_2048": build_exact_reduction_ir("q5", 2048),
        "q5_768": build_exact_reduction_ir("q5", 768),
    }

    def ir_invariants():
        for name, ir in irs.items():
            assert len(ir.accumulators) == 256, name
            assert len(ir.nodes) == 512, name
            assert sum(node.opcode == "ordered_add" for node in ir.nodes) == 255
            assert ir.fma_policy == FMA_POLICY
            assert ir.final_cast == FINAL_CAST
            assert_source_partition(ir)
        return {name: ir.digest() for name, ir in irs.items()}

    run("exact_ir_invariants_and_source_partition", ir_invariants)

    schedules = {
        f"{name}_w{width}": schedule_exact_reduction(ir, width)
        for name, ir in irs.items()
        for width in WIDTHS
    }

    def isomorphism_all_widths():
        details = {}
        for name, ir in irs.items():
            details[name] = {}
            for width in WIDTHS:
                schedule = schedules[f"{name}_w{width}"]
                result = verify_graph_isomorphism(ir, schedule)
                assert result.passed, (name, width, result.reasons)
                assert result.compared_nodes == 512
                stages = {
                    stage: sum(item.stage == stage for item in schedule.scheduled_adds)
                    for stage in ("lane_local", "cross_warp_shared", "warp_shuffle")
                }
                if width == 64:
                    assert stages["cross_warp_shared"] == 32
                else:
                    assert stages["cross_warp_shared"] == 0
                details[name][str(width)] = {
                    "rows_per_block": schedule.rows_per_block,
                    "virtual_accumulators_per_lane": schedule.virtual_accumulators_per_lane,
                    "stage_node_counts": stages,
                }
        return details

    run("ordered_graph_isomorphism_q8_q5_all_widths", isomorphism_all_widths)

    def cpu_bit_equality():
        rng = np.random.default_rng(SEED)
        cases = adversarial_inputs(rng)
        comparisons = 0
        for values in cases:
            for name, ir in irs.items():
                expected = evaluate_reference_tree(ir, values)
                for width in WIDTHS:
                    observed = evaluate_physical_schedule(schedules[f"{name}_w{width}"], values)
                    assert float_bits(observed) == float_bits(expected), (
                        name,
                        width,
                        float_bits(expected),
                        float_bits(observed),
                    )
                    comparisons += 1
        return {
            "seed": SEED,
            "input_vectors": len(cases),
            "shapes": list(irs),
            "widths": list(WIDTHS),
            "bitwise_comparisons": comparisons,
        }

    run("cpu_random_and_adversarial_bit_equality", cpu_bit_equality)

    def negative_mutations():
        ir = irs["q8_4096"]
        schedule = schedules["q8_4096_w16"]
        node_map = schedule.node_map()
        add_id = node_map[schedule.root].inputs[0]
        add_node = node_map[add_id]
        swapped = mutate_schedule_node(
            schedule, add_id, inputs=(add_node.inputs[1], add_node.inputs[0])
        )
        bad_source = list(schedule.source_orders)
        bad_source[0] = tuple(reversed(bad_source[0]))
        bad_lane = list(schedule.lane_mapping)
        bad_lane[0], bad_lane[1] = bad_lane[1], bad_lane[0]
        mutations = {
            "ordered_children_swapped": swapped,
            "source_order_reversed": replace(schedule, source_orders=tuple(bad_source)),
            "final_cast_changed": replace(schedule, final_cast="round_fp16"),
            "fma_policy_changed": replace(schedule, fma_policy="forced-fma"),
            "lane_mapping_swapped": replace(schedule, lane_mapping=tuple(bad_lane)),
        }
        rejected = {}
        for name, mutation in mutations.items():
            result = verify_graph_isomorphism(ir, mutation)
            assert not result.passed, name
            rejected[name] = list(result.reasons)

        w64 = schedules["q8_4096_w64"]
        changed_adds = list(w64.scheduled_adds)
        target = next(i for i, item in enumerate(changed_adds) if item.stage == "cross_warp_shared")
        changed_adds[target] = replace(changed_adds[target], stage="warp_shuffle")
        bad_cross = replace(w64, scheduled_adds=tuple(changed_adds))
        result = verify_graph_isomorphism(ir, bad_cross)
        assert not result.passed
        rejected["cross_warp_node_removed"] = list(result.reasons)
        return rejected

    run("negative_semantic_mutations_rejected", negative_mutations)

    p6b_text = P6B_SOURCE.read_text(encoding="utf-8")
    p7_text = P7_SOURCE.read_text(encoding="utf-8")
    n1c_text = N1C_SOURCE.read_text(encoding="utf-8")

    def manual_source_audit():
        p7 = audit_manual_p7_source(p7_text)
        n1c = audit_manual_n1c_source(n1c_text)
        assert p7.structural_contract_pass, p7.reasons
        assert n1c.structural_contract_pass, n1c.reasons
        baseline_markers = (
            "__shared__ float reduction[256];",
            "for (int stride = 128; stride > 0; stride >>= 1)",
            "reduction[threadIdx.x] += reduction[threadIdx.x + stride]",
            "round_bf16(reduction[0])",
        )
        missing = [marker for marker in baseline_markers if marker not in p6b_text]
        assert not missing, missing
        return {
            "p6b_reference_markers": list(baseline_markers),
            "p7_widths": list(p7.widths),
            "n1c_widths": list(n1c.widths),
        }

    run("mechanical_manual_p6b_p7_n1c_source_audit", manual_source_audit)

    def generated_source_gate():
        specs = [
            (irs["q8_2048"], width) for width in WIDTHS
        ] + [
            (irs["q5_2048"], width) for width in WIDTHS
        ] + [
            (irs["q5_768"], width) for width in WIDTHS
        ]
        source = generate_cuda_source(specs)
        source_again = generate_cuda_source(specs)
        assert source == source_again
        for family in ("q8", "q5"):
            for width in WIDTHS:
                suffix = f"ergv_{family}_row_w{width}"
                assert suffix in source, suffix
                assert source.count(f"float {suffix}") == 1, f"duplicate symbol: {suffix}"
        for marker in (
            "int tid = lane + WIDTH * virtual_index;",
            "partial[index] += partial[index + stride / WIDTH];",
            "value += __shfl_down_sync(mask, value, offset, WIDTH);",
            "add lane+32 into lanes 0..31",
        ):
            assert marker in source, marker
        return {
            "source_bytes": len(source.encode("utf-8")),
            "sha256": source_sha256(source),
            "families": ["q8", "q5"],
            "widths": list(WIDTHS),
            "q5_shapes": [2048, 768],
        }

    run("deterministic_q8_q5_cuda_codegen", generated_source_gate)

    def frozen_n1c_plan_gate():
        choices = n1c_frozen_choices()
        observed = {
            (choice.family, choice.projection): choice.width for choice in choices
        }
        expected = {
            ("q8", "head"): 16,
            ("q8", "k"): 64,
            ("q8", "o"): 16,
            ("q8", "q"): 16,
            ("q8", "router"): 64,
            ("q8", "v"): 64,
            ("q5", "gate_up"): 8,
            ("q5", "down"): 8,
        }
        assert observed == expected
        for choice in choices:
            assert choice.rows_per_block == 256 // choice.width
            ir = irs["q8_2048"] if choice.family == "q8" else irs["q5_2048"]
            assert verify_graph_isomorphism(
                ir, schedule_exact_reduction(ir, choice.width)
            ).passed
        return [
            {
                "family": choice.family,
                "projection": choice.projection,
                "width": choice.width,
                "rows_per_block": choice.rows_per_block,
            }
            for choice in choices
        ]

    run("frozen_n1c_choices_representable", frozen_n1c_plan_gate)

    passed = sum(item["passed"] for item in tests)
    result = {
        "kind": "ergv_compiler_c0_cpu_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": passed == len(tests),
        "tests_passed": passed,
        "tests_total": len(tests),
        "tests": tests,
        "source_provenance": {
            "preregistration_sha256": sha256(PREREG),
            "compiler_sha256": sha256(ROOT / "src" / "moe_lab" / "ergv_compiler.py"),
            "test_runner_sha256": sha256(Path(__file__)),
            "p6b_source_sha256": sha256(P6B_SOURCE),
            "manual_p7_source_sha256": sha256(P7_SOURCE),
            "n1c_source_sha256": sha256(N1C_SOURCE),
        },
        "protocol": {
            "seed": SEED,
            "families": ["q8", "q5"],
            "columns": {"q8": [2048, 4096], "q5": [2048, 768]},
            "widths": list(WIDTHS),
            "reduction_strides": list(REDUCTION_STRIDES),
            "gpu_compilation_or_timing": False,
        },
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Restricted IR/codegen CPU proof only; no generated GPU compilation, "
            "real-weight equality, performance, second architecture, public baseline, or novelty claim."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "overall_pass": result["overall_pass"],
                "tests_passed": passed,
                "tests_total": len(tests),
                "failed": [item["name"] for item in tests if not item["passed"]],
                "wall_seconds": result["wall_seconds"],
            },
            indent=2,
        )
    )
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
