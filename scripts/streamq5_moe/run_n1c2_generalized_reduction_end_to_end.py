from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.streamq5_moe.run_p12_32g_4k_endurance as p12
import scripts.streamq5_moe.run_p13c_evt_pm_32g_endurance as p13c
from scripts.streamq5_moe.run_n1c_generalized_exact_reduction_autotuner import ERVF_SOURCE as N1C_SOURCE


R = ROOT / "reports/streamq5_moe"
PREREG = R / "N1C2_GENERALIZED_REDUCTION_END_TO_END_PREREGISTRATION.md"
N1C = R / "n1c_generalized_exact_reduction_autotuner.json"
P7_TEST = R / "p7c_ervf_end_to_end_test.json"
OUTPUT = R / "n1c2_generalized_reduction_end_to_end.json"
TOKENS = 128
WARMUP = 16
FROZEN_Q8 = {"head": 16, "k": 64, "o": 16, "q": 16, "router": 64, "v": 64}
FROZEN_Q5 = {"gate_up": 8, "down": 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha_device(array: cp.ndarray) -> str:
    observed = cp.asnumpy(array)
    return hashlib.sha256(observed.view(np.uint8)).hexdigest()


def runtime_class():
    source = p13c.P7_RUNNER.read_text(encoding="utf-8")
    marker = "for old, new in replacements.items():"
    additions = {
        p13c.p12r.OLD_PIN: p13c.p12r.NEW_PIN,
        p13c.p12r.OLD_COPY: p13c.p12r.NEW_COPY,
        "class Runtime:": p13c.p12r2.MAPPED_CODE,
        "pin_expert_bank(self.expert_bank)": "map_expert_bank(self.expert_bank)",
        "copy_expert(": "copy_expert_mapped(",
        "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE,":
            "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + ATTENTION_SOURCE + N1C_SOURCE,",
        '"attention_scores",': '"attention_scores_evt8", "attention_softmax_materialize",',
        '"attention_values",': '"attention_values_materialized",',
        '"q8_ervf16",': '"q8_ervf16", "q8_n1c_16", "q8_n1c_64",',
        '"q5_gate_up_ervf16",': '"q5_gate_up_ervf16", "q5_gate_up_n1c_8",',
        '"q5_down_ervf16",': '"q5_down_ervf16", "q5_down_n1c_8",',
        'self.k["attention_scores"]((Q_HEADS * context,), (HEAD_DIM,), (':
            'self.k["attention_scores_evt8"](((Q_HEADS * context + 7) // 8,), (256,), (',
        'self.k["attention_values"]((Q_HEADS,), (HEAD_DIM,), (':
            'self.k["attention_softmax_materialize"]((Q_HEADS,), (HEAD_DIM,), (\n'
            '                self.scores, np.int32(context),\n'
            '            ), stream=self.compute)\n'
            '            self.k["attention_values_materialized"]((Q_HEADS,), (HEAD_DIM,), (',
    }
    pairs = ", ".join(repr(old) + ": " + repr(new) for old, new in additions.items())
    if marker not in source:
        raise RuntimeError("P7 replacement marker missing")
    source = source.replace(marker, "replacements.update({" + pairs + "})\n" + marker)
    old_exec = (
        'exec(compile(source, str(source_path), "exec"), '
        '{"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})'
    )
    if old_exec not in source:
        raise RuntimeError("P7 runtime exec transform target missing")
    namespace = {
        "__name__": "n1c2_runtime",
        "__file__": str(p13c.P7_RUNNER),
        "ATTENTION_SOURCE": p13c.ATTENTION_SOURCE,
        "N1C_SOURCE": N1C_SOURCE,
    }
    transformed = source.replace(
        old_exec, 'exec(compile(source, str(source_path), "exec"), globals())'
    )
    exec(compile(transformed, str(p13c.P7_RUNNER), "exec"), namespace)
    return namespace["Runtime"]


def stats(values: list[float]) -> dict[str, float]:
    observed = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(observed.mean()),
        "p50": float(np.percentile(observed, 50)),
        "p95": float(np.percentile(observed, 95)),
        "p99": float(np.percentile(observed, 99)),
        "min": float(observed.min()),
        "max": float(observed.max()),
    }


def dynamic_snapshot(runtime) -> list[list[tuple]]:
    return [list(layer.items()) for layer in runtime.dynamic]


def snapshot(runtime) -> dict:
    runtime.compute.synchronize()
    runtime.copy.synchronize()
    return {
        "dynamic": dynamic_snapshot(runtime),
        "total_misses": runtime.total_misses,
        "total_miss_bytes": runtime.total_miss_bytes,
        "kv_layer_position_writes": runtime.kv_layer_position_writes,
        "route_unique_failures": runtime.route_unique_failures,
        "route_weight_error_max": runtime.route_weight_error_max,
    }


def restore(runtime, saved: dict) -> None:
    runtime.dynamic = [OrderedDict(items) for items in saved["dynamic"]]
    runtime.total_misses = saved["total_misses"]
    runtime.total_miss_bytes = saved["total_miss_bytes"]
    runtime.kv_layer_position_writes = saved["kv_layer_position_writes"]
    runtime.route_unique_failures = saved["route_unique_failures"]
    runtime.route_weight_error_max = saved["route_weight_error_max"]


def install_candidate_dispatch(runtime) -> None:
    baseline_q8 = runtime.q8
    baseline_experts = runtime.launch_expert_group
    runtime._n1c2_candidate = False

    def q8_dispatch(self, layer, name, source, output):
        if not self._n1c2_candidate:
            return baseline_q8(layer, name, source, output)
        width = FROZEN_Q8[name]
        _index, record = self.record_by_key[(layer, name)]
        base = self.device_offsets[(layer, name)]
        groups = 256 // width
        self.k[f"q8_n1c_{width}"](
            ((record["rows"] + groups - 1) // groups,), (256,),
            (
                source, self.trunk, np.int64(base), np.int64(record["code_bytes"]),
                np.int32(record["rows"]), np.int32(record["cols"]), output,
            ),
            stream=self.compute,
        )

    def expert_dispatch(self, slots, positions, count):
        if not self._n1c2_candidate:
            return baseline_experts(slots, positions, count)
        if count == 0:
            return
        gate_width = FROZEN_Q5["gate_up"]
        down_width = FROZEN_Q5["down"]
        gate_groups = 256 // gate_width
        down_groups = 256 // down_width
        self.k[f"q5_gate_up_n1c_{gate_width}"](
            ((count * 1536 + gate_groups - 1) // gate_groups,), (256,),
            (self.normed, self.expert_cache, slots, positions, self.gate, self.up),
            stream=self.compute,
        )
        self.k["swiglu_n"](
            (count * 3,), (256,), (self.gate, self.up, positions), stream=self.compute
        )
        self.k[f"q5_down_n1c_{down_width}"](
            ((count * 2048 + down_groups - 1) // down_groups,), (256,),
            (self.gate, self.expert_cache, slots, positions, self.down),
            stream=self.compute,
        )

    runtime.q8 = MethodType(q8_dispatch, runtime)
    runtime.launch_expert_group = MethodType(expert_dispatch, runtime)


def select(runtime, candidate: bool) -> None:
    runtime._n1c2_candidate = candidate


def observed_decode(runtime, token: int, position: int, candidate: bool) -> dict:
    select(runtime, candidate)
    row = runtime.decode(token, position)
    return {
        "prediction": int(row["prediction"]),
        "misses": int(row["misses"]),
        "wall_ms": float(row["wall_ms"]),
        "kv": runtime.kv_digest(position + 1),
        "dynamic": dynamic_snapshot(runtime),
        "logits_sha256": array_sha_device(runtime.logits),
        "state_sha256": array_sha_device(runtime.state),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    n1c = json.loads(N1C.read_text(encoding="utf-8"))
    expected = {"q8": FROZEN_Q8, "q5": FROZEN_Q5}
    if not n1c["overall_pass"] or n1c["selected"] != expected:
        raise RuntimeError("N1C pass and exact frozen configuration required")

    Runtime = runtime_class()
    lock = json.loads(p12.P6_LOCK.read_text(encoding="utf-8"))
    runtime = Runtime(lock)
    activation_ms = runtime.activate_domain("general")
    install_candidate_dispatch(runtime)
    physical = runtime.physical()

    rollout = json.loads(P7_TEST.read_text(encoding="utf-8"))["rollout"]
    prompt = [int(value) for value in rollout["prompt_ids"]]
    token = prompt[0]
    for position in range(len(prompt) - 1):
        select(runtime, False)
        runtime.decode(token, position)
        token = prompt[position + 1]

    pairs = []
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    for step in range(TOKENS):
        position = len(prompt) - 1 + step
        saved = snapshot(runtime)
        order = (False, True) if step % 2 == 0 else (True, False)
        observed = {}
        for candidate in order:
            restore(runtime, saved)
            name = "candidate" if candidate else "baseline"
            observed[name] = observed_decode(runtime, token, position, candidate)

        # Only a fresh baseline execution controls the next token and state.
        restore(runtime, saved)
        select(runtime, False)
        canonical = runtime.decode(token, position)

        baseline = observed["baseline"]
        candidate = observed["candidate"]
        exact = {
            "exact_prediction": baseline["prediction"] == candidate["prediction"],
            "exact_misses": baseline["misses"] == candidate["misses"],
            "exact_kv": baseline["kv"] == candidate["kv"],
            "exact_dynamic": baseline["dynamic"] == candidate["dynamic"],
            "exact_logits": baseline["logits_sha256"] == candidate["logits_sha256"],
            "exact_state": baseline["state_sha256"] == candidate["state_sha256"],
        }
        pairs.append({
            "step": step,
            "order": ["candidate" if value else "baseline" for value in order],
            "baseline": {key: value for key, value in baseline.items() if key != "dynamic"},
            "candidate": {key: value for key, value in candidate.items() if key != "dynamic"},
            **exact,
        })
        if step >= WARMUP:
            baseline_times.append(baseline["wall_ms"])
            candidate_times.append(candidate["wall_ms"])
        token = int(canonical["prediction"])

    baseline_stats = stats(baseline_times)
    candidate_stats = stats(candidate_times)
    ratios = {
        name: candidate_stats[name] / baseline_stats[name]
        for name in ("mean", "p50", "p95")
    }
    exact_names = (
        "exact_prediction", "exact_misses", "exact_kv", "exact_dynamic",
        "exact_logits", "exact_state",
    )
    exactness = {name: all(pair[name] for pair in pairs) for name in exact_names}
    gates = {
        "tokens_128": len(pairs) == TOKENS,
        "warmup_16": len(baseline_times) == TOKENS - WARMUP,
        **exactness,
        "mean_ratio_le_0_98": ratios["mean"] <= 0.98,
        "p50_ratio_le_0_98": ratios["p50"] <= 0.98,
        "p95_ratio_le_1_00": ratios["p95"] <= 1.00,
    }
    result = {
        "kind": "streamq5_moe_n1c2_generalized_reduction_end_to_end",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "n1c_sha256": sha256(N1C),
            "p7_test_sha256": sha256(P7_TEST),
        },
        "configuration": expected,
        "physical": physical,
        "workload": {
            "prompt_tokens": len(prompt),
            "paired_tokens": TOKENS,
            "warmup_pairs": WARMUP,
            "timed_pairs": len(baseline_times),
            "activation_ms": activation_ms,
            "order": "ABBA_by_adjacent_token_pairs",
        },
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "ratios": ratios,
        "exactness": exactness,
        "gates": gates,
        "pairs": pairs,
        "overall_pass": all(gates.values()),
        "claim_boundary": (
            "Paired 128-token same-runtime P13 integration of the frozen N1C graph; "
            "no 10K, cross-model, cross-GPU, energy, novelty or SOTA claim."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "ratios": ratios,
        "exactness": exactness,
        "gates": gates,
        "overall_pass": result["overall_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
