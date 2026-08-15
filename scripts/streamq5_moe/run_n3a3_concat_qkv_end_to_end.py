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
from scripts.streamq5_moe.run_n3a2_attention_projection_flow import SOURCE as N3A2_SOURCE


R = ROOT / "reports/streamq5_moe"
PREREG = R / "N3A3_CONCAT_QKV_END_TO_END_PREREGISTRATION.md"
N3A2 = R / "n3a2_attention_projection_flow.json"
P7_TEST = R / "p7c_ervf_end_to_end_test.json"
OUTPUT = R / "n3a3_concat_qkv_end_to_end.json"
TOKENS = 128
WARMUP = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha_device(array: cp.ndarray) -> str:
    return hashlib.sha256(cp.asnumpy(array).view(np.uint8)).hexdigest()


def runtime_class():
    source = p13c.P7_RUNNER.read_text(encoding="utf-8")
    marker = "for old, new in replacements.items():"
    old_qkv = (
        '            self.q8(layer, "q", self.normed, self.q)\n'
        '            self.q8(layer, "k", self.normed, self.key)\n'
        '            self.q8(layer, "v", self.normed, self.value)'
    )
    new_qkv = (
        '            if self._n3a3_candidate:\n'
        '                self.qkv_concat(layer)\n'
        '            else:\n'
        '                self.q8(layer, "q", self.normed, self.q)\n'
        '                self.q8(layer, "k", self.normed, self.key)\n'
        '                self.q8(layer, "v", self.normed, self.value)'
    )
    additions = {
        p13c.p12r.OLD_PIN: p13c.p12r.NEW_PIN,
        p13c.p12r.OLD_COPY: p13c.p12r.NEW_COPY,
        "class Runtime:": p13c.p12r2.MAPPED_CODE,
        "pin_expert_bank(self.expert_bank)": "map_expert_bank(self.expert_bank)",
        "copy_expert(": "copy_expert_mapped(",
        "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE,":
            "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + ATTENTION_SOURCE + N3A2_SOURCE,",
        '"attention_scores",': '"attention_scores_evt8", "attention_softmax_materialize",',
        '"attention_values",': '"attention_values_materialized",',
        '"q8_ervf16",': '"q8_ervf16", "n3a2_qkv_concat",',
        'self.k["attention_scores"]((Q_HEADS * context,), (HEAD_DIM,), (':
            'self.k["attention_scores_evt8"](((Q_HEADS * context + 7) // 8,), (256,), (',
        'self.k["attention_values"]((Q_HEADS,), (HEAD_DIM,), (':
            'self.k["attention_softmax_materialize"]((Q_HEADS,), (HEAD_DIM,), (\n'
            '                self.scores, np.int32(context),\n'
            '            ), stream=self.compute)\n'
            '            self.k["attention_values_materialized"]((Q_HEADS,), (HEAD_DIM,), (',
        old_qkv: new_qkv,
    }
    pairs = ", ".join(repr(old) + ": " + repr(new) for old, new in additions.items())
    if marker not in source:
        raise RuntimeError("P7 replacement marker missing")
    source = source.replace(marker, "replacements.update({" + pairs + "})\n" + marker)
    old_exec = 'exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})'
    if old_exec not in source:
        raise RuntimeError("P7 runtime exec transform target missing")
    namespace = {"__name__": "n3a3_runtime", "__file__": str(p13c.P7_RUNNER),
                 "ATTENTION_SOURCE": p13c.ATTENTION_SOURCE, "N3A2_SOURCE": N3A2_SOURCE}
    transformed = source.replace(old_exec, 'exec(compile(source, str(source_path), "exec"), globals())')
    exec(compile(transformed, str(p13c.P7_RUNNER), "exec"), namespace)
    return namespace["Runtime"]


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "min": float(x.min()), "max": float(x.max())}


def dynamic_snapshot(runtime):
    return [list(layer.items()) for layer in runtime.dynamic]


def snapshot(runtime):
    runtime.compute.synchronize(); runtime.copy.synchronize()
    return {"dynamic": dynamic_snapshot(runtime), "total_misses": runtime.total_misses,
            "total_miss_bytes": runtime.total_miss_bytes,
            "kv_layer_position_writes": runtime.kv_layer_position_writes,
            "route_unique_failures": runtime.route_unique_failures,
            "route_weight_error_max": runtime.route_weight_error_max}


def restore(runtime, saved):
    runtime.dynamic = [OrderedDict(items) for items in saved["dynamic"]]
    for name in ("total_misses", "total_miss_bytes", "kv_layer_position_writes",
                 "route_unique_failures", "route_weight_error_max"):
        setattr(runtime, name, saved[name])


def install(runtime):
    runtime._n3a3_candidate = False

    def qkv_concat(self, layer):
        records = []
        for name in ("q", "k", "v"):
            _index, record = self.record_by_key[(layer, name)]
            records.append((self.device_offsets[(layer, name)], record))
        (qb, qr), (kb, kr), (vb, vr) = records
        self.k["n3a2_qkv_concat"]((320,), (256,), (
            self.normed, self.trunk,
            np.int64(qb), np.int64(qr["code_bytes"]),
            np.int64(kb), np.int64(kr["code_bytes"]),
            np.int64(vb), np.int64(vr["code_bytes"]),
            self.q, self.key, self.value,
        ), stream=self.compute)

    runtime.qkv_concat = MethodType(qkv_concat, runtime)


def select(runtime, candidate: bool):
    runtime._n3a3_candidate = candidate


def observe(runtime, token: int, position: int, candidate: bool):
    select(runtime, candidate)
    row = runtime.decode(token, position)
    return {"prediction": int(row["prediction"]), "misses": int(row["misses"]),
            "wall_ms": float(row["wall_ms"]), "kv": runtime.kv_digest(position + 1),
            "dynamic": dynamic_snapshot(runtime), "logits_sha256": array_sha_device(runtime.logits),
            "state_sha256": array_sha_device(runtime.state)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    n3a2 = json.loads(N3A2.read_text(encoding="utf-8"))
    if not n3a2["overall_pass"] or n3a2["selected"] != "concat_qkv":
        raise RuntimeError("N3A2 concat_qkv pass required")
    Runtime = runtime_class()
    lock = json.loads(p12.P6_LOCK.read_text(encoding="utf-8"))
    runtime = Runtime(lock); activation_ms = runtime.activate_domain("general"); install(runtime)
    physical = runtime.physical()
    rollout = json.loads(P7_TEST.read_text(encoding="utf-8"))["rollout"]
    prompt = [int(value) for value in rollout["prompt_ids"]]
    token = prompt[0]
    for position in range(len(prompt) - 1):
        select(runtime, False); runtime.decode(token, position); token = prompt[position + 1]
    pairs = []; baseline_times = []; candidate_times = []
    for step in range(TOKENS):
        position = len(prompt) - 1 + step; saved = snapshot(runtime)
        order = (False, True) if step % 2 == 0 else (True, False); observed = {}
        for candidate in order:
            restore(runtime, saved)
            observed["candidate" if candidate else "baseline"] = observe(runtime, token, position, candidate)
        restore(runtime, saved); select(runtime, False); canonical = runtime.decode(token, position)
        baseline = observed["baseline"]; candidate = observed["candidate"]
        exact = {"exact_prediction": baseline["prediction"] == candidate["prediction"],
                 "exact_misses": baseline["misses"] == candidate["misses"],
                 "exact_kv": baseline["kv"] == candidate["kv"],
                 "exact_dynamic": baseline["dynamic"] == candidate["dynamic"],
                 "exact_logits": baseline["logits_sha256"] == candidate["logits_sha256"],
                 "exact_state": baseline["state_sha256"] == candidate["state_sha256"]}
        pairs.append({"step": step, "order": ["candidate" if flag else "baseline" for flag in order],
                      "baseline": {key: value for key, value in baseline.items() if key != "dynamic"},
                      "candidate": {key: value for key, value in candidate.items() if key != "dynamic"}, **exact})
        if step >= WARMUP:
            baseline_times.append(baseline["wall_ms"]); candidate_times.append(candidate["wall_ms"])
        token = int(canonical["prediction"])
    baseline = stats(baseline_times); candidate = stats(candidate_times)
    ratios = {name: candidate[name] / baseline[name] for name in ("mean", "p50", "p95")}
    names = ("exact_prediction", "exact_misses", "exact_kv", "exact_dynamic", "exact_logits", "exact_state")
    exactness = {name: all(pair[name] for pair in pairs) for name in names}
    gates = {"tokens_128": len(pairs) == TOKENS, "warmup_16": len(baseline_times) == TOKENS - WARMUP,
             **exactness, "mean_ratio_le_0_98": ratios["mean"] <= 0.98,
             "p50_ratio_le_0_98": ratios["p50"] <= 0.98, "p95_ratio_le_1_00": ratios["p95"] <= 1.00}
    result = {"kind": "streamq5_moe_n3a3_concat_qkv_end_to_end", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
                         "n3a2_sha256": sha256(N3A2), "p7_test_sha256": sha256(P7_TEST)},
              "configuration": {"candidate": "concat_qkv", "changed_launches_per_layer": "3_to_1"},
              "physical": physical,
              "workload": {"prompt_tokens": len(prompt), "paired_tokens": TOKENS, "warmup_pairs": WARMUP,
                           "timed_pairs": len(baseline_times), "activation_ms": activation_ms,
                           "order": "ABBA_by_adjacent_token_pairs"},
              "baseline": baseline, "candidate": candidate, "ratios": ratios,
              "exactness": exactness, "gates": gates, "pairs": pairs, "overall_pass": all(gates.values()),
              "claim_boundary": "Paired 128-token same-runtime P13 concat-QKV integration; no 10K, cross-model, cross-GPU, energy, novelty or SOTA claim."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline, "candidate": candidate, "ratios": ratios,
                      "exactness": exactness, "gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
