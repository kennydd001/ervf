from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.streamq5_moe.run_p12_32g_4k_endurance as p12
import scripts.streamq5_moe.run_p13c_evt_pm_32g_endurance as p13c


R = ROOT / "reports/streamq5_moe"
PREREG = R / "N1A2_Q5_STAGING_END_TO_END_PREREGISTRATION.md"
N1A = R / "n1a_shared_activation_ervf.json"
P7_TEST = R / "p7c_ervf_end_to_end_test.json"
OUTPUT = R / "n1a2_q5_staging_end_to_end.json"
TOKENS = 256


STAGED_SOURCE = r'''
extern "C" __global__ void q5_gate_up_ervf16_staged(
    const float* x, const unsigned char* cache, const int* slots,
    const int* positions, float* gate, float* up) {
    __shared__ float staged[2048];
    for (int col = (int)threadIdx.x; col < 2048; col += 256) staged[col] = x[col];
    __syncthreads();
    const int group = (int)threadIdx.x >> 4;
    const int lane = (int)threadIdx.x & 15;
    const int global_row = (int)blockIdx.x * 16 + group;
    if (global_row >= 8 * 1536) return;
    const int expert = global_row / 1536;
    const int local = global_row - expert * 1536;
    const int projection = local >= 768;
    const int row = local - projection * 768;
    const int output_expert = positions[expert];
    const long long base = (long long)slots[expert] * 3035136LL + (long long)projection * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_ervf_row<16>(staged, packed, scales, row, 2048, lane);
    if (lane == 0) {
        if (projection) up[output_expert * 768 + row] = round_bf16(value);
        else gate[output_expert * 768 + row] = round_bf16(value);
    }
}
extern "C" __global__ void q5_down_ervf16_staged(
    const float* activation, const unsigned char* cache, const int* slots,
    const int* positions, float* down) {
    __shared__ float staged[768];
    const int first_row = (int)blockIdx.x * 16;
    const int expert = first_row / 2048;
    const int output_expert = positions[expert];
    for (int col = (int)threadIdx.x; col < 768; col += 256)
        staged[col] = activation[output_expert * 768 + col];
    __syncthreads();
    const int group = (int)threadIdx.x >> 4;
    const int lane = (int)threadIdx.x & 15;
    const int global_row = first_row + group;
    if (global_row >= 8 * 2048) return;
    const int row = global_row - expert * 2048;
    const long long base = (long long)slots[expert] * 3035136LL + 2LL * 1011712LL;
    const unsigned char* packed = cache + base + 64;
    const unsigned short* scales = (const unsigned short*)(cache + base + 64 + 983040);
    float value = q5_ervf_row<16>(staged, packed, scales, row, 768, lane);
    if (lane == 0) down[output_expert * 2048 + row] = round_bf16(value);
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_class(staged: bool):
    if not staged:
        return p13c.load_runtime_class_evt_pm()
    original_source = p13c.ATTENTION_SOURCE
    original_loader = p13c.load_runtime_class_evt_pm
    def loader():
        source = p13c.P7_RUNNER.read_text(encoding="utf-8")
        marker = "for old, new in replacements.items():"
        additions = {
            p13c.p12r.OLD_PIN: p13c.p12r.NEW_PIN,
            p13c.p12r.OLD_COPY: p13c.p12r.NEW_COPY,
            "class Runtime:": p13c.p12r2.MAPPED_CODE,
            "pin_expert_bank(self.expert_bank)": "map_expert_bank(self.expert_bank)",
            "copy_expert(": "copy_expert_mapped(",
            "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE,": "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + ATTENTION_SOURCE + STAGED_SOURCE,",
            '"attention_scores",': '"attention_scores_evt8", "attention_softmax_materialize",',
            '"attention_values",': '"attention_values_materialized",',
            '"q5_gate_up_ervf16",': '"q5_gate_up_ervf16", "q5_gate_up_ervf16_staged",',
            '"q5_down_ervf16",': '"q5_down_ervf16", "q5_down_ervf16_staged",',
            'self.k["attention_scores"]((Q_HEADS * context,), (HEAD_DIM,), (':
                'self.k["attention_scores_evt8"](((Q_HEADS * context + 7) // 8,), (256,), (',
            'self.k["attention_values"]((Q_HEADS,), (HEAD_DIM,), (':
                'self.k["attention_softmax_materialize"]((Q_HEADS,), (HEAD_DIM,), (\n'
                '                self.scores, np.int32(context),\n'
                '            ), stream=self.compute)\n'
                '            self.k["attention_values_materialized"]((Q_HEADS,), (HEAD_DIM,), (',
            'self.k["q5_gate_up_ervf16"](((count * 1536 + 15) // 16,), (256,), (':
                'self.k["q5_gate_up_ervf16_staged"](((count * 1536 + 15) // 16,), (256,), (',
            'self.k["q5_down_ervf16"](((count * 2048 + 15) // 16,), (256,), (':
                'self.k["q5_down_ervf16_staged"](((count * 2048 + 15) // 16,), (256,), (',
        }
        pairs = ", ".join(repr(old) + ": " + repr(new) for old, new in additions.items())
        source = source.replace(marker, "replacements.update({" + pairs + "})\n" + marker)
        old_exec = 'exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})'
        if old_exec not in source: raise RuntimeError("P7 transform target missing")
        namespace = {"__name__": "n1a2_runtime", "__file__": str(p13c.P7_RUNNER),
                     "ATTENTION_SOURCE": original_source, "STAGED_SOURCE": STAGED_SOURCE}
        exec(compile(source.replace(old_exec, 'exec(compile(source, str(source_path), "exec"), globals())'), str(p13c.P7_RUNNER), "exec"), namespace)
        return namespace["Runtime"]
    return loader()


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def execute(Runtime, inputs):
    lock = json.loads(p12.P6_LOCK.read_text(encoding="utf-8"))
    runtime = Runtime(lock); activation_ms = runtime.activate_domain("general")
    outputs = []; measured = []
    for position, token in enumerate(inputs):
        row = runtime.decode(int(token), position)
        outputs.append({"prediction": int(row["prediction"]), "misses": int(row["misses"]), "wall_ms": float(row["wall_ms"]), "finite": bool(row["finite"])})
        if position >= len(inputs) - TOKENS: measured.append(float(row["wall_ms"]))
    return {"activation_ms": activation_ms, "tokens": outputs, "timing": stats(measured),
            "kv_digest": runtime.kv_digest(len(inputs)), "finite": all(row["finite"] for row in outputs)}


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    n1a = json.loads(N1A.read_text(encoding="utf-8"))
    if not n1a["component_pass"]["q5"]: raise RuntimeError("N1A Q5 pass required")
    p7 = json.loads(P7_TEST.read_text(encoding="utf-8"))["rollout"]
    inputs = [int(x) for x in p7["prompt_ids"]] + [int(x) for x in p7["feedback_ids"][:TOKENS]]
    baseline = execute(runtime_class(False), inputs)
    candidate = execute(runtime_class(True), inputs)
    exact_predictions = [x["prediction"] for x in baseline["tokens"]] == [x["prediction"] for x in candidate["tokens"]]
    exact_misses = [x["misses"] for x in baseline["tokens"]] == [x["misses"] for x in candidate["tokens"]]
    exact_kv = baseline["kv_digest"] == candidate["kv_digest"]
    ratios = {"mean": candidate["timing"]["mean"] / baseline["timing"]["mean"],
              "p95": candidate["timing"]["p95"] / baseline["timing"]["p95"]}
    gates = {"tokens_256": len(candidate["tokens"]) - len(p7["prompt_ids"]) == TOKENS,
             "finite": baseline["finite"] and candidate["finite"], "predictions_exact": exact_predictions,
             "misses_exact": exact_misses, "kv_exact": exact_kv,
             "mean_ratio_le_0_95": ratios["mean"] <= 0.95, "p95_ratio_le_0_95": ratios["p95"] <= 0.95}
    result = {"kind": "streamq5_moe_n1a2_q5_staging_end_to_end", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"preregistration_sha256": sha256(PREREG), "n1a_sha256": sha256(N1A), "p7_test_sha256": sha256(P7_TEST),
                         "p13c_result_sha256": sha256(R / "p13c_evt_pm_32g_endurance.json")},
              "workload": {"prompt_tokens": len(p7["prompt_ids"]), "measured_tokens": TOKENS, "total_calls": len(inputs)},
              "baseline": baseline, "candidate": candidate, "ratios": ratios, "gates": gates, "overall_pass": all(gates.values()),
              "claim_boundary": "Controlled 256-token exact-input replay; not a sampled rollout or 10K endurance run."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_timing": baseline["timing"], "candidate_timing": candidate["timing"],
                      "ratios": ratios, "gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__": main()
