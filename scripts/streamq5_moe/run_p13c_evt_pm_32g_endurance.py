from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.streamq5_moe.run_p12_32g_4k_endurance as p12
import scripts.streamq5_moe.run_p12r_residency_lifetime_staging as p12r
import scripts.streamq5_moe.run_p12r2_pinned_window_streaming as p12r2
from scripts.streamq5_moe.run_p13a_exact_virtual_attention import SOURCE as ATTENTION_BASE


P7_RUNNER = ROOT / "scripts/streamq5_moe/run_p7c_ervf_end_to_end.py"
PREREG = ROOT / "reports/streamq5_moe/P13C_EVT_PM_32G_ENDURANCE_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/streamq5_moe/p13c_evt_pm_32g_endurance.json"
P12R2 = ROOT / "reports/streamq5_moe/p12r2_pinned_window_streaming.json"


OLD_ADD = '''    partial[0] += partial[2];
    partial[1] += partial[3];
    float value = partial[0] + partial[1];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffU, value, offset, 32);'''
NEW_ADD = '''    partial[0] = __fadd_rn(partial[0], partial[2]);
    partial[1] = __fadd_rn(partial[1], partial[3]);
    float value = __fadd_rn(partial[0], partial[1]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffU, value, offset, 32);
        value = __fadd_rn(value, other);
    }'''
ATTENTION_SOURCE = ATTENTION_BASE.replace(OLD_ADD, NEW_ADD)


def load_runtime_class_evt_pm():
    source = P7_RUNNER.read_text(encoding="utf-8")
    marker = "for old, new in replacements.items():"
    additions = {
        p12r.OLD_PIN: p12r.NEW_PIN,
        p12r.OLD_COPY: p12r.NEW_COPY,
        "class Runtime:": p12r2.MAPPED_CODE,
        "pin_expert_bank(self.expert_bank)": "map_expert_bank(self.expert_bank)",
        "copy_expert(": "copy_expert_mapped(",
        "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE,": "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + ATTENTION_SOURCE,",
        '"attention_scores",': '"attention_scores_evt8", "attention_softmax_materialize",',
        '"attention_values",': '"attention_values_materialized",',
        'self.k["attention_scores"]((Q_HEADS * context,), (HEAD_DIM,), (':
            'self.k["attention_scores_evt8"](((Q_HEADS * context + 7) // 8,), (256,), (',
        'self.k["attention_values"]((Q_HEADS,), (HEAD_DIM,), (':
            'self.k["attention_softmax_materialize"]((Q_HEADS,), (HEAD_DIM,), (\n'
            '                self.scores, np.int32(context),\n'
            '            ), stream=self.compute)\n'
            '            self.k["attention_values_materialized"]((Q_HEADS,), (HEAD_DIM,), (',
    }
    pairs = ", ".join(repr(old) + ": " + repr(new) for old, new in additions.items())
    source = source.replace(marker, "replacements.update({" + pairs + "})\n" + marker)
    old_exec = 'exec(compile(source, str(source_path), "exec"), {"__name__": "__main__", "__file__": __file__, "ERVF_SOURCE": ERVF_SOURCE})'
    new_exec = 'exec(compile(source, str(source_path), "exec"), globals())'
    if old_exec not in source: raise RuntimeError("P7 import transform target missing")
    namespace = {"__name__": "p13c_runtime", "__file__": str(P7_RUNNER), "ATTENTION_SOURCE": ATTENTION_SOURCE}
    exec(compile(source.replace(old_exec, new_exec), str(P7_RUNNER), "exec"), namespace)
    return namespace["Runtime"]


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


if __name__ == "__main__":
    p12.PREREG = PREREG
    p12.OUTPUT = OUTPUT
    p12.load_runtime_class = load_runtime_class_evt_pm
    p12.__file__ = __file__
    p12.main()
    if OUTPUT.exists():
        result = json.loads(OUTPUT.read_text(encoding="utf-8"))
        baseline = json.loads(P12R2.read_text(encoding="utf-8"))
        first = stats(result["wall_ms"][16:1016])
        paired_late = stats(result["wall_ms"][8192 + 16:8192 + 1016])
        paired = {"first_cycle": first, "third_cycle": paired_late,
                  "mean_ratio": paired_late["mean"] / first["mean"],
                  "p95_ratio": paired_late["p95"] / first["p95"]}
        exactness = {
            "predictions_sha256_equal": result["predictions_sha256"] == baseline["predictions_sha256"],
            "misses_equal": result["misses"] == baseline["misses"],
            "kv_4k_sha256_equal": result["kv_4k"]["sha256"] == baseline["kv_4k"]["sha256"],
        }
        result["kind"] = "streamq5_moe_p13c_evt_pm_32g_endurance"
        result["baseline_p12r2_sha256"] = p12.sha256(P12R2)
        result["exactness_vs_p12r2"] = exactness
        result["context_paired_thermal"] = paired
        result["gates"].pop("last_mean_le_110pct_first", None)
        result["gates"].pop("last_p95_le_110pct_first", None)
        result["gates"]["paired_mean_le_110pct"] = paired["mean_ratio"] <= 1.10
        result["gates"]["paired_p95_le_110pct"] = paired["p95_ratio"] <= 1.10
        result["gates"]["predictions_exact"] = exactness["predictions_sha256_equal"]
        result["gates"]["misses_exact"] = exactness["misses_equal"]
        result["gates"]["kv_4k_exact"] = exactness["kv_4k_sha256_equal"]
        result["overall_pass"] = all(result["gates"].values())
        OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"p13c_output": str(OUTPUT), "timing": result["timing"], "tokens_per_second": result["tokens_per_second"], "exactness": exactness, "context_paired_thermal": paired, "gates": result["gates"], "overall_pass": result["overall_pass"]}, indent=2), flush=True)
