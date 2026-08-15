from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.streamq5_moe.run_p12_32g_4k_endurance as p12
import scripts.streamq5_moe.run_p13c_evt_pm_32g_endurance as p13c
from scripts.streamq5_moe.run_n1b_q5_vectorized_loads import VECTOR_SOURCE

R = ROOT / "reports/streamq5_moe"
PREREG = R / "N1BI_Q5_VECTORLOAD_END_TO_END_PREREGISTRATION.md"
N1B = R / "n1b_q5_vectorized_loads.json"
P7_TEST = R / "p7c_ervf_end_to_end_test.json"
OUTPUT = R / "n1bi_q5_vectorload_end_to_end.json"
TOKENS = 128
WARMUP = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_class():
    source = p13c.P7_RUNNER.read_text(encoding="utf-8")
    marker = "for old, new in replacements.items():"
    additions = {
        p13c.p12r.OLD_PIN: p13c.p12r.NEW_PIN,
        p13c.p12r.OLD_COPY: p13c.p12r.NEW_COPY,
        "class Runtime:": p13c.p12r2.MAPPED_CODE,
        "pin_expert_bank(self.expert_bank)": "map_expert_bank(self.expert_bank)",
        "copy_expert(": "copy_expert_mapped(",
        "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE,": "module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + ATTENTION_SOURCE + VECTOR_SOURCE,",
        '"attention_scores",': '"attention_scores_evt8", "attention_softmax_materialize",',
        '"attention_values",': '"attention_values_materialized",',
        '"q5_gate_up_ervf16",': '"q5_gate_up_ervf16", "q5_gate_up_aligned32x2",',
        '"q5_down_ervf16",': '"q5_down_ervf16", "q5_down_aligned32x2",',
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
    if old_exec not in source:
        raise RuntimeError("P7 transform target missing")
    namespace = {"__name__": "n1bi_runtime", "__file__": str(p13c.P7_RUNNER),
                 "ATTENTION_SOURCE": p13c.ATTENTION_SOURCE, "VECTOR_SOURCE": VECTOR_SOURCE}
    exec(compile(source.replace(old_exec, 'exec(compile(source, str(source_path), "exec"), globals())'), str(p13c.P7_RUNNER), "exec"), namespace)
    return namespace["Runtime"]


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def dynamic_snapshot(runtime):
    return [list(layer.items()) for layer in runtime.dynamic]


def restore(runtime, snap):
    runtime.dynamic = [OrderedDict(items) for items in snap["dynamic"]]
    runtime.total_misses = snap["total_misses"]
    runtime.total_miss_bytes = snap["total_miss_bytes"]
    runtime.kv_layer_position_writes = snap["kv_layer_position_writes"]
    runtime.route_unique_failures = snap["route_unique_failures"]
    runtime.route_weight_error_max = snap["route_weight_error_max"]


def snapshot(runtime):
    runtime.compute.synchronize(); runtime.copy.synchronize()
    return {"dynamic": dynamic_snapshot(runtime), "total_misses": runtime.total_misses,
            "total_miss_bytes": runtime.total_miss_bytes, "kv_layer_position_writes": runtime.kv_layer_position_writes,
            "route_unique_failures": runtime.route_unique_failures, "route_weight_error_max": runtime.route_weight_error_max}


def select(runtime, candidate: bool):
    runtime.k["q5_gate_up_ervf16"] = runtime.k["q5_gate_up_aligned32x2"] if candidate else runtime._baseline_gate
    runtime.k["q5_down_ervf16"] = runtime.k["q5_down_aligned32x2"] if candidate else runtime._baseline_down


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    n1b = json.loads(N1B.read_text(encoding="utf-8"))
    if not n1b["overall_pass"] or n1b["selected"] != "aligned32x2":
        raise RuntimeError("N1B aligned32x2 pass required")
    Runtime = runtime_class(); lock = json.loads(p12.P6_LOCK.read_text(encoding="utf-8"))
    runtime = Runtime(lock); activation_ms = runtime.activate_domain("general")
    runtime._baseline_gate = runtime.k["q5_gate_up_ervf16"]; runtime._baseline_down = runtime.k["q5_down_ervf16"]
    rollout = json.loads(P7_TEST.read_text(encoding="utf-8"))["rollout"]
    prompt = [int(x) for x in rollout["prompt_ids"]]
    token = prompt[0]
    for position in range(len(prompt) - 1):
        select(runtime, False); runtime.decode(token, position); token = prompt[position + 1]
    pairs=[]; baseline_times=[]; candidate_times=[]
    for step in range(TOKENS):
        position = len(prompt) - 1 + step
        pre = snapshot(runtime)
        order = (False, True) if step % 2 == 0 else (True, False)
        observed = {}
        for candidate in order:
            restore(runtime, pre); select(runtime, candidate)
            row = runtime.decode(token, position)
            observed["candidate" if candidate else "baseline"] = {"prediction": int(row["prediction"]),
                "misses": int(row["misses"]), "wall_ms": float(row["wall_ms"]), "kv": runtime.kv_digest(position + 1),
                "dynamic": dynamic_snapshot(runtime)}
        # Canonical continuation is baseline state, independent of measurement order.
        restore(runtime, pre); select(runtime, False)
        canonical = runtime.decode(token, position)
        base=observed["baseline"]; cand=observed["candidate"]
        pairs.append({"step":step,"order":["candidate" if x else "baseline" for x in order],
                      "baseline":{k:v for k,v in base.items() if k!="dynamic"},"candidate":{k:v for k,v in cand.items() if k!="dynamic"},
                      "exact_prediction":base["prediction"]==cand["prediction"],"exact_misses":base["misses"]==cand["misses"],
                      "exact_kv":base["kv"]==cand["kv"],"exact_dynamic":base["dynamic"]==cand["dynamic"]})
        if step >= WARMUP:
            baseline_times.append(base["wall_ms"]); candidate_times.append(cand["wall_ms"])
        token=int(canonical["prediction"])
    baseline=stats(baseline_times); candidate=stats(candidate_times)
    ratios={name:candidate[name]/baseline[name] for name in ("mean","p50","p95")}
    exact={name:all(row[name] for row in pairs) for name in ("exact_prediction","exact_misses","exact_kv","exact_dynamic")}
    gates={"tokens_128":len(pairs)==TOKENS,"warmup_16":len(baseline_times)==TOKENS-WARMUP,**exact,
           "mean_ratio_le_0_98":ratios["mean"]<=0.98,"p50_ratio_le_0_98":ratios["p50"]<=0.98,"p95_ratio_le_1_00":ratios["p95"]<=1.00}
    result={"kind":"streamq5_moe_n1bi_q5_vectorload_end_to_end","completed_utc":datetime.now(timezone.utc).isoformat(),
            "inputs":{"preregistration_sha256":sha256(PREREG),"n1b_sha256":sha256(N1B),"p7_test_sha256":sha256(P7_TEST)},
            "workload":{"prompt_tokens":len(prompt),"paired_tokens":TOKENS,"warmup_pairs":WARMUP,"activation_ms":activation_ms},
            "baseline":baseline,"candidate":candidate,"ratios":ratios,"exactness":exact,"gates":gates,"pairs":pairs,
            "overall_pass":all(gates.values()),"claim_boundary":"Paired 128-token same-runtime P13 integration; no 10K, cross-model, cross-GPU or SOTA claim."}
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"baseline":baseline,"candidate":candidate,"ratios":ratios,"exactness":exact,"gates":gates,"overall_pass":result["overall_pass"]},indent=2),flush=True)


if __name__=="__main__":main()
