#!/usr/bin/env python3
"""Independent fail-closed verifier for frozen PORT80B T0-R4-REF-R2 artifacts."""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
RUN_DIR = REPORTS / "runs" / "streamq5_moe" / "port80b_t0r4r2_official_layer0"
RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_t0r4r2_official_layer0_reference.py"
LOCK = REPORTS / "port80b_t0r4r2_runner_lock.json"
VLOCK = REPORTS / "port80b_t0r4r2_verifier_lock.json"
BANK = RUN_DIR / "layer0_513_real_q5_records.bin"
BANK_BYTES = 1_040_117_760

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""): h.update(block)
    return h.hexdigest()

def tbytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()

def manifest_entry(name: str, value: torch.Tensor) -> dict:
    return {"semantic_key": name, "dtype": str(value.dtype), "shape": list(value.shape),
            "bytes": value.numel() * value.element_size(),
            "sha256": hashlib.sha256(tbytes(value)).hexdigest()}

def preflight() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    vlock = json.loads(VLOCK.read_text(encoding="utf-8"))
    checks = {
        "runner_bound": RUNNER.is_file() and sha(RUNNER) == lock["runner_sha256"],
        "verifier_bound_runner_lock": sha(Path(__file__)) == lock["verifier_sha256"],
        "verifier_bound_verifier_lock": sha(Path(__file__)) == vlock["verifier_sha256"],
        "verifier_lock_bound": sha(VLOCK) == lock["verifier_lock_sha256"],
        "execution_outputs_absent": not RUN_DIR.exists() or not any(RUN_DIR.glob("t0r4r2_run_*_result.json")),
        "schema_contract": vlock["schema_version"] == "PORT80B_T0R4R2_REF_V1",
    }
    return {"kind":"port80b_t0r4r2_independent_verifier_preflight", "pass":all(checks.values()),
            "checks":checks, "checks_passed":sum(checks.values()), "checks_total":len(checks),
            "claim_boundary":"Source/provenance/schema preflight only; no model execution or output claim."}

def verify(index: int) -> dict:
    result_path = RUN_DIR / f"t0r4r2_run_{index}_result.json"
    raw_path = RUN_DIR / f"t0r4r2_run_{index}_raw.safetensors"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["result_identity"] = result["kind"] == "port80b_t0r4r2_official_layer0_reference_and_bank_stage" and result["run_index"] == index
    checks["raw_artifact_hash"] = sha(raw_path) == result["raw_artifact_sha256"]
    observed = {}
    tensors = {}
    with safe_open(raw_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            tensors[key] = value
            observed[key] = manifest_entry(key, value)
    checks["raw_manifest_exact"] = observed == result["raw_tensor_manifest"]
    checks["all_raw_finite"] = all(bool(torch.isfinite(v.float()).all()) for v in tensors.values())
    cache = result["cache_state_schema"]
    checks["cache_row_count"] = len(cache) == 64
    checks["cache_schema"] = all(row["conv"]["dtype"] == "torch.bfloat16" and row["conv"]["shape"] == [1,8192,4]
                                  and row["recurrent"]["dtype"] == "torch.float32"
                                  and row["recurrent"]["shape"] == [1,32,128,128] for row in cache)
    checks["cache_prefix_coverage"] = {(row["prompt"],row["step"]) for row in cache} == {(p,s) for p in range(4) for s in range(1,17)}
    route_ok = True; margins = []
    for prompt in range(4):
        ids=tensors[f"p{prompt}_whole_router_ids"]; pre=tensors[f"p{prompt}_whole_router_weights_precast_fp32"]
        native=tensors[f"p{prompt}_whole_router_weights_native_bf16"]; margin=tensors[f"p{prompt}_whole_router_top10_top11_margin_fp32"]
        route_ok &= bool((ids>=0).all() and (ids<512).all() and (pre>0).all() and (native>0).all()
                         and torch.isfinite(pre).all() and torch.isfinite(native.float()).all()
                         and (pre[:,:-1]>=pre[:,1:]).all() and (native[:,:-1]>=native[:,1:]).all()
                         and all(torch.unique(row).numel()==10 for row in ids) and (margin>0).all())
        margins.append(float(margin.min()))
    checks["routes_positive_finite_nonincreasing"] = route_ok
    checks["minimum_margin_exact"] = result["minimum_top10_top11_margin_fp32"] == min(margins)
    checks["manual_ulp"] = len(result["manual_moe_max_bf16_ulp"]) == 4 and max(result["manual_moe_max_bf16_ulp"]) <= 1
    checks["fresh_cache_ladder"] = len(result["state_equivalence"]) == 4 and all(x["all_final_outputs_bitwise_equal"] for x in result["state_equivalence"])
    rt=result["runtime_contract"]
    checks["runtime_contract"] = (rt["affinity"]==list(range(16)) and rt["torch_threads"]==1 and rt["torch_interop_threads"]==1
        and rt["deterministic_algorithms"] and rt["float32_matmul_precision"]=="highest" and rt["mkldnn_enabled"]
        and rt["flush_denormal"] is False and rt["flush_denormal_nonzero_subnormal_probe"]
        and rt["cpu_identity"]=="Intel64 Family 6 Model 197 Stepping 2, GenuineIntel" and rt["torch_cpu_capability"]=="AVX2"
        and rt["inference_mode_inside_compute"] and rt["autocast_cpu_inside_compute"] is False and result["cuda_initialized_after"] is False)
    records=result["record_artifact"]
    checks["record_bank"] = records["bytes"]==BANK_BYTES and BANK.stat().st_size==BANK_BYTES and sha(BANK)==records["sha256"]
    recs=records["records"]
    checks["record_manifest_bindings"] = (len(recs)==513 and all(e["revision"]=="a19358a7659bd1f564300250ee189120c49a562f"
        and e["layer"]==0 and e["expert"]==n and e["shared"]==(n==512) and e["expert_total_bytes"]==2_027_520
        and len(e["projections"])==3 and all(p["revision"]==e["revision"] and p["layer"]==0 and p["expert"]==n
            and p["shared"]==e["shared"] and p["projection"]==j and p["total_record_bytes"]==675_840
            for j,p in enumerate(e["projections"])) for n,e in enumerate(recs)))
    checks["resources"] = (result["resources"]["windows_peak_working_set_bytes"] <= 12*2**30
                           and result["resources"]["minimum_available_ram_bytes"] >= 2*2**30
                           and result["resources"]["projected_steady_working_set_bytes"] <= int(10.5*2**30))
    if index==2:
        checks["clean_replay"] = all(result["clean_replay"].values())
    return {"kind":"port80b_t0r4r2_independent_result_verification", "run_index":index,
            "pass":all(checks.values()), "checks":checks, "checks_passed":sum(checks.values()), "checks_total":len(checks),
            "claim_boundary":"Independent verification of the BF16 reference/bank-build stage only; no Q5 execution or T0-P4 claim."}

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("preflight","verify"),required=True); parser.add_argument("--run-index",type=int,choices=(1,2)); args=parser.parse_args()
    if args.phase=="preflight": result=preflight()
    else:
        if args.run_index is None: raise SystemExit("--run-index required for verify")
        result=verify(args.run_index)
    print(json.dumps(result,indent=2)); return 0 if result["pass"] else 2

if __name__=="__main__": raise SystemExit(main())
