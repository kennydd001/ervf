from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7a_kernel_roofline import load_q8
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE, comparison, stats


R = ROOT / "reports/streamq5_moe"
PREREG = R / "N3A4_O_RESIDUAL_FUSION_PREREGISTRATION.md"
BANK = R / "p6a_exact_runtime_bank_result.json"
OUTPUT = R / "n3a4_o_residual_fusion.json"
SEED = 120825
LAYERS = 48


SOURCE = r'''
extern "C" __global__ void n3a4_o_residual_ervf16(
    const float* attention, const float* residual,
    const unsigned char* bank, long long base, long long code_bytes,
    float* state) {
    int group=(int)threadIdx.x>>4; int lane=(int)threadIdx.x&15;
    int row=(int)blockIdx.x*16+group; if(row>=2048)return;
    const signed char* codes=(const signed char*)(bank+base);
    const unsigned short* scales=(const unsigned short*)(bank+base+code_bytes);
    float value=q8_ervf_row<16>(attention,codes,scales,row,4096,lane);
    if(lane==0){
        float projected=round_bf16(value);
        state[row]=round_bf16(residual[row]+projected);
    }
}
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def paired_measure(stream, baseline, candidate, warmups: int, iterations: int):
    for iteration in range(warmups):
        order = (baseline, candidate) if iteration % 2 == 0 else (candidate, baseline)
        for launch in order:
            launch()
    stream.synchronize()
    values = {"baseline": [], "candidate": []}
    for iteration in range(iterations):
        order = (("baseline", baseline), ("candidate", candidate)) if iteration % 2 == 0 else (("candidate", candidate), ("baseline", baseline))
        for name, launch in order:
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record(stream); launch(); end.record(stream); end.synchronize()
            values[name].append(float(cp.cuda.get_elapsed_time(begin, end)))
    result = {name: {"event_ms": rows, "stats": stats(rows)} for name, rows in values.items()}
    result["p50_ratio"] = result["candidate"]["stats"]["p50"] / result["baseline"]["stats"]["p50"]
    result["p95_ratio"] = result["candidate"]["stats"]["p95"] / result["baseline"]["stats"]["p95"]
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest, q8_pin, q8_host, q8_mem, q8, q8_records, q8_sha = load_q8()
    records = [(base, row) for base, row in q8_records if row["name"] == "o" and row["layer"] < LAYERS]
    if len(records) != LAYERS:
        raise RuntimeError("48 physical O records required")
    names = ("q8_ervf16", "residual_add", "n3a4_o_residual_ervf16")
    module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + SOURCE, options=("--std=c++11",), name_expressions=names)
    k = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    rng = np.random.default_rng(SEED)
    validation_attention = cp.asarray(rng.standard_normal((LAYERS, 4096), dtype=np.float32))
    validation_residual = cp.asarray(rng.standard_normal((LAYERS, 2048), dtype=np.float32))
    test_attention = cp.asarray(rng.standard_normal((LAYERS, 4096), dtype=np.float32))
    test_residual = cp.asarray(rng.standard_normal((LAYERS, 2048), dtype=np.float32))
    projected = cp.empty(2048, dtype=cp.float32); state = cp.empty(2048, dtype=cp.float32)

    def baseline_plane(attention, residual, capture=False):
        observed = np.empty((LAYERS, 2048), dtype=np.float32) if capture else None
        for layer, (base, row) in enumerate(records):
            k["q8_ervf16"]((128,), (256,),
                (attention[layer], q8, np.int64(base), np.int64(row["code_bytes"]), np.int32(2048), np.int32(4096), projected), stream=stream)
            k["residual_add"]((8,), (256,), (residual[layer], projected, state, np.int32(2048)), stream=stream)
            if capture:
                stream.synchronize(); observed[layer] = cp.asnumpy(state)
        return observed

    def candidate_plane(attention, residual, capture=False):
        observed = np.empty((LAYERS, 2048), dtype=np.float32) if capture else None
        for layer, (base, row) in enumerate(records):
            k["n3a4_o_residual_ervf16"]((128,), (256,),
                (attention[layer], residual[layer], q8, np.int64(base), np.int64(row["code_bytes"]), state), stream=stream)
            if capture:
                stream.synchronize(); observed[layer] = cp.asnumpy(state)
        return observed

    reference = baseline_plane(validation_attention, validation_residual, True)
    observed = candidate_plane(validation_attention, validation_residual, True)
    validation_correctness = comparison(observed, reference)
    validation = paired_measure(stream,
        lambda: baseline_plane(validation_attention, validation_residual),
        lambda: candidate_plane(validation_attention, validation_residual), 5, 30)
    test_opened = bool(validation_correctness["bitwise_equal"] and validation_correctness["finite"] and validation["p50_ratio"] <= 0.98)
    test_correctness = None; test = None
    if test_opened:
        reference = baseline_plane(test_attention, test_residual, True)
        observed = candidate_plane(test_attention, test_residual, True)
        test_correctness = comparison(observed, reference)
        test = paired_measure(stream,
            lambda: baseline_plane(test_attention, test_residual),
            lambda: candidate_plane(test_attention, test_residual), 10, 120)
        test["speedup_p50"] = 1.0 / test["p50_ratio"]
        test["pass"] = bool(test_correctness["bitwise_equal"] and test_correctness["finite"] and test["p50_ratio"] <= 0.97 and test["p95_ratio"] <= 1.00)
    props = cp.cuda.runtime.getDeviceProperties(0)
    result = {"kind": "streamq5_moe_n3a4_o_residual_fusion", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"preregistration_sha256": sha256(PREREG), "script_sha256": sha256(Path(__file__)),
                         "bank_sha256": sha256(BANK), "q8_pinned_aggregate_sha256": q8_sha,
                         "seed": SEED, "layers": LAYERS, "o_records": len(records)},
              "device": {"name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"]},
              "validation_correctness": validation_correctness, "validation": validation,
              "test_opened": test_opened, "test_correctness": test_correctness, "test": test,
              "overall_pass": bool(test and test["pass"]),
              "claim_boundary": "Physical resident 48-layer Q8 O-projection plus residual component only; no full attention, expert path, decoder, quality, cross-GPU or SOTA claim."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_correctness": validation_correctness,
                      "validation_ratios": {"p50": validation["p50_ratio"], "p95": validation["p95_ratio"]},
                      "test_opened": test_opened, "test_correctness": test_correctness,
                      "test": None if test is None else {key: value for key, value in test.items() if key not in ("baseline", "candidate")},
                      "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
