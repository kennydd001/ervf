from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
for entry in (str(ROOT), str(SCRIPTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import het_next_l0_ph0r3_common as common
import het_next_l0_ph0r3_intel as intel_base


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_exploratory_real_projection"
RESULT = RUN / "ph0x_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_EXPLORATORY_REAL_PROJECTION_PREREGISTRATION_2026-08-13.md"


INTEL_SOURCE = intel_base.SRC.replace(
    "for(int d=16;d>=1;d>>=1){#pragma unroll for(int i=0;i<d;i++)p[i]=p[i]+p[i+d];}",
    "for(int d=16;d>=1;d>>=1){\n #pragma unroll\n for(int i=0;i<d;i++)p[i]=p[i]+p[i+d];\n }",
)

CUDA_SOURCE = r'''
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__device__ __forceinline__ float b2f(unsigned short x){return __uint_as_float(((unsigned)x)<<16);}
__device__ __forceinline__ unsigned short rbf(float x){unsigned b=__float_as_uint(x),l=(b>>16)&1U;b+=0x7fffU+l;return (unsigned short)(b>>16);}
__device__ __forceinline__ float rbff(float x){return b2f(rbf(x));}
extern "C" __global__ void ph0(const unsigned char* rec,const unsigned short* x,unsigned short* out,unsigned* count){
 cg::thread_block block=cg::this_thread_block();
 auto tile=cg::tiled_partition<8>(block);
 int lane=(int)tile.thread_rank(),row=(int)blockIdx.x*32+(int)threadIdx.x/8;
 if(row>=512)return;
 const unsigned char* code=rec+64;
 const unsigned short* scale=(const unsigned short*)(rec+64+655360);
 float p[32];
 #pragma unroll
 for(int v=0;v<32;v++){
  int pack=lane+8*v,col=pack*8;
  const unsigned char* s=code+(long long)row*1280LL+(long long)pack*5LL;
  unsigned long long w=(unsigned long long)s[0]|(unsigned long long)s[1]<<8|(unsigned long long)s[2]<<16|(unsigned long long)s[3]<<24|(unsigned long long)s[4]<<32;
  float a=0.0f,sc=b2f(scale[row*16+(col>>7)]);
  #pragma unroll
  for(int k=0;k<8;k++){
   int q=(int)((w>>(5*k))&31ULL)-15;
   a=fmaf(rbff((float)q*sc),b2f(x[col+k]),a);
  }
  p[v]=a;
 }
 #pragma unroll
 for(int d=16;d>=1;d>>=1){
  #pragma unroll
  for(int i=0;i<d;i++)p[i]=__fadd_rn(p[i],p[i+d]);
 }
 float z=p[0];
 #pragma unroll
 for(int d=4;d>=1;d>>=1){float o=tile.shfl_down(z,d);if(lane<d)z=__fadd_rn(z,o);}
 if(lane==0){out[row]=rbf(z);atomicAdd(&count[row],1U);}
}
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_nvidia(record: bytes, input_bytes: bytes) -> dict[str, object]:
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    pci = cp.cuda.runtime.deviceGetPCIBusId(0)
    if isinstance(pci, bytes):
        pci = pci.decode()
    if cp.cuda.runtime.getDeviceCount() != 1 or name != "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU" or pci != "0000:01:00.0":
        raise RuntimeError(f"nvidia_identity:{name}:{pci}")
    module = cp.RawModule(
        code=CUDA_SOURCE,
        backend="nvrtc",
        options=("--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"),
        name_expressions=("ph0",),
    )
    kernel = module.get_function("ph0")
    stream = cp.cuda.Stream(non_blocking=True)
    pinned: list[tuple[str, object, int]] = []
    device: list[tuple[str, object, int]] = []
    ledger: list[dict[str, object]] = []
    errors: list[str] = []
    try:
        for label, size in (("record", common.RECORD_BYTES), ("input", common.INPUT_BYTES), ("output", common.ROWS * 2), ("counters", common.COUNTER_BYTES)):
            host = cp.cuda.alloc_pinned_memory(size)
            dev = cp.cuda.alloc(size)
            pinned.append((label, host, size))
            device.append((label, dev, size))
            ledger.append({"kind": "allocation", "name": label, "bytes": size, "host_pointer": int(host.ptr), "device_pointer": int(dev.ptr)})
        memoryview(pinned[0][1])[:] = record
        memoryview(pinned[1][1])[:] = input_bytes
        stream.memset_async(device[2][1].ptr, 0xFF, common.ROWS * 2)
        stream.memset_async(device[3][1].ptr, 0, common.COUNTER_BYTES)
        ledger.extend(({"op": "memset", "target": "output", "bytes": common.ROWS * 2}, {"op": "memset", "target": "counters", "bytes": common.COUNTER_BYTES}))
        cp.cuda.runtime.memcpyAsync(device[0][1].ptr, pinned[0][1].ptr, common.RECORD_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
        cp.cuda.runtime.memcpyAsync(device[1][1].ptr, pinned[1][1].ptr, common.INPUT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
        ledger.extend(({"op": "H2D", "target": "record", "bytes": common.RECORD_BYTES}, {"op": "H2D", "target": "input", "bytes": common.INPUT_BYTES}))
        kernel((16,), (256,), (device[0][1], device[1][1], device[2][1], device[3][1]), stream=stream)
        ledger.append({"op": "kernel", "grid": [16], "block": [256]})
        cp.cuda.runtime.memcpyAsync(pinned[2][1].ptr, device[2][1].ptr, common.ROWS * 2, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        cp.cuda.runtime.memcpyAsync(pinned[3][1].ptr, device[3][1].ptr, common.COUNTER_BYTES, cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
        ledger.extend(({"op": "D2H", "target": "output", "bytes": common.ROWS * 2}, {"op": "D2H", "target": "counters", "bytes": common.COUNTER_BYTES}))
        stream.synchronize()
        ledger.append({"op": "synchronize", "code": 0})
        output = bytes(memoryview(pinned[2][1]))
        counters = bytes(memoryview(pinned[3][1]))
        return {
            "identity": {"name": name, "pci": pci, "driver_version": cp.cuda.runtime.driverGetVersion(), "runtime_version": cp.cuda.runtime.runtimeGetVersion()},
            "output_hex": output.hex(),
            "counters_hex": counters.hex(),
            "source_sha256": sha(CUDA_SOURCE.encode()),
            "ledger": ledger,
        }
    finally:
        try:
            stream.synchronize()
        except Exception as exc:
            errors.append(str(exc))
        for label, dev, _ in reversed(device):
            try:
                dev.mem.free()
                ledger.append({"release": f"device_{label}", "code": 0})
            except Exception as exc:
                errors.append(str(exc))
        for label, host, _ in reversed(pinned):
            try:
                host.free()
                ledger.append({"release": f"pinned_{label}", "code": 0})
            except Exception as exc:
                errors.append(str(exc))
        ledger.append({"cleanup_complete": not errors, "errors": errors})


def compare(result: dict[str, object], oracle: np.ndarray) -> dict[str, object]:
    output = np.frombuffer(bytes.fromhex(str(result["output_hex"])), "<u2")
    counters = np.frombuffer(bytes.fromhex(str(result["counters_hex"])), "<u4")
    return {
        "words": int(output.size),
        "different_words": int(np.count_nonzero(output != oracle)) if output.size == oracle.size else -1,
        "output_sha256": sha(output.tobytes()),
        "counters_all_one": bool(counters.size == common.ROWS and np.all(counters == 1)),
        "sentinel_overwritten": bool(output.size == common.ROWS and np.all(output != 0xFFFF)),
    }


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    RUN.mkdir(parents=True)
    result: dict[str, object] = {
        "kind": "het_next_l0_ph0x_exploratory_real_projection",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {"prereg_sha256": common.file_digest(PREREG), "runner_sha256": common.file_digest(Path(__file__))},
    }
    error = None
    try:
        source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
        input_bytes = common.read_exact(common.D2, common.INPUT_OFFSET, common.INPUT_BYTES)
        record, evidence = common.build_record(source)
        safe = common.safe_check(record, input_bytes)
        controls = common.controls(record, input_bytes)
        oracle = common.cpu_oracle(record, input_bytes)
        result.update({"record_evidence": evidence, "safe_trace": safe["trace"], "controls": controls, "cpu_output_hex": oracle.tobytes().hex(), "cpu_output_sha256": sha(oracle.tobytes())})
        intel_base.SRC = INTEL_SOURCE
        intel = intel_base.run(record, input_bytes)
        intel_cmp = compare(intel, oracle)
        nvidia = run_nvidia(record, input_bytes)
        nvidia_cmp = compare(nvidia, oracle)
        result.update({"intel": intel, "intel_comparison": intel_cmp, "nvidia": nvidia, "nvidia_comparison": nvidia_cmp})
        gates = {
            "controls_all_pass": len(controls) == 8 and all(bool(row.get("pass")) for row in controls),
            "intel_exact": intel_cmp["different_words"] == 0,
            "nvidia_exact": nvidia_cmp["different_words"] == 0,
            "intel_counters": intel_cmp["counters_all_one"],
            "nvidia_counters": nvidia_cmp["counters_all_one"],
            "intel_sentinel": intel_cmp["sentinel_overwritten"],
            "nvidia_sentinel": nvidia_cmp["sentinel_overwritten"],
            "intel_cleanup": bool(intel["ledger"][-1]["cleanup_complete"]),
            "nvidia_cleanup": bool(nvidia["ledger"][-1]["cleanup_complete"]),
            "distinct_pci": intel["identity"]["pci"] != nvidia["identity"]["pci"],
        }
        result["gates"] = gates
        result["positive"] = all(gates.values())
        result["status"] = "exploratory_single_real_projection_positive" if result["positive"] else "exploratory_single_real_projection_negative"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update({"positive": False, "status": "exploratory_single_real_projection_failure", "error": error, "traceback": traceback.format_exc()})
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["claim_boundary"] = "One real projection/input, exploratory only; no full expert/layer/model/performance/concurrency/deployment/novelty/breakthrough claim."
    common.write_atomic_new(RESULT, common.canonical(result))
    print(json.dumps({"status": result["status"], "positive": result["positive"], "gates": result.get("gates"), "error": error}, indent=2))
    return 0 if error is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
