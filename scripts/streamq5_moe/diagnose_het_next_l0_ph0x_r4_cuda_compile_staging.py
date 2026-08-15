from __future__ import annotations

import ctypes
import hashlib
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import het_next_l0_ph0r3_common as common


OUT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r4_cuda_compile_staging_diagnostic.json"
R3_RESULT = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r3_exploratory_real_projection/ph0x_r3_result.json"
R3_RESULT_SHA = "e5fea8e2609f11dd294733645c9a4ecb08892c9d2070de33baacbd1a74b0df7c"


CUDA_SOURCE = r'''
__device__ __forceinline__ float b2f(unsigned short x){return __uint_as_float(((unsigned)x)<<16);}
__device__ __forceinline__ unsigned short rbf(float x){unsigned b=__float_as_uint(x),l=(b>>16)&1U;b+=0x7fffU+l;return (unsigned short)(b>>16);}
__device__ __forceinline__ float rbff(float x){return b2f(rbf(x));}
extern "C" __global__ void ph0(const unsigned char* rec,const unsigned short* x,unsigned short* out,unsigned* count){
 int lane=(int)threadIdx.x&7,subgroup=((int)threadIdx.x>>3)&31,row=(int)blockIdx.x*32+subgroup;
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
 unsigned mask=__activemask();
 #pragma unroll
 for(int d=4;d>=1;d>>=1){float o=__shfl_down_sync(mask,z,d,8);if(lane<d)z=__fadd_rn(z,o);}
 if(lane==0){out[row]=rbf(z);atomicAdd(&count[row],1U);}
}
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    result: dict[str, object] = {
        "kind": "ph0x_r4_cuda_compile_staging_diagnostic",
        "r3_result_sha256": common.file_digest(R3_RESULT),
        "expected_r3_result_sha256": R3_RESULT_SHA,
        "cuda_source_sha256": sha(CUDA_SOURCE.encode()),
        "kernel_launched": False,
        "intel_opened": False,
        "h2d_calls": 0,
        "d2h_calls": 0,
    }
    pinned = None
    error = None
    try:
        if result["r3_result_sha256"] != R3_RESULT_SHA:
            raise RuntimeError("r3_result_hash_drift")
        source = common.read_exact(common.SHARD, common.SOURCE_OFFSET, common.SOURCE_BYTES)
        record, evidence = common.build_record(source)
        if evidence["record_sha256"] != "e3b10ab3fe1381a78065ff8231510c831693da549d697ac66945a92def25e1a9":
            raise RuntimeError("record_hash_drift")
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        pci = cp.cuda.runtime.deviceGetPCIBusId(0)
        if isinstance(pci, bytes):
            pci = pci.decode()
        result["identity"] = {
            "name": name,
            "pci": pci,
            "driver": cp.cuda.runtime.driverGetVersion(),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
        }
        pinned = cp.cuda.alloc_pinned_memory(common.RECORD_BYTES)
        ctypes.memmove(int(pinned.ptr), record, len(record))
        staged = ctypes.string_at(int(pinned.ptr), common.RECORD_BYTES)
        result["staging"] = {
            "bytes": len(staged),
            "sha256": sha(staged),
            "matches_record": staged == record,
        }
        module = cp.RawModule(
            code=CUDA_SOURCE,
            backend="nvrtc",
            options=("--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"),
            name_expressions=("ph0",),
        )
        function = module.get_function("ph0")
        result["compile"] = {"success": function is not None}
        result["diagnostic_pass"] = bool(result["staging"]["matches_record"] and result["compile"]["success"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result.update({"diagnostic_pass": False, "error": error, "traceback": traceback.format_exc()})
    finally:
        cleanup_errors: list[str] = []
        if pinned is not None:
            try:
                pinned.mem.free()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        result["cleanup"] = {"pinned_release_attempted": pinned is not None, "errors": cleanup_errors}
    common.write_atomic_new(OUT, common.canonical(result))
    print(json.dumps({"diagnostic_pass": result["diagnostic_pass"], "error": error, "cleanup": result["cleanup"]}, indent=2))
    return 0 if result["diagnostic_pass"] and not result["cleanup"]["errors"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
