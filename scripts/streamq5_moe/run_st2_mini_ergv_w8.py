from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import traceback

from scripts.streamq5_moe import run_st2_mini_host_usm_q5 as base


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "reports/streamq5_moe/ST2_MINI_ERVG_W8_CONFIRMATION_PREREGISTRATION_2026-08-12.md"
OUTPUT = ROOT / "reports/streamq5_moe/st2_mini_ergv_w8_result.json"


KERNEL_SOURCE = r"""
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define RECORD_BYTES 1011712UL
#define HEADER_BYTES 64UL
#define CODE_BYTES 983040UL
#define WIDTH 8
#define VIRTUAL 32
#define ROWS_PER_BLOCK 32

inline float bf16_to_float(ushort value) {
    return as_float(((uint)value) << 16);
}

inline float round_bf16(float value) {
    uint bits = as_uint(value);
    uint lsb = (bits >> 16) & 1U;
    bits += 0x7fffU + lsb;
    return as_float(bits & 0xffff0000U);
}

inline ushort round_bf16_bits(float value) {
    uint bits = as_uint(value);
    uint lsb = (bits >> 16) & 1U;
    bits += 0x7fffU + lsb;
    return (ushort)(bits >> 16);
}

__kernel
__attribute__((reqd_work_group_size(256, 1, 1)))
__attribute__((intel_reqd_sub_group_size(8)))
void q5_ergv_host_usm(
    __global const uchar* bank,
    __global const int* record_indices,
    __global const float* x,
    __global ushort* output,
    int rows,
    int cols,
    int batch_records)
{
    const int groups_per_record = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const int linear_group = (int)get_group_id(0);
    const int item = linear_group / groups_per_record;
    const int block = linear_group - item * groups_per_record;
    const int subrow = (int)get_sub_group_id();
    const int lane = (int)get_sub_group_local_id();
    const int row = block * ROWS_PER_BLOCK + subrow;
    if (item >= batch_records || row >= rows)
        return;

    const ulong base = ((ulong)record_indices[item]) * RECORD_BYTES;
    __global const uchar* packed = bank + base + HEADER_BYTES;
    __global const ushort* scales = (__global const ushort*)(bank + base + HEADER_BYTES + CODE_BYTES);
    const int packs = cols >> 3;
    const int scale_groups = cols >> 7;
    const int row_code_bytes = packs * 5;
    float partial[VIRTUAL];

    #pragma unroll
    for (int virtual_index = 0; virtual_index < VIRTUAL; ++virtual_index) {
        const int tid = lane + WIDTH * virtual_index;
        float sum = 0.0f;
        if (tid < packs) {
            __global const uchar* source = packed + ((ulong)row * (ulong)row_code_bytes) + ((ulong)tid * 5UL);
            ulong word = ((ulong)source[0])
                | ((ulong)source[1] << 8)
                | ((ulong)source[2] << 16)
                | ((ulong)source[3] << 24)
                | ((ulong)source[4] << 32);
            const int column = tid << 3;
            const float scale = bf16_to_float(scales[row * scale_groups + (column >> 7)]);
            #pragma unroll
            for (int part = 0; part < 8; ++part) {
                const int code = (int)((word >> (part * 5)) & 31UL) - 15;
                const float weight = round_bf16(((float)code) * scale);
                sum = fma(weight, x[column + part], sum);
            }
        }
        partial[virtual_index] = sum;
    }

    #pragma unroll
    for (int stride = 128; stride >= WIDTH; stride >>= 1) {
        #pragma unroll
        for (int index = 0; index < stride / WIDTH; ++index)
            partial[index] = partial[index] + partial[index + stride / WIDTH];
    }
    float value = partial[0];
    #pragma unroll
    for (int offset = WIDTH / 2; offset > 0; offset >>= 1) {
        float other = intel_sub_group_shuffle_down(value, value, (uint)offset);
        if (lane < offset)
            value = value + other;
    }
    if (lane == 0)
        output[item * rows + row] = round_bf16_bits(value);
}
"""


class Width8OpenCL(base.OpenCL):
    def launch(self, record_count: int, rows: int, cols: int) -> tuple[float, float]:
        self.set_int_arg(4, rows)
        self.set_int_arg(5, cols)
        self.set_int_arg(6, record_count)
        rows_per_block = 32
        groups = record_count * math.ceil(rows / rows_per_block)
        global_size = (C.c_size_t * 1)(groups * base.LOCAL_SIZE)
        local_size = (C.c_size_t * 1)(base.LOCAL_SIZE)
        event = C.c_void_p()
        wall_start = base.time.perf_counter_ns()
        base.check(self.lib.clEnqueueNDRangeKernel(self.queue, self.kernel, 1, None, global_size, local_size, 0, None, C.byref(event)), "clEnqueueNDRangeKernel")
        base.check(self.lib.clFinish(self.queue), "clFinish")
        wall_ms = (base.time.perf_counter_ns() - wall_start) / 1e6
        started = C.c_ulonglong()
        ended = C.c_ulonglong()
        base.check(self.lib.clGetEventProfilingInfo(event, base.CL_PROFILING_COMMAND_START, C.sizeof(started), C.byref(started), None), "event start")
        base.check(self.lib.clGetEventProfilingInfo(event, base.CL_PROFILING_COMMAND_END, C.sizeof(ended), C.byref(ended), None), "event end")
        base.check(self.lib.clReleaseEvent(event), "clReleaseEvent")
        return (ended.value - started.value) / 1e6, wall_ms


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    base.PREREG = PREREG
    base.OUTPUT = OUTPUT
    base.KERNEL_SOURCE = KERNEL_SOURCE
    base.OpenCL = Width8OpenCL
    payload = {
        "kind": "streamq5_moe_st2_mini_ergv_width8_confirmation",
        "started_utc": base.utc_now(),
        "preregistration_sha256": sha256(PREREG),
        "primary_st2_result_sha256": sha256(ROOT / "reports/streamq5_moe/st2_mini_host_usm_q5_result.json"),
        "p1d_manifest_sha256": sha256(base.P1D),
        "runner_sha256": sha256(Path(__file__)),
        "imported_runner_sha256": sha256(ROOT / "scripts/streamq5_moe/run_st2_mini_host_usm_q5.py"),
        "kernel_source_sha256": hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest(),
        "execution_lock": {
            "width": 8,
            "rows_per_workgroup": 32,
            "ring_records": base.RING_RECORDS,
            "ring_bytes": base.RING_RECORDS * base.RECORD_BYTES,
            "batch_records": base.BATCH_RECORDS,
            "warmup_batches": base.WARMUP_BATCHES,
            "timed_iterations": base.TIMED_ITERATIONS,
            "minimum_p95_side_gbps": base.MIN_P95_SIDE_GBPS,
        },
        "nvidia_gpu_kernel_or_transfer_calls": 0,
    }
    try:
        payload.update(base.execute())
    except Exception as exc:
        payload.update({"status": "blocked_or_runtime_failure", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    payload["completed_utc"] = base.utc_now()
    payload["claim_boundary"] = "Intel width-8 ERGV host-USM Q5 component only; no dGPU or SplitTree-layer claim."
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "performance"}, indent=2))
    if "performance" in payload:
        print(json.dumps({"performance": {key: value for key, value in payload["performance"].items() if not key.startswith("raw_")}}, indent=2))


if __name__ == "__main__":
    main()

