from __future__ import annotations

import ctypes as C
from ctypes import wintypes
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import struct
import threading
import time
import traceback
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
RUNS = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
PREREG = REPORTS / "ST2_MINI_PREREGISTRATION_2026-08-12.md"
CAPABILITY = REPORTS / "st2_mini_opencl_capability_probe.json"
P1D = REPORTS / "p1d_physical_bank_result.json"
OUTPUT = REPORTS / "st2_mini_host_usm_q5_result.json"

RECORD_BYTES = 1_011_712
HEADER_BYTES = 64
CODE_BYTES = 983_040
SCALE_BYTES = 24_576
RING_RECORDS = 531
EXTRA_DOWN_RECORDS = 1
ALLOC_RECORDS = RING_RECORDS + EXTRA_DOWN_RECORDS
ALLOC_BYTES = ALLOC_RECORDS * RECORD_BYTES
EFFECTIVE_BYTES_PER_RECORD = CODE_BYTES + SCALE_BYTES
BATCH_RECORDS = 16
WARMUP_BATCHES = math.ceil(RING_RECORDS / BATCH_RECORDS)
TIMED_ITERATIONS = 1_000
MIN_P95_SIDE_GBPS = 21.63
LOCAL_SIZE = 256
HEADER = struct.Struct("<4sHHHBBIIH2xIII28s")

CL_SUCCESS = 0
CL_TRUE = 1
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_DEVICE_TYPE = 0x1000
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DEVICE_EXTENSIONS = 0x1030
CL_CONTEXT_PLATFORM = 0x1084
CL_QUEUE_PROFILING_ENABLE = 1 << 1
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_READ_ONLY = 1 << 2
CL_PROGRAM_BUILD_LOG = 0x1183
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283
CL_MEM_ALLOC_TYPE_INTEL = 0x419A
CL_MEM_ALLOC_BASE_PTR_INTEL = 0x419B
CL_MEM_ALLOC_SIZE_INTEL = 0x419C
CL_MEM_ALLOC_DEVICE_INTEL = 0x419D
CL_MEM_TYPE_HOST_INTEL = 0x4197


KERNEL_SOURCE = r"""
#pragma OPENCL FP_CONTRACT ON

#define RECORD_BYTES 1011712UL
#define HEADER_BYTES 64UL
#define CODE_BYTES 983040UL

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

__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void q5_ergv_host_usm(
    __global const uchar* bank,
    __global const int* record_indices,
    __global const float* x,
    __global ushort* output,
    int rows,
    int cols,
    int batch_records)
{
    const int group = (int)get_group_id(0);
    const int item = group / rows;
    const int row = group - item * rows;
    const int tid = (int)get_local_id(0);
    if (item >= batch_records)
        return;

    const ulong base = ((ulong)record_indices[item]) * RECORD_BYTES;
    __global const uchar* packed = bank + base + HEADER_BYTES;
    __global const ushort* scales = (__global const ushort*)(bank + base + HEADER_BYTES + CODE_BYTES);
    const int packs = cols >> 3;
    const int groups = cols >> 7;
    const int row_code_bytes = packs * 5;
    float sum = 0.0f;
    if (tid < packs) {
        __global const uchar* source = packed + ((ulong)row * (ulong)row_code_bytes) + ((ulong)tid * 5UL);
        ulong word = ((ulong)source[0])
            | ((ulong)source[1] << 8)
            | ((ulong)source[2] << 16)
            | ((ulong)source[3] << 24)
            | ((ulong)source[4] << 32);
        const int column = tid << 3;
        const float scale = bf16_to_float(scales[row * groups + (column >> 7)]);
        #pragma unroll
        for (int part = 0; part < 8; ++part) {
            const int code = (int)((word >> (part * 5)) & 31UL) - 15;
            const float weight = round_bf16(((float)code) * scale);
            sum = fma(weight, x[column + part], sum);
        }
    }

    volatile __local float partial[256];
    partial[tid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 128; stride >= 1; stride >>= 1) {
        if (tid < stride)
            partial[tid] = partial[tid] + partial[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
        output[item * rows + row] = round_bf16_bits(partial[0]);
}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def check(code: int, operation: str) -> None:
    if code != CL_SUCCESS:
        raise RuntimeError(f"{operation} failed with OpenCL error {code}")


class _PDHValueUnion(C.Union):
    _fields_ = [("long_value", wintypes.LONG), ("double_value", C.c_double), ("large_value", C.c_longlong)]


class _PDHValue(C.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("status", wintypes.DWORD), ("value", _PDHValueUnion)]


class HardPageReadSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, float | str]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="st2-mini-pdh", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        query = C.c_void_p()
        page_reads = C.c_void_p()
        pages_input = C.c_void_p()
        try:
            pdh = C.WinDLL("pdh", use_last_error=True)
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, C.c_void_p, C.POINTER(C.c_void_p)]
            pdh.PdhAddEnglishCounterW.argtypes = [C.c_void_p, wintypes.LPCWSTR, C.c_void_p, C.POINTER(C.c_void_p)]
            pdh.PdhCollectQueryData.argtypes = [C.c_void_p]
            pdh.PdhGetFormattedCounterValue.argtypes = [C.c_void_p, wintypes.DWORD, C.c_void_p, C.POINTER(_PDHValue)]
            pdh.PdhCloseQuery.argtypes = [C.c_void_p]
            if pdh.PdhOpenQueryW(None, None, C.byref(query)):
                raise RuntimeError("PdhOpenQueryW failed")
            for counter, name in ((page_reads, r"\Memory\Page Reads/sec"), (pages_input, r"\Memory\Pages Input/sec")):
                if pdh.PdhAddEnglishCounterW(query, name, None, C.byref(counter)):
                    raise RuntimeError(f"PdhAddEnglishCounterW failed for {name}")
            if pdh.PdhCollectQueryData(query):
                raise RuntimeError("initial PdhCollectQueryData failed")
            while not self._stop.wait(1.0):
                if pdh.PdhCollectQueryData(query):
                    raise RuntimeError("PdhCollectQueryData failed")
                row: dict[str, float | str] = {"utc": utc_now(), "monotonic_seconds": time.perf_counter()}
                for counter, key in ((page_reads, "page_reads_per_sec"), (pages_input, "pages_input_per_sec")):
                    value = _PDHValue()
                    code = pdh.PdhGetFormattedCounterValue(counter, 0x00000200, None, C.byref(value))
                    if code or value.status:
                        raise RuntimeError(f"PdhGetFormattedCounterValue failed for {key}")
                    row[key] = float(value.double_value)
                self.samples.append(row)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if query:
                try:
                    C.WinDLL("pdh").PdhCloseQuery(query)
                except Exception:
                    pass


class OpenCL:
    def __init__(self) -> None:
        self.lib = C.WinDLL("OpenCL.dll")
        self._bind()
        self.platform, self.device, self.identity = self._select_intel_arc()
        self.context = None
        self.queue = None
        self.program = None
        self.kernel = None
        self.buffers: list[int] = []
        self.usm_ptr: int | None = None
        self._load_usm_functions()

    def _bind(self) -> None:
        l = self.lib
        l.clGetPlatformIDs.argtypes = [C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)]
        l.clGetPlatformIDs.restype = C.c_int
        l.clGetPlatformInfo.argtypes = [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
        l.clGetPlatformInfo.restype = C.c_int
        l.clGetDeviceIDs.argtypes = [C.c_void_p, C.c_ulonglong, C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)]
        l.clGetDeviceIDs.restype = C.c_int
        l.clGetDeviceInfo.argtypes = [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
        l.clGetDeviceInfo.restype = C.c_int
        l.clGetExtensionFunctionAddressForPlatform.argtypes = [C.c_void_p, C.c_char_p]
        l.clGetExtensionFunctionAddressForPlatform.restype = C.c_void_p
        l.clCreateContext.argtypes = [C.POINTER(C.c_ssize_t), C.c_uint, C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]
        l.clCreateContext.restype = C.c_void_p
        l.clCreateCommandQueue.argtypes = [C.c_void_p, C.c_void_p, C.c_ulonglong, C.POINTER(C.c_int)]
        l.clCreateCommandQueue.restype = C.c_void_p
        l.clCreateProgramWithSource.argtypes = [C.c_void_p, C.c_uint, C.POINTER(C.c_char_p), C.POINTER(C.c_size_t), C.POINTER(C.c_int)]
        l.clCreateProgramWithSource.restype = C.c_void_p
        l.clBuildProgram.argtypes = [C.c_void_p, C.c_uint, C.POINTER(C.c_void_p), C.c_char_p, C.c_void_p, C.c_void_p]
        l.clBuildProgram.restype = C.c_int
        l.clGetProgramBuildInfo.argtypes = [C.c_void_p, C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
        l.clGetProgramBuildInfo.restype = C.c_int
        l.clCreateKernel.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int)]
        l.clCreateKernel.restype = C.c_void_p
        l.clCreateBuffer.argtypes = [C.c_void_p, C.c_ulonglong, C.c_size_t, C.c_void_p, C.POINTER(C.c_int)]
        l.clCreateBuffer.restype = C.c_void_p
        l.clSetKernelArg.argtypes = [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p]
        l.clSetKernelArg.restype = C.c_int
        l.clEnqueueWriteBuffer.argtypes = [C.c_void_p, C.c_void_p, C.c_uint, C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p]
        l.clEnqueueWriteBuffer.restype = C.c_int
        l.clEnqueueReadBuffer.argtypes = [C.c_void_p, C.c_void_p, C.c_uint, C.c_size_t, C.c_size_t, C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p]
        l.clEnqueueReadBuffer.restype = C.c_int
        l.clEnqueueNDRangeKernel.argtypes = [C.c_void_p, C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), C.c_uint, C.c_void_p, C.POINTER(C.c_void_p)]
        l.clEnqueueNDRangeKernel.restype = C.c_int
        l.clFinish.argtypes = [C.c_void_p]
        l.clFinish.restype = C.c_int
        l.clGetEventProfilingInfo.argtypes = [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
        l.clGetEventProfilingInfo.restype = C.c_int
        l.clReleaseEvent.argtypes = [C.c_void_p]
        l.clReleaseEvent.restype = C.c_int
        l.clReleaseMemObject.argtypes = [C.c_void_p]
        l.clReleaseMemObject.restype = C.c_int
        l.clReleaseKernel.argtypes = [C.c_void_p]
        l.clReleaseProgram.argtypes = [C.c_void_p]
        l.clReleaseCommandQueue.argtypes = [C.c_void_p]
        l.clReleaseContext.argtypes = [C.c_void_p]

    def _string_info(self, function, handle, param: int) -> str:
        size = C.c_size_t()
        check(function(handle, param, 0, None, C.byref(size)), f"info size 0x{param:x}")
        buffer = C.create_string_buffer(size.value)
        check(function(handle, param, size.value, buffer, None), f"info value 0x{param:x}")
        return buffer.value.decode("utf-8", errors="replace")

    def _select_intel_arc(self):
        count = C.c_uint()
        check(self.lib.clGetPlatformIDs(0, None, C.byref(count)), "clGetPlatformIDs count")
        platforms = (C.c_void_p * count.value)()
        check(self.lib.clGetPlatformIDs(count.value, platforms, None), "clGetPlatformIDs")
        candidates = []
        for p_value in platforms:
            p = C.c_void_p(p_value)
            ndev = C.c_uint()
            if self.lib.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, None, C.byref(ndev)) != CL_SUCCESS:
                continue
            devices = (C.c_void_p * ndev.value)()
            check(self.lib.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndev.value, devices, None), "clGetDeviceIDs")
            for d_value in devices:
                d = C.c_void_p(d_value)
                name = self._string_info(self.lib.clGetDeviceInfo, d, CL_DEVICE_NAME)
                vendor = self._string_info(self.lib.clGetDeviceInfo, d, CL_DEVICE_VENDOR)
                extensions = self._string_info(self.lib.clGetDeviceInfo, d, CL_DEVICE_EXTENSIONS).split()
                dtype = C.c_ulonglong()
                check(self.lib.clGetDeviceInfo(d, CL_DEVICE_TYPE, C.sizeof(dtype), C.byref(dtype), None), "device type")
                if "intel" in vendor.lower() and "arc" in name.lower() and "cl_intel_unified_shared_memory" in extensions:
                    candidates.append((p, d, {"name": name, "vendor": vendor, "type": dtype.value}))
        if len(candidates) != 1:
            raise RuntimeError(f"expected exactly one Intel Arc USM device, found {len(candidates)}")
        return candidates[0]

    def _load_usm_functions(self) -> None:
        def address(name: str) -> int:
            value = self.lib.clGetExtensionFunctionAddressForPlatform(self.platform, name.encode("ascii"))
            if not value:
                raise RuntimeError(f"missing {name}")
            return int(value)

        call = C.WINFUNCTYPE
        self.host_alloc = call(C.c_void_p, C.c_void_p, C.POINTER(C.c_ssize_t), C.c_size_t, C.c_uint, C.POINTER(C.c_int))(address("clHostMemAllocINTEL"))
        self.mem_free = call(C.c_int, C.c_void_p, C.c_void_p)(address("clMemFreeINTEL"))
        self.set_arg_pointer = call(C.c_int, C.c_void_p, C.c_uint, C.c_void_p)(address("clSetKernelArgMemPointerINTEL"))
        self.get_alloc_info = call(C.c_int, C.c_void_p, C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t))(address("clGetMemAllocInfoINTEL"))

    def setup(self) -> dict:
        err = C.c_int()
        props = (C.c_ssize_t * 3)(CL_CONTEXT_PLATFORM, int(self.platform.value), 0)
        devices = (C.c_void_p * 1)(self.device.value)
        self.context = self.lib.clCreateContext(props, 1, devices, None, None, C.byref(err))
        check(err.value, "clCreateContext")
        self.queue = self.lib.clCreateCommandQueue(self.context, self.device, CL_QUEUE_PROFILING_ENABLE, C.byref(err))
        check(err.value, "clCreateCommandQueue")
        source = KERNEL_SOURCE.encode("utf-8")
        strings = (C.c_char_p * 1)(source)
        lengths = (C.c_size_t * 1)(len(source))
        self.program = self.lib.clCreateProgramWithSource(self.context, 1, strings, lengths, C.byref(err))
        check(err.value, "clCreateProgramWithSource")
        code = self.lib.clBuildProgram(self.program, 1, devices, b"-cl-std=CL3.0", None, None)
        if code != CL_SUCCESS:
            size = C.c_size_t()
            self.lib.clGetProgramBuildInfo(self.program, self.device, CL_PROGRAM_BUILD_LOG, 0, None, C.byref(size))
            log = C.create_string_buffer(size.value)
            self.lib.clGetProgramBuildInfo(self.program, self.device, CL_PROGRAM_BUILD_LOG, size.value, log, None)
            raise RuntimeError(f"clBuildProgram failed {code}: {log.value.decode(errors='replace')}")
        self.kernel = self.lib.clCreateKernel(self.program, b"q5_ergv_host_usm", C.byref(err))
        check(err.value, "clCreateKernel")
        self.usm_ptr = int(self.host_alloc(self.context, None, ALLOC_BYTES, 4096, C.byref(err)))
        check(err.value, "clHostMemAllocINTEL")
        if not self.usm_ptr:
            raise MemoryError("clHostMemAllocINTEL returned null")
        alloc_type = C.c_uint()
        alloc_size = C.c_size_t()
        base = C.c_void_p()
        check(self.get_alloc_info(self.context, C.c_void_p(self.usm_ptr), CL_MEM_ALLOC_TYPE_INTEL, C.sizeof(alloc_type), C.byref(alloc_type), None), "USM alloc type")
        check(self.get_alloc_info(self.context, C.c_void_p(self.usm_ptr), CL_MEM_ALLOC_SIZE_INTEL, C.sizeof(alloc_size), C.byref(alloc_size), None), "USM alloc size")
        check(self.get_alloc_info(self.context, C.c_void_p(self.usm_ptr), CL_MEM_ALLOC_BASE_PTR_INTEL, C.sizeof(base), C.byref(base), None), "USM base")
        return {"type": alloc_type.value, "type_is_host": alloc_type.value == CL_MEM_TYPE_HOST_INTEL, "size": alloc_size.value, "base_pointer_matches": int(base.value) == self.usm_ptr, "alignment": 4096}

    def buffer(self, flags: int, size: int) -> int:
        err = C.c_int()
        value = self.lib.clCreateBuffer(self.context, flags, size, None, C.byref(err))
        check(err.value, "clCreateBuffer")
        result = int(value)
        self.buffers.append(result)
        return result

    def set_mem_arg(self, index: int, memory: int) -> None:
        value = C.c_void_p(memory)
        check(self.lib.clSetKernelArg(self.kernel, index, C.sizeof(value), C.byref(value)), f"set mem arg {index}")

    def set_int_arg(self, index: int, value: int) -> None:
        item = C.c_int(value)
        check(self.lib.clSetKernelArg(self.kernel, index, C.sizeof(item), C.byref(item)), f"set int arg {index}")

    def write(self, memory: int, array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array)
        check(self.lib.clEnqueueWriteBuffer(self.queue, C.c_void_p(memory), CL_TRUE, 0, contiguous.nbytes, C.c_void_p(contiguous.ctypes.data), 0, None, None), "clEnqueueWriteBuffer")

    def read(self, memory: int, array: np.ndarray) -> None:
        check(self.lib.clEnqueueReadBuffer(self.queue, C.c_void_p(memory), CL_TRUE, 0, array.nbytes, C.c_void_p(array.ctypes.data), 0, None, None), "clEnqueueReadBuffer")

    def launch(self, record_count: int, rows: int, cols: int) -> tuple[float, float]:
        self.set_int_arg(4, rows)
        self.set_int_arg(5, cols)
        self.set_int_arg(6, record_count)
        global_size = (C.c_size_t * 1)(record_count * rows * LOCAL_SIZE)
        local_size = (C.c_size_t * 1)(LOCAL_SIZE)
        event = C.c_void_p()
        wall_start = time.perf_counter_ns()
        check(self.lib.clEnqueueNDRangeKernel(self.queue, self.kernel, 1, None, global_size, local_size, 0, None, C.byref(event)), "clEnqueueNDRangeKernel")
        check(self.lib.clFinish(self.queue), "clFinish")
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        started = C.c_ulonglong()
        ended = C.c_ulonglong()
        check(self.lib.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, C.sizeof(started), C.byref(started), None), "event start")
        check(self.lib.clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, C.sizeof(ended), C.byref(ended), None), "event end")
        check(self.lib.clReleaseEvent(event), "clReleaseEvent")
        return (ended.value - started.value) / 1e6, wall_ms

    def bind(self, indices: int, x: int, output: int) -> None:
        check(self.set_arg_pointer(self.kernel, 0, C.c_void_p(self.usm_ptr)), "clSetKernelArgMemPointerINTEL")
        self.set_mem_arg(1, indices)
        self.set_mem_arg(2, x)
        self.set_mem_arg(3, output)

    def close(self) -> None:
        if self.queue:
            try:
                self.lib.clFinish(self.queue)
            except Exception:
                pass
        for buffer in reversed(self.buffers):
            try:
                self.lib.clReleaseMemObject(C.c_void_p(buffer))
            except Exception:
                pass
        if self.usm_ptr and self.context:
            try:
                self.mem_free(self.context, C.c_void_p(self.usm_ptr))
            except Exception:
                pass
        for handle, release in ((self.kernel, self.lib.clReleaseKernel), (self.program, self.lib.clReleaseProgram), (self.queue, self.lib.clReleaseCommandQueue), (self.context, self.lib.clReleaseContext)):
            if handle:
                try:
                    release(handle)
                except Exception:
                    pass


def parse_header(data: bytes) -> dict:
    values = HEADER.unpack_from(data, 0)
    return {
        "magic": values[0].decode("ascii"),
        "version": values[1],
        "layer": values[2],
        "expert": values[3],
        "projection": values[4],
        "bits": values[5],
        "rows": values[6],
        "cols": values[7],
        "group": values[8],
        "code_bytes": values[9],
        "scale_bytes": values[10],
        "crc32": values[11],
    }


def load_real_records(destination: int) -> tuple[dict, dict[str, bytes]]:
    manifest = json.loads(P1D.read_text(encoding="utf-8"))
    hasher = hashlib.sha256()
    descriptors = []
    correctness: dict[str, bytes] = {}
    slot = 0
    for layer in range(3):
        path = RUNS / f"layer_{layer:02d}.q5bin"
        expected = manifest["manifests"][str(layer)]["artifact_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"layer {layer} SHA mismatch")
        with path.open("rb") as handle:
            for expert in range(128):
                for projection in (0, 1):
                    if slot >= RING_RECORDS:
                        break
                    offset = (expert * 3 + projection) * RECORD_BYTES
                    handle.seek(offset)
                    data = handle.read(RECORD_BYTES)
                    if len(data) != RECORD_BYTES:
                        raise EOFError(f"short record layer={layer} expert={expert} projection={projection}")
                    header = parse_header(data)
                    expected_tuple = ("SQ5M", layer, expert, projection, 5, 768, 2048, 128, CODE_BYTES, SCALE_BYTES)
                    observed_tuple = (header["magic"], header["layer"], header["expert"], header["projection"], header["bits"], header["rows"], header["cols"], header["group"], header["code_bytes"], header["scale_bytes"])
                    if observed_tuple != expected_tuple:
                        raise ValueError(f"header mismatch {observed_tuple} != {expected_tuple}")
                    C.memmove(destination + slot * RECORD_BYTES, data, RECORD_BYTES)
                    hasher.update(data)
                    if layer == 0 and expert == 0 and projection == 0:
                        correctness["gate"] = data
                    if layer == 0 and expert == 0 and projection == 1:
                        correctness["up"] = data
                    descriptors.append((layer, expert, projection))
                    slot += 1
                if slot >= RING_RECORDS:
                    break
            if slot >= RING_RECORDS:
                break
    if slot != RING_RECORDS:
        raise RuntimeError(f"loaded {slot} ring records instead of {RING_RECORDS}")
    down_path = RUNS / "layer_00.q5bin"
    with down_path.open("rb") as handle:
        handle.seek(2 * RECORD_BYTES)
        down = handle.read(RECORD_BYTES)
    down_header = parse_header(down)
    expected_down = ("SQ5M", 0, 0, 2, 5, 2048, 768, 128, CODE_BYTES, SCALE_BYTES)
    observed_down = (down_header["magic"], down_header["layer"], down_header["expert"], down_header["projection"], down_header["bits"], down_header["rows"], down_header["cols"], down_header["group"], down_header["code_bytes"], down_header["scale_bytes"])
    if observed_down != expected_down:
        raise ValueError(f"down header mismatch {observed_down}")
    C.memmove(destination + RING_RECORDS * RECORD_BYTES, down, RECORD_BYTES)
    hasher.update(down)
    correctness["down"] = down
    selection_payload = json.dumps(descriptors, separators=(",", ":")).encode("ascii")
    return {
        "ring_records": RING_RECORDS,
        "extra_down_records": EXTRA_DOWN_RECORDS,
        "allocation_bytes": ALLOC_BYTES,
        "ring_record_bytes": RING_RECORDS * RECORD_BYTES,
        "ring_mib": RING_RECORDS * RECORD_BYTES / 2**20,
        "usm_content_sha256": hasher.hexdigest(),
        "selection_sha256": hashlib.sha256(selection_payload).hexdigest(),
        "first": descriptors[0],
        "last": descriptors[-1],
        "verified_layer_sha256": {str(layer): manifest["manifests"][str(layer)]["artifact_sha256"] for layer in range(3)},
    }, correctness


def bf16_round_float(values: np.ndarray) -> np.ndarray:
    work = np.asarray(values, dtype=np.float32)
    bits = work.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def bf16_round_bits(values: np.ndarray) -> np.ndarray:
    work = np.asarray(values, dtype=np.float32)
    bits = work.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)


def cpu_q5_oracle(record: bytes, x: np.ndarray) -> np.ndarray:
    header = parse_header(record)
    rows, cols = header["rows"], header["cols"]
    packs = cols // 8
    groups = cols // 128
    packed = np.frombuffer(record, dtype=np.uint8, count=CODE_BYTES, offset=HEADER_BYTES)[: rows * packs * 5].reshape(rows, packs, 5)
    word = np.zeros((rows, packs), dtype=np.uint64)
    for byte in range(5):
        word |= packed[:, :, byte].astype(np.uint64) << np.uint64(byte * 8)
    shifts = (np.arange(8, dtype=np.uint64) * np.uint64(5)).reshape(1, 1, 8)
    codes = (((word[:, :, None] >> shifts) & np.uint64(31)).astype(np.int16) - 15).astype(np.float32)
    scale_bits = np.frombuffer(record, dtype="<u2", count=rows * groups, offset=HEADER_BYTES + CODE_BYTES).reshape(rows, groups)
    scales = (scale_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    pack_groups = np.arange(packs) // 16
    weights = bf16_round_float(codes * scales[:, pack_groups, None])
    vector = np.asarray(x[:cols], dtype=np.float32).reshape(packs, 8)
    accumulators = np.zeros((rows, 256), dtype=np.float32)
    lane = accumulators[:, :packs]
    for part in range(8):
        lane[:] = np.asarray(lane.astype(np.float64) + weights[:, :, part].astype(np.float64) * vector[:, part].astype(np.float64), dtype=np.float32)
    active = accumulators
    for stride in (128, 64, 32, 16, 8, 4, 2, 1):
        active[:, :stride] = np.asarray(active[:, :stride] + active[:, stride : 2 * stride], dtype=np.float32)
    return bf16_round_bits(active[:, 0])


def inputs() -> dict[str, np.ndarray]:
    index = np.arange(2048)
    exponents = (index % 7) - 3
    normal = np.ldexp(np.ones(2048, dtype=np.float32), exponents).astype(np.float32)
    normal[(index * 17 + 3) % 5 < 2] *= np.float32(-1)
    cancellation = np.ldexp(np.ones(2048, dtype=np.float32), (index // 2) % 5 - 2).astype(np.float32)
    cancellation[index % 2 == 1] *= np.float32(-1)
    edge_values = np.asarray([0.0, -0.0, 2.0**-120, -(2.0**-120), 2.0**-126, -(2.0**-126), 1.0, -1.0], dtype=np.float32)
    edge = np.resize(edge_values, 2048).astype(np.float32)
    return {"normal": normal, "cancellation": cancellation, "edge": edge}


def execute() -> dict:
    capability = json.loads(CAPABILITY.read_text(encoding="utf-8"))
    if not capability.get("capability_pass"):
        raise RuntimeError("capability probe did not pass")
    cl = OpenCL()
    try:
        allocation = cl.setup()
        if not allocation["type_is_host"] or not allocation["base_pointer_matches"] or allocation["size"] != ALLOC_BYTES:
            raise RuntimeError(f"host-USM allocation attestation failed: {allocation}")
        loaded, correctness_records = load_real_records(cl.usm_ptr)
        indices_mem = cl.buffer(CL_MEM_READ_ONLY, BATCH_RECORDS * np.dtype(np.int32).itemsize)
        x_mem = cl.buffer(CL_MEM_READ_ONLY, 2048 * np.dtype(np.float32).itemsize)
        output_mem = cl.buffer(CL_MEM_WRITE_ONLY, BATCH_RECORDS * 2048 * np.dtype(np.uint16).itemsize)
        cl.bind(indices_mem, x_mem, output_mem)

        correctness_rows = []
        input_bank = inputs()
        record_slots = {"gate": 0, "up": 1, "down": RING_RECORDS}
        for input_name, vector in input_bank.items():
            cl.write(x_mem, vector)
            for projection_name in ("gate", "up", "down"):
                record = correctness_records[projection_name]
                header = parse_header(record)
                expected = cpu_q5_oracle(record, vector)
                selected = np.zeros(BATCH_RECORDS, dtype=np.int32)
                selected[0] = record_slots[projection_name]
                cl.write(indices_mem, selected)
                event_ms, wall_ms = cl.launch(1, header["rows"], header["cols"])
                observed = np.empty(header["rows"], dtype=np.uint16)
                cl.read(output_mem, observed)
                differing = int(np.count_nonzero(observed != expected))
                correctness_rows.append({
                    "input": input_name,
                    "projection": projection_name,
                    "rows": header["rows"],
                    "cols": header["cols"],
                    "bit_differences": differing,
                    "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                    "observed_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                    "event_ms": event_ms,
                    "wall_ms": wall_ms,
                })
        bit_differences = sum(row["bit_differences"] for row in correctness_rows)
        if bit_differences:
            return {
                "status": "negative_cross_backend_q5_semantics",
                "device": cl.identity,
                "allocation": allocation,
                "loaded": loaded,
                "correctness": {"rows": correctness_rows, "bit_differences": bit_differences, "pass": False},
                "performance": {"status": "not_run_correctness_hard_stop"},
            }

        perf_input = input_bank["normal"]
        cl.write(x_mem, perf_input)
        for batch in range(WARMUP_BATCHES):
            start = (batch * BATCH_RECORDS) % RING_RECORDS
            selected = np.asarray([(start + offset) % RING_RECORDS for offset in range(BATCH_RECORDS)], dtype=np.int32)
            cl.write(indices_mem, selected)
            cl.launch(BATCH_RECORDS, 768, 2048)

        sampler = HardPageReadSampler()
        sampler.start()
        event_ms_values: list[float] = []
        wall_ms_values: list[float] = []
        try:
            for iteration in range(TIMED_ITERATIONS):
                start = (iteration * 17) % RING_RECORDS
                selected = np.asarray([(start + offset) % RING_RECORDS for offset in range(BATCH_RECORDS)], dtype=np.int32)
                cl.write(indices_mem, selected)
                event_ms, wall_ms = cl.launch(BATCH_RECORDS, 768, 2048)
                event_ms_values.append(event_ms)
                wall_ms_values.append(wall_ms)
        finally:
            sampler.stop()
        final_output = np.empty(BATCH_RECORDS * 768, dtype=np.uint16)
        cl.read(output_mem, final_output)
        bytes_per_event = BATCH_RECORDS * EFFECTIVE_BYTES_PER_RECORD
        event_gbps = [bytes_per_event / (value * 1e6) for value in event_ms_values]
        wall_gbps = [bytes_per_event / (value * 1e6) for value in wall_ms_values]
        p95_side = min(percentile(event_gbps, 5), percentile(wall_gbps, 5))
        page_values = [float(row["page_reads_per_sec"]) for row in sampler.samples]
        page_gate = sampler.error is None and bool(page_values) and max(page_values) == 0.0
        performance = {
            "status": "complete",
            "ring_records": RING_RECORDS,
            "ring_bytes": RING_RECORDS * RECORD_BYTES,
            "ring_mib": RING_RECORDS * RECORD_BYTES / 2**20,
            "batch_records": BATCH_RECORDS,
            "warmup_batches": WARMUP_BATCHES,
            "iterations": TIMED_ITERATIONS,
            "effective_bytes_per_record": EFFECTIVE_BYTES_PER_RECORD,
            "effective_bytes_per_event": bytes_per_event,
            "event_ms": stats(event_ms_values),
            "wall_ms": stats(wall_ms_values),
            "event_gbps": stats(event_gbps),
            "wall_gbps": stats(wall_gbps),
            "conservative_p95_latency_side_gbps": p95_side,
            "minimum_gbps": MIN_P95_SIDE_GBPS,
            "throughput_pass": p95_side >= MIN_P95_SIDE_GBPS,
            "raw_event_ms": event_ms_values,
            "raw_wall_ms": wall_ms_values,
            "final_output_sha256": hashlib.sha256(final_output.tobytes()).hexdigest(),
            "pdh": {"samples": sampler.samples, "error": sampler.error, "all_page_reads_zero": page_gate},
        }
        overall = p95_side >= MIN_P95_SIDE_GBPS and page_gate
        return {
            "status": "pass" if overall else "negative_throughput_or_page_gate",
            "device": cl.identity,
            "allocation": allocation,
            "loaded": loaded,
            "correctness": {"rows": correctness_rows, "bit_differences": 0, "pass": True},
            "performance": performance,
            "hidden_copy_audit": {
                "weight_api": "clHostMemAllocINTEL + clSetKernelArgMemPointerINTEL",
                "weight_cl_mem_buffers_created": 0,
                "weight_enqueue_write_calls": 0,
                "weight_enqueue_copy_calls": 0,
                "weight_enqueue_migrate_calls": 0,
                "private_weight_copy_requested": False,
                "host_usm_allocation_attested": allocation["type_is_host"] and allocation["base_pointer_matches"],
                "boundary": "API-level proof that the benchmark supplies host USM directly; hardware caches remain normal and are not a private full-bank copy.",
            },
        }
    finally:
        cl.close()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    base = {
        "kind": "streamq5_moe_st2_mini_host_usm_q5",
        "started_utc": utc_now(),
        "preregistration_sha256": sha256(PREREG),
        "capability_probe_sha256": sha256(CAPABILITY),
        "p1d_manifest_sha256": sha256(P1D),
        "runner_sha256": sha256(Path(__file__)),
        "kernel_source_sha256": hashlib.sha256(KERNEL_SOURCE.encode("utf-8")).hexdigest(),
        "execution_lock": {
            "ring_records": RING_RECORDS,
            "allocation_records": ALLOC_RECORDS,
            "allocation_bytes": ALLOC_BYTES,
            "batch_records": BATCH_RECORDS,
            "warmup_batches": WARMUP_BATCHES,
            "timed_iterations": TIMED_ITERATIONS,
            "minimum_p95_side_gbps": MIN_P95_SIDE_GBPS,
        },
        "nvidia_gpu_kernel_or_transfer_calls": 0,
    }
    try:
        result = execute()
        base.update(result)
    except Exception as exc:
        base.update({
            "status": "blocked_or_runtime_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    base["completed_utc"] = utc_now()
    base["claim_boundary"] = "Intel-iGPU host-USM Q5 component test only; no dGPU, cross-device SplitTree, real-80B, model-quality or end-to-end claim."
    OUTPUT.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in base.items() if key not in ("performance",)}, indent=2))
    if "performance" in base:
        summary = {key: value for key, value in base["performance"].items() if not key.startswith("raw_")}
        print(json.dumps({"performance": summary}, indent=2))


if __name__ == "__main__":
    main()
