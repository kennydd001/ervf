#!/usr/bin/env python3
"""PH1 Intel Arc host-USM backend. Import is device-free; run() is lock-gated."""
from __future__ import annotations

import ctypes as C
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
LOCK = REPORTS / "het_next_l0_ph1_intel_execution_lock.json"
COMPILE_LOCK = REPORTS / "het_next_l0_ph1_intel_compile_lock.json"

CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 4
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_PCI_BUS_INFO_KHR = 0x410F
CL_CONTEXT_PLATFORM = 0x1084
CL_PROGRAM_BUILD_LOG = 0x1183
CL_PROGRAM_BINARY_SIZES = 0x1165
CL_PROGRAM_BINARIES = 0x1166
CL_MEM_ALLOC_TYPE_INTEL = 0x419A
CL_MEM_ALLOC_BASE_PTR_INTEL = 0x419B
CL_MEM_ALLOC_SIZE_INTEL = 0x419C
CL_MEM_TYPE_HOST_INTEL = 0x4197

BUFFER_TABLE = (
    ("gate_record", 675_840),
    ("up_record", 675_840),
    ("down_record", 675_840),
    ("natural_input", 4_096),
    ("silu_lut", 131_072),
    ("gate", 1_024),
    ("up", 1_024),
    ("silu", 1_024),
    ("activation", 1_024),
    ("down", 4_096),
    ("gate_counters", 2_048),
    ("up_counters", 2_048),
    ("activation_counters", 2_048),
    ("down_counters", 8_192),
)
if sum(size for _, size in BUFFER_TABLE) != 2_185_216:
    raise RuntimeError("buffer_table")

SRC = r'''
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_int64 : enable
#define CODE_BYTES 655360UL

inline float bf16_to_float(ushort value) {
    return as_float(((uint)value) << 16);
}
inline ushort float_to_bf16(float value) {
    uint bits = as_uint(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (ushort)(bits >> 16);
}
inline float rounded_bf16_float(float value) {
    return bf16_to_float(float_to_bf16(value));
}
inline ulong round_shift_even(ulong number, int shift) {
    if (shift <= 0) return number << (-shift);
    if (shift >= 64) return 0UL;
    ulong quotient = number >> shift;
    ulong mask = (1UL << shift) - 1UL;
    ulong remainder = number & mask;
    ulong half = 1UL << (shift - 1);
    return quotient + (remainder > half || (remainder == half && (quotient & 1UL)));
}
inline ushort multiply_bf16_exact(ushort a, ushort b, __private uint* ok) {
    uint sign = ((uint)(a ^ b)) & 0x8000U;
    uint ae = ((uint)a >> 7) & 255U, be = ((uint)b >> 7) & 255U;
    uint af = ((uint)a) & 127U, bf = ((uint)b) & 127U;
    if (ae == 255U || be == 255U) { *ok = 0U; return (ushort)0xffffU; }
    if ((ae == 0U && af == 0U) || (be == 0U && bf == 0U)) return (ushort)sign;
    ulong an = ae == 0U ? (ulong)af : (ulong)(128U + af);
    ulong bn = be == 0U ? (ulong)bf : (ulong)(128U + bf);
    int ax = ae == 0U ? -133 : (int)ae - 134;
    int bx = be == 0U ? -133 : (int)be - 134;
    ulong number = an * bn;
    int exponent = ax + bx;
    int highest = 63 - (int)clz(number);
    int top = highest + exponent;
    if (top > 127) { *ok = 0U; return (ushort)0xffffU; }
    if (top >= -126) {
        int shift = highest - 7;
        ulong significand = round_shift_even(number, shift);
        if (significand == 256UL) { significand = 128UL; shift += 1; }
        int unbiased = exponent + shift + 7;
        if (unbiased > 127) { *ok = 0U; return (ushort)0xffffU; }
        return (ushort)(sign | ((uint)(unbiased + 127) << 7) | ((uint)significand & 127U));
    }
    int shift = -133 - exponent;
    ulong fraction = round_shift_even(number, shift);
    if (fraction == 0UL) return (ushort)sign;
    if (fraction >= 128UL) return (ushort)(sign | 0x0080U);
    return (ushort)(sign | (uint)fraction);
}

inline void linear_2048(
    __global const uchar* record, __global const ushort* input,
    __global ushort* output, __global uint* counters) {
    int subgroup = (int)get_sub_group_id();
    int lane = (int)get_sub_group_local_id();
    int row = (int)get_group_id(0) * 32 + subgroup;
    if (row >= 512) return;
    __global const uchar* codes = record + 64;
    __global const ushort* scales = (__global const ushort*)(record + 64 + CODE_BYTES);
    float partial[32];
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 32; ++virtual_index) {
        int pack = lane + 8 * virtual_index;
        int column = pack * 8;
        __global const uchar* source = codes + (ulong)row * 1280UL + (ulong)pack * 5UL;
        ulong fields = (ulong)source[0] | (ulong)source[1] << 8 | (ulong)source[2] << 16 |
                       (ulong)source[3] << 24 | (ulong)source[4] << 32;
        float accumulator = 0.0f;
        float scale = bf16_to_float(scales[row * 16 + (column >> 7)]);
        #pragma unroll
        for (int field = 0; field < 8; ++field) {
            int q = (int)((fields >> (5 * field)) & 31UL) - 15;
            float weight = rounded_bf16_float((float)q * scale);
            accumulator = fma(weight, bf16_to_float(input[column + field]), accumulator);
        }
        partial[virtual_index] = accumulator;
    }
    #pragma unroll
    for (int distance = 16; distance >= 1; distance >>= 1) {
        #pragma unroll
        for (int index = 0; index < distance; ++index) partial[index] = partial[index] + partial[index + distance];
    }
    float value = partial[0];
    #pragma unroll
    for (int distance = 4; distance >= 1; distance >>= 1) {
        float other = intel_sub_group_shuffle_down(value, value, (uint)distance);
        if (lane < distance) value = value + other;
    }
    if (lane == 0) {
        output[row] = float_to_bf16(value);
        atomic_inc((volatile __global unsigned int*)&counters[row]);
    }
}

inline void linear_512(
    __global const uchar* record, __global const ushort* input,
    __global ushort* output, __global uint* counters) {
    int subgroup = (int)get_sub_group_id();
    int lane = (int)get_sub_group_local_id();
    int row = (int)get_group_id(0) * 32 + subgroup;
    if (row >= 2048) return;
    __global const uchar* codes = record + 64;
    __global const ushort* scales = (__global const ushort*)(record + 64 + CODE_BYTES);
    float partial[8];
    #pragma unroll
    for (int virtual_index = 0; virtual_index < 8; ++virtual_index) {
        int pack = lane + 8 * virtual_index;
        int column = pack * 8;
        __global const uchar* source = codes + (ulong)row * 320UL + (ulong)pack * 5UL;
        ulong fields = (ulong)source[0] | (ulong)source[1] << 8 | (ulong)source[2] << 16 |
                       (ulong)source[3] << 24 | (ulong)source[4] << 32;
        float accumulator = 0.0f;
        float scale = bf16_to_float(scales[row * 4 + (column >> 7)]);
        #pragma unroll
        for (int field = 0; field < 8; ++field) {
            int q = (int)((fields >> (5 * field)) & 31UL) - 15;
            float weight = rounded_bf16_float((float)q * scale);
            accumulator = fma(weight, bf16_to_float(input[column + field]), accumulator);
        }
        partial[virtual_index] = accumulator;
    }
    #pragma unroll
    for (int distance = 4; distance >= 1; distance >>= 1) {
        #pragma unroll
        for (int index = 0; index < distance; ++index) partial[index] = partial[index] + partial[index + distance];
    }
    float value = partial[0];
    #pragma unroll
    for (int distance = 4; distance >= 1; distance >>= 1) {
        float other = intel_sub_group_shuffle_down(value, value, (uint)distance);
        if (lane < distance) value = value + other;
    }
    if (lane == 0) {
        output[row] = float_to_bf16(value);
        atomic_inc((volatile __global unsigned int*)&counters[row]);
    }
}

__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void gate_linear(__global const uchar* record, __global const ushort* input,
                 __global ushort* output, __global uint* counters) {
    linear_2048(record, input, output, counters);
}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void up_linear(__global const uchar* record, __global const ushort* input,
               __global ushort* output, __global uint* counters) {
    linear_2048(record, input, output, counters);
}
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void activation(__global const ushort* gate, __global const ushort* up,
                __global const ushort* lut, __global ushort* silu,
                __global ushort* activated, __global uint* counters) {
    int row = (int)get_global_id(0);
    if (row >= 512) return;
    ushort gate_word = gate[row], up_word = up[row];
    if ((((uint)gate_word >> 7) & 255U) == 255U || (((uint)up_word >> 7) & 255U) == 255U) return;
    ushort silu_word = lut[(uint)gate_word];
    uint ok = 1U;
    ushort activation_word = multiply_bf16_exact(silu_word, up_word, &ok);
    if (!ok) return;
    silu[row] = silu_word;
    activated[row] = activation_word;
    atomic_inc((volatile __global unsigned int*)&counters[row]);
}
__kernel __attribute__((reqd_work_group_size(256,1,1))) __attribute__((intel_reqd_sub_group_size(8)))
void down_linear(__global const uchar* record, __global const ushort* input,
                 __global ushort* output, __global uint* counters) {
    linear_512(record, input, output, counters);
}
'''


class PCI(C.Structure):
    _fields_ = [("domain", C.c_uint), ("bus", C.c_uint), ("device", C.c_uint), ("function", C.c_uint)]


class IntelRunFailure(RuntimeError):
    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


def _check(code: int, operation: str) -> None:
    if code != CL_SUCCESS:
        raise RuntimeError(f"{operation}:{code}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_gate(expected: dict) -> dict:
    lock = json.loads(LOCK.read_text())
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_execution_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == "PH1_INTEL_AFTER_SOURCE_AND_PREFLIGHT_GO"
        and lock.get("backend_sha256") == _file_sha(Path(__file__))
        and lock.get("cpu_commit_sha256") == expected["cpu_commit_sha256"]
        and lock.get("cpu_verification_sha256") == expected["cpu_verification_sha256"]
        and lock.get("source_sha256") == expected["source_sha256"] == _sha(SRC.encode())
        and lock.get("binary_sha256") == expected["binary_sha256"]
        and lock.get("build_log_sha256") == expected["build_log_sha256"]
    ):
        raise RuntimeError("intel_execution_lock")
    return {"lock_sha256": _file_sha(LOCK), "lock": lock}


def _compile_runtime_gate(expected: dict) -> dict:
    lock = json.loads(COMPILE_LOCK.read_text())
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == "PH1_INTEL_COMPILE_AFTER_SOURCE_AND_PREFLIGHT_GO"
        and lock.get("backend_sha256") == _file_sha(Path(__file__))
        and lock.get("cpu_commit_sha256") == expected["cpu_commit_sha256"]
        and lock.get("cpu_verification_sha256") == expected["cpu_verification_sha256"]
        and lock.get("source_sha256") == expected["source_sha256"] == _sha(SRC.encode())
    ):
        raise RuntimeError("intel_compile_lock")
    return {"lock_sha256": _file_sha(COMPILE_LOCK), "lock": lock}


class IntelBackend:
    def __init__(self):
        self.library = None
        self.context = self.queue = self.program = None
        self.kernels: list[tuple[str, object]] = []
        self.allocations: list[tuple[str, int, int, object]] = []
        self.ledger: list[dict] = []
        self.cleanup_errors: list[str] = []

    def _bind(self) -> None:
        library = self.library
        pointer, uint, size, integer = C.c_void_p, C.c_uint, C.c_size_t, C.c_int
        library.clGetPlatformIDs.argtypes = [uint, C.POINTER(pointer), C.POINTER(uint)]
        library.clGetPlatformIDs.restype = integer
        library.clGetDeviceIDs.argtypes = [pointer, C.c_ulonglong, uint, C.POINTER(pointer), C.POINTER(uint)]
        library.clGetDeviceIDs.restype = integer
        library.clGetDeviceInfo.argtypes = [pointer, uint, size, pointer, C.POINTER(size)]
        library.clGetDeviceInfo.restype = integer
        library.clGetExtensionFunctionAddressForPlatform.argtypes = [pointer, C.c_char_p]
        library.clGetExtensionFunctionAddressForPlatform.restype = pointer
        library.clCreateContext.argtypes = [C.POINTER(C.c_ssize_t), uint, C.POINTER(pointer), pointer, pointer, C.POINTER(integer)]
        library.clCreateContext.restype = pointer
        library.clCreateCommandQueue.argtypes = [pointer, pointer, C.c_ulonglong, C.POINTER(integer)]
        library.clCreateCommandQueue.restype = pointer
        library.clCreateProgramWithSource.argtypes = [pointer, uint, C.POINTER(C.c_char_p), C.POINTER(size), C.POINTER(integer)]
        library.clCreateProgramWithSource.restype = pointer
        library.clBuildProgram.argtypes = [pointer, uint, C.POINTER(pointer), C.c_char_p, pointer, pointer]
        library.clBuildProgram.restype = integer
        library.clGetProgramBuildInfo.argtypes = [pointer, pointer, uint, size, pointer, C.POINTER(size)]
        library.clGetProgramBuildInfo.restype = integer
        library.clGetProgramInfo.argtypes = [pointer, uint, size, pointer, C.POINTER(size)]
        library.clGetProgramInfo.restype = integer
        library.clCreateKernel.argtypes = [pointer, C.c_char_p, C.POINTER(integer)]
        library.clCreateKernel.restype = pointer
        library.clEnqueueNDRangeKernel.argtypes = [pointer, pointer, uint, pointer, C.POINTER(size), C.POINTER(size), uint, pointer, pointer]
        library.clEnqueueNDRangeKernel.restype = integer
        library.clFinish.argtypes = [pointer]
        library.clFinish.restype = integer
        for name in ("clReleaseKernel", "clReleaseProgram", "clReleaseCommandQueue", "clReleaseContext"):
            getattr(library, name).argtypes = [pointer]
            getattr(library, name).restype = integer

    def _info(self, device, parameter: int) -> str:
        size = C.c_size_t()
        _check(self.library.clGetDeviceInfo(device, parameter, 0, None, C.byref(size)), "device_info_size")
        buffer = C.create_string_buffer(size.value)
        _check(self.library.clGetDeviceInfo(device, parameter, size.value, buffer, None), "device_info")
        return buffer.value.decode(errors="replace")

    def _select(self):
        count = C.c_uint()
        _check(self.library.clGetPlatformIDs(0, None, C.byref(count)), "platform_count")
        platforms = (C.c_void_p * count.value)()
        _check(self.library.clGetPlatformIDs(count.value, platforms, None), "platforms")
        found = []
        for raw_platform in platforms:
            platform_handle = C.c_void_p(raw_platform)
            device_count = C.c_uint()
            if self.library.clGetDeviceIDs(platform_handle, CL_DEVICE_TYPE_GPU, 0, None, C.byref(device_count)):
                continue
            devices = (C.c_void_p * device_count.value)()
            _check(self.library.clGetDeviceIDs(platform_handle, CL_DEVICE_TYPE_GPU, device_count.value, devices, None), "devices")
            for raw_device in devices:
                device = C.c_void_p(raw_device)
                name = self._info(device, CL_DEVICE_NAME)
                extensions = self._info(device, CL_DEVICE_EXTENSIONS)
                if name == "Intel(R) Arc(TM) Pro 140T GPU (32GB)" and "cl_intel_unified_shared_memory" in extensions:
                    found.append((platform_handle, device, extensions))
        if len(found) != 1:
            raise RuntimeError(f"intel_cardinality:{len(found)}")
        platform_handle, device, extensions = found[0]
        pci = PCI()
        _check(self.library.clGetDeviceInfo(device, CL_DEVICE_PCI_BUS_INFO_KHR, C.sizeof(pci), C.byref(pci), None), "pci")
        identity = {
            "name": self._info(device, CL_DEVICE_NAME),
            "vendor": self._info(device, CL_DEVICE_VENDOR),
            "driver": self._info(device, CL_DRIVER_VERSION),
            "pci": f"{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}.{pci.function}",
            "extensions": extensions.split(),
        }
        if identity["vendor"] != "Intel(R) Corporation" or identity["driver"] != "32.0.101.8517" or identity["pci"] != "0000:00:02.0":
            raise RuntimeError("intel_identity")
        self.ledger.append({"op": "identity", "identity": identity})
        return platform_handle, device, identity

    def _extension(self, platform_handle, name: str, prototype):
        address = self.library.clGetExtensionFunctionAddressForPlatform(platform_handle, name.encode())
        if not address:
            raise RuntimeError("missing_extension:" + name)
        return prototype(address)

    def run(self, records: dict[str, bytes], input_bytes: bytes, lut: bytes, eligibility: dict) -> dict:
        authorization = _runtime_gate(eligibility)  # before OpenCL.dll
        evidence = {"authorization": authorization, "ledger": self.ledger, "source_sha256": _sha(SRC.encode())}
        identity = {}
        binary = build_log = b""
        outputs: dict[str, bytes] = {}
        try:
            if set(records) != {"gate", "up", "down"} or any(len(records[name]) != 675_840 for name in records):
                raise RuntimeError("record_package")
            if len(input_bytes) != 4_096 or len(lut) != 131_072:
                raise RuntimeError("input_or_lut")
            self.library = C.WinDLL("OpenCL.dll")
            self._bind()
            platform_handle, device, identity = self._select()
            error = C.c_int()
            properties = (C.c_ssize_t * 3)(CL_CONTEXT_PLATFORM, int(platform_handle.value), 0)
            devices = (C.c_void_p * 1)(device.value)
            self.context = self.library.clCreateContext(properties, 1, devices, None, None, C.byref(error))
            _check(error.value, "context")
            self.ledger.append({"op": "context_create", "pointer": int(self.context)})
            self.queue = self.library.clCreateCommandQueue(self.context, device, 0, C.byref(error))
            _check(error.value, "queue")
            self.ledger.append({"op": "queue_create", "pointer": int(self.queue), "in_order": True})
            source = SRC.encode()
            strings, lengths = (C.c_char_p * 1)(source), (C.c_size_t * 1)(len(source))
            self.program = self.library.clCreateProgramWithSource(self.context, 1, strings, lengths, C.byref(error))
            _check(error.value, "program")
            self.ledger.append({"op": "program_create", "source_bytes": len(source), "source_sha256": _sha(source)})
            options = b"-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt"
            build_code = self.library.clBuildProgram(self.program, 1, devices, options, None, None)
            log_size = C.c_size_t()
            _check(self.library.clGetProgramBuildInfo(self.program, device, CL_PROGRAM_BUILD_LOG, 0, None, C.byref(log_size)), "build_log_size")
            log_buffer = C.create_string_buffer(log_size.value)
            _check(self.library.clGetProgramBuildInfo(self.program, device, CL_PROGRAM_BUILD_LOG, log_size.value, log_buffer, None), "build_log")
            build_log = log_buffer.raw
            self.ledger.append({"op": "program_build", "code": int(build_code), "options": options.decode(), "log_sha256": _sha(build_log)})
            _check(build_code, "program_build")
            binary_sizes = (C.c_size_t * 1)()
            _check(self.library.clGetProgramInfo(self.program, CL_PROGRAM_BINARY_SIZES, C.sizeof(binary_sizes), binary_sizes, None), "binary_size")
            binary_buffer = C.create_string_buffer(binary_sizes[0])
            binary_pointers = (C.c_void_p * 1)(C.cast(binary_buffer, C.c_void_p))
            _check(self.library.clGetProgramInfo(self.program, CL_PROGRAM_BINARIES, C.sizeof(binary_pointers), binary_pointers, None), "binary")
            binary = binary_buffer.raw
            if _sha(source) != eligibility["source_sha256"] or _sha(build_log) != eligibility["build_log_sha256"] or _sha(binary) != eligibility["binary_sha256"]:
                raise RuntimeError("compiled_program_drift")
            for kernel_name in ("gate_linear", "up_linear", "activation", "down_linear"):
                kernel = self.library.clCreateKernel(self.program, kernel_name.encode(), C.byref(error))
                _check(error.value, "kernel:" + kernel_name)
                self.kernels.append((kernel_name, kernel))
                self.ledger.append({"op": "kernel_create", "name": kernel_name, "pointer": int(kernel)})

            host_alloc = self._extension(platform_handle, "clHostMemAllocINTEL", C.WINFUNCTYPE(C.c_void_p, C.c_void_p, C.POINTER(C.c_ssize_t), C.c_size_t, C.c_uint, C.POINTER(C.c_int)))
            memory_free = self._extension(platform_handle, "clMemFreeINTEL", C.WINFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p))
            set_pointer = self._extension(platform_handle, "clSetKernelArgMemPointerINTEL", C.WINFUNCTYPE(C.c_int, C.c_void_p, C.c_uint, C.c_void_p))
            get_alloc_info = self._extension(platform_handle, "clGetMemAllocInfoINTEL", C.WINFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)))
            pointer_by_name: dict[str, int] = {}
            for name, size in BUFFER_TABLE:
                pointer = int(host_alloc(self.context, None, size, 4096, C.byref(error)))
                _check(error.value, "allocate:" + name)
                self.allocations.append((name, pointer, size, memory_free))
                pointer_by_name[name] = pointer
                allocation_type, allocation_base, allocation_size = C.c_uint(), C.c_void_p(), C.c_size_t()
                _check(get_alloc_info(self.context, C.c_void_p(pointer), CL_MEM_ALLOC_TYPE_INTEL, C.sizeof(allocation_type), C.byref(allocation_type), None), "alloc_type")
                _check(get_alloc_info(self.context, C.c_void_p(pointer), CL_MEM_ALLOC_BASE_PTR_INTEL, C.sizeof(allocation_base), C.byref(allocation_base), None), "alloc_base")
                _check(get_alloc_info(self.context, C.c_void_p(pointer), CL_MEM_ALLOC_SIZE_INTEL, C.sizeof(allocation_size), C.byref(allocation_size), None), "alloc_size")
                if allocation_type.value != CL_MEM_TYPE_HOST_INTEL or allocation_base.value != pointer or allocation_size.value != size or pointer % 4096:
                    raise RuntimeError("allocation_attestation:" + name)
                self.ledger.append({"op": "host_usm_allocate", "name": name, "bytes": size, "alignment": 4096, "pointer": pointer, "queried_type": "host", "queried_base": allocation_base.value, "queried_size": allocation_size.value})

            writes = {
                "gate_record": records["gate"], "up_record": records["up"], "down_record": records["down"],
                "natural_input": input_bytes, "silu_lut": lut,
            }
            for name in ("gate_record", "up_record", "down_record", "natural_input", "silu_lut"):
                C.memmove(pointer_by_name[name], writes[name], len(writes[name]))
                self.ledger.append({"op": "cpu_direct_write", "name": name, "bytes": len(writes[name]), "sha256": _sha(writes[name])})
            for name in ("gate", "up", "silu", "activation", "down"):
                size = dict(BUFFER_TABLE)[name]
                C.memset(pointer_by_name[name], 0xFF, size)
                self.ledger.append({"op": "cpu_initialize", "name": name, "bytes": size, "value": "ff"})
            for name in ("gate_counters", "up_counters", "activation_counters", "down_counters"):
                size = dict(BUFFER_TABLE)[name]
                C.memset(pointer_by_name[name], 0, size)
                self.ledger.append({"op": "cpu_initialize", "name": name, "bytes": size, "value": "00"})

            argument_maps = (
                ("gate_linear", ("gate_record", "natural_input", "gate", "gate_counters")),
                ("up_linear", ("up_record", "natural_input", "up", "up_counters")),
                ("activation", ("gate", "up", "silu_lut", "silu", "activation", "activation_counters")),
                ("down_linear", ("down_record", "activation", "down", "down_counters")),
            )
            kernel_by_name = dict(self.kernels)
            for kernel_name, names in argument_maps:
                for index, name in enumerate(names):
                    _check(set_pointer(kernel_by_name[kernel_name], index, C.c_void_p(pointer_by_name[name])), f"setarg:{kernel_name}:{index}")
                    self.ledger.append({"op": "set_pointer_arg", "kernel": kernel_name, "index": index, "name": name, "pointer": pointer_by_name[name]})
            launches = (("gate_linear", 4096, 256), ("up_linear", 4096, 256), ("activation", 512, 256), ("down_linear", 16384, 256))
            for kernel_name, global_size, local_size in launches:
                global_array, local_array = (C.c_size_t * 1)(global_size), (C.c_size_t * 1)(local_size)
                _check(self.library.clEnqueueNDRangeKernel(self.queue, kernel_by_name[kernel_name], 1, None, global_array, local_array, 0, None, None), "enqueue:" + kernel_name)
                self.ledger.append({"op": "enqueue", "kernel": kernel_name, "global": global_size, "local": local_size, "event_requested": False})
            _check(self.library.clFinish(self.queue), "finish")
            self.ledger.append({"op": "finish", "code": 0})
            for name in ("gate", "up", "silu", "activation", "down", "gate_counters", "up_counters", "activation_counters", "down_counters"):
                size = dict(BUFFER_TABLE)[name]
                data = C.string_at(pointer_by_name[name], size)
                outputs[name] = data
                self.ledger.append({"op": "cpu_direct_read", "name": name, "bytes": size, "sha256": _sha(data), "after_finish": True})

        except Exception as exc:
            evidence.update({"identity": identity, "outputs": {name: data.hex() for name, data in outputs.items()}, "build_log_hex": build_log.hex(), "binary_hex": binary.hex(), "error": f"{type(exc).__name__}: {exc}"})
            raise IntelRunFailure(str(exc), evidence) from exc
        finally:
            self._close()

        evidence.update({
            "identity": identity,
            "outputs": {name: data.hex() for name, data in outputs.items()},
            "build_log_hex": build_log.hex(), "build_log_sha256": _sha(build_log),
            "binary_hex": binary.hex(), "binary_sha256": _sha(binary),
            "forbidden_calls": {name: 0 for name in ("cl_mem", "CreateBuffer", "enqueue_read", "enqueue_write", "enqueue_copy", "migrate", "prefetch")},
        })
        if self.cleanup_errors:
            evidence["error"] = "cleanup_errors"
            raise IntelRunFailure("cleanup_errors", evidence)
        return evidence

    def compile_only(self, eligibility: dict) -> dict:
        authorization = _compile_runtime_gate(eligibility)  # before OpenCL.dll
        evidence = {"authorization": authorization, "ledger": self.ledger, "source": SRC, "source_sha256": _sha(SRC.encode())}
        identity = {}
        binary = build_log = b""
        try:
            self.library = C.WinDLL("OpenCL.dll")
            self._bind()
            platform_handle, device, identity = self._select()
            error = C.c_int()
            properties = (C.c_ssize_t * 3)(CL_CONTEXT_PLATFORM, int(platform_handle.value), 0)
            devices = (C.c_void_p * 1)(device.value)
            self.context = self.library.clCreateContext(properties, 1, devices, None, None, C.byref(error))
            _check(error.value, "context")
            self.ledger.append({"op": "context_create", "pointer": int(self.context)})
            source = SRC.encode()
            strings, lengths = (C.c_char_p * 1)(source), (C.c_size_t * 1)(len(source))
            self.program = self.library.clCreateProgramWithSource(self.context, 1, strings, lengths, C.byref(error))
            _check(error.value, "program")
            self.ledger.append({"op": "program_create", "source_bytes": len(source), "source_sha256": _sha(source)})
            options = b"-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt"
            build_code = self.library.clBuildProgram(self.program, 1, devices, options, None, None)
            log_size = C.c_size_t()
            _check(self.library.clGetProgramBuildInfo(self.program, device, CL_PROGRAM_BUILD_LOG, 0, None, C.byref(log_size)), "build_log_size")
            log_buffer = C.create_string_buffer(log_size.value)
            _check(self.library.clGetProgramBuildInfo(self.program, device, CL_PROGRAM_BUILD_LOG, log_size.value, log_buffer, None), "build_log")
            build_log = log_buffer.raw
            self.ledger.append({"op": "program_build", "code": int(build_code), "options": options.decode(), "log_sha256": _sha(build_log)})
            _check(build_code, "program_build")
            sizes = (C.c_size_t * 1)()
            _check(self.library.clGetProgramInfo(self.program, CL_PROGRAM_BINARY_SIZES, C.sizeof(sizes), sizes, None), "binary_size")
            buffer = C.create_string_buffer(sizes[0])
            pointers = (C.c_void_p * 1)(C.cast(buffer, C.c_void_p))
            _check(self.library.clGetProgramInfo(self.program, CL_PROGRAM_BINARIES, C.sizeof(pointers), pointers, None), "binary")
            binary = buffer.raw
        except Exception as exc:
            evidence.update({"identity": identity, "build_log_hex": build_log.hex(), "binary_hex": binary.hex(), "error": f"{type(exc).__name__}: {exc}"})
            raise IntelRunFailure(str(exc), evidence) from exc
        finally:
            self._close()
        evidence.update({"identity": identity, "build_log_hex": build_log.hex(), "build_log_sha256": _sha(build_log), "binary_hex": binary.hex(), "binary_sha256": _sha(binary), "options": "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt", "payload_read": False, "allocations": 0, "kernels_created": 0, "kernels_launched": 0})
        if self.cleanup_errors:
            evidence["error"] = "cleanup_errors"
            raise IntelRunFailure("cleanup_errors", evidence)
        return evidence

    def _close(self) -> None:
        for name, pointer, _size, memory_free in reversed(self.allocations):
            try:
                code = int(memory_free(self.context, C.c_void_p(pointer)))
                self.ledger.append({"op": "release", "name": name, "code": code})
                _check(code, "release:" + name)
            except Exception as exc:
                self.cleanup_errors.append(f"{name}:{exc}")
        for name, handle in reversed(self.kernels):
            try:
                code = int(self.library.clReleaseKernel(handle))
                self.ledger.append({"op": "release", "name": "kernel:" + name, "code": code})
                _check(code, "release_kernel:" + name)
            except Exception as exc:
                self.cleanup_errors.append(f"kernel:{name}:{exc}")
        for name, handle, function in (
            ("program", self.program, "clReleaseProgram"),
            ("queue", self.queue, "clReleaseCommandQueue"),
            ("context", self.context, "clReleaseContext"),
        ):
            if handle:
                try:
                    code = int(getattr(self.library, function)(handle))
                    self.ledger.append({"op": "release", "name": name, "code": code})
                    _check(code, "release:" + name)
                except Exception as exc:
                    self.cleanup_errors.append(f"{name}:{exc}")
        self.ledger.append({"op": "cleanup", "cleanup_complete": not self.cleanup_errors, "errors": self.cleanup_errors, "live_owned_resources": 0 if not self.cleanup_errors else None})


def run(records: dict[str, bytes], input_bytes: bytes, lut: bytes, eligibility: dict) -> dict:
    return IntelBackend().run(records, input_bytes, lut, eligibility)


def compile_only(eligibility: dict) -> dict:
    return IntelBackend().compile_only(eligibility)
