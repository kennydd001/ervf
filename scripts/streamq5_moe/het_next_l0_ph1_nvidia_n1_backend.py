#!/usr/bin/env python3
"""Direct NVRTC compiler and CUDA Driver backend for PH1 NVIDIA N1.

Import is inert.  compile_one_program() is compiler-only.  DriverBackend.run()
is physical-only and consumes an already frozen cubin.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import threading
from pathlib import Path

from het_next_l0_ph1_nvidia_n1_common import (
    ARGUMENT_MAPS, BUFFER_TABLE, LAUNCHES, RESOURCE_STAGES, host_sample, sha,
)

ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = Path(__file__).with_name("het_next_l0_ph1_nvidia_n1_kernels.cu")
NVRTC_DLL = ROOT / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll"
NVRTC_DLL_SHA = "c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe"
NVRTC_HEADER = ROOT / ".venv/Lib/site-packages/nvidia/cu13/include/nvrtc.h"
NVRTC_HEADER_SHA = "316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21"
NVCUDA_DLL = Path(r"C:/Windows/System32/nvcuda.dll")
NVCUDA_DLL_SHA = "86b41599a673f1aa4699ab458dc5c1e02b57da64d17221f45327af0393fd59a5"
NVCUDA_BYTES = 4_466_920
OPTIONS = (
    "--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true",
    "--ftz=false", "--gpu-architecture=sm_120",
    "--device-as-default-execution-space",
)
EXPECTED_NAME = "NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU"
EXPECTED_PCI = "0000:01:00.0"
EXPECTED_DRIVER = 13020
EXPECTED_CC = (12, 0)
CU_STREAM_NON_BLOCKING = 1
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
MIN_FREE = 64 * 2**20


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(code: int, operation: str):
    if int(code) != 0:
        raise RuntimeError(f"{operation}:{int(code)}")


class CompilerFailure(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message); self.evidence = evidence


def _bind_nvrtc(library):
    p, i, z, cp = C.c_void_p, C.c_int, C.c_size_t, C.c_char_p
    signatures = {
        "nvrtcVersion": ([C.POINTER(i), C.POINTER(i)], i),
        "nvrtcCreateProgram": ([C.POINTER(p), cp, cp, i, C.POINTER(cp), C.POINTER(cp)], i),
        "nvrtcCompileProgram": ([p, i, C.POINTER(cp)], i),
        "nvrtcGetProgramLogSize": ([p, C.POINTER(z)], i),
        "nvrtcGetProgramLog": ([p, p], i),
        "nvrtcGetPTXSize": ([p, C.POINTER(z)], i),
        "nvrtcGetPTX": ([p, p], i),
        "nvrtcGetCUBINSize": ([p, C.POINTER(z)], i),
        "nvrtcGetCUBIN": ([p, p], i),
        "nvrtcDestroyProgram": ([C.POINTER(p)], i),
    }
    for name, (args, result) in signatures.items():
        function = getattr(library, name); function.argtypes = args; function.restype = result
    return signatures


def compile_one_program(source: bytes):
    """Compile one sm_120 program and retrieve log, PTX and cubin from it."""
    evidence = {"kind": "ph1_nvidia_n1_one_program_compile", "options": list(OPTIONS), "ledger": [], "artifacts": {}, "cudart_loaded": False}
    program = C.c_void_p(); failure = None
    if file_sha(NVRTC_DLL) != NVRTC_DLL_SHA or file_sha(NVRTC_HEADER) != NVRTC_HEADER_SHA:
        raise CompilerFailure("nvrtc_identity", evidence)
    library = C.CDLL(str(NVRTC_DLL)); signatures = _bind_nvrtc(library)
    evidence["nvrtc"] = {"path": str(NVRTC_DLL.resolve()), "sha256": NVRTC_DLL_SHA, "header_sha256": NVRTC_HEADER_SHA, "calling_convention": "cdecl", "abi_functions": sorted(signatures)}

    def row(op, code, **values):
        item = {"sequence": len(evidence["ledger"]), "op": op, "attempted": True, "code": int(code), **values}; evidence["ledger"].append(item); return item

    try:
        major, minor = C.c_int(), C.c_int(); code = library.nvrtcVersion(C.byref(major), C.byref(minor)); row("nvrtcVersion", code, major=major.value, minor=minor.value); check(code, "nvrtcVersion")
        if (major.value, minor.value) != (13, 3):
            raise RuntimeError(f"nvrtc_version:{major.value}.{minor.value}")
        code = library.nvrtcCreateProgram(C.byref(program), source, b"ph1_nvidia_n1.cu", 0, None, None)
        row("nvrtcCreateProgram", code, program_identity=int(program.value or 0), registered_owned=bool(program.value))
        if not program.value:
            check(code, "nvrtcCreateProgram"); raise RuntimeError("nvrtc_null_program")
        check(code, "nvrtcCreateProgram")
        encoded = [value.encode("ascii") for value in OPTIONS]; option_array = (C.c_char_p * len(encoded))(*encoded)
        code = library.nvrtcCompileProgram(program, len(encoded), option_array); row("nvrtcCompileProgram", code, program_identity=int(program.value))
        compile_code = int(code)
        log_size = C.c_size_t(); log_code = library.nvrtcGetProgramLogSize(program, C.byref(log_size)); row("nvrtcGetProgramLogSize", log_code, program_identity=int(program.value), bytes=log_size.value); check(log_code, "log_size")
        log_buffer = C.create_string_buffer(max(1, log_size.value)); log_code = library.nvrtcGetProgramLog(program, C.cast(log_buffer, C.c_void_p)); row("nvrtcGetProgramLog", log_code, program_identity=int(program.value), bytes=log_size.value); check(log_code, "log")
        log_bytes = bytes(log_buffer.raw[:log_size.value]); evidence["artifacts"]["log"] = {"bytes": len(log_bytes), "sha256": sha(log_bytes), "data": log_bytes}
        check(compile_code, "nvrtcCompileProgram")
        ptx_size = C.c_size_t(); code = library.nvrtcGetPTXSize(program, C.byref(ptx_size)); row("nvrtcGetPTXSize", code, program_identity=int(program.value), bytes=ptx_size.value); check(code, "ptx_size")
        if ptx_size.value <= 1:
            raise RuntimeError("empty_ptx")
        ptx_buffer = C.create_string_buffer(ptx_size.value); code = library.nvrtcGetPTX(program, C.cast(ptx_buffer, C.c_void_p)); row("nvrtcGetPTX", code, program_identity=int(program.value), bytes=ptx_size.value); check(code, "ptx")
        ptx = bytes(ptx_buffer.raw[:ptx_size.value]); evidence["artifacts"]["ptx"] = {"bytes": len(ptx), "sha256": sha(ptx), "data": ptx, "label": "PTX_from_sm_120_targeted_compile"}
        cubin_size = C.c_size_t(); code = library.nvrtcGetCUBINSize(program, C.byref(cubin_size)); row("nvrtcGetCUBINSize", code, program_identity=int(program.value), bytes=cubin_size.value); check(code, "cubin_size")
        if cubin_size.value == 0:
            raise RuntimeError("empty_cubin")
        cubin_buffer = (C.c_ubyte * cubin_size.value)(); code = library.nvrtcGetCUBIN(program, C.cast(cubin_buffer, C.c_void_p)); row("nvrtcGetCUBIN", code, program_identity=int(program.value), bytes=cubin_size.value); check(code, "cubin")
        cubin = bytes(cubin_buffer); evidence["artifacts"]["cubin"] = {"bytes": len(cubin), "sha256": sha(cubin), "data": cubin}
        if not cubin.startswith(b"\x7fELF"):
            raise RuntimeError("cubin_not_elf")
    except Exception as exc:
        failure = exc
    finally:
        if program.value:
            identity = int(program.value)
            try:
                code = library.nvrtcDestroyProgram(C.byref(program)); row("nvrtcDestroyProgram", code, program_identity=identity, pointer_after=int(program.value or 0)); check(code, "nvrtcDestroyProgram")
            except Exception as exc:
                failure = failure or exc
    if failure:
        evidence["status"] = "compile_failure"; raise CompilerFailure(str(failure), evidence) from failure
    evidence["status"] = "compile_positive"; return evidence


class CUuuid(C.Structure):
    _fields_ = [("bytes", C.c_char * 16)]


class DriverFailure(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message); self.evidence = evidence


class DriverBackend:
    """One-owner-thread direct Driver execution with explicit primary context."""
    def __init__(self):
        self.lib = None; self.context = C.c_void_p(); self.pushed = False; self.retained = False; self.primary_released = False
        self.stream = C.c_void_p(); self.module = C.c_void_p(); self.functions = {}; self.pinned = []; self.device = []
        self.context_ledger = []; self.ledger = []; self.cleanup_errors = []; self.resources = []; self.cubin_buffer = None; self.parameter_storage = []
        self.owner_tid = threading.get_native_id(); self.device_identity = {}; self.device = []

    def _bind(self):
        l = self.lib; p, u, i, z, d = C.c_void_p, C.c_uint, C.c_int, C.c_size_t, C.c_uint64
        table = {
            "cuInit": ([u], i), "cuDriverGetVersion": ([C.POINTER(i)], i), "cuDeviceGetCount": ([C.POINTER(i)], i), "cuDeviceGet": ([C.POINTER(i), i], i),
            "cuDeviceGetName": ([C.c_char_p, i, i], i), "cuDeviceGetUuid_v2": ([C.POINTER(CUuuid), i], i), "cuDeviceGetPCIBusId": ([C.c_char_p, i, i], i),
            "cuDeviceGetAttribute": ([C.POINTER(i), i, i], i), "cuDeviceTotalMem_v2": ([C.POINTER(z), i], i), "cuMemGetInfo_v2": ([C.POINTER(z), C.POINTER(z)], i),
            "cuCtxGetCurrent": ([C.POINTER(p)], i), "cuDevicePrimaryCtxGetState": ([i, C.POINTER(u), C.POINTER(i)], i), "cuDevicePrimaryCtxRetain": ([C.POINTER(p), i], i),
            "cuCtxPushCurrent_v2": ([p], i), "cuCtxPopCurrent_v2": ([C.POINTER(p)], i), "cuDevicePrimaryCtxRelease_v2": ([i], i),
            "cuStreamCreate": ([C.POINTER(p), u], i), "cuStreamSynchronize": ([p], i), "cuStreamDestroy_v2": ([p], i),
            "cuModuleLoadDataEx": ([C.POINTER(p), p, u, C.POINTER(i), C.POINTER(p)], i), "cuModuleGetFunction": ([C.POINTER(p), p, C.c_char_p], i), "cuModuleUnload": ([p], i),
            "cuMemHostAlloc": ([C.POINTER(p), z, u], i), "cuMemFreeHost": ([p], i), "cuMemAlloc_v2": ([C.POINTER(d), z], i), "cuMemFree_v2": ([d], i),
            "cuMemcpyHtoDAsync_v2": ([d, p, z, p], i), "cuMemcpyDtoHAsync_v2": ([p, d, z, p], i), "cuMemsetD8Async": ([d, C.c_ubyte, z, p], i),
            "cuLaunchKernel": ([p, u,u,u, u,u,u, u, p, C.POINTER(p), C.POINTER(p)], i),
        }
        for name, (args, result) in table.items():
            function = getattr(l, name); function.argtypes = args; function.restype = result
        self.abi = {name: [str(arg) for arg in args] for name, (args, _) in table.items()}

    def _load(self):
        if NVCUDA_DLL.stat().st_size != NVCUDA_BYTES or file_sha(NVCUDA_DLL) != NVCUDA_DLL_SHA:
            raise RuntimeError("nvcuda_identity")
        self.lib = C.WinDLL(str(NVCUDA_DLL), use_last_error=True, winmode=0x00000800); self._bind()
        kernel32 = C.WinDLL("kernel32.dll", use_last_error=True); kernel32.GetModuleFileNameW.argtypes = [C.c_void_p, C.c_wchar_p, C.c_uint]; kernel32.GetModuleFileNameW.restype = C.c_uint
        buffer = C.create_unicode_buffer(32768); length = kernel32.GetModuleFileNameW(C.c_void_p(self.lib._handle), buffer, len(buffer))
        if not length:
            raise RuntimeError("loaded_module_path")
        loaded = Path(buffer.value)
        if os.path.normcase(str(loaded.resolve())) != os.path.normcase(str(NVCUDA_DLL.resolve())) or file_sha(loaded) != NVCUDA_DLL_SHA:
            raise RuntimeError("loaded_module_identity")
        self.ledger.append({"op": "driver_load", "path": str(loaded.resolve()), "bytes": NVCUDA_BYTES, "sha256": NVCUDA_DLL_SHA, "calling_convention": "WinDLL", "winmode": 0x800, "cudart_loaded": False})

    def _ctx(self, op, code, **values):
        self.context_ledger.append({"sequence": len(self.context_ledger), "op": op, "code": int(code), "owner_tid": self.owner_tid, **values})

    def _ordinary(self, op, code, **values):
        self.ledger.append({"sequence": len(self.ledger), "op": op, "code": int(code), "owner_tid": self.owner_tid, **values})

    def _sample_device(self, stage):
        sample = host_sample(stage, "attempted"); free, total = C.c_size_t(), C.c_size_t(); code = self.lib.cuMemGetInfo_v2(C.byref(free), C.byref(total))
        sample.update({"device_query_state": "attempted", "device_free_bytes": int(free.value), "device_total_bytes": int(total.value), "cuMemGetInfo_return": int(code)}); self.resources.append(sample); check(code, "cuMemGetInfo_v2")
        return sample

    def _identity(self):
        count = C.c_int(); check(self.lib.cuDeviceGetCount(C.byref(count)), "device_count")
        if count.value != 1:
            raise RuntimeError(f"device_count:{count.value}")
        device = C.c_int(); check(self.lib.cuDeviceGet(C.byref(device), 0), "device_get")
        name = C.create_string_buffer(256); pci = C.create_string_buffer(32); uuid = CUuuid(); driver = C.c_int(); major = C.c_int(); minor = C.c_int(); total = C.c_size_t()
        check(self.lib.cuDeviceGetName(name, len(name), device.value), "device_name"); check(self.lib.cuDeviceGetPCIBusId(pci, len(pci), device.value), "device_pci"); check(self.lib.cuDeviceGetUuid_v2(C.byref(uuid), device.value), "device_uuid"); check(self.lib.cuDriverGetVersion(C.byref(driver)), "driver_version")
        check(self.lib.cuDeviceGetAttribute(C.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device.value), "cc_major"); check(self.lib.cuDeviceGetAttribute(C.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device.value), "cc_minor"); check(self.lib.cuDeviceTotalMem_v2(C.byref(total), device.value), "total_mem")
        identity = {"ordinal": 0, "device": device.value, "name": name.value.decode(), "pci": pci.value.decode(), "uuid_hex": C.string_at(C.byref(uuid), C.sizeof(uuid)).hex(), "driver_version": driver.value, "runtime_version": "not_applicable_driver_api_only", "compute_capability": [major.value, minor.value], "total_memory": total.value, "cudart_loaded": False}
        if identity["name"] != EXPECTED_NAME or identity["pci"] != EXPECTED_PCI or identity["driver_version"] != EXPECTED_DRIVER or tuple(identity["compute_capability"]) != EXPECTED_CC:
            raise RuntimeError("device_identity")
        self.device_identity = identity; self._ordinary("identity", 0, identity=identity); return device.value

    def _acquire_stream(self):
        out = C.c_void_p(); row = {"op": "stream_create", "requested_flags": CU_STREAM_NON_BLOCKING, "returned": 0, "registered_owned": False, "code": None, "owner_tid": self.owner_tid}; self.ledger.append(row)
        code = self.lib.cuStreamCreate(C.byref(out), CU_STREAM_NON_BLOCKING); row.update(code=int(code), returned=int(out.value or 0))
        if out.value: self.stream = out; row["registered_owned"] = True
        check(code, "stream_create")
        if not out.value: raise RuntimeError("stream_null")

    def _acquire_module(self, cubin: bytes):
        self.cubin_buffer = (C.c_ubyte * len(cubin)).from_buffer_copy(cubin); out = C.c_void_p(); row = {"op": "module_load", "bytes": len(cubin), "sha256": sha(cubin), "numOptions": 0, "options": None, "optionValues": None, "returned": 0, "registered_owned": False, "code": None, "owner_tid": self.owner_tid}; self.ledger.append(row)
        code = self.lib.cuModuleLoadDataEx(C.byref(out), C.cast(self.cubin_buffer, C.c_void_p), 0, None, None); row.update(code=int(code), returned=int(out.value or 0))
        if out.value: self.module = out; row["registered_owned"] = True
        check(code, "module_load")
        if not out.value: raise RuntimeError("module_null")
        for entry in ("q5_linear", "bf16_lut_activation"):
            function = C.c_void_p(); code = self.lib.cuModuleGetFunction(C.byref(function), self.module, entry.encode()); self._ordinary("module_get_function", code, entry=entry, pointer=int(function.value or 0)); check(code, "module_get_function")
            if not function.value: raise RuntimeError("function_null")
            self.functions[entry] = function

    def _allocations(self):
        for name, size in BUFFER_TABLE:
            out = C.c_void_p(); row = {"op": "pinned_allocate", "name": name, "bytes": size, "flags": 0, "returned": 0, "registered_owned": False, "code": None, "owner_tid": self.owner_tid}; self.ledger.append(row)
            code = self.lib.cuMemHostAlloc(C.byref(out), size, 0); row.update(code=int(code), returned=int(out.value or 0))
            if out.value: self.pinned.append((name, int(out.value), size)); row["registered_owned"] = True
            check(code, "pinned_allocate")
            if not out.value: raise RuntimeError("pinned_null")
        for name, size in BUFFER_TABLE:
            out = C.c_uint64(); row = {"op": "device_allocate", "name": name, "bytes": size, "returned": 0, "registered_owned": False, "code": None, "owner_tid": self.owner_tid}; self.ledger.append(row)
            code = self.lib.cuMemAlloc_v2(C.byref(out), size); row.update(code=int(code), returned=int(out.value))
            if out.value: self.device.append((name, int(out.value), size)); row["registered_owned"] = True
            check(code, "device_allocate")
            if not out.value: raise RuntimeError("device_null")
        if len({x[1] for x in self.pinned}) != 14 or len({x[1] for x in self.device}) != 14:
            raise RuntimeError("allocation_alias")

    def _launch(self, function_name, label, names, grid, block, pointers):
        variables = [C.c_uint64(pointers[name]) for name in names]
        params = (C.c_void_p * len(variables))(*(C.cast(C.byref(value), C.c_void_p) for value in variables))
        self.parameter_storage.append((variables, params))
        code = self.lib.cuLaunchKernel(self.functions[function_name], *grid, *block, 0, self.stream, params, None)
        self._ordinary("launch", code, label=label, function=function_name, grid=list(grid), block=list(block), sharedMemBytes=0, stream=int(self.stream.value), extra=None, argument_names=list(names), argument_values=[int(value.value) for value in variables], parameter_slots=[int(params[index]) for index in range(len(params))]); check(code, f"launch:{label}")

    def _release_ordinary(self):
        rows = []
        def attempt(name, function):
            row = {"op": "release", "attempt_index": len(rows), "name": name, "attempted": True, "owned_before": True, "code": None, "exception": None, "owned_after": True, "owner_tid": self.owner_tid}; rows.append(row); self.ledger.append(row)
            try:
                code = int(function()); row["code"] = code
                if code == 0: row["owned_after"] = False
                else: self.cleanup_errors.append(f"{name}:code:{code}")
            except Exception as exc:
                row["exception"] = f"{type(exc).__name__}:{exc}"; self.cleanup_errors.append(f"{name}:{row['exception']}")
            return row
        remaining_device = []
        for item in reversed(self.device):
            name, pointer, _ = item; row = attempt("device:" + name, lambda p=pointer: self.lib.cuMemFree_v2(C.c_uint64(p)))
            if row["owned_after"]: remaining_device.append(item)
        self.device = list(reversed(remaining_device))
        remaining_pinned = []
        for item in reversed(self.pinned):
            name, pointer, _ = item; row = attempt("pinned:" + name, lambda p=pointer: self.lib.cuMemFreeHost(C.c_void_p(p)))
            if row["owned_after"]: remaining_pinned.append(item)
        self.pinned = list(reversed(remaining_pinned))
        if self.module.value:
            pointer = int(self.module.value); row = attempt("module", lambda p=pointer: self.lib.cuModuleUnload(C.c_void_p(p)))
            if not row["owned_after"]: self.module = C.c_void_p()
        if self.stream.value:
            pointer = int(self.stream.value); row = attempt("stream", lambda p=pointer: self.lib.cuStreamDestroy_v2(C.c_void_p(p)))
            if not row["owned_after"]: self.stream = C.c_void_p()
        return rows

    def _finish_context(self, device):
        if self.pushed:
            popped = C.c_void_p(); code = self.lib.cuCtxPopCurrent_v2(C.byref(popped)); self._ctx("pop", code, popped=int(popped.value or 0), owned=int(self.context.value or 0));
            if int(code) or popped.value != self.context.value: self.cleanup_errors.append(f"pop:{int(code)}:{int(popped.value or 0)}")
            else: self.pushed = False
            restored = C.c_void_p(); code = self.lib.cuCtxGetCurrent(C.byref(restored)); self._ctx("restored_current", code, pointer=int(restored.value or 0));
            if int(code) or restored.value: self.cleanup_errors.append(f"restore:{int(code)}:{int(restored.value or 0)}")
        if self.retained:
            code = self.lib.cuDevicePrimaryCtxRelease_v2(device); self._ctx("primary_release", code, context=int(self.context.value or 0));
            if int(code): self.cleanup_errors.append(f"primary_release:{int(code)}")
            else: self.retained = False; self.primary_released = True

    def run(self, records, input_bytes: bytes, lut: bytes, cubin: bytes, initial_resources):
        evidence = {"ledger": self.ledger, "context_ledger": self.context_ledger, "resources": self.resources, "outputs": {}, "cleanup_errors": self.cleanup_errors, "forbidden_calls": {name: 0 for name in ("cudart", "cupy", "default_stream", "managed", "pool", "peer", "graph", "event")}, "runtime_version": "not_applicable_driver_api_only", "cudart_loaded": False}
        self.resources.extend(initial_resources); failure = None; device = None; ordinary_released = False
        try:
            self._load(); check(self.lib.cuInit(0), "cuInit"); device = self._identity()
            prior = C.c_void_p(); code = self.lib.cuCtxGetCurrent(C.byref(prior)); self._ctx("prior_current", code, pointer=int(prior.value or 0)); check(code, "prior_current")
            if prior.value: raise RuntimeError("prior_context_nonnull")
            flags, active = C.c_uint(), C.c_int(); code = self.lib.cuDevicePrimaryCtxGetState(device, C.byref(flags), C.byref(active)); self._ctx("primary_state", code, flags=flags.value, active=active.value); check(code, "primary_state")
            context = C.c_void_p(); code = self.lib.cuDevicePrimaryCtxRetain(C.byref(context), device); self._ctx("retain", code, context=int(context.value or 0), registered_owned=bool(context.value))
            if context.value: self.context = context; self.retained = True
            check(code, "primary_retain")
            if not context.value: raise RuntimeError("primary_context_null")
            code = self.lib.cuCtxPushCurrent_v2(self.context); self._ctx("push", code, context=int(self.context.value)); check(code, "context_push"); self.pushed = True
            current = C.c_void_p(); code = self.lib.cuCtxGetCurrent(C.byref(current)); self._ctx("post_push_current", code, pointer=int(current.value or 0), owned=int(self.context.value)); check(code, "post_push_current")
            if current.value != self.context.value: raise RuntimeError("current_context_identity")
            first = self._sample_device("post_context_push")
            if first["device_free_bytes"] < MIN_FREE: raise RuntimeError("device_start_free")
            self._acquire_stream(); self._acquire_module(cubin); self._sample_device("post_module_stream_preallocation")
            self._allocations(); self._sample_device("post_allocations")
            pinned = {name: pointer for name, pointer, _ in self.pinned}; device_ptr = {name: pointer for name, pointer, _ in self.device}; sizes = dict(BUFFER_TABLE)
            payload = {"gate_record": records["gate"], "up_record": records["up"], "down_record": records["down"], "natural_input": input_bytes, "silu_lut": lut}
            for name, data in payload.items():
                C.memmove(pinned[name], data, len(data)); self._ordinary("pinned_write", 0, name=name, bytes=len(data), sha256=sha(data), pointer=pinned[name])
            for name in ("gate", "up", "silu", "activation", "down"):
                code = self.lib.cuMemsetD8Async(C.c_uint64(device_ptr[name]), 0xFF, sizes[name], self.stream); self._ordinary("memset", code, name=name, value=255, bytes=sizes[name], stream=int(self.stream.value)); check(code, "memset")
            for name in ("gate_counters", "up_counters", "activation_counters", "down_counters"):
                code = self.lib.cuMemsetD8Async(C.c_uint64(device_ptr[name]), 0, sizes[name], self.stream); self._ordinary("memset", code, name=name, value=0, bytes=sizes[name], stream=int(self.stream.value)); check(code, "memset")
            for name in ("gate_record", "up_record", "down_record", "natural_input", "silu_lut"):
                code = self.lib.cuMemcpyHtoDAsync_v2(C.c_uint64(device_ptr[name]), C.c_void_p(pinned[name]), sizes[name], self.stream); self._ordinary("h2d", code, name=name, bytes=sizes[name], source=pinned[name], destination=device_ptr[name], stream=int(self.stream.value)); check(code, "h2d")
            self._sample_device("post_memset_h2d")
            launch_by_label = {name: (grid, block) for name, grid, block in LAUNCHES}
            for label, names in ARGUMENT_MAPS:
                function = "bf16_lut_activation" if label == "bf16_lut_activation" else "q5_linear"; grid, block = launch_by_label[label]; self._launch(function, label, names, grid, block, device_ptr)
            self._sample_device("post_launches_queued")
            for name in ("gate", "up", "silu", "activation", "down", "gate_counters", "up_counters", "activation_counters", "down_counters"):
                code = self.lib.cuMemcpyDtoHAsync_v2(C.c_void_p(pinned[name]), C.c_uint64(device_ptr[name]), sizes[name], self.stream); self._ordinary("d2h", code, name=name, bytes=sizes[name], source=device_ptr[name], destination=pinned[name], stream=int(self.stream.value)); check(code, "d2h")
            code = self.lib.cuStreamSynchronize(self.stream); self._ordinary("stream_synchronize", code, stream=int(self.stream.value)); check(code, "stream_synchronize")
            for name in ("gate", "up", "silu", "activation", "down", "gate_counters", "up_counters", "activation_counters", "down_counters"):
                data = C.string_at(pinned[name], sizes[name]); evidence["outputs"][name] = data.hex(); self._ordinary("pinned_read", 0, name=name, bytes=len(data), sha256=sha(data), pointer=pinned[name], after_sync=True)
            self._sample_device("post_d2h_sync")
            releases = self._release_ordinary(); ordinary_released = True
            if len(releases) != 30: raise RuntimeError(f"release_cardinality:{len(releases)}")
            self._sample_device("post_ordinary_releases_pre_pop")
        except Exception as exc:
            failure = exc
        finally:
            if self.lib:
                if not ordinary_released: self._release_ordinary()
                if device is not None: self._finish_context(device)
                try:
                    final_host = host_sample("post_context_release", "not_attempted"); final_host["driver_context_calls_after_primary_release"] = 0 if self.primary_released else None; final_host["primary_release_succeeded"] = self.primary_released; self.resources.append(final_host)
                except Exception as exc:
                    self.cleanup_errors.append(f"post_context_resource:{type(exc).__name__}:{exc}")
        evidence.update({"identity": self.device_identity, "owner_tid": self.owner_tid, "abi": getattr(self, "abi", {}), "loaded_driver": bool(self.lib), "live_owned_resources": len(self.device) + len(self.pinned) + int(bool(self.module.value)) + int(bool(self.stream.value)) + int(self.retained) + int(self.pushed), "primary_released": self.primary_released})
        if failure or self.cleanup_errors:
            raise DriverFailure(str(failure or "cleanup_failure"), evidence) from failure
        return evidence
