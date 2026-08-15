#!/usr/bin/env python3
"""Standalone HET-NEXT-CAP0 physical capability runner (execution closed)."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import struct
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
REPORTS = ROOT / "reports/streamq5_moe"
RUN_DIR = ROOT / "reports/runs/streamq5_moe/het_next_cap0r4_dual_device_cohabitation"
RESULT = RUN_DIR / "cap0r4_result.json"
COMMIT = RUN_DIR / "cap0r4_commit.json"
FAILURE = RUN_DIR / "cap0r4_failure.json"
LOCK = REPORTS / "het_next_cap0r4_runner_lock.json"
VERIFIER_LOCK = REPORTS / "het_next_cap0r4_verifier_lock.json"
PREREG = REPORTS / "HET_NEXT_CAP0R4_DUAL_DEVICE_COHABITATION_PREREGISTRATION_2026-08-13.md"
DESIGN = REPORTS / "HET_NEXT_CAP0R4_STATIC_PREFLIGHT_DESIGN_2026-08-13.md"
PROTOCOL_PATH = SCRIPTS / "het_next_cap0r4_protocol.py"
KERNEL_PATH = SCRIPTS / "het_next_cap0r4_kernels.py"
VERIFIER = SCRIPTS / "verify_het_next_cap0r4_dual_device_cohabitation.py"
PREFLIGHT = SCRIPTS / "preflight_het_next_cap0r4_static.py"

SEED = 0x4845544E45585430
WORD_COUNT = 1024
BUFFER_BYTES = 4096
REPETITIONS = 3
THREAD_LPS = {"coordinator": 0, "intel": 2, "nvidia": 4, "monitor": 6}
COUNTERS = (r"\Memory\Page Reads/sec", r"\Memory\Pages Input/sec", r"\Paging File(_Total)\% Usage")
INPUT_SHA256 = "a9d32afd712f6ac80ef7739b11c2baa59e4f84c2067e20307f175de4e8a1acca"
INTEL_SHA256 = "c83e434be87333bc6bf15d3f0ee492c3e3f9d65b847902bea55310165a42923f"
NVIDIA_SHA256 = "f07c3d87d952d1dc82c65d90f467af87426c1658267b7d94f359122e73eafd5f"
ACK_PENDING = "PENDING_INDEPENDENT_SOURCE_AUDIT"
_EMERGENCY_CLEANUP = None
_FAILURE_EVIDENCE = {}


def available_ram_bytes():
    class MEMORYSTATUSEX(C.Structure):
        _fields_ = [("length", C.c_ulong), ("memory_load", C.c_ulong), ("total_phys", C.c_ulonglong), ("avail_phys", C.c_ulonglong), ("total_page", C.c_ulonglong), ("avail_page", C.c_ulonglong), ("total_virtual", C.c_ulonglong), ("avail_virtual", C.c_ulonglong), ("avail_extended", C.c_ulonglong)]
    value = MEMORYSTATUSEX(); value.length = C.sizeof(value)
    fn=C.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx;fn.argtypes=[C.POINTER(MEMORYSTATUSEX)];fn.restype=C.c_int
    if not fn(C.byref(value)): raise OSError(C.get_last_error(), "GlobalMemoryStatusEx")
    return int(value.avail_phys)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def input_words():
    words = [SEED & 0xFFFFFFFF]
    for _ in range(1, WORD_COUNT): words.append((1664525 * words[-1] + 1013904223) & 0xFFFFFFFF)
    return words


def expected_words(device, words):
    if device == "intel":
        return [((((v ^ 0xA5A5A5A5) << 7) | ((v ^ 0xA5A5A5A5) >> 25)) + 0x3C6EF372) & 0xFFFFFFFF for v in words]
    return [((((((v + 0x9E3779B9) & 0xFFFFFFFF) >> 11) | (((v + 0x9E3779B9) & 0xFFFFFFFF) << 21)) & 0xFFFFFFFF) ^ 0xC3C3C3C3) for v in words]


def packed_sha(words): return hashlib.sha256(struct.pack("<1024I", *words)).hexdigest()


class Ledger:
    def __init__(self): self.rows = []; self._next = 0; self._lock = threading.Lock()
    def create(self, kind, owner, detail=None):
        with self._lock:
            self._next += 1; row = {"id": self._next, "kind": kind, "owner": owner, "creator_thread_id": threading.get_native_id(), "detail": detail or {}, "create_qpc_ns": time.perf_counter_ns(), "release_attempts": 0, "release_code": None, "final_state": "live"}; self.rows.append(row); return row
    def release(self, row, function):
        with self._lock: row["release_attempts"] += 1; row["release_attempt_qpc_ns"] = time.perf_counter_ns()
        try:
            code = function(); code = 0 if code is None else int(code); row["release_code"] = code
            row["final_state"] = "released" if code == 0 else "release_error"
        except BaseException as exc:
            row["release_code"] = f"{type(exc).__name__}: {exc}"; row["final_state"] = "release_exception"; raise


def process_identity():
    k = C.WinDLL("kernel32", use_last_error=True)
    class FILETIME(C.Structure): _fields_ = [("low", C.c_ulong), ("high", C.c_ulong)]
    k.GetCurrentProcess.restype = C.c_void_p; k.GetProcessTimes.argtypes = [C.c_void_p, C.POINTER(FILETIME), C.POINTER(FILETIME), C.POINTER(FILETIME), C.POINTER(FILETIME)]; k.GetProcessTimes.restype = C.c_int
    created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
    if not k.GetProcessTimes(k.GetCurrentProcess(), C.byref(created), C.byref(exited), C.byref(kernel), C.byref(user)): raise OSError(C.get_last_error(), "GetProcessTimes")
    return {"pid": os.getpid(), "create_filetime": (created.high << 32) | created.low, "argv": list(sys.argv)}


def pin_and_identify(logical_processor):
    """Pin group 0, then parse relation-core records to bind LP to a physical core."""
    from ctypes import wintypes
    k = C.WinDLL("kernel32", use_last_error=True)
    class GROUP_AFFINITY(C.Structure): _fields_ = [("Mask", C.c_size_t), ("Group", C.c_ushort), ("Reserved", C.c_ushort * 3)]
    k.GetCurrentThread.restype = C.c_void_p; k.SetThreadGroupAffinity.argtypes = [C.c_void_p, C.POINTER(GROUP_AFFINITY), C.POINTER(GROUP_AFFINITY)]; k.SetThreadGroupAffinity.restype = wintypes.BOOL
    target, previous = GROUP_AFFINITY(1 << logical_processor, 0, (C.c_ushort * 3)(0, 0, 0)), GROUP_AFFINITY()
    if not k.SetThreadGroupAffinity(k.GetCurrentThread(), C.byref(target), C.byref(previous)): raise OSError(C.get_last_error(), "SetThreadGroupAffinity")
    k.GetCurrentProcessorNumberEx.argtypes = [C.c_void_p]; k.GetCurrentProcessorNumberEx.restype = None
    class PROCESSOR_NUMBER(C.Structure): _fields_ = [("Group", C.c_ushort), ("Number", C.c_ubyte), ("Reserved", C.c_ubyte)]
    current = PROCESSOR_NUMBER(); k.GetCurrentProcessorNumberEx(C.byref(current))
    if current.Group != 0 or current.Number != logical_processor: raise RuntimeError("affinity_not_observed")
    k.GetLogicalProcessorInformationEx.argtypes = [C.c_int, C.c_void_p, C.POINTER(wintypes.DWORD)]; k.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    size = wintypes.DWORD(); k.GetLogicalProcessorInformationEx(0, None, C.byref(size)); buffer = C.create_string_buffer(size.value)
    if not k.GetLogicalProcessorInformationEx(0, buffer, C.byref(size)): raise OSError(C.get_last_error(), "GetLogicalProcessorInformationEx")
    offset, core_index = 0, None
    while offset < size.value:
        relationship, record_size = struct.unpack_from("<II", buffer.raw, offset)
        if relationship == 0:
            group_count = struct.unpack_from("<H", buffer.raw, offset + 30)[0]
            for group_slot in range(group_count):
                mask, group = struct.unpack_from("<QH", buffer.raw, offset + 32 + 16 * group_slot)
                if group == 0 and mask & (1 << logical_processor): core_index = offset
        offset += record_size
    if core_index is None: raise RuntimeError("physical_core_not_found")
    return {"logical_processor": logical_processor, "processor_group": 0, "physical_core_record_offset": core_index, "thread_id": threading.get_native_id(), "start_qpc_ns": time.perf_counter_ns()}


def intel_backend(ledger):
    from ctypes import wintypes
    from het_next_cap0r4_kernels import INTEL_BUILD_OPTIONS, INTEL_SOURCE
    CL_GPU = 4; CL_CONTEXT_PLATFORM = 0x1084; CL_QUEUE_PROFILING_ENABLE = 2
    CL_DEVICE_NAME = 0x102B; CL_DEVICE_VENDOR = 0x102C; CL_DRIVER_VERSION = 0x102D; CL_DEVICE_VERSION = 0x102F; CL_DEVICE_OPENCL_C_VERSION = 0x103D; CL_DEVICE_EXTENSIONS = 0x1030
    CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F; CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010; CL_DEVICE_ADDRESS_BITS = 0x100D; CL_DEVICE_QUEUE_PROPERTIES = 0x102A; CL_DEVICE_PCI_BUS_INFO_KHR = 0x410F
    USM_INFO = {"host": 0x4190, "device": 0x4191, "single_shared": 0x4192, "cross_shared": 0x4193, "system_shared": 0x4194}
    CL_PROGRAM_BINARY_SIZES = 0x1165; CL_PROGRAM_BINARIES = 0x1166; CL_BUILD_LOG = 0x1183
    CL_ALLOC_TYPE = 0x419A; CL_ALLOC_BASE = 0x419B; CL_ALLOC_SIZE = 0x419C; CL_MEM_TYPE_HOST = 0x4197
    cl = C.WinDLL("OpenCL.dll"); cleanup = []; log = ledger
    def bind(name, args, result=C.c_int): fn = getattr(cl, name); fn.argtypes = args; fn.restype = result; return fn
    get_platforms = bind("clGetPlatformIDs", [C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)])
    get_platform_info = bind("clGetPlatformInfo", [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)])
    get_devices = bind("clGetDeviceIDs", [C.c_void_p, C.c_ulonglong, C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)])
    get_device_info = bind("clGetDeviceInfo", [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)])
    create_context = bind("clCreateContext", [C.POINTER(C.c_ssize_t), C.c_uint, C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.POINTER(C.c_int)], C.c_void_p)
    create_queue = bind("clCreateCommandQueue", [C.c_void_p, C.c_void_p, C.c_ulonglong, C.POINTER(C.c_int)], C.c_void_p)
    create_program = bind("clCreateProgramWithSource", [C.c_void_p, C.c_uint, C.POINTER(C.c_char_p), C.POINTER(C.c_size_t), C.POINTER(C.c_int)], C.c_void_p)
    build_program = bind("clBuildProgram", [C.c_void_p, C.c_uint, C.POINTER(C.c_void_p), C.c_char_p, C.c_void_p, C.c_void_p])
    get_build = bind("clGetProgramBuildInfo", [C.c_void_p, C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)])
    get_program = bind("clGetProgramInfo", [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)])
    create_kernel = bind("clCreateKernel", [C.c_void_p, C.c_char_p, C.POINTER(C.c_int)], C.c_void_p)
    set_arg = bind("clSetKernelArg", [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p])
    enqueue = bind("clEnqueueNDRangeKernel", [C.c_void_p, C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), C.c_uint, C.c_void_p, C.POINTER(C.c_void_p)])
    finish = bind("clFinish", [C.c_void_p]); get_profile = bind("clGetEventProfilingInfo", [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)])
    extension = bind("clGetExtensionFunctionAddressForPlatform", [C.c_void_p, C.c_char_p], C.c_void_p)
    releases = {name: bind(name, [C.c_void_p]) for name in ("clReleaseEvent", "clReleaseKernel", "clReleaseProgram", "clReleaseCommandQueue", "clReleaseContext")}
    def check(code, name):
        if int(code) != 0: raise RuntimeError(f"{name}:{int(code)}")
    def text_info(obj, fn, param):
        size = C.c_size_t(); check(fn(obj, param, 0, None, C.byref(size)), "info_size"); data = C.create_string_buffer(size.value); check(fn(obj, param, size.value, data, None), "info"); return data.value.decode(errors="replace")
    def scalar_info(device, param, ctype):
        value = ctype(); check(get_device_info(device, param, C.sizeof(value), C.byref(value), None), "device_scalar"); return int(value.value)
    count = C.c_uint(); check(get_platforms(0, None, C.byref(count)), "platform_count"); platforms = (C.c_void_p * count.value)(); check(get_platforms(count, platforms, None), "platforms")
    candidates = []
    for platform_index, platform in enumerate(platforms):
        dc = C.c_uint(); code = get_devices(platform, CL_GPU, 0, None, C.byref(dc))
        if code != 0: continue
        devices = (C.c_void_p * dc.value)(); check(get_devices(platform, CL_GPU, dc, devices, None), "devices")
        for device in devices:
            vendor, name = text_info(device, get_device_info, CL_DEVICE_VENDOR), text_info(device, get_device_info, CL_DEVICE_NAME)
            extensions = text_info(device, get_device_info, CL_DEVICE_EXTENSIONS).split()
            if "Intel" in vendor and "Arc" in name and "cl_intel_unified_shared_memory" in extensions: candidates.append((platform_index, platform, device, vendor, name, extensions))
    if len(candidates) != 1: raise RuntimeError(f"intel_candidate_count:{len(candidates)}")
    platform_index, platform, device, vendor, name, extensions = candidates[0]; error = C.c_int(); devices1 = (C.c_void_p * 1)(device)
    context = queue = program = kernel = None; pointer = None; mem_free = None
    class PCI_BUS_INFO(C.Structure): _fields_ = [("domain", C.c_uint), ("bus", C.c_uint), ("device", C.c_uint), ("function", C.c_uint)]
    pci = PCI_BUS_INFO()
    if "cl_khr_pci_bus_info" not in extensions or get_device_info(device, CL_DEVICE_PCI_BUS_INFO_KHR, C.sizeof(pci), C.byref(pci), None) != 0: raise RuntimeError("intel_pci_identity_unavailable")
    identity = {"enumerated_platform_count": count.value, "eligible_device_count": len(candidates), "chosen_platform_index": platform_index, "platform_name": text_info(platform, get_platform_info, 0x0902), "platform_vendor": text_info(platform, get_platform_info, 0x0903), "platform_version": text_info(platform, get_platform_info, 0x0901), "vendor": vendor, "name": name, "pci": {"domain": pci.domain, "bus": pci.bus, "device": pci.device, "function": pci.function}, "driver": text_info(device, get_device_info, CL_DRIVER_VERSION), "device_version": text_info(device, get_device_info, CL_DEVICE_VERSION), "opencl_c": text_info(device, get_device_info, CL_DEVICE_OPENCL_C_VERSION), "extensions": extensions, "global_mem": scalar_info(device, CL_DEVICE_GLOBAL_MEM_SIZE, C.c_ulonglong), "max_alloc": scalar_info(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, C.c_ulonglong), "address_bits": scalar_info(device, CL_DEVICE_ADDRESS_BITS, C.c_uint), "queue_properties": scalar_info(device, CL_DEVICE_QUEUE_PROPERTIES, C.c_ulonglong), "usm_capabilities": {key: scalar_info(device, value, C.c_ulonglong) for key, value in USM_INFO.items()}}
    def ext(name, restype, *args):
        address = extension(platform, name.encode())
        if not address: raise RuntimeError(f"missing_extension_function:{name}")
        return C.WINFUNCTYPE(restype, *args)(address)
    try:
        props = (C.c_ssize_t * 3)(CL_CONTEXT_PLATFORM, int(platform), 0); context = create_context(props, 1, devices1, None, None, C.byref(error)); check(error.value, "create_context"); cleanup.append((log.create("intel_context", "intel", {"handle":int(context)}), lambda: releases["clReleaseContext"](context)))
        queue = create_queue(context, device, CL_QUEUE_PROFILING_ENABLE, C.byref(error)); check(error.value, "create_queue"); cleanup.append((log.create("intel_queue", "intel", {"handle":int(queue),"in_order": True, "profiling": True}), lambda: releases["clReleaseCommandQueue"](queue)))
        source = INTEL_SOURCE.encode(); source_ptr = C.c_char_p(source); source_len = C.c_size_t(len(source)); program = create_program(context, 1, C.byref(source_ptr), C.byref(source_len), C.byref(error)); check(error.value, "create_program"); cleanup.append((log.create("intel_program", "intel"), lambda: releases["clReleaseProgram"](program)))
        build_code = build_program(program, 1, devices1, INTEL_BUILD_OPTIONS.encode(), None, None); log_size = C.c_size_t(); get_build(program, device, CL_BUILD_LOG, 0, None, C.byref(log_size)); log_buffer = C.create_string_buffer(max(1, log_size.value)); get_build(program, device, CL_BUILD_LOG, log_size.value, log_buffer, None); build_log = log_buffer.value.decode(errors="replace"); check(build_code, "build_program")
        binary_size = C.c_size_t(); check(get_program(program, CL_PROGRAM_BINARY_SIZES, C.sizeof(binary_size), C.byref(binary_size), None), "binary_size"); binary = C.create_string_buffer(binary_size.value); binary_ptrs = (C.c_void_p * 1)(C.addressof(binary)); check(get_program(program, CL_PROGRAM_BINARIES, C.sizeof(binary_ptrs), binary_ptrs, None), "binary")
        kernel = create_kernel(program, b"cap0_intel_bijection", C.byref(error)); check(error.value, "create_kernel"); cleanup.append((log.create("intel_kernel", "intel", {"handle":int(kernel)}), lambda: releases["clReleaseKernel"](kernel)))
        host_alloc = ext("clHostMemAllocINTEL", C.c_void_p, C.c_void_p, C.POINTER(C.c_longlong), C.c_size_t, C.c_uint, C.POINTER(C.c_int)); mem_free = ext("clMemFreeINTEL", C.c_int, C.c_void_p, C.c_void_p); set_pointer = ext("clSetKernelArgMemPointerINTEL", C.c_int, C.c_void_p, C.c_uint, C.c_void_p); get_alloc = ext("clGetMemAllocInfoINTEL", C.c_int, C.c_void_p, C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t))
        pointer = host_alloc(context, None, BUFFER_BYTES, 4096, C.byref(error)); check(error.value, "host_usm_alloc")
        if not pointer: raise MemoryError("null_host_usm")
        cleanup.append((log.create("intel_host_usm", "intel", {"bytes": BUFFER_BYTES, "alignment": 4096}), lambda: mem_free(context, pointer)))
        allocation_type = C.c_uint(); allocation_base = C.c_void_p(); allocation_size = C.c_size_t()
        check(get_alloc(context, pointer, CL_ALLOC_TYPE, C.sizeof(allocation_type), C.byref(allocation_type), None), "alloc_type"); check(get_alloc(context, pointer, CL_ALLOC_BASE, C.sizeof(allocation_base), C.byref(allocation_base), None), "alloc_base"); check(get_alloc(context, pointer, CL_ALLOC_SIZE, C.sizeof(allocation_size), C.byref(allocation_size), None), "alloc_size")
        if allocation_type.value != CL_MEM_TYPE_HOST or allocation_base.value != pointer or allocation_size.value != BUFFER_BYTES: raise RuntimeError("host_usm_identity")
        check(set_pointer(kernel, 0, pointer), "set_usm_pointer"); word_count = C.c_uint(WORD_COUNT); check(set_arg(kernel, 1, C.sizeof(word_count), C.byref(word_count)), "set_count")
        init = {**identity, "source_sha256": hashlib.sha256(INTEL_SOURCE.encode()).hexdigest(), "build_options": INTEL_BUILD_OPTIONS, "build_log": build_log, "binary_bytes": binary_size.value, "binary_sha256": hashlib.sha256(binary.raw[:binary_size.value]).hexdigest(), "binary_hex": binary.raw[:binary_size.value].hex(), "allocation": {"type": allocation_type.value, "base_matches": True, "bytes": allocation_size.value, "api": "clHostMemAllocINTEL", "cl_mem": False, "write_read_buffer": False, "migrate_prefetch": False}}
        def run_once(words):
            raw = struct.pack("<1024I", *words); C.memmove(pointer, raw, BUFFER_BYTES); event = C.c_void_p(); global_size, local_size = C.c_size_t(WORD_COUNT), C.c_size_t(256); submit = time.perf_counter_ns(); check(enqueue(queue, kernel, 1, None, C.byref(global_size), C.byref(local_size), 0, None, C.byref(event)), "enqueue"); event_row = log.create("intel_event", "intel", {"handle":int(event.value)})
            try:
                check(finish(queue), "finish"); done = time.perf_counter_ns(); start_ns, end_ns = C.c_ulonglong(), C.c_ulonglong(); check(get_profile(event, 0x1282, C.sizeof(start_ns), C.byref(start_ns), None), "profile_start"); check(get_profile(event, 0x1283, C.sizeof(end_ns), C.byref(end_ns), None), "profile_end"); output = C.string_at(pointer, BUFFER_BYTES)
                return {"submit_qpc_ns": submit, "done_qpc_ns": done, "device_start_ns": start_ns.value, "device_end_ns": end_ns.value, "output_sha256": hashlib.sha256(output).hexdigest(), "output_words": list(struct.unpack("<1024I", output))}
            finally: log.release(event_row, lambda: releases["clReleaseEvent"](event))
        def close():
            errors = []
            for row, release in reversed(cleanup):
                try: log.release(row, release)
                except BaseException as exc: errors.append(f"{row['kind']}:{type(exc).__name__}:{exc}")
            if errors: raise RuntimeError(";".join(errors))
        return init, run_once, close
    except BaseException:
        for row, release in reversed(cleanup):
            try: log.release(row, release)
            except BaseException: pass
        raise


def nvidia_backend(ledger):
    from het_next_cap0r4_kernels import NVIDIA_NVRTC_OPTIONS, NVIDIA_SOURCE
    driver=C.WinDLL('nvcuda.dll');nvrtc_path=ROOT/'.venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll';nvrtc=C.CDLL(str(nvrtc_path));cleanup=[]
    def bind(lib,name,args,result=C.c_int):fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
    cuInit=bind(driver,'cuInit',[C.c_uint]);cuDriverGetVersion=bind(driver,'cuDriverGetVersion',[C.POINTER(C.c_int)]);cuDeviceGetCount=bind(driver,'cuDeviceGetCount',[C.POINTER(C.c_int)]);cuDeviceGet=bind(driver,'cuDeviceGet',[C.POINTER(C.c_int),C.c_int]);cuDeviceGetName=bind(driver,'cuDeviceGetName',[C.c_char_p,C.c_int,C.c_int]);cuDeviceGetUuid=bind(driver,'cuDeviceGetUuid_v2',[C.c_void_p,C.c_int]);cuDeviceGetPCIBusId=bind(driver,'cuDeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int]);cuDeviceGetAttribute=bind(driver,'cuDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int]);cuDeviceTotalMem=bind(driver,'cuDeviceTotalMem_v2',[C.POINTER(C.c_size_t),C.c_int]);cuMemGetInfo=bind(driver,'cuMemGetInfo_v2',[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)])
    cuCtxCreate=bind(driver,'cuCtxCreate_v2',[C.POINTER(C.c_void_p),C.c_uint,C.c_int]);cuCtxDestroy=bind(driver,'cuCtxDestroy_v2',[C.c_void_p]);cuModuleLoadDataEx=bind(driver,'cuModuleLoadDataEx',[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint,C.c_void_p,C.c_void_p]);cuModuleUnload=bind(driver,'cuModuleUnload',[C.c_void_p]);cuModuleGetFunction=bind(driver,'cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]);cuStreamCreate=bind(driver,'cuStreamCreate',[C.POINTER(C.c_void_p),C.c_uint]);cuStreamDestroy=bind(driver,'cuStreamDestroy_v2',[C.c_void_p]);cuStreamSynchronize=bind(driver,'cuStreamSynchronize',[C.c_void_p]);cuEventCreate=bind(driver,'cuEventCreate',[C.POINTER(C.c_void_p),C.c_uint]);cuEventDestroy=bind(driver,'cuEventDestroy_v2',[C.c_void_p]);cuEventRecord=bind(driver,'cuEventRecord',[C.c_void_p,C.c_void_p]);cuEventElapsedTime=bind(driver,'cuEventElapsedTime',[C.POINTER(C.c_float),C.c_void_p,C.c_void_p]);cuMemAlloc=bind(driver,'cuMemAlloc_v2',[C.POINTER(C.c_ulonglong),C.c_size_t]);cuMemFree=bind(driver,'cuMemFree_v2',[C.c_ulonglong]);cuMemHostAlloc=bind(driver,'cuMemHostAlloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_uint]);cuMemFreeHost=bind(driver,'cuMemFreeHost',[C.c_void_p]);cuMemcpyHtoDAsync=bind(driver,'cuMemcpyHtoDAsync_v2',[C.c_ulonglong,C.c_void_p,C.c_size_t,C.c_void_p]);cuMemcpyDtoHAsync=bind(driver,'cuMemcpyDtoHAsync_v2',[C.c_void_p,C.c_ulonglong,C.c_size_t,C.c_void_p]);cuLaunchKernel=bind(driver,'cuLaunchKernel',[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)])
    nvrtcVersion=bind(nvrtc,'nvrtcVersion',[C.POINTER(C.c_int),C.POINTER(C.c_int)]);nvrtcCreateProgram=bind(nvrtc,'nvrtcCreateProgram',[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p]);nvrtcCompileProgram=bind(nvrtc,'nvrtcCompileProgram',[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)]);nvrtcGetPTXSize=bind(nvrtc,'nvrtcGetPTXSize',[C.c_void_p,C.POINTER(C.c_size_t)]);nvrtcGetPTX=bind(nvrtc,'nvrtcGetPTX',[C.c_void_p,C.c_void_p]);nvrtcGetProgramLogSize=bind(nvrtc,'nvrtcGetProgramLogSize',[C.c_void_p,C.POINTER(C.c_size_t)]);nvrtcGetProgramLog=bind(nvrtc,'nvrtcGetProgramLog',[C.c_void_p,C.c_void_p]);nvrtcDestroyProgram=bind(nvrtc,'nvrtcDestroyProgram',[C.POINTER(C.c_void_p)])
    def ck(code,name):
        if int(code)!=0:raise RuntimeError(f'{name}:{int(code)}')
    ck(cuInit(0),'cuInit');count=C.c_int();ck(cuDeviceGetCount(C.byref(count)),'cuDeviceGetCount');devices=[]
    for ordinal in range(count.value):
        dev=C.c_int();ck(cuDeviceGet(C.byref(dev),ordinal),'cuDeviceGet');name=C.create_string_buffer(256);ck(cuDeviceGetName(name,256,dev.value),'cuDeviceGetName');
        if b'NVIDIA' in name.value:devices.append((ordinal,dev.value,name.value.decode(errors='replace')))
    if len(devices)!=1:raise RuntimeError(f'nvidia_candidate_count:{len(devices)}')
    ordinal,dev,name=devices[0];attrs={};
    for key,code in {'major':75,'minor':76,'pci_bus':33,'pci_device':34,'pci_domain':50,'can_map_host':19,'concurrent_kernels':31,'async_engines':40}.items():v=C.c_int();ck(cuDeviceGetAttribute(C.byref(v),code,dev),f'attribute:{key}');attrs[key]=v.value
    uuid_bytes=(C.c_ubyte*16)();ck(cuDeviceGetUuid(uuid_bytes,dev),'cuDeviceGetUuid');pci_buffer=C.create_string_buffer(32);ck(cuDeviceGetPCIBusId(pci_buffer,32,dev),'cuDeviceGetPCIBusId');pci=pci_buffer.value.decode().lower();
    if not __import__('re').fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',pci):raise RuntimeError('invalid_cuda_pci_bdf')
    total=C.c_size_t();ck(cuDeviceTotalMem(C.byref(total),dev),'cuDeviceTotalMem');ctx=C.c_void_p();ck(cuCtxCreate(C.byref(ctx),0,dev),'cuCtxCreate');cleanup.append((ledger.create('nvidia_context','nvidia',{'handle':int(ctx.value)}),lambda:cuCtxDestroy(ctx)))
    program=C.c_void_p();module=C.c_void_p();stream=C.c_void_p();host=C.c_void_p();dptr=C.c_ulonglong();kernel=C.c_void_p();ptx=b'';compile_log=''
    try:
        ck(nvrtcCreateProgram(C.byref(program),NVIDIA_SOURCE.encode(),b'cap0r4.cu',0,None,None),'nvrtcCreateProgram');cleanup.append((ledger.create('nvidia_nvrtc_program','nvidia'),lambda:nvrtcDestroyProgram(C.byref(program))))
        options=(C.c_char_p*len(NVIDIA_NVRTC_OPTIONS))(*[x.encode() for x in NVIDIA_NVRTC_OPTIONS]);compile_code=nvrtcCompileProgram(program,len(options),options);ls=C.c_size_t();ck(nvrtcGetProgramLogSize(program,C.byref(ls)),'nvrtcGetProgramLogSize');lb=C.create_string_buffer(max(1,ls.value));ck(nvrtcGetProgramLog(program,lb),'nvrtcGetProgramLog');compile_log=lb.value.decode(errors='replace');ck(compile_code,'nvrtcCompileProgram');ps=C.c_size_t();ck(nvrtcGetPTXSize(program,C.byref(ps)),'nvrtcGetPTXSize');pb=C.create_string_buffer(ps.value);ck(nvrtcGetPTX(program,pb),'nvrtcGetPTX');ptx=pb.raw[:ps.value]
        free,total_live=C.c_size_t(),C.c_size_t();ck(cuMemGetInfo(C.byref(free),C.byref(total_live)),'cuMemGetInfo');
        if free.value<64<<20:raise MemoryError('nvidia_free_vram_below_64MiB')
        ck(cuModuleLoadDataEx(C.byref(module),C.cast(pb,C.c_void_p),0,None,None),'cuModuleLoadDataEx');cleanup.append((ledger.create('nvidia_module','nvidia',{'handle':int(module.value)}),lambda:cuModuleUnload(module)));ck(cuModuleGetFunction(C.byref(kernel),module,b'cap0_nvidia_bijection'),'cuModuleGetFunction');ck(cuStreamCreate(C.byref(stream),1),'cuStreamCreate');cleanup.append((ledger.create('nvidia_stream','nvidia',{'handle':int(stream.value),'nondefault':True}),lambda:cuStreamDestroy(stream)));ck(cuMemHostAlloc(C.byref(host),BUFFER_BYTES,0),'cuMemHostAlloc');cleanup.append((ledger.create('nvidia_pinned_host','nvidia',{'bytes':BUFFER_BYTES,'pointer':int(host.value)}),lambda:cuMemFreeHost(host)));ck(cuMemAlloc(C.byref(dptr),BUFFER_BYTES),'cuMemAlloc');cleanup.append((ledger.create('nvidia_device_memory','nvidia',{'bytes':BUFFER_BYTES,'deviceptr':int(dptr.value)}),lambda:cuMemFree(dptr.value)))
        dv=C.c_int();ck(cuDriverGetVersion(C.byref(dv)),'cuDriverGetVersion');ma,mi=C.c_int(),C.c_int();ck(nvrtcVersion(C.byref(ma),C.byref(mi)),'nvrtcVersion')
        identity={'enumerated_device_count':count.value,'eligible_device_count':len(devices),'device_ordinal':ordinal,'name':name,'uuid':bytes(uuid_bytes).hex(),'pci_bus_id':pci,'driver_version':dv.value,'runtime_version':'not_applicable_driver_api_only','nvrtc_version':[ma.value,mi.value],'compute_capability':[attrs['major'],attrs['minor']],'total_memory':total.value,'free_memory_start':free.value,'can_map_host_memory':attrs['can_map_host'],'concurrent_kernels':attrs['concurrent_kernels'],'async_engine_count':attrs['async_engines'],'source_sha256':hashlib.sha256(NVIDIA_SOURCE.encode()).hexdigest(),'compile_options':list(NVIDIA_NVRTC_OPTIONS),'compile_log':compile_log,'binary_bytes':len(ptx),'binary_sha256':hashlib.sha256(ptx).hexdigest(),'binary_hex':ptx.hex(),'binary_kind':'PTX','allocation':{'pinned_host_bytes':BUFFER_BYTES,'device_bytes':BUFFER_BYTES,'managed':False}}
    except BaseException:
        for row,release in reversed(cleanup):
            try:ledger.release(row,release)
            except BaseException:pass
        raise
    def run_once(words):
        raw=struct.pack('<1024I',*words);C.memmove(host,raw,BUFFER_BYTES);start_event,end_event=C.c_void_p(),C.c_void_p();erows=[]
        try:
            ck(cuEventCreate(C.byref(start_event),0),'cuEventCreate_start');erows.append((ledger.create('nvidia_start_event','nvidia',{'handle':int(start_event.value)}),lambda:cuEventDestroy(start_event)))
            ck(cuEventCreate(C.byref(end_event),0),'cuEventCreate_end');erows.append((ledger.create('nvidia_end_event','nvidia',{'handle':int(end_event.value)}),lambda:cuEventDestroy(end_event)))
        except BaseException:
            for row,release in reversed(erows):
                try:ledger.release(row,release)
                except BaseException:pass
            raise
        submit = time.perf_counter_ns()
        try:
            ck(cuEventRecord(start_event,stream),'cuEventRecord_start');ck(cuMemcpyHtoDAsync(dptr.value,host,BUFFER_BYTES,stream),'cuMemcpyHtoDAsync');arg0=C.c_ulonglong(dptr.value);arg1=C.c_uint(WORD_COUNT);params=(C.c_void_p*2)(C.cast(C.byref(arg0),C.c_void_p),C.cast(C.byref(arg1),C.c_void_p));ck(cuLaunchKernel(kernel,4,1,1,256,1,1,0,stream,params,None),'cuLaunchKernel');ck(cuMemcpyDtoHAsync(host,dptr.value,BUFFER_BYTES,stream),'cuMemcpyDtoHAsync');ck(cuEventRecord(end_event,stream),'cuEventRecord_end');ck(cuStreamSynchronize(stream),'cuStreamSynchronize');done=time.perf_counter_ns();elapsed=C.c_float();ck(cuEventElapsedTime(C.byref(elapsed),start_event,end_event),'cuEventElapsedTime');output=C.string_at(host,BUFFER_BYTES)
            return {'submit_qpc_ns':submit,'done_qpc_ns':done,'device_elapsed_ms':elapsed.value,'output_sha256':hashlib.sha256(output).hexdigest(),'output_words':list(struct.unpack('<1024I',output))}
        finally:
            errors=[]
            for row, release in reversed(erows):
                try:ledger.release(row, release)
                except BaseException as exc:errors.append(f'{row["kind"]}:{type(exc).__name__}:{exc}')
            if errors:raise RuntimeError(';'.join(errors))
    def close():
        errors = []
        for row, release in reversed(cleanup):
            try: ledger.release(row, release)
            except BaseException as exc: errors.append(f"{row['kind']}:{type(exc).__name__}:{exc}")
        if errors: raise RuntimeError(";".join(errors))
    return identity, run_once, close


def pdh_monitor(stop_event, p, ledger):
    from ctypes import wintypes
    class VALUE_UNION(C.Union): _fields_ = [("long_value", wintypes.LONG), ("double_value", C.c_double), ("large_value", C.c_longlong)]
    class VALUE(C.Structure): _anonymous_ = ("value",); _fields_ = [("status", wintypes.DWORD), ("value", VALUE_UNION)]
    pdh = C.WinDLL("pdh", use_last_error=True); query = C.c_void_p(); counters = []
    pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, C.c_void_p, C.POINTER(C.c_void_p)]; pdh.PdhOpenQueryW.restype = C.c_long
    pdh.PdhAddEnglishCounterW.argtypes = [C.c_void_p, wintypes.LPCWSTR, C.c_void_p, C.POINTER(C.c_void_p)]; pdh.PdhAddEnglishCounterW.restype = C.c_long
    pdh.PdhCollectQueryData.argtypes = [C.c_void_p]; pdh.PdhCollectQueryData.restype = C.c_long
    pdh.PdhGetFormattedCounterValue.argtypes = [C.c_void_p, wintypes.DWORD, C.c_void_p, C.POINTER(VALUE)]; pdh.PdhGetFormattedCounterValue.restype = C.c_long
    pdh.PdhRemoveCounter.argtypes = [C.c_void_p]; pdh.PdhRemoveCounter.restype = C.c_long
    pdh.PdhCloseQuery.argtypes = [C.c_void_p]; pdh.PdhCloseQuery.restype = C.c_long
    k = C.WinDLL("kernel32", use_last_error=True); k.CreateWaitableTimerW.argtypes = [C.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]; k.CreateWaitableTimerW.restype = wintypes.HANDLE; k.SetWaitableTimer.argtypes = [wintypes.HANDLE, C.POINTER(C.c_longlong), wintypes.LONG, C.c_void_p, C.c_void_p, wintypes.BOOL]; k.SetWaitableTimer.restype = wintypes.BOOL; k.CancelWaitableTimer.argtypes = [wintypes.HANDLE]; k.CancelWaitableTimer.restype = wintypes.BOOL; k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD;k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
    query_row = ledger.create("pdh_query", "monitor"); timer = None; timer_row = None; samples = []; query_open_qpc_ns = 0; query_close_qpc_ns = 0
    try:
        if pdh.PdhOpenQueryW(None, None, C.byref(query)): raise RuntimeError("PdhOpenQueryW")
        query_open_qpc_ns=time.perf_counter_ns()
        for path in COUNTERS:
            counter = C.c_void_p()
            if pdh.PdhAddEnglishCounterW(query, path, None, C.byref(counter)): raise RuntimeError(f"PdhAddEnglishCounterW:{path}")
            counters.append(counter); ledger.create("pdh_counter", "monitor", {"path": path})
        if pdh.PdhCollectQueryData(query): raise RuntimeError("PdhCollectQueryData_initial")
        timer = k.CreateWaitableTimerW(None, False, "cap0_pdh_100ms");
        if not timer: raise OSError(C.get_last_error(), "CreateWaitableTimerW")
        timer_row = ledger.create("pdh_waitable_timer", "monitor"); due = C.c_longlong(-1_000_000)
        if not k.SetWaitableTimer(timer, C.byref(due), 100, None, None, False): raise OSError(C.get_last_error(), "SetWaitableTimer")
        scheduled = time.perf_counter_ns() + 100_000_000
        while not p.wait_one(stop_event, 0):
            if int(k.WaitForSingleObject(timer, 200)) != 0: raise TimeoutError("pdh_timer")
            actual = time.perf_counter_ns()
            if pdh.PdhCollectQueryData(query): raise RuntimeError("PdhCollectQueryData")
            values, statuses = {}, {}
            for path, counter in zip(COUNTERS, counters):
                value = VALUE(); code = int(pdh.PdhGetFormattedCounterValue(counter, 0x200, None, C.byref(value))); values[path] = float(value.double_value); statuses[path] = {"return": code, "cstatus": int(value.status)}
            samples.append({"scheduled_qpc_ns": scheduled, "actual_qpc_ns": actual, "lateness_ms": (actual - scheduled) / 1e6, "values": values, "statuses": statuses}); scheduled += 100_000_000
        return {"paths": list(COUNTERS), "samples": samples, "query_open_qpc_ns":query_open_qpc_ns, "query_close_qpc_ns":None}
    finally:
        cleanup_errors=[]
        if timer:
            try:k.CancelWaitableTimer(timer)
            except BaseException as exc:cleanup_errors.append(f'cancel_timer:{exc}')
            if timer_row:
                try:ledger.release(timer_row, lambda: 0 if k.CloseHandle(timer) else C.get_last_error() or 1)
                except BaseException as exc:cleanup_errors.append(f'close_timer:{exc}')
        for row,counter in zip([r for r in ledger.rows if r["kind"] == "pdh_counter" and r["final_state"] == "live"], counters):
            try:ledger.release(row, lambda c=counter: pdh.PdhRemoveCounter(c))
            except BaseException as exc:cleanup_errors.append(f'remove_counter:{exc}')
        try:ledger.release(query_row, lambda: pdh.PdhCloseQuery(query));query_close_qpc_ns=time.perf_counter_ns()
        except BaseException as exc:cleanup_errors.append(f'close_query:{exc}')
        if samples:samples[-1]['query_close_qpc_ns']=query_close_qpc_ns
        if cleanup_errors:raise RuntimeError(';'.join(cleanup_errors))


def lockcheck():
    lock = json.loads(LOCK.read_text())
    expected = {"runner_sha256": sha256_file(__file__), "protocol_sha256": sha256_file(PROTOCOL_PATH), "kernel_sha256": sha256_file(KERNEL_PATH), "verifier_sha256": sha256_file(VERIFIER), "verifier_lock_sha256": sha256_file(VERIFIER_LOCK), "preflight_sha256": sha256_file(PREFLIGHT), "prereg_sha256": sha256_file(PREREG), "design_sha256": sha256_file(DESIGN)}
    return lock, expected, all(lock.get(key) == value for key, value in expected.items())


def run_capability(ack):
    global _EMERGENCY_CLEANUP, _FAILURE_EVIDENCE
    from het_next_cap0r4_protocol import DualDeviceProtocol, Win32Primitives, atomic_create_json
    lock, expected, bound = lockcheck()
    if not bound or lock.get("execution_open") is not True or lock.get("audit_token") != ack or ack == ACK_PENDING: raise PermissionError("execution_closed_or_hash_drift")
    if RUN_DIR.exists():
        if RESULT.exists() and COMMIT.exists() and not FAILURE.exists():
            marker=json.loads(COMMIT.read_text());valid=marker.get('result')=={'bytes':RESULT.stat().st_size,'sha256':sha256_file(RESULT)}
            if valid:return {'status':'already_complete','result':str(RESULT),'commit':str(COMMIT)}
        raise FileExistsError("output_directory_not_absent")
    start_ram = available_ram_bytes()
    if start_ram < (2 << 30): raise MemoryError("available_ram_below_2GiB")
    words = input_words(); intel_expected, nvidia_expected = expected_words("intel", words), expected_words("nvidia", words)
    if (packed_sha(words), packed_sha(intel_expected), packed_sha(nvidia_expected)) != (INPUT_SHA256, INTEL_SHA256, NVIDIA_SHA256): raise RuntimeError("fixture_drift")
    page_size = int(os.sysconf("SC_PAGE_SIZE")) if os.name != "nt" else 4096
    if page_size != 4096: raise RuntimeError("unexpected_page_size")
    ledger = Ledger(); p = Win32Primitives(ledger); protocol = None
    bootstrap = p.event(True, False, "cap0_bootstrap"); monitor_stop = p.event(True, False, "cap0_monitor_stop"); thread_ready = {name: p.event(True, False, f"cap0_{name}_thread_ready") for name in ("intel", "nvidia", "monitor")}; cleanup_done = {name: p.event(True, False, f"cap0_{name}_cleanup_done") for name in ("intel", "nvidia")}
    protocol = DualDeviceProtocol(p); state = {"thread_rows": {}, "results": {"intel": [], "nvidia": []}, "initialization": {}, "errors": {}, "monitor": None}; close_functions = {}
    def worker(name, backend):
        try:
            state["thread_rows"][name] = pin_and_identify(THREAD_LPS[name]); p.set(thread_ready[name]);
            if not p.wait_one(bootstrap): raise TimeoutError("bootstrap")
            init, execute, close = backend(ledger); close_functions[name] = close; state["initialization"][name] = init; protocol.worker_initialized(name, init)
            for epoch in range(1, 4):
                descriptor = protocol.worker_descriptor(name); protocol.worker_start(name, descriptor["epoch"]); row = execute(words); row.update({"epoch": epoch, "worker": name}); state["results"][name].append(row); protocol.worker_finish(name, epoch, row)
            if not p.wait_one(protocol.channels[name].stop): raise TimeoutError("stop")
        except BaseException as exc:
            state["errors"][name] = f"{type(exc).__name__}: {exc}"; state.setdefault("tracebacks", {})[name] = traceback.format_exc()
            try: protocol.worker_initialized(name, {"error": state["errors"][name]})
            except BaseException: pass
        finally:
            try:
                if name in close_functions: close_functions[name]()
            except BaseException as exc: state["errors"][name + "_cleanup"] = f"{type(exc).__name__}: {exc}"
            state["thread_rows"].setdefault(name, {})["end_qpc_ns"] = time.perf_counter_ns(); p.set(cleanup_done[name])
    def monitor():
        try:
            state["thread_rows"]["monitor"] = pin_and_identify(THREAD_LPS["monitor"]); p.set(thread_ready["monitor"])
            if not p.wait_one(bootstrap): raise TimeoutError("bootstrap")
            state["monitor"] = pdh_monitor(monitor_stop, p, ledger)
        except BaseException as exc: state["errors"]["monitor"] = f"{type(exc).__name__}: {exc}"; state.setdefault("tracebacks", {})["monitor"] = traceback.format_exc()
        finally: state["thread_rows"].setdefault("monitor", {})["end_qpc_ns"] = time.perf_counter_ns()
    threads = {"intel": threading.Thread(target=worker, args=("intel", intel_backend), name="cap0-intel"), "nvidia": threading.Thread(target=worker, args=("nvidia", nvidia_backend), name="cap0-nvidia"), "monitor": threading.Thread(target=monitor, name="cap0-pdh")}
    closed = {"value": False}
    def emergency_cleanup():
        if closed["value"]: return {"already_closed": True}
        errors=[]
        for action in (lambda:p.set(bootstrap),lambda:p.set(monitor_stop),lambda:p.set(protocol.start),protocol.request_stop):
            try:action()
            except BaseException as exc:errors.append(f'{type(exc).__name__}:{exc}')
        for thread in threads.values():
            if thread.ident is not None:thread.join(timeout=35)
        hung=[name for name,thread in threads.items() if thread.is_alive()]
        if not hung:
            try:protocol.close()
            except BaseException as exc:errors.append(f'protocol_close:{type(exc).__name__}:{exc}')
            for event in [bootstrap,monitor_stop,*thread_ready.values(),*cleanup_done.values()]:
                try:p.close(event)
                except BaseException as exc:errors.append(f'event_close:{type(exc).__name__}:{exc}')
        closed['value']=not hung
        return {'cleanup_errors':errors,'hung_threads':hung,'ledger':ledger.rows,'thread_rows':state['thread_rows'],'partial_results':state['results'],'worker_errors':state['errors'],'closed':closed['value']}
    _EMERGENCY_CLEANUP=emergency_cleanup
    for thread in threads.values(): thread.start()
    state["thread_rows"]["coordinator"] = pin_and_identify(THREAD_LPS["coordinator"])
    if not p.wait_all(list(thread_ready.values())): raise TimeoutError("thread_ready")
    rows = state["thread_rows"]
    if len({rows[name]["thread_id"] for name in rows}) != 4 or len({rows[name]["physical_core_record_offset"] for name in rows}) != 4 or any(rows[name]["processor_group"] != 0 for name in rows): raise RuntimeError("topology_not_four_distinct_cores")
    p.set(bootstrap); time.sleep(0.5); initialization = protocol.wait_initialized()
    if state["errors"] or any("error" in initialization[name] for name in initialization): raise RuntimeError(f"initialization_failure:{state['errors']}:{initialization}")
    intel_pci = initialization['intel']['pci']; nvidia_pci = initialization['nvidia']['pci_bus_id'].lower()
    intel_bdf = f"{intel_pci['domain']:04x}:{intel_pci['bus']:02x}:{intel_pci['device']:02x}.{intel_pci['function']}"
    if intel_bdf.lower() == nvidia_pci: raise RuntimeError("devices_share_pci_identity")
    lifecycle = []
    for epoch in range(1, 4):
        protocol.publish(epoch); t0 = protocol.coordinator_release(epoch); outputs, t1 = protocol.collect(epoch)
        i, n = outputs["intel"], outputs["nvidia"]; overlap = max(i["submit_qpc_ns"], n["submit_qpc_ns"]) < min(i["done_qpc_ns"], n["done_qpc_ns"])
        lifecycle.append({"epoch": epoch, "coordinator_t0": t0, "coordinator_t1": t1, "strict_work_interval_overlap": overlap})
    protocol.request_stop()
    if not p.wait_all(list(cleanup_done.values())): raise TimeoutError("cleanup")
    cleanup_complete_qpc_ns=time.perf_counter_ns();time.sleep(0.5);monitor_stop_qpc_ns=time.perf_counter_ns(); p.set(monitor_stop)
    for thread in threads.values(): thread.join(timeout=35)
    if any(thread.is_alive() for thread in threads.values()): raise TimeoutError("thread_join")
    state["thread_rows"]["coordinator"]["end_qpc_ns"] = time.perf_counter_ns()
    for name, expected_output, digest in (("intel", intel_expected, INTEL_SHA256), ("nvidia", nvidia_expected, NVIDIA_SHA256)):
        for row in state["results"][name]:
            row["different_words"] = sum(a != b for a, b in zip(row["output_words"], expected_output)); row["expected_sha256"] = digest
    samples = (state["monitor"] or {}).get("samples", []); intervals = [(samples[i]["actual_qpc_ns"] - samples[i - 1]["actual_qpc_ns"]) / 1e6 for i in range(1, len(samples))]
    monitor_valid = len(samples) >= 11 and all(80 <= value <= 120 for value in intervals) and all(row["lateness_ms"] <= 20 for row in samples) and all(status["return"] == 0 and status["cstatus"] == 0 for row in samples for status in row["statuses"].values())
    close_errors=[]
    try:protocol.close()
    except BaseException as exc:close_errors.append(f'protocol:{type(exc).__name__}:{exc}')
    for event in [bootstrap, monitor_stop, *thread_ready.values(), *cleanup_done.values()]:
        try:p.close(event)
        except BaseException as exc:close_errors.append(f'external_event:{type(exc).__name__}:{exc}')
    closed['value']=True;_EMERGENCY_CLEANUP=None
    cleanup_errors = [row for row in ledger.rows if row["release_attempts"] != 1 or row["release_code"] != 0 or row["final_state"] != "released"]
    monitor_boundaries={"monitor_initialized_qpc_ns":samples[0]["actual_qpc_ns"] if samples else 0,"first_release_qpc_ns":lifecycle[0]["coordinator_t0"],"cleanup_complete_qpc_ns":cleanup_complete_qpc_ns,"monitor_stop_qpc_ns":monitor_stop_qpc_ns,"pre_ms":(lifecycle[0]["coordinator_t0"]-samples[0]["actual_qpc_ns"])/1e6 if samples else -1,"post_ms":(monitor_stop_qpc_ns-cleanup_complete_qpc_ns)/1e6}
    monitor_valid=monitor_valid and monitor_boundaries['pre_ms']>=500 and monitor_boundaries['post_ms']>=500
    positive = not state["errors"] and not close_errors and not cleanup_errors and monitor_valid and all(row["strict_work_interval_overlap"] for row in lifecycle) and all(row["different_words"] == 0 and row["output_sha256"] == row["expected_sha256"] for name in ("intel", "nvidia") for row in state["results"][name])
    result = {"kind": "het_next_cap0r4_dual_device_cohabitation", "status": "dual_device_cohabitation_positive" if positive else "capability_negative", "claim_boundary": "4KiB dual-device lifecycle/correctness only; no performance or model claim", "process": process_identity(), "resources": {"available_ram_start": start_ram, "minimum_ram": 2 << 30}, "pci_distinct": True, "bindings": expected, "thread_rows": state["thread_rows"], "initialization": state["initialization"], "repetitions": state["results"], "lifecycle": lifecycle, "protocol_log": protocol.log, "primitive_calls": p.calls, "monitor": {**(state["monitor"] or {}), "interval_ms": intervals, "boundaries":monitor_boundaries,"valid_protocol": monitor_valid}, "ledger": ledger.rows, "cleanup_errors": cleanup_errors, "close_errors":close_errors,"errors": state["errors"], "fixtures": {"seed": SEED, "word_count": WORD_COUNT, "input_sha256": INPUT_SHA256, "intel_sha256": INTEL_SHA256, "nvidia_sha256": NVIDIA_SHA256}}
    RUN_DIR.mkdir(parents=True, exist_ok=False); atomic_create_json(RESULT, canonical_bytes(result)); commit = {"kind": "het_next_cap0_commit", "result": {"bytes": RESULT.stat().st_size, "sha256": sha256_file(RESULT)}, "status": result["status"]}; atomic_create_json(COMMIT, canonical_bytes(commit))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("capability",), required=True); parser.add_argument("--ack", required=True); args = parser.parse_args()
    try:
        result = run_capability(args.ack); print(json.dumps({"status": result["status"], "result": str(RESULT), "commit": str(COMMIT)}, sort_keys=True)); return 0 if result["status"] in ("dual_device_cohabitation_positive","already_complete") else 2
    except BaseException as exc:
        from het_next_cap0r4_protocol import atomic_create_json, recover_transaction
        cleanup = _EMERGENCY_CLEANUP() if _EMERGENCY_CLEANUP is not None else {"cleanup_not_initialized": True}; dispositions = recover_transaction(RUN_DIR) if RUN_DIR.exists() else []
        payload = {"kind": "het_next_cap0_failure", "status": "valid_failure", "stage": "capability", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "runner_sha256": sha256_file(__file__), "scope_provenance": {"opened_paths": [], "forbidden_payload_count": 0}, "cleanup": cleanup, "dispositions": dispositions}
        try:
            if not RUN_DIR.exists(): RUN_DIR.mkdir(parents=True, exist_ok=False)
            if not FAILURE.exists(): atomic_create_json(FAILURE, canonical_bytes(payload))
        except BaseException: pass
        raise


if __name__ == "__main__": raise SystemExit(main())
