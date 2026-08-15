from __future__ import annotations

import ctypes as C
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
PREREG = REPORTS / "ST2_MINI_PREREGISTRATION_2026-08-12.md"
OUTPUT = REPORTS / "st2_mini_opencl_capability_probe.json"

CL_SUCCESS = 0
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_PLATFORM_PROFILE = 0x0900
CL_PLATFORM_VERSION = 0x0901
CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_PLATFORM_EXTENSIONS = 0x0904
CL_DEVICE_TYPE = 0x1000
CL_DEVICE_VENDOR_ID = 0x1001
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_HOST_UNIFIED_MEMORY = 0x1035
CL_DEVICE_SVM_CAPABILITIES = 0x1053
CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL = 0x4190
CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL = 0x4191
CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL = 0x4192
CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL = 0x4193
CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL = 0x4194


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(code: int, operation: str) -> None:
    if code != CL_SUCCESS:
        raise RuntimeError(f"{operation} failed with OpenCL error {code}")


def string_info(function, handle, param: int) -> str:
    size = C.c_size_t()
    check(function(handle, param, 0, None, C.byref(size)), f"info-size 0x{param:x}")
    buffer = C.create_string_buffer(size.value)
    check(
        function(handle, param, size.value, buffer, None),
        f"info-value 0x{param:x}",
    )
    return buffer.value.decode("utf-8", errors="replace")


def scalar_info(function, handle, param: int, ctype):
    value = ctype()
    check(
        function(handle, param, C.sizeof(value), C.byref(value), None),
        f"scalar-info 0x{param:x}",
    )
    return value.value


def optional_scalar_info(function, handle, param: int, ctype):
    value = ctype()
    code = function(handle, param, C.sizeof(value), C.byref(value), None)
    return {"available": code == CL_SUCCESS, "error": int(code), "value": value.value if code == CL_SUCCESS else None}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    ocl = C.WinDLL("OpenCL.dll")
    ocl.clGetPlatformIDs.argtypes = [C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)]
    ocl.clGetPlatformIDs.restype = C.c_int
    ocl.clGetPlatformInfo.argtypes = [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
    ocl.clGetPlatformInfo.restype = C.c_int
    ocl.clGetDeviceIDs.argtypes = [C.c_void_p, C.c_ulonglong, C.c_uint, C.POINTER(C.c_void_p), C.POINTER(C.c_uint)]
    ocl.clGetDeviceIDs.restype = C.c_int
    ocl.clGetDeviceInfo.argtypes = [C.c_void_p, C.c_uint, C.c_size_t, C.c_void_p, C.POINTER(C.c_size_t)]
    ocl.clGetDeviceInfo.restype = C.c_int
    ocl.clGetExtensionFunctionAddressForPlatform.argtypes = [C.c_void_p, C.c_char_p]
    ocl.clGetExtensionFunctionAddressForPlatform.restype = C.c_void_p

    count = C.c_uint()
    check(ocl.clGetPlatformIDs(0, None, C.byref(count)), "clGetPlatformIDs count")
    platforms = (C.c_void_p * count.value)()
    check(ocl.clGetPlatformIDs(count.value, platforms, None), "clGetPlatformIDs values")
    records = []
    selected = None
    function_names = (
        "clHostMemAllocINTEL",
        "clMemFreeINTEL",
        "clSetKernelArgMemPointerINTEL",
        "clGetMemAllocInfoINTEL",
        "clEnqueueMemFillINTEL",
        "clEnqueueMemcpyINTEL",
        "clEnqueueMigrateMemINTEL",
    )
    for p_index, platform_handle in enumerate(platforms):
        p = C.c_void_p(platform_handle)
        platform_row = {
            "index": p_index,
            "name": string_info(ocl.clGetPlatformInfo, p, CL_PLATFORM_NAME),
            "vendor": string_info(ocl.clGetPlatformInfo, p, CL_PLATFORM_VENDOR),
            "version": string_info(ocl.clGetPlatformInfo, p, CL_PLATFORM_VERSION),
            "extensions": string_info(ocl.clGetPlatformInfo, p, CL_PLATFORM_EXTENSIONS).split(),
            "devices": [],
        }
        ndev = C.c_uint()
        code = ocl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, None, C.byref(ndev))
        if code != CL_SUCCESS:
            platform_row["device_query_error"] = int(code)
            records.append(platform_row)
            continue
        devices = (C.c_void_p * ndev.value)()
        check(ocl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndev.value, devices, None), "clGetDeviceIDs values")
        functions = {
            name: bool(ocl.clGetExtensionFunctionAddressForPlatform(p, name.encode("ascii")))
            for name in function_names
        }
        for d_index, device_handle in enumerate(devices):
            d = C.c_void_p(device_handle)
            extensions = string_info(ocl.clGetDeviceInfo, d, CL_DEVICE_EXTENSIONS).split()
            row = {
                "index": d_index,
                "name": string_info(ocl.clGetDeviceInfo, d, CL_DEVICE_NAME),
                "vendor": string_info(ocl.clGetDeviceInfo, d, CL_DEVICE_VENDOR),
                "vendor_id": scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_VENDOR_ID, C.c_uint),
                "type": scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_TYPE, C.c_ulonglong),
                "driver_version": string_info(ocl.clGetDeviceInfo, d, CL_DRIVER_VERSION),
                "device_version": string_info(ocl.clGetDeviceInfo, d, CL_DEVICE_VERSION),
                "compute_units": scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_MAX_COMPUTE_UNITS, C.c_uint),
                "max_work_group_size": scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_MAX_WORK_GROUP_SIZE, C.c_size_t),
                "global_mem_bytes": scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_GLOBAL_MEM_SIZE, C.c_ulonglong),
                "max_mem_alloc_bytes": scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, C.c_ulonglong),
                "local_mem_bytes": scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_LOCAL_MEM_SIZE, C.c_ulonglong),
                "host_unified_memory": bool(scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_HOST_UNIFIED_MEMORY, C.c_uint)),
                "svm_capabilities": optional_scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_SVM_CAPABILITIES, C.c_ulonglong),
                "extensions": extensions,
                "intel_usm_capabilities": {
                    "host": optional_scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL, C.c_ulonglong),
                    "device": optional_scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, C.c_ulonglong),
                    "single_device_shared": optional_scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, C.c_ulonglong),
                    "cross_device_shared": optional_scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, C.c_ulonglong),
                    "shared_system": optional_scalar_info(ocl.clGetDeviceInfo, d, CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL, C.c_ulonglong),
                },
                "extension_functions": functions,
            }
            is_intel_arc = "intel" in row["vendor"].lower() and "arc" in row["name"].lower()
            host_cap = row["intel_usm_capabilities"]["host"]
            capability_pass = (
                is_intel_arc
                and "cl_intel_unified_shared_memory" in extensions
                and host_cap["available"]
                and bool(host_cap["value"] & 1)
                and functions["clHostMemAllocINTEL"]
                and functions["clMemFreeINTEL"]
                and functions["clSetKernelArgMemPointerINTEL"]
                and functions["clGetMemAllocInfoINTEL"]
            )
            row["preregistered_device_match"] = is_intel_arc
            row["capability_pass"] = capability_pass
            if capability_pass and selected is None:
                selected = {"platform_index": p_index, "device_index": d_index, "name": row["name"]}
            platform_row["devices"].append(row)
        records.append(platform_row)

    result = {
        "kind": "streamq5_moe_st2_mini_opencl_capability_probe",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "os": platform.platform(),
        "preregistration_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "platform_count": count.value,
        "platforms": records,
        "selected": selected,
        "capability_pass": selected is not None,
        "next_action": "run_host_usm_q5_gate" if selected else "blocked_no_auditable_host_usm_path",
        "claim_boundary": "Read-only capability query; no kernel, allocation, GPU timing, or correctness claim.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

