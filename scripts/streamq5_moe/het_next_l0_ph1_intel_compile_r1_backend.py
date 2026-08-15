#!/usr/bin/env python3
"""PH1-R1 Intel compile-only backend. Import is device-free and payload-free."""
from __future__ import annotations

import ast
import ctypes as C
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/streamq5_moe"
R0_BACKEND = ROOT / "scripts/streamq5_moe/het_next_l0_ph1_intel_backend.py"
LOCK = REPORTS / "het_next_l0_ph1_intel_compile_r1_lock.json"
R0_BACKEND_SHA256 = "1c70d4248bdf64404589916a6be624594e8343442a64c57e926e52926f51ceac"
SOURCE_SHA256 = "06be3a9ba863d5f01d4025dc8d5e5679cdcc9827c13db5663c149227f3254528"
OPTIONS = "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt"
ACK = "PH1_INTEL_COMPILE_R1_AFTER_INDEPENDENT_SOURCE_AND_PREFLIGHT_GO"

CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 4
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_PCI_BUS_INFO_KHR = 0x410F
CL_CONTEXT_PLATFORM = 0x1084
CL_PROGRAM_NUM_DEVICES = 0x1162
CL_PROGRAM_BINARY_SIZES = 0x1165
CL_PROGRAM_BINARIES = 0x1166
CL_PROGRAM_BUILD_LOG = 0x1183


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_source() -> str:
    """Reconstruct the frozen kernel source with only the two preregistered pragma edits."""
    if file_sha256(R0_BACKEND) != R0_BACKEND_SHA256:
        raise RuntimeError("r0_backend_hash_drift")
    tree = ast.parse(R0_BACKEND.read_text(encoding="utf-8"), filename=str(R0_BACKEND))
    original = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SRC" for t in node.targets):
            original = ast.literal_eval(node.value)
            break
    if not isinstance(original, str):
        raise RuntimeError("r0_source_literal_missing")
    wrong = (
        "#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable\n"
        "#pragma OPENCL EXTENSION cl_khr_int64 : enable\n"
    )
    corrected = "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n"
    if original.count(wrong) != 1:
        raise RuntimeError("r0_pragma_contract")
    source = original.replace(wrong, corrected)
    if sha256_bytes(source.encode()) != SOURCE_SHA256:
        raise RuntimeError("r1_source_hash_drift")
    return source


SRC = canonical_source()


class PCI(C.Structure):
    _fields_ = [
        ("domain", C.c_uint),
        ("bus", C.c_uint),
        ("device", C.c_uint),
        ("function", C.c_uint),
    ]


class CompileFailure(RuntimeError):
    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


def check(code: int, operation: str) -> None:
    if int(code) != CL_SUCCESS:
        raise RuntimeError(f"{operation}:{int(code)}")


def authorize(expected: dict) -> dict:
    """Fail before OpenCL.dll is opened unless the immutable R1 lock is open and exact."""
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if not (
        lock.get("kind") == "het_next_l0_ph1_intel_compile_r1_lock"
        and lock.get("execution_open") is True
        and lock.get("audit_token") == ACK
        and lock.get("backend_sha256") == file_sha256(Path(__file__))
        and lock.get("source_sha256") == SOURCE_SHA256 == sha256_bytes(SRC.encode())
        and lock.get("cpu_commit_sha256") == expected.get("cpu_commit_sha256")
        and lock.get("cpu_verification_sha256") == expected.get("cpu_verification_sha256")
        and lock.get("prior_audit_sha256") == expected.get("prior_audit_sha256")
    ):
        raise RuntimeError("compile_r1_authorization")
    return {"lock_sha256": file_sha256(LOCK), "audit_token": ACK}


class IntelCompileOnlyR1:
    """Own exactly a context and program; queues, kernels, memory and launches do not exist here."""

    def __init__(self) -> None:
        self.library = None
        self.context = None
        self.program = None
        self.ledger: list[dict] = []
        self.cleanup_errors: list[str] = []

    def _bind(self) -> None:
        lib = self.library
        pointer, uint, size, integer = C.c_void_p, C.c_uint, C.c_size_t, C.c_int
        lib.clGetPlatformIDs.argtypes = [uint, C.POINTER(pointer), C.POINTER(uint)]
        lib.clGetPlatformIDs.restype = integer
        lib.clGetDeviceIDs.argtypes = [pointer, C.c_ulonglong, uint, C.POINTER(pointer), C.POINTER(uint)]
        lib.clGetDeviceIDs.restype = integer
        lib.clGetDeviceInfo.argtypes = [pointer, uint, size, pointer, C.POINTER(size)]
        lib.clGetDeviceInfo.restype = integer
        lib.clCreateContext.argtypes = [C.POINTER(C.c_ssize_t), uint, C.POINTER(pointer), pointer, pointer, C.POINTER(integer)]
        lib.clCreateContext.restype = pointer
        lib.clCreateProgramWithSource.argtypes = [pointer, uint, C.POINTER(C.c_char_p), C.POINTER(size), C.POINTER(integer)]
        lib.clCreateProgramWithSource.restype = pointer
        lib.clBuildProgram.argtypes = [pointer, uint, C.POINTER(pointer), C.c_char_p, pointer, pointer]
        lib.clBuildProgram.restype = integer
        lib.clGetProgramBuildInfo.argtypes = [pointer, pointer, uint, size, pointer, C.POINTER(size)]
        lib.clGetProgramBuildInfo.restype = integer
        lib.clGetProgramInfo.argtypes = [pointer, uint, size, pointer, C.POINTER(size)]
        lib.clGetProgramInfo.restype = integer
        for name in ("clReleaseProgram", "clReleaseContext"):
            getattr(lib, name).argtypes = [pointer]
            getattr(lib, name).restype = integer

    def _info(self, device, parameter: int) -> str:
        size = C.c_size_t()
        check(self.library.clGetDeviceInfo(device, parameter, 0, None, C.byref(size)), "device_info_size")
        if size.value <= 1:
            raise RuntimeError(f"empty_device_info:{parameter}")
        buffer = C.create_string_buffer(size.value)
        check(self.library.clGetDeviceInfo(device, parameter, size.value, buffer, None), "device_info")
        return buffer.value.decode(errors="strict")

    def _select(self):
        count = C.c_uint()
        check(self.library.clGetPlatformIDs(0, None, C.byref(count)), "platform_count")
        platforms = (C.c_void_p * count.value)()
        check(self.library.clGetPlatformIDs(count.value, platforms, None), "platforms")
        found = []
        for raw_platform in platforms:
            platform = C.c_void_p(raw_platform)
            number = C.c_uint()
            code = self.library.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, None, C.byref(number))
            if code != CL_SUCCESS:
                continue
            devices = (C.c_void_p * number.value)()
            check(self.library.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, number.value, devices, None), "devices")
            for raw_device in devices:
                device = C.c_void_p(raw_device)
                name = self._info(device, CL_DEVICE_NAME)
                extensions = self._info(device, CL_DEVICE_EXTENSIONS).split()
                if name == "Intel(R) Arc(TM) Pro 140T GPU (32GB)" and "cl_intel_unified_shared_memory" in extensions:
                    found.append((platform, device, extensions))
        if len(found) != 1:
            raise RuntimeError(f"intel_cardinality:{len(found)}")
        platform, device, extensions = found[0]
        pci = PCI()
        check(self.library.clGetDeviceInfo(device, CL_DEVICE_PCI_BUS_INFO_KHR, C.sizeof(pci), C.byref(pci), None), "pci")
        identity = {
            "name": self._info(device, CL_DEVICE_NAME),
            "vendor": self._info(device, CL_DEVICE_VENDOR),
            "driver": self._info(device, CL_DRIVER_VERSION),
            "pci": f"{pci.domain:04x}:{pci.bus:02x}:{pci.device:02x}.{pci.function}",
            "extensions": extensions,
        }
        if (
            identity["vendor"] != "Intel(R) Corporation"
            or identity["driver"] != "32.0.101.8517"
            or identity["pci"] != "0000:00:02.0"
        ):
            raise RuntimeError("intel_identity")
        self.ledger.append({"op": "identity", "identity": identity})
        return platform, device, identity

    def _release_all(self) -> None:
        for name, handle, function in (
            ("program", self.program, "clReleaseProgram"),
            ("context", self.context, "clReleaseContext"),
        ):
            if handle:
                try:
                    code = int(getattr(self.library, function)(handle))
                    self.ledger.append({"op": "release", "name": name, "code": code})
                    check(code, f"release:{name}")
                except Exception as exc:
                    self.cleanup_errors.append(f"{name}:{type(exc).__name__}:{exc}")
        self.ledger.append(
            {
                "op": "cleanup",
                "cleanup_complete": not self.cleanup_errors,
                "errors": list(self.cleanup_errors),
                "live_owned_resources": 0 if not self.cleanup_errors else None,
            }
        )

    def compile_only(self, eligibility: dict) -> dict:
        authorization = authorize(eligibility)
        evidence = {
            "authorization": authorization,
            "source": SRC,
            "source_sha256": SOURCE_SHA256,
            "options": OPTIONS,
            "ledger": self.ledger,
            "payload_read": False,
            "queues_created": 0,
            "kernels_created": 0,
            "events_created": 0,
            "memory_objects_created": 0,
            "allocations": 0,
            "kernels_launched": 0,
        }
        identity: dict = {}
        build_log = b""
        binary = b""
        declared_binary_bytes = 0
        queried_program_devices = 0
        try:
            self.library = C.WinDLL("OpenCL.dll")
            self._bind()
            platform, device, identity = self._select()
            error = C.c_int()
            properties = (C.c_ssize_t * 3)(CL_CONTEXT_PLATFORM, int(platform.value), 0)
            devices = (C.c_void_p * 1)(device.value)
            self.context = self.library.clCreateContext(properties, 1, devices, None, None, C.byref(error))
            check(error.value, "context")
            if not self.context:
                raise RuntimeError("null_context")
            self.ledger.append({"op": "context_create", "pointer": int(self.context)})

            source = SRC.encode()
            strings = (C.c_char_p * 1)(source)
            lengths = (C.c_size_t * 1)(len(source))
            self.program = self.library.clCreateProgramWithSource(self.context, 1, strings, lengths, C.byref(error))
            check(error.value, "program")
            if not self.program:
                raise RuntimeError("null_program")
            self.ledger.append({"op": "program_create", "source_bytes": len(source), "source_sha256": sha256_bytes(source)})

            build_code = int(self.library.clBuildProgram(self.program, 1, devices, OPTIONS.encode(), None, None))
            log_size = C.c_size_t()
            check(self.library.clGetProgramBuildInfo(self.program, device, CL_PROGRAM_BUILD_LOG, 0, None, C.byref(log_size)), "build_log_size")
            if log_size.value:
                log_buffer = C.create_string_buffer(log_size.value)
                check(self.library.clGetProgramBuildInfo(self.program, device, CL_PROGRAM_BUILD_LOG, log_size.value, log_buffer, None), "build_log")
                build_log = bytes(log_buffer.raw[: log_size.value])
            self.ledger.append(
                {
                    "op": "program_build",
                    "code": build_code,
                    "options": OPTIONS,
                    "log_bytes": len(build_log),
                    "log_sha256": sha256_bytes(build_log),
                }
            )
            check(build_code, "program_build")

            device_count = C.c_uint()
            check(self.library.clGetProgramInfo(self.program, CL_PROGRAM_NUM_DEVICES, C.sizeof(device_count), C.byref(device_count), None), "program_device_count")
            queried_program_devices = int(device_count.value)
            if queried_program_devices != 1:
                raise RuntimeError(f"program_device_count:{queried_program_devices}")
            sizes = (C.c_size_t * queried_program_devices)()
            check(self.library.clGetProgramInfo(self.program, CL_PROGRAM_BINARY_SIZES, C.sizeof(sizes), sizes, None), "binary_sizes")
            declared_binary_bytes = int(sizes[0])
            if declared_binary_bytes <= 0:
                raise RuntimeError("empty_program_binary")
            binary_buffer = (C.c_ubyte * declared_binary_bytes)()
            binary_pointers = (C.c_void_p * queried_program_devices)(C.cast(binary_buffer, C.c_void_p))
            check(self.library.clGetProgramInfo(self.program, CL_PROGRAM_BINARIES, C.sizeof(binary_pointers), binary_pointers, None), "program_binaries")
            binary = bytes(binary_buffer)
            if len(binary) != declared_binary_bytes or not binary:
                raise RuntimeError("binary_query_read_length_mismatch")
            self.ledger.append(
                {
                    "op": "program_binary_read",
                    "program_devices": queried_program_devices,
                    "declared_bytes": declared_binary_bytes,
                    "read_bytes": len(binary),
                    "sha256": sha256_bytes(binary),
                    "nonempty": True,
                }
            )
        except Exception as exc:
            evidence.update(
                {
                    "identity": identity,
                    "build_log_hex": build_log.hex(),
                    "binary_hex": binary.hex(),
                    "declared_binary_bytes": declared_binary_bytes,
                    "queried_program_devices": queried_program_devices,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            failure = exc
        else:
            failure = None
        finally:
            if self.library is not None:
                self._release_all()

        evidence.update(
            {
                "identity": identity,
                "build_log_hex": build_log.hex(),
                "build_log_sha256": sha256_bytes(build_log),
                "binary_hex": binary.hex(),
                "binary_sha256": sha256_bytes(binary),
                "binary_nonempty": bool(binary),
                "declared_binary_bytes": declared_binary_bytes,
                "read_binary_bytes": len(binary),
                "queried_program_devices": queried_program_devices,
                "cleanup_errors": list(self.cleanup_errors),
            }
        )
        if failure is not None or self.cleanup_errors:
            message = str(failure) if failure is not None else "cleanup_errors"
            raise CompileFailure(message, evidence) from failure
        if not (
            evidence["binary_nonempty"] is True
            and evidence["queried_program_devices"] == 1
            and evidence["declared_binary_bytes"] == evidence["read_binary_bytes"] > 0
            and evidence["binary_sha256"] == sha256_bytes(binary)
        ):
            raise CompileFailure("binary_positive_gate", evidence)
        return evidence


def compile_only(eligibility: dict) -> dict:
    return IntelCompileOnlyR1().compile_only(eligibility)
