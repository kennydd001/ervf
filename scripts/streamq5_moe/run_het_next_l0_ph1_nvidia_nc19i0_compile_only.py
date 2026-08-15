#!/usr/bin/env python3
"""One-shot NC19I0 NVRTC compile-only runner; import is inert."""
from __future__ import annotations

import ctypes as C
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
SCRIPT = Path(__file__).resolve()
CONTRACT_PATH = SCRIPT.with_name("het_next_l0_ph1_nvidia_nc19i0_compile_contract.py")
_SPEC = importlib.util.spec_from_file_location("nc19i0_compile_contract_runtime", CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(CONTRACT_PATH)
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
SOURCE = ROOT / "scripts/streamq5_moe/het_next_l0_ph1_nvidia_n5_kernels.cu"
SOURCE_SHA = "9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781"
SOURCE_BYTES = 6173
NVRTC = ROOT / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc64_130_0.dll"
NVRTC_SHA = "c7af6b5dbd001852d1b4a18effc6fbcfc94787eddadffea629a8333cb25b05fe"
BUILTINS = ROOT / ".venv/Lib/site-packages/nvidia/cu13/bin/x86_64/nvrtc-builtins64_133.dll"
BUILTINS_SHA = "82c703802846329d3bab3d8df06f8c956516a0eeec568033092d6c0a69b2733a"
HEADER = ROOT / ".venv/Lib/site-packages/nvidia/cu13/include/nvrtc.h"
HEADER_SHA = "316a1375c18c69c5f1857dfc154c47a58a6795ffe462d2fcb50f5272ea472d21"
PYTHON = ROOT / ".venv/Scripts/python.exe"
PYTHON_SHA = "0b471133e110cfb53a061cad528ce8e517d7b9ac41a0a396c39ad795a487fc14"
SOURCE_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_source_lock.json"
PREFLIGHT_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_preflight_lock.json"
AUTH_LOCK = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_authorization_bootstrap_lock.json"
PREFLIGHT_RESULT = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_static_preflight_result.json"
OUT = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_compile_only"
NEGATIVE = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_compile_only_negative"
FAILURES = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_compile_only_incidental_failures"
QUARANTINE = REPORTS / "het_next_l0_ph1_nvidia_nc19i0_compile_only_quarantine"
INDEPENDENT = ROOT / "scripts/streamq5_moe/verify_het_next_l0_ph1_nvidia_nc19i0_compile_only.py"
ACK = "ACK_HET_NEXT_L0_PH1_NVIDIA_NC19I0_COMPILE_ONLY_ONCE"
KIND = "het_next_l0_ph1_nvidia_nc19i0_compile_only"
LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x100
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x1000
LOAD_FLAGS = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _kernel32():
    k = C.WinDLL("kernel32", use_last_error=True)
    BOOL, DWORD, HANDLE = C.c_int32, C.c_uint32, C.c_void_p
    table = {
        "AddDllDirectory": ([C.c_wchar_p], C.c_void_p),
        "RemoveDllDirectory": ([C.c_void_p], BOOL),
        "LoadLibraryExW": ([C.c_wchar_p, HANDLE, DWORD], C.c_void_p),
        "FreeLibrary": ([C.c_void_p], BOOL),
        "GetProcAddress": ([C.c_void_p, C.c_char_p], C.c_void_p),
        "GetModuleHandleW": ([C.c_wchar_p], C.c_void_p),
        "GetCommandLineW": ([], C.c_wchar_p),
        "LocalFree": ([C.c_void_p], C.c_void_p),
    }
    for name, (args, result) in table.items():
        fn = getattr(k, name); fn.argtypes = args; fn.restype = result
    return k, table


def invocation() -> dict:
    raw = ""; native = []; parse_error = None
    try:
        kernel, _ = _kernel32(); raw = kernel.GetCommandLineW()
        shell = C.WinDLL("shell32", use_last_error=True)
        shell.CommandLineToArgvW.argtypes=[C.c_wchar_p,C.POINTER(C.c_int)]
        shell.CommandLineToArgvW.restype=C.POINTER(C.c_wchar_p)
        count=C.c_int(); pointer=shell.CommandLineToArgvW(raw,C.byref(count))
        if not pointer: raise OSError(C.get_last_error(),"CommandLineToArgvW")
        try: native=[pointer[i] for i in range(count.value)]
        finally:
            if kernel.LocalFree(C.cast(pointer,C.c_void_p)): raise OSError(C.get_last_error(),"LocalFree")
    except Exception as exc:
        parse_error = f"{type(exc).__name__}:{exc}"
    orig = list(getattr(sys, "orig_argv", []))
    argv = list(sys.argv)
    direct = (__name__ == "__main__" and __spec__ is None and __package__ in (None, "")
              and Path(__file__).resolve() == SCRIPT and argv == [str(SCRIPT), ACK]
              and len(orig) == 5 and orig[1:] == ["-I", "-B", str(SCRIPT), ACK]
              and len(native) == 5 and native[1:] == ["-I", "-B", str(SCRIPT), ACK]
              and parse_error is None and "-c" not in orig and "-m" not in orig
              and Path(sys.executable).resolve() == PYTHON.resolve())
    return {"raw": raw, "native_argv":native, "parse_error":parse_error,
            "orig_argv": orig, "sys_argv": argv,
            "sys_executable": str(Path(sys.executable).resolve()),
            "base_executable":str(Path(getattr(sys,"_base_executable",sys.executable)).resolve()),
            "prefix":str(Path(sys.prefix).resolve()),"base_prefix":str(Path(sys.base_prefix).resolve()),
            "name": __name__, "spec_is_none": __spec__ is None,
            "package": __package__, "file": str(SCRIPT), "direct": direct}


def validate_authorization(ack: str) -> tuple[bool, dict]:
    observed = {"ack": ack, "invocation": invocation()}
    if ack != ACK or not observed["invocation"]["direct"]:
        return False, observed
    try:
        lock = json.loads(AUTH_LOCK.read_text(encoding="utf-8"))
        source_lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
        preflight_lock = json.loads(PREFLIGHT_LOCK.read_text(encoding="utf-8"))
        preflight = json.loads(PREFLIGHT_RESULT.read_text(encoding="utf-8"))
        observed.update({"auth_lock": lock, "source_lock_sha256": file_sha(SOURCE_LOCK),
                         "preflight_lock": preflight_lock, "preflight_result": preflight})
        checks = (
            lock.get("kind") == "het_next_l0_ph1_nvidia_nc19i0_authorization_bootstrap_lock"
            and lock.get("execution_open") is True and lock.get("authorization_token") == ACK
            and lock.get("source_lock_sha256") == observed["source_lock_sha256"]
            and preflight_lock.get("preflight_open") is True
            and preflight.get("kind") == "het_next_l0_ph1_nvidia_nc19i0_static_preflight_result"
            and preflight.get("pass") is True and preflight.get("device_opened") is False
            and preflight.get("compiler_loaded") is False and preflight.get("payload_bytes_read") == 0
            and source_lock.get("revision") == "NC19I0" and source_lock.get("compile_open") is True
        )
        return bool(checks), observed
    except Exception as exc:
        observed["error"] = f"{type(exc).__name__}:{exc}"
        return False, observed


class WinNvrtcAdapter:
    """Direct kernel32 ownership plus cdecl wrappers; no Driver/runtime API."""
    ABI = {
        "nvrtcVersion": ([C.POINTER(C.c_int), C.POINTER(C.c_int)], C.c_int),
        "nvrtcCreateProgram": ([C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int,
                                C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)], C.c_int),
        "nvrtcCompileProgram": ([C.c_void_p, C.c_int, C.POINTER(C.c_char_p)], C.c_int),
        "nvrtcGetProgramLogSize": ([C.c_void_p, C.POINTER(C.c_size_t)], C.c_int),
        "nvrtcGetProgramLog": ([C.c_void_p, C.c_void_p], C.c_int),
        "nvrtcGetPTXSize": ([C.c_void_p, C.POINTER(C.c_size_t)], C.c_int),
        "nvrtcGetPTX": ([C.c_void_p, C.c_void_p], C.c_int),
        "nvrtcGetCUBINSize": ([C.c_void_p, C.POINTER(C.c_size_t)], C.c_int),
        "nvrtcGetCUBIN": ([C.c_void_p, C.c_void_p], C.c_int),
        "nvrtcDestroyProgram": ([C.POINTER(C.c_void_p)], C.c_int),
    }

    def __init__(self):
        self.kernel, self.kernel_abi = _kernel32()
        self.cookie = C.c_void_p(); self.module = C.c_void_p(); self.program = C.c_void_p()
        self.wrappers = {}; self.options = None; self.log_size = C.c_size_t()
        self.ptx_size = C.c_size_t(); self.cubin_size = C.c_size_t()
        self.source_buffer = None; self.name_buffer = None
        self.ownership = []; self.cleanup = []; self.version = None

    def _last_error_call(self, name, *args):
        C.set_last_error(0); value = getattr(self.kernel, name)(*args); error = C.get_last_error()
        return value, error

    def load(self):
        before = {name: int(self.kernel.GetModuleHandleW(name) or 0)
                  for name in (NVRTC.name, BUILTINS.name)}
        if any(before.values()):
            raise RuntimeError("compiler_module_preloaded")
        value, error = self._last_error_call("AddDllDirectory", str(NVRTC.parent.resolve()))
        self.cookie = C.c_void_p(value)
        self.ownership.append({"resource": "dll_directory_cookie", "identity": int(value or 0),
                               "registered": bool(value), "code": 0 if value else error})
        if not value:
            raise OSError(error, "AddDllDirectory")
        value, error = self._last_error_call("LoadLibraryExW", str(NVRTC.resolve()), None, LOAD_FLAGS)
        self.module = C.c_void_p(value)
        self.ownership.append({"resource": "nvrtc_hmodule", "identity": int(value or 0),
                               "registered": bool(value), "code": 0 if value else error})
        if not value:
            raise OSError(error, "LoadLibraryExW")
        for name, (args, result) in self.ABI.items():
            address, error = self._last_error_call("GetProcAddress", self.module, name.encode("ascii"))
            if not address:
                raise OSError(error, name)
            self.wrappers[name] = C.CFUNCTYPE(result, *args)(address)
        return {"before": before, "module": int(self.module.value or 0),
                "cookie": int(self.cookie.value or 0), "flags": LOAD_FLAGS}

    def call(self, op, *, handle, source, program_name, options):
        fn = self.wrappers[op]
        if op == "nvrtcVersion":
            major, minor = C.c_int(), C.c_int(); code = fn(C.byref(major), C.byref(minor))
            self.version = [major.value, minor.value]; return {"code": code}
        if op == "nvrtcCreateProgram":
            self.source_buffer = C.create_string_buffer(source)
            self.name_buffer = C.create_string_buffer(program_name)
            if C.sizeof(self.source_buffer) != len(source) + 1 or self.source_buffer.raw != source + b"\0":
                raise RuntimeError("source_buffer")
            if C.sizeof(self.name_buffer) != len(program_name) + 1 or self.name_buffer.raw != program_name + b"\0":
                raise RuntimeError("name_buffer")
            code = fn(C.byref(self.program), C.cast(self.source_buffer, C.c_char_p),
                      C.cast(self.name_buffer, C.c_char_p), 0, None, None)
            self.ownership.append({"resource": "nvrtc_program", "identity": int(self.program.value or 0),
                                   "registered": bool(self.program.value), "code": int(code)})
            return {"code": code, "handle": int(self.program.value or 0)}
        if op == "nvrtcCompileProgram":
            encoded = [x.encode("ascii") for x in options]
            self.options = (C.c_char_p * len(encoded))(*encoded)
            return {"code": fn(self.program, len(encoded), self.options)}
        if op == "nvrtcGetProgramLogSize":
            return {"code": fn(self.program, C.byref(self.log_size))}
        if op == "nvrtcGetProgramLog":
            buf = C.create_string_buffer(max(1, self.log_size.value))
            code = fn(self.program, C.cast(buf, C.c_void_p))
            return {"code": code, "log": bytes(buf.raw[:self.log_size.value])}
        if op == "nvrtcGetPTXSize":
            return {"code": fn(self.program, C.byref(self.ptx_size))}
        if op == "nvrtcGetPTX":
            buf = C.create_string_buffer(max(1, self.ptx_size.value))
            code = fn(self.program, C.cast(buf, C.c_void_p))
            return {"code": code, "ptx": bytes(buf.raw[:self.ptx_size.value])}
        if op == "nvrtcGetCUBINSize":
            return {"code": fn(self.program, C.byref(self.cubin_size))}
        if op == "nvrtcGetCUBIN":
            buf = (C.c_ubyte * max(1, self.cubin_size.value))()
            code = fn(self.program, C.cast(buf, C.c_void_p))
            return {"code": code, "cubin": bytes(buf[:self.cubin_size.value])}
        if op == "nvrtcDestroyProgram":
            code = fn(C.byref(self.program))
            return {"code": code, "handle": int(self.program.value or 0)}
        raise ValueError(op)

    def close(self):
        self.wrappers.clear()
        self.cleanup.append({"resource": "wrappers", "attempted": True, "code": 0})
        if self.module.value:
            identity = int(self.module.value)
            value, error = self._last_error_call("FreeLibrary", self.module)
            self.cleanup.append({"resource": "nvrtc_hmodule", "identity": identity,
                                 "attempted": True, "code": 0 if value else error})
            self.module = C.c_void_p()
        else:
            self.cleanup.append({"resource": "nvrtc_hmodule", "identity": 0,
                                 "attempted": False, "code": "not_attempted"})
        if self.cookie.value:
            identity = int(self.cookie.value)
            value, error = self._last_error_call("RemoveDllDirectory", self.cookie)
            self.cleanup.append({"resource": "dll_directory_cookie", "identity": identity,
                                 "attempted": True, "code": 0 if value else error})
            self.cookie = C.c_void_p()
        else:
            self.cleanup.append({"resource": "dll_directory_cookie", "identity": 0,
                                 "attempted": False, "code": "not_attempted"})
        after = {name: int(self.kernel.GetModuleHandleW(name) or 0)
                 for name in (NVRTC.name, BUILTINS.name)}
        self.cleanup.append({"resource": "postrelease_module_check", "attempted": True,
                             "code": 0 if not any(after.values()) else "module_still_loaded",
                             "modules": after})
        return after


def verify_candidate(directory: Path) -> bool:
    try:
        command = [str(PYTHON.resolve()), "-I", "-B", str(INDEPENDENT.resolve()),
                   "--candidate", str(Path(directory).resolve()), "--mode", "precommit",
                   "--no-write", "--token", "NC19I0_PRECOMMIT_INDEPENDENT_VERIFY"]
        completed = subprocess.run(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                   capture_output=True, check=False, timeout=120,
                                   creationflags=0x08000000)
        result = json.loads(completed.stdout)
        return completed.returncode == 0 and result.get("pass") is True and result.get("total") == 25
    except Exception:
        return False


def execute(authorization: dict) -> int:
    source = SOURCE.read_bytes()
    identities = {
        "source": {"path": str(SOURCE.resolve()), "bytes": len(source), "sha256": contract.sha256(source)},
        "nvrtc": {"path": str(NVRTC.resolve()), "bytes": NVRTC.stat().st_size, "sha256": file_sha(NVRTC)},
        "builtins": {"path": str(BUILTINS.resolve()), "bytes": BUILTINS.stat().st_size, "sha256": file_sha(BUILTINS)},
        "header": {"path": str(HEADER.resolve()), "bytes": HEADER.stat().st_size, "sha256": file_sha(HEADER)},
        "python": {"path": str(PYTHON.resolve()), "bytes": PYTHON.stat().st_size, "sha256": file_sha(PYTHON)},
    }
    if identities["source"]["bytes"] != SOURCE_BYTES or identities["source"]["sha256"] != SOURCE_SHA:
        raise RuntimeError("source_identity")
    if identities["nvrtc"]["sha256"] != NVRTC_SHA or identities["builtins"]["sha256"] != BUILTINS_SHA or identities["header"]["sha256"] != HEADER_SHA or identities["python"]["sha256"] != PYTHON_SHA:
        raise RuntimeError("toolchain_identity")
    work = REPORTS / f"het_next_l0_ph1_nvidia_nc19i0_compile_work.inprogress.{os.getpid()}.{uuid.uuid4().hex[:16]}"
    work.mkdir(parents=False)
    private = work / "private_cache"; private.mkdir()
    for subdir in contract.ENV_SUBDIRS.values():
        (private / subdir).mkdir()
    captured = contract.capture_environment(os.environ)
    replacements = contract.apply_private_environment(os.environ, private)
    history = []
    def snap(label):
        entries = contract.cache_entries(private)
        history.append({"stage": label, "entries": entries,
                        "tree_digest": contract.cache_tree_digest(entries)})
    adapter = WinNvrtcAdapter(); compile_evidence = None; primary = None; modules = None
    try:
        snap("pre_load"); modules = adapter.load()
        compile_evidence = contract.compile_with_adapter(adapter, source, snap)
    except Exception as exc:
        primary = f"{type(exc).__name__}:{exc}"
        compile_evidence = compile_evidence or {"ledger": contract.not_attempted_ledger(),
                                                "artifacts": {}, "primary": {"state": "failure", "value": primary}, "secondary": []}
    finally:
        after = adapter.close(); snap("post_release")
        restore_rows = contract.restore_environment(os.environ, captured)
    try:
        if len(history) != 12:
            raise RuntimeError("cache_history_count")
        cache_files = [x for row in history for x in row["entries"] if x["type"] == "file"]
        cleanup_ok = all(x["code"] == 0 for x in adapter.cleanup) and all(x["code"] == 0 for x in restore_rows)
        checks = contract.validate_compile_evidence(compile_evidence)
        if compile_evidence["primary"]["state"] == "none" and (not all(checks.values()) or adapter.version != [13,3]):
            failed = sorted(name for name,value in checks.items() if not value)
            if adapter.version != [13,3]: failed.append("nvrtc_version")
            compile_evidence["primary"] = {"state":"failure","value":"artifact_contract:"+",".join(failed)}
        positive = primary is None and compile_evidence["primary"]["state"] == "none" and all(checks.values()) and cleanup_ok and not cache_files
        status = "compile_positive" if positive else "compile_valid_negative" if cleanup_ok and compile_evidence["ledger"][2].get("attempted") else "incidental_failure"
        compile_record = dict(compile_evidence)
        compile_record["artifacts"] = {name:{"bytes":len(raw),"sha256":contract.sha256(raw)}
                                       for name,raw in compile_evidence["artifacts"].items()}
        result = {
            "kind": KIND if positive else KIND + "_negative", "revision": "NC19I0", "status": status,
            "terminal_valid": status in {"compile_positive", "compile_valid_negative"},
            "positive": positive, "authorization": authorization,
            "invocation": invocation(), "source_identity": identities["source"],
            "toolchain_identity": identities, "options": list(contract.OPTIONS),
            "program_name": contract.PROGRAM_NAME.decode("ascii"),
            "create_operands": {"source_bytes": len(source), "source_buffer_bytes": len(source)+1,
                                "source_terminal_nul": 1, "program_name_bytes": len(contract.PROGRAM_NAME),
                                "program_name_buffer_bytes": len(contract.PROGRAM_NAME)+1,
                                "num_headers": 0, "headers": None, "include_names": None},
            "compile": compile_record, "compile_checks": checks,
            "loader": {"calling_convention": "WinDLL_kernel32_plus_cdecl_CFUNCTYPE",
                       "load_flags": LOAD_FLAGS, "modules": modules,
                       "nvrtc_version": adapter.version,
                       "kernel32_abi": {n:{"argtypes":[getattr(t,'__name__',str(t)) for t in a],"restype":getattr(r,'__name__',str(r))} for n,(a,r) in adapter.kernel_abi.items() if n not in {"GetCommandLineW","LocalFree"}},
                       "invocation_abi": {n:{"argtypes":[getattr(t,'__name__',str(t)) for t in a],"restype":getattr(r,'__name__',str(r))} for n,(a,r) in adapter.kernel_abi.items() if n in {"GetCommandLineW","LocalFree"}},
                       "nvrtc_abi": {n:{"argtypes":[getattr(t,'__name__',str(t)) for t in a],"restype":getattr(r,'__name__',str(r))} for n,(a,r) in adapter.ABI.items()},
                       "nvrtc_abi_names": list(adapter.ABI)},
            "ownership": adapter.ownership, "cleanup": adapter.cleanup,
            "cache": {"private_root": str(private.resolve()), "environment_original": captured,
                      "environment_applied": replacements, "environment_restore": restore_rows,
                      "history": history, "history_digest": contract.cache_history_digest(history)},
            "exclusions": {"payload_bytes_read": 0, "nvcuda_driver_calls": 0,
                           "cuda_runtime_calls": 0, "device_calls": 0},
        }
        artifact_files = {}
        if positive:
            artifact_files = {"source.cu": source, "build.log": compile_evidence["artifacts"]["log"],
                              "ptx.bin": compile_evidence["artifacts"]["ptx"],
                              "cubin.bin": compile_evidence["artifacts"]["cubin"]}
        else:
            artifact_files = {"source.cu": source}
            for name, raw in compile_evidence["artifacts"].items():
                artifact_files[{"log":"build.log","ptx":"ptx.bin","cubin":"cubin.bin"}[name]] = raw
        kind = KIND if positive else KIND + "_negative"
        destination = OUT if positive else NEGATIVE
        bundle = contract.build_bundle(result, artifact_files, kind)
        contract.publish_transaction(destination, bundle, kind, verify_candidate)
        return 0 if positive else 3
    finally:
        if work.exists():
            shutil.rmtree(work)


def main() -> int:
    ack = sys.argv[1] if len(sys.argv) == 2 else ""
    authorized, evidence = validate_authorization(ack)
    if not authorized:
        return 2
    if OUT.exists() and verify_candidate(OUT):
        return 0
    if any(path.exists() for path in (OUT, NEGATIVE, FAILURES, QUARANTINE)):
        return 3
    try:
        return execute(evidence)
    except Exception as exc:
        contract.write_incidental_failure(FAILURES, {
            "kind": "het_next_l0_ph1_nvidia_nc19i0_incidental_failure", "revision": "NC19I0",
            "status": "incidental_failure", "stage": "authorized_compile",
            "error_type": type(exc).__name__, "error": str(exc),
            "device_opened": False, "driver_loaded": False,
            "compiler_loaded": False, "payload_bytes_read": 0,
            "dispositions": [], "attempt_consumed": True, "next_invocation_allowed": False,
        })
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
