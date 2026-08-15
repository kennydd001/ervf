#!/usr/bin/env python3
"""R8P2 closed preflight: Windows venv launcher/base dual identity; no device."""
from __future__ import annotations

import copy, ctypes as C, hashlib, json, os, sys, tempfile, traceback, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import preflight_het_next_l0_ph1_intel_execution_r8p1 as prior

LOCK = R / "het_next_l0_ph1_intel_execution_r8p2_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_PREREGISTRATION_2026-08-14.md"
VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8p2.py"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p2_static_preflight.json"
MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p2_static_preflight.manifest.json"
COMMIT = R / "het_next_l0_ph1_intel_execution_r8p2_static_preflight.commit.json"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p2_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r8p2_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p2_quarantine"
DIAGNOSIS = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_EXACT_INVOCATION_FAILURE_DIAGNOSIS_2026-08-14.md"
R8P1_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
R8P_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
ACK = "PH1_INTEL_EXECUTION_R8P2_DUAL_IDENTITY_CPU_PREPARATION_CLOSED"
VENV = ROOT / ".venv"; VENV_PYTHON = VENV / "Scripts/python.exe"; PYVENV = VENV / "pyvenv.cfg"
ALIAS = Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")
BASE_PREFIX = Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
BASE_BINARY = BASE_PREFIX / "python.exe"
SCRIPT = Path(__file__).resolve()
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT), "--ack", ACK]
EXPECTED_ARGV = [str(SCRIPT), "--ack", ACK]
CORE = (RESULT, MANIFEST, COMMIT)
KIND = "ph1_intel_execution_r8p2_static_preflight"
CHAIN = {
    "preflight_sha256": SCRIPT, "verifier_sha256": VERIFIER, "prereg_sha256": PREREG,
    "r8p1_diagnosis_sha256": DIAGNOSIS, "r8p1_source_audit_sha256": R8P1_AUDIT,
    "r8p_source_audit_sha256": R8P_AUDIT, "r8p1_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p1.py",
    "r8p1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p1.py",
    "r8p1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_PREREGISTRATION_2026-08-14.md",
    "r8p1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p1_lock.json",
    **{name: path for name, path in prior.CHAIN.items() if name not in {"preflight_sha256", "verifier_sha256", "prereg_sha256", "audit_sha256"}},
}
STATIC = {**prior.base.runner.LOCK_STATIC, "base_binary_sha256": "5365b422ee178f691988eb937b7abca5f48910b148f76fcce6dbaf5585c948d0", "base_binary_bytes": 172912,
          "base_alias": str(ALIAS), "base_prefix": str(BASE_PREFIX), "venv_python": str(VENV_PYTHON.resolve()), "venv_prefix": str(VENV.resolve())}

def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha256(path: Path) -> str: return sha_bytes(path.read_bytes())
def canon(value: object) -> bytes: return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
def same(a: object, b: object) -> bool: return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def parse_windows_commandline(raw: str) -> list[str]:
    kernel = C.WinDLL("kernel32", use_last_error=True); shell = C.WinDLL("shell32", use_last_error=True)
    parse = shell.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p)
    free = kernel.LocalFree; free.argtypes = (C.c_void_p,); free.restype = C.c_void_p
    count = C.c_int(); ptr = parse(raw, C.byref(count))
    if not ptr: raise C.WinError(C.get_last_error())
    try: vector = [ptr[i] for i in range(count.value)]
    finally:
        if free(C.cast(ptr, C.c_void_p)): raise C.WinError(C.get_last_error())
    return vector

def win32_vector() -> tuple[str, list[str]]:
    kernel = C.WinDLL("kernel32", use_last_error=True); get = kernel.GetCommandLineW; get.argtypes = (); get.restype = C.c_wchar_p
    raw = get(); return raw, parse_windows_commandline(raw)

def pyvenv_contract() -> dict:
    rows = {}
    for line in PYVENV.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1); rows[key.strip()] = value.strip()
    return {"sha256": sha256(PYVENV), "home": rows.get("home"), "executable": rows.get("executable"), "version": rows.get("version")}

def dual_identity() -> dict:
    raw, native = win32_vector()
    return {"native_raw": raw, "native_argv": native, "orig_argv": list(sys.orig_argv), "argv": list(sys.argv),
            "sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_executable": getattr(sys, "_base_executable", None),
            "base_prefix": sys.base_prefix, "venv_launcher_sha256": sha256(VENV_PYTHON), "pyvenv": pyvenv_contract(),
            "base_binary_sha256": sha256(BASE_BINARY), "base_binary_bytes": BASE_BINARY.stat().st_size,
            "direct_entry": __spec__ is None and (__package__ is None or __package__ == "")}

IDENTITY_KEYS = {"native_raw", "native_argv", "orig_argv", "argv", "sys_executable", "sys_prefix", "base_executable", "base_prefix", "venv_launcher_sha256", "pyvenv", "base_binary_sha256", "base_binary_bytes", "direct_entry"}
def identity_valid(x: dict) -> bool:
    return set(x) == IDENTITY_KEYS and isinstance(x["native_raw"], str) and bool(x["native_raw"]) and parse_windows_commandline(x["native_raw"]) == EXPECTED_NATIVE and x["native_argv"] == EXPECTED_NATIVE and x["orig_argv"] == EXPECTED_NATIVE and x["argv"] == EXPECTED_ARGV and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == prior.base.runner.PYTHON_SHA and x["pyvenv"] == {"sha256": prior.base.runner.PYVENV_SHA, "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is True

def identity_mutations(row: dict) -> list[str]:
    cases = {
        "native_alias": lambda x: x["native_argv"].__setitem__(0, "C:/wrong/python.exe"), "orig_alias": lambda x: x["orig_argv"].__setitem__(0, "C:/wrong/python.exe"),
        "venv_launcher": lambda x: x.__setitem__("sys_executable", "C:/wrong/python.exe"), "venv_launcher_hash": lambda x: x.__setitem__("venv_launcher_sha256", "0" * 64), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "C:/wrong"),
        "base_executable": lambda x: x.__setitem__("base_executable", "C:/wrong/python.exe"), "base_prefix": lambda x: x.__setitem__("base_prefix", "C:/wrong"),
        "pyvenv_home": lambda x: x["pyvenv"].__setitem__("home", "C:/wrong"), "pyvenv_executable": lambda x: x["pyvenv"].__setitem__("executable", "C:/wrong"), "pyvenv_hash": lambda x: x["pyvenv"].__setitem__("sha256", "0" * 64), "base_hash": lambda x: x.__setitem__("base_binary_sha256", "0" * 64), "base_bytes": lambda x: x.__setitem__("base_binary_bytes", 0),
        "flag_order": lambda x: x["native_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["native_argv"].__setitem__(3, "-c"),
        "script": lambda x: x["argv"].__setitem__(0, "C:/wrong.py"), "ack": lambda x: x["argv"].__setitem__(2, "WRONG"),
        "extra": lambda x: x["orig_argv"].append("extra"), "parsed_extra": lambda x: x["native_argv"].append("extra"), "raw_extra": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "not_direct": lambda x: x.__setitem__("direct_entry", False),
    }
    rejected = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not identity_valid(candidate): rejected.append(name)
    return rejected

def r8p1_negative() -> dict:
    absent = tuple(prior.CORE) + (prior.VERIFY_RESULT, prior.FAILED, prior.QUARANTINE)
    return {"exit_code": 1, "failure_artifact": None, "error": "RuntimeError:exact_invocation", "traceback_marker": "preflight_het_next_l0_ph1_intel_execution_r8p1.py:188", "diagnosis_sha256": sha256(DIAGNOSIS), "diagnosis_bytes": DIAGNOSIS.stat().st_size,
            "result_paths_absent": all(not p.exists() for p in absent), "device_opened": False, "cpu_payload_read": False}

def topology() -> dict:
    absent = tuple(prior.BASE_R8_PATHS) + tuple(prior.CORE) + (prior.VERIFY_RESULT, prior.FAILED, prior.QUARANTINE) + CORE + (VERIFY_RESULT, FAILED, QUARANTINE)
    return {"absent": {str(p): p.exists() for p in absent}, "temps": sorted(str(p) for p in R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*")),
            "family": sorted(str(p) for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8"))}

def topology_clean(x: dict) -> bool:
    expected = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", prior.LOCK, LOCK))
    return set(x) == {"absent", "temps", "family"} and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == expected

def atomic_create(path: Path, data: bytes) -> None:
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as h: h.write(data); h.flush(); os.fsync(h.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()

def verify_bundle(result=RESULT, manifest=MANIFEST, commit=COMMIT) -> bool:
    if not all(p.is_file() for p in (result, manifest, commit)): return False
    rb = result.read_bytes(); mb = manifest.read_bytes()
    return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(commit.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}

def quarantine_core(paths: tuple[Path, ...], root=QUARANTINE) -> list[dict]:
    existing = [p for p in paths if p.exists()]
    if not existing: return []
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); rows = []
    for path in existing:
        target = attempt / path.name; os.replace(path, target); rows.append({"name": path.name, "sha256": sha256(target), "bytes": target.stat().st_size})
    return rows

def publish(row: dict, result=RESULT, manifest=MANIFEST, commit=COMMIT) -> None:
    if any(p.exists() for p in (result, manifest, commit)): raise FileExistsError("bundle_target")
    rb = canon(row); mb = canon({"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]}); cb = canon({"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)})
    try:
        atomic_create(result, rb); atomic_create(manifest, mb); atomic_create(commit, cb)
        if not verify_bundle(result, manifest, commit): raise RuntimeError("bundle_verify")
    except Exception:
        quarantine_core((result, manifest, commit), QUARANTINE); raise

def atomic_failure(stage: str, error: BaseException, invocation: dict | None, root=FAILED) -> Path:
    row = {"kind": "ph1_intel_execution_r8p2_early_failure", "status": "valid_protocol_negative", "stage": stage,
           "error": f"{type(error).__name__}:{error}", "traceback": traceback.format_exc()[-32768:], "device_opened": False,
           "compiler_opened": False, "cpu_payload_read": False, "invocation": invocation, "disposition": "bounded_create_new", "diagnosis_sha256": sha256(DIAGNOSIS)}
    data = canon(row)
    if len(data) > 65536: raise RuntimeError("failure_cap")
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir()
    try: atomic_create(attempt / "failure.json", data)
    except Exception:
        if attempt.exists() and not any(attempt.iterdir()): attempt.rmdir()
        raise
    return attempt / "failure.json"

def failure_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "failed"; fake = RuntimeError("fixture")
        try: raise fake
        except RuntimeError as exc: path = atomic_failure("exact_invocation", exc, {"fixture": True}, root)
        row = json.loads(path.read_text()); first = path.read_bytes(); second = atomic_failure("runtime", fake, None, root)
        return {"one_file_each": len(list(root.rglob("failure.json"))) == 2, "bounded": len(first) <= 65536 and second.stat().st_size <= 65536,
                "schema": set(row) == {"kind", "status", "stage", "error", "traceback", "device_opened", "compiler_opened", "cpu_payload_read", "invocation", "disposition", "diagnosis_sha256"},
                "no_overwrite": path.read_bytes() == first and path != second}

def main() -> int:
    if sys.argv != EXPECTED_ARGV: return 3
    invocation = None; stage = "exact_invocation"
    try:
        invocation = dual_identity()
        if not identity_valid(invocation): raise RuntimeError("exact_invocation")
        pre = topology()
        if not topology_clean(pre): raise RuntimeError("pre_run_topology")
        stage = "runtime"; runtime = prior.base.runner.collect_runtime()
        if runtime["available"] < 16 * 2**30: raise RuntimeError("start_ram")
        wheels = {"psutil": prior.base.verify_wheel_record(prior.base.runner.RUNTIME_FILES["psutil_record"][0]), "numpy": prior.base.verify_wheel_record(prior.base.runner.RUNTIME_FILES["numpy_record"][0])}
        stage = "cpu_preparation"; preparation = prior.base.preparation_summary(); runtime_ok, runtime_rejected = prior.base.runtime_mutations(runtime)
        stage = "simulations"; transaction = prior.transaction_simulation(); failure = failure_simulation(); negative = r8p1_negative()
        lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}
        names = list({"native_alias", "orig_alias", "venv_launcher", "venv_launcher_hash", "venv_prefix", "base_executable", "base_prefix", "pyvenv_home", "pyvenv_executable", "pyvenv_hash", "base_hash", "base_bytes", "flag_order", "trampoline", "script", "ack", "extra", "parsed_extra", "raw_extra", "not_direct"})
        checks = {"dual_identity": identity_valid(invocation), "identity_mutations": set(identity_mutations(invocation)) == set(names), "hash_bindings": all(lock.get(k) == v for k, v in observed.items()),
                  "closed_pending": lock.get("kind") == "ph1_intel_execution_r8p2_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING",
                  "runtime_lock": all(lock.get(k) == v for k, v in STATIC.items()), "exact_runtime": prior.base.runner.validate_runtime(runtime), "start_ram": runtime["available"] >= 16 * 2**30,
                  "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899,
                  "runtime_mutations": runtime_ok and len(runtime_rejected) == 10, "r7d1_failure": prior.base.runner.prior_failure_valid(), "r8p1_protocol_negative": all((negative["exit_code"] == 1, negative["failure_artifact"] is None, negative["result_paths_absent"], negative["diagnosis_sha256"] == "ef42c92407142893532daab1ea5dd7463bec7b384e796fcfe56df59dbbf7a6a7")),
                  "cpu_preparation": prior.base.validate_preparation(preparation), "transaction_simulation": all(transaction.values()), "failure_simulation": all(failure.values()), "pre_run_topology": topology_clean(pre), "base_clean": prior.base.runner.clean_now()}
        row = {"kind": KIND, "ack": ACK, "identity": invocation, "pre_run_topology": pre, "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks),
               "runtime": runtime, "wheel_records": wheels, "preparation": preparation, "preparation_digest": prior.base.sha_bytes(prior.base.canon(preparation)), "transaction_simulation": transaction, "failure_simulation": failure,
               "r8p1_protocol_negative": negative, "runtime_mutations_rejected": runtime_rejected, "no_compiler_device": True, "cpu_payload_read": True}
        stage = "publish"; publish(row); print(json.dumps(row, indent=2)); return 0 if row["pass"] else 3
    except Exception as exc:
        if not FAILED.exists():
            try: atomic_failure(stage, exc, invocation)
            except Exception: pass
        raise

if __name__ == "__main__": raise SystemExit(main())
