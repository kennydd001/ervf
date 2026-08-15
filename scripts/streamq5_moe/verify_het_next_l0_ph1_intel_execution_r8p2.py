#!/usr/bin/env python3
"""Independent R8P2 verifier with its own exact dual-identity vector."""
from __future__ import annotations

import copy, ctypes as C, hashlib, json, os, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import verify_het_next_l0_ph1_intel_execution_r8p1 as prior

PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r8p2.py"
LOCK = R / "het_next_l0_ph1_intel_execution_r8p2_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_PREREGISTRATION_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p2_static_preflight.json"
MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p2_static_preflight.manifest.json"
COMMIT = R / "het_next_l0_ph1_intel_execution_r8p2_static_preflight.commit.json"
OUTPUT = R / "het_next_l0_ph1_intel_execution_r8p2_independent_verification.json"
FAILED = R / "het_next_l0_ph1_intel_execution_r8p2_failed_attempts"
QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p2_quarantine"
DIAGNOSIS = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_EXACT_INVOCATION_FAILURE_DIAGNOSIS_2026-08-14.md"
R8P1_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
R8P_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
ACK = "PH1_INTEL_EXECUTION_R8P2_DUAL_IDENTITY_CPU_PREPARATION_CLOSED"
ALIAS = Path(r"C:\Users\de_do\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")
BASE_PREFIX = Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0")
BASE_BINARY = BASE_PREFIX / "python.exe"; VENV = ROOT / ".venv"; VENV_PYTHON = VENV / "Scripts/python.exe"; PYVENV = VENV / "pyvenv.cfg"
SCRIPT = Path(__file__).resolve(); EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT)]; EXPECTED_ARGV = [str(SCRIPT)]
PREFLIGHT_NATIVE = [str(ALIAS), "-I", "-B", str(PREFLIGHT.resolve()), "--ack", ACK]; PREFLIGHT_ARGV = [str(PREFLIGHT.resolve()), "--ack", ACK]
KIND = "ph1_intel_execution_r8p2_static_preflight"
CHECK_NAMES = {"dual_identity", "identity_mutations", "hash_bindings", "closed_pending", "runtime_lock", "exact_runtime", "start_ram", "wheel_records", "runtime_mutations", "r7d1_failure", "r8p1_protocol_negative", "cpu_preparation", "transaction_simulation", "failure_simulation", "pre_run_topology", "base_clean"}
PRE_ABSENT = (
    R / "het_next_l0_ph1_intel_execution_r8", R / "het_next_l0_ph1_intel_execution_r8_failed_attempts", R / "het_next_l0_ph1_intel_execution_r8_quarantine",
    R / "het_next_l0_ph1_intel_execution_r8_independent_verification.json", R / "het_next_l0_ph1_intel_execution_r8_static_preflight.json", R / "het_next_l0_ph1_intel_execution_r8p_independent_verification.json",
    prior.RESULT, prior.MANIFEST, prior.COMMIT, prior.OUTPUT, prior.FAILED, prior.QUARANTINE,
    RESULT, MANIFEST, COMMIT, OUTPUT, FAILED, QUARANTINE,
)
CHAIN = {
    "preflight_sha256": PREFLIGHT, "verifier_sha256": SCRIPT, "prereg_sha256": PREREG,
    "r8p1_diagnosis_sha256": DIAGNOSIS, "r8p1_source_audit_sha256": R8P1_AUDIT, "r8p_source_audit_sha256": R8P_AUDIT,
    "r8p1_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p1.py", "r8p1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p1.py",
    "r8p1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_PREREGISTRATION_2026-08-14.md", "r8p1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p1_lock.json",
    "r8_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8.py",
    "r8_preflight_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p.py", "r8_physical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8.py",
    "r8_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md", "r8_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8_lock.json",
    "runtime_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md",
    "r7d1_failure_sha256": prior.FAILURE, "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py", "numerical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py",
    "cpu_result_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.json", "cpu_raw_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.safetensors",
}
STATIC = {**prior.LOCK_STATIC, "base_binary_sha256": "5365b422ee178f691988eb937b7abca5f48910b148f76fcce6dbaf5585c948d0", "base_binary_bytes": 172912,
          "base_alias": str(ALIAS), "base_prefix": str(BASE_PREFIX), "venv_python": str(VENV_PYTHON.resolve()), "venv_prefix": str(VENV.resolve())}

def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha256(path: Path) -> str: return sha_bytes(path.read_bytes())
def canon(x: object) -> bytes: return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()
def same(a: object, b: object) -> bool: return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def parse_windows_commandline(raw: str) -> list[str]:
    k = C.WinDLL("kernel32", use_last_error=True); s = C.WinDLL("shell32", use_last_error=True)
    parse = s.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p)
    free = k.LocalFree; free.argtypes = (C.c_void_p,); free.restype = C.c_void_p
    n = C.c_int(); ptr = parse(raw, C.byref(n))
    if not ptr: raise C.WinError(C.get_last_error())
    try: vector = [ptr[i] for i in range(n.value)]
    finally:
        if free(C.cast(ptr, C.c_void_p)): raise C.WinError(C.get_last_error())
    return vector

def native_vector() -> tuple[str, list[str]]:
    k = C.WinDLL("kernel32", use_last_error=True); get = k.GetCommandLineW; get.argtypes = (); get.restype = C.c_wchar_p
    raw = get(); return raw, parse_windows_commandline(raw)

def cfg() -> dict:
    values = {}
    for line in PYVENV.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1); values[k.strip()] = v.strip()
    return values

def verifier_identity() -> bool:
    raw, parsed = native_vector(); values = cfg()
    return bool(raw) and parsed == EXPECTED_NATIVE and list(sys.orig_argv) == EXPECTED_NATIVE and list(sys.argv) == EXPECTED_ARGV and same(sys.executable, str(VENV_PYTHON.resolve())) and same(sys.prefix, str(VENV.resolve())) and same(getattr(sys, "_base_executable", None), str(ALIAS)) and same(sys.base_prefix, str(BASE_PREFIX)) and sha256(VENV_PYTHON) == STATIC["python_sha256"] and sha256(PYVENV) == STATIC["pyvenv_sha256"] and values.get("home", "").casefold() == str(ALIAS.parent).casefold() and values.get("executable", "").casefold() == str(ALIAS).casefold() and BASE_BINARY.stat().st_size == 172912 and sha256(BASE_BINARY) == STATIC["base_binary_sha256"] and __spec__ is None and (__package__ is None or __package__ == "")

IDENTITY_KEYS = {"native_raw", "native_argv", "orig_argv", "argv", "sys_executable", "sys_prefix", "base_executable", "base_prefix", "venv_launcher_sha256", "pyvenv", "base_binary_sha256", "base_binary_bytes", "direct_entry"}
def stored_identity(x: dict) -> bool:
    return set(x) == IDENTITY_KEYS and bool(x["native_raw"]) and parse_windows_commandline(x["native_raw"]) == PREFLIGHT_NATIVE and x["native_argv"] == PREFLIGHT_NATIVE and x["orig_argv"] == PREFLIGHT_NATIVE and x["argv"] == PREFLIGHT_ARGV and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == 172912 and x["direct_entry"] is True

def lock_valid() -> bool:
    lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}
    return set(lock) == {"kind", "execution_open", "audit_token", *STATIC, *observed} and lock["kind"] == "ph1_intel_execution_r8p2_lock" and lock["execution_open"] is False and lock["audit_token"] == "PENDING" and all(lock.get(k) == v for k, v in STATIC.items()) and all(lock.get(k) == v for k, v in observed.items()) and observed["r8p1_diagnosis_sha256"] == "ef42c92407142893532daab1ea5dd7463bec7b384e796fcfe56df59dbbf7a6a7" and observed["r8p1_source_audit_sha256"] == "85e03d967700d500e3a51d791901110f30ad8be6b9b62723f6f84f9fc610a28e"

def bundle_valid() -> bool:
    if not all(p.is_file() for p in (RESULT, MANIFEST, COMMIT)): return False
    rb = RESULT.read_bytes(); mb = MANIFEST.read_bytes()
    return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": RESULT.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(COMMIT.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}

def topology_valid() -> bool:
    absent = tuple(prior.RESULT.parent / n for n in ())
    forbidden = (prior.RESULT, prior.MANIFEST, prior.COMMIT, prior.OUTPUT, prior.FAILED, prior.QUARANTINE, FAILED, QUARANTINE, OUTPUT)
    expected = {R / "het_next_l0_ph1_intel_execution_r8_lock.json", R / "het_next_l0_ph1_intel_execution_r8p1_lock.json", LOCK, RESULT, MANIFEST, COMMIT}
    family = {p for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8")}
    return all(not p.exists() for p in forbidden) and family == expected and not list(R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*"))

def protocol_negative_valid(x: dict) -> bool:
    return x == {"exit_code": 1, "failure_artifact": None, "error": "RuntimeError:exact_invocation", "traceback_marker": "preflight_het_next_l0_ph1_intel_execution_r8p1.py:188", "diagnosis_sha256": "ef42c92407142893532daab1ea5dd7463bec7b384e796fcfe56df59dbbf7a6a7", "diagnosis_bytes": 4637, "result_paths_absent": True, "device_opened": False, "cpu_payload_read": False}

def pre_topology_valid(x: dict) -> bool:
    return x == {"absent": {str(p): False for p in PRE_ABSENT}, "temps": [], "family": sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", R / "het_next_l0_ph1_intel_execution_r8p1_lock.json", LOCK))}

def result_valid(row: dict, preparation: dict, wheels: dict) -> bool:
    checks = row.get("checks", {})
    return set(row) == {"kind", "ack", "identity", "pre_run_topology", "checks", "pass", "passed", "total", "runtime", "wheel_records", "preparation", "preparation_digest", "transaction_simulation", "failure_simulation", "r8p1_protocol_negative", "runtime_mutations_rejected", "no_compiler_device", "cpu_payload_read"} and row["kind"] == KIND and row["ack"] == ACK and stored_identity(row["identity"]) and pre_topology_valid(row["pre_run_topology"]) and set(checks) == CHECK_NAMES and all(v is True for v in checks.values()) and row["pass"] is True and row["passed"] == row["total"] == len(CHECK_NAMES) and prior.base.runtime_static_valid(row["runtime"]) and row["runtime"]["available"] >= 16 * 2**30 and row["wheel_records"] == wheels and row["preparation"] == preparation and row["preparation_digest"] == prior.base.PREPARATION_DIGEST == sha_bytes(canon(preparation)) and all(row["transaction_simulation"].values()) and set(row["failure_simulation"]) == {"one_file_each", "bounded", "schema", "no_overwrite"} and all(row["failure_simulation"].values()) and protocol_negative_valid(row["r8p1_protocol_negative"]) and row["runtime_mutations_rejected"] == ["python_path", "python_hash", "isolation", "bytecode", "pyvenv", "psutil_native", "psutil_record", "numpy_version", "numpy_record", "ram"] and row["no_compiler_device"] is True and row["cpu_payload_read"] is True

def mutations(row: dict, preparation: dict, wheels: dict) -> list[str]:
    cases = {"kind": lambda x: x.__setitem__("kind", "wrong"), "native": lambda x: x["identity"]["native_argv"].append("extra"), "venv": lambda x: x["identity"].__setitem__("sys_executable", "wrong"), "base": lambda x: x["identity"].__setitem__("base_binary_sha256", "0" * 64), "topology": lambda x: x["pre_run_topology"]["temps"].append("stale"), "check": lambda x: x["checks"].__setitem__("dual_identity", False), "prior": lambda x: x["r8p1_protocol_negative"].__setitem__("exit_code", 0), "failure": lambda x: x["failure_simulation"].__setitem__("bounded", False), "preparation": lambda x: x.__setitem__("preparation_digest", "0" * 64)}
    out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not result_valid(candidate, preparation, wheels): out.append(name)
    return out

def atomic_create(path: Path, data: bytes) -> None:
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as h: h.write(data); h.flush(); os.fsync(h.fileno())
        os.link(temp, path)
    finally:
        if temp.exists(): temp.unlink()

def main() -> int:
    if not verifier_identity(): return 3
    raw = RESULT.read_bytes(); row = json.loads(raw); preparation = prior.base.independent_preparation(); wheels = {"psutil": prior.base.wheel_record(prior.base.FILES["psutil_record"][0]), "numpy": prior.base.wheel_record(prior.base.FILES["numpy_record"][0])}
    rejected = mutations(row, preparation, wheels)
    checks = {"exact_verifier_dual_identity": True, "lock_chain": lock_valid(), "bundle": bundle_valid(), "topology": topology_valid(), "r7d1_failure_bundle": prior.failure_bundle_valid(prior.FAILURE_ROOT),
              "r8p1_negative_bound": sha256(DIAGNOSIS) == "ef42c92407142893532daab1ea5dd7463bec7b384e796fcfe56df59dbbf7a6a7" and not any(p.exists() for p in (prior.RESULT, prior.MANIFEST, prior.COMMIT, prior.OUTPUT, prior.FAILED, prior.QUARANTINE)),
              "runtime": prior.base.runtime_static_valid(prior.base.collect_runtime()), "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899,
              "preparation": sha_bytes(canon(preparation)) == prior.base.PREPARATION_DIGEST, "result": result_valid(row, preparation, wheels), "mutations": rejected == ["kind", "native", "venv", "base", "topology", "check", "prior", "failure", "preparation"]}
    output = {"kind": "ph1_intel_execution_r8p2_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "result_sha256": sha_bytes(raw), "manifest_sha256": sha256(MANIFEST), "commit_sha256": sha256(COMMIT), "no_compiler_device": True}
    atomic_create(OUTPUT, canon(output)); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3

if __name__ == "__main__": raise SystemExit(main())
