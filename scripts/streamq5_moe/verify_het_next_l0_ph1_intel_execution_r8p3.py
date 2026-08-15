#!/usr/bin/env python3
"""Independent R8P3 verifier: live identity, exact result and own transactions."""
from __future__ import annotations

import ast, copy, ctypes as C, hashlib, json, os, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import verify_het_next_l0_ph1_intel_execution_r8p2 as prior

SCRIPT = Path(__file__).resolve(); PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r8p3.py"; LOCK = R / "het_next_l0_ph1_intel_execution_r8p3_lock.json"
PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P3_PREREGISTRATION_2026-08-14.md"; AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p3_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p3_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p3_static_preflight.commit.json"
OUTPUT = R / "het_next_l0_ph1_intel_execution_r8p3_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p3_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p3_quarantine"
ACK = "PH1_INTEL_EXECUTION_R8P3_CURRENT_TRANSACTION_STATIC_CPU_PREPARATION_CLOSED"; KIND = "ph1_intel_execution_r8p3_static_preflight"
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; PYVENV = prior.PYVENV
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT)]; EXPECTED_ARGV = [str(SCRIPT)]
PREFLIGHT_NATIVE = [str(ALIAS), "-I", "-B", str(PREFLIGHT.resolve()), "--ack", ACK]; PREFLIGHT_ARGV = [str(PREFLIGHT.resolve()), "--ack", ACK]
STATIC = dict(prior.STATIC); TX_KEYS = {"clean_commit", "repeat_rejected_unchanged", "stale_temp_cleaned", "prelink_failure_clean", "postlink_interruption_recovered", "partial_publish_quarantined", "committed_bytes_immutable"}
VERIFY_TX_KEYS = {"clean_create", "repeat_preserved", "stale_cleaned", "prelink_clean", "postlink_recovered", "topology_mutations"}
CHECK_NAMES = {"dual_identity", "identity_mutations", "hash_bindings", "closed_pending", "runtime_lock", "runtime", "start_ram", "wheel_records", "runtime_mutations", "r7d1_failure", "cpu_preparation", "current_transactions", "static_no_model_compiler_opencl_device", "topology", "base_clean"}
CHAIN = {"preflight_sha256": PREFLIGHT, "verifier_sha256": SCRIPT, "prereg_sha256": PREREG, "r8p2_audit_sha256": AUDIT,
         "r8p2_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p2.py", "r8p2_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p2.py",
         "r8p2_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_PREREGISTRATION_2026-08-14.md", "r8p2_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p2_lock.json",
         "r8p1_diagnosis_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_EXACT_INVOCATION_FAILURE_DIAGNOSIS_2026-08-14.md", "r8p1_source_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md",
         "r8p_source_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r8p1_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p1.py",
         "r8p1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p1.py", "r8p1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_PREREGISTRATION_2026-08-14.md", "r8p1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p1_lock.json",
         "r8_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p.py",
         "r8_physical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8.py", "r8_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md", "r8_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8_lock.json",
         "runtime_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md", "r7d1_failure_sha256": prior.prior.FAILURE,
         "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py", "numerical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py", "cpu_result_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.json", "cpu_raw_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.safetensors"}
SOURCE_PATHS = (PREFLIGHT, SCRIPT, S / "preflight_het_next_l0_ph1_intel_execution_r8p2.py", S / "preflight_het_next_l0_ph1_intel_execution_r8p1.py", S / "preflight_het_next_l0_ph1_intel_execution_r8.py", S / "run_het_next_l0_ph1_intel_execution_r8.py")

def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha256(path: Path) -> str: return sha_bytes(path.read_bytes())
def canon(x: object) -> bytes: return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()
def same(a: object, b: object) -> bool: return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def parse_commandline(raw: str) -> list[str]:
    k = C.WinDLL("kernel32", use_last_error=True); s = C.WinDLL("shell32", use_last_error=True); parse = s.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p); free = k.LocalFree; free.argtypes = (C.c_void_p,); free.restype = C.c_void_p
    n = C.c_int(); ptr = parse(raw, C.byref(n))
    if not ptr: raise C.WinError(C.get_last_error())
    try: return [ptr[i] for i in range(n.value)]
    finally:
        if free(C.cast(ptr, C.c_void_p)): raise C.WinError(C.get_last_error())

IDENTITY_KEYS = {"native_raw", "native_argv", "orig_argv", "argv", "sys_executable", "sys_prefix", "base_executable", "base_prefix", "venv_launcher_sha256", "pyvenv", "base_binary_sha256", "base_binary_bytes", "direct_entry"}
def make_live_identity() -> dict:
    k = C.WinDLL("kernel32", use_last_error=True); get = k.GetCommandLineW; get.argtypes = (); get.restype = C.c_wchar_p; raw = get(); values = {}
    for line in PYVENV.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1); values[key.strip()] = value.strip()
    return {"native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv), "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix,
            "venv_launcher_sha256": sha256(VENV_PYTHON), "pyvenv": {"sha256": sha256(PYVENV), "home": values.get("home"), "executable": values.get("executable"), "version": values.get("version")}, "base_binary_sha256": sha256(BASE_BINARY), "base_binary_bytes": BASE_BINARY.stat().st_size, "direct_entry": __spec__ is None and (__package__ is None or __package__ == "")}

def identity_valid(x: dict, native: list[str], argv: list[str]) -> bool:
    return set(x) == IDENTITY_KEYS and bool(x["native_raw"]) and parse_commandline(x["native_raw"]) == native and x["native_argv"] == native and x["orig_argv"] == native and x["argv"] == argv and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is True

def live_identity_mutations(row: dict) -> list[str]:
    cases = {"native": lambda x: x["native_argv"].append("x"), "raw": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "orig": lambda x: x["orig_argv"].append("x"), "argv": lambda x: x["argv"].append("x"), "venv": lambda x: x.__setitem__("sys_executable", "wrong"), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "wrong"), "base": lambda x: x.__setitem__("base_executable", "wrong"), "base_prefix": lambda x: x.__setitem__("base_prefix", "wrong"), "flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "script": lambda x: x["argv"].__setitem__(0, "wrong"), "direct": lambda x: x.__setitem__("direct_entry", False)}
    rejected = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not identity_valid(candidate, EXPECTED_NATIVE, EXPECTED_ARGV): rejected.append(name)
    return rejected

def atomic_create(path: Path, data: bytes, *, link_fn=os.link, unlink_fn=Path.unlink) -> None:
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as h: h.write(data); h.flush(); os.fsync(h.fileno())
        link_fn(temp, path)
    finally:
        if temp.exists(): unlink_fn(temp)

def cleanup_temps(root: Path, stem: str) -> list[str]:
    out = []
    for path in sorted(root.glob(stem + ".inprogress.*")): path.unlink(); out.append(path.name)
    return out

def own_transaction_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); target = root / "verify.json"; atomic_create(target, b"one"); frozen = target.read_bytes(); repeat = False
        try: atomic_create(target, b"two")
        except FileExistsError: repeat = target.read_bytes() == frozen
        stale = root / "stale.json.inprogress.fixed"; stale.write_bytes(b"s"); stale_ok = cleanup_temps(root, "stale.json") == [stale.name]
        def bad_link(_a, _b): raise OSError("link")
        pre = root / "pre.json"; pre_ok = False
        try: atomic_create(pre, b"x", link_fn=bad_link)
        except OSError: pre_ok = not pre.exists() and not list(root.glob(pre.name + ".inprogress.*"))
        def bad_unlink(_p): raise OSError("unlink")
        post = root / "post.json"; post_ok = False
        try: atomic_create(post, b"y", unlink_fn=bad_unlink)
        except OSError:
            before = list(root.glob(post.name + ".inprogress.*")); cleanup_temps(root, post.name); post_ok = post.read_bytes() == b"y" and len(before) == 1
        topology = {"baseline": topology_with(set(EXPECTED_FAMILY_AFTER)), "extra": not topology_with(set(EXPECTED_FAMILY_AFTER) | {R / "unexpected"}), "missing": not topology_with(set(EXPECTED_FAMILY_AFTER) - {RESULT})}
        return {"clean_create": target.is_file(), "repeat_preserved": repeat, "stale_cleaned": stale_ok, "prelink_clean": pre_ok, "postlink_recovered": post_ok, "topology_mutations": all(topology.values())}

def bundle_valid() -> bool:
    if not all(p.is_file() for p in (RESULT, MANIFEST, COMMIT)): return False
    rb = RESULT.read_bytes(); mb = MANIFEST.read_bytes()
    return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": RESULT.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(COMMIT.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}

EXPECTED_FAMILY_AFTER = {R / "het_next_l0_ph1_intel_execution_r8_lock.json", R / "het_next_l0_ph1_intel_execution_r8p1_lock.json", R / "het_next_l0_ph1_intel_execution_r8p2_lock.json", LOCK, RESULT, MANIFEST, COMMIT}
def topology_with(family: set[Path]) -> bool: return family == EXPECTED_FAMILY_AFTER
def topology_valid() -> bool:
    family = {p for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8")}
    return topology_with(family) and not FAILED.exists() and not QUARANTINE.exists() and not OUTPUT.exists() and not list(R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*"))

def static_contract() -> bool:
    forbidden_imports = {"pyopencl", "cupy", "torch", "transformers"}; forbidden_calls = {"Popen", "run", "check_call", "check_output", "system", "startfile", "CDLL", "LoadLibrary", "clBuildProgram", "clCreateContext", "clEnqueueNDRangeKernel", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"}; literals = set()
    for path in SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8")); imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in ([*n.names] if isinstance(n, ast.Import) else [ast.alias(name=n.module or "")])}; calls = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                calls.append(n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else "")
                if isinstance(n.func, ast.Attribute) and n.func.attr == "WinDLL" and n.args and isinstance(n.args[0], ast.Constant): literals.add(n.args[0].value)
        if imports & forbidden_imports or set(calls) & forbidden_calls: return False
    return literals == {"kernel32", "shell32"}

def lock_valid() -> bool:
    lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}
    return set(lock) == {"kind", "execution_open", "audit_token", *STATIC, *observed} and lock["kind"] == "ph1_intel_execution_r8p3_lock" and lock["execution_open"] is False and lock["audit_token"] == "PENDING" and all(lock.get(k) == v for k, v in STATIC.items()) and all(lock.get(k) == v for k, v in observed.items()) and observed["r8p2_audit_sha256"] == "9cb07a4bb35c335afb0be388c67167944e414c44e3d37685027b838b3d0c5be5"

def stored_topology_valid(x: dict) -> bool:
    absent_paths = tuple(prior.prior.RESULT.parent / n for n in ())
    expected_locks = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", R / "het_next_l0_ph1_intel_execution_r8p1_lock.json", R / "het_next_l0_ph1_intel_execution_r8p2_lock.json", LOCK))
    return set(x) == {"absent", "temps", "family"} and all(v is False for v in x["absent"].values()) and len(x["absent"]) == 24 and x["temps"] == [] and x["family"] == expected_locks

def result_valid(row: dict, preparation: dict, wheels: dict) -> bool:
    checks = row.get("checks", {}); tx = row.get("transaction_simulation")
    return set(row) == {"kind", "ack", "identity", "pre_run_topology", "checks", "pass", "passed", "total", "runtime", "wheel_records", "preparation", "preparation_digest", "transaction_simulation", "static_contract", "identity_mutations_rejected", "runtime_mutations_rejected", "cpu_frozen_slice_read", "model_forward", "compiler_opened", "opencl_opened", "device_opened"} and row["kind"] == KIND and row["ack"] == ACK and identity_valid(row["identity"], PREFLIGHT_NATIVE, PREFLIGHT_ARGV) and stored_topology_valid(row["pre_run_topology"]) and set(checks) == CHECK_NAMES and all(v is True for v in checks.values()) and row["pass"] is True and row["passed"] == row["total"] == len(CHECK_NAMES) and prior.prior.base.runtime_static_valid(row["runtime"]) and row["runtime"]["available"] >= 16 * 2**30 and row["wheel_records"] == wheels and row["preparation"] == preparation and row["preparation_digest"] == prior.prior.base.PREPARATION_DIGEST == sha_bytes(canon(preparation)) and isinstance(tx, dict) and set(tx) == TX_KEYS and all(v is True for v in tx.values()) and isinstance(row["static_contract"], dict) and row["static_contract"].get("pass") is True and row["identity_mutations_rejected"] == ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "direct"] and row["runtime_mutations_rejected"] == ["python_path", "python_hash", "isolation", "bytecode", "pyvenv", "psutil_native", "psutil_record", "numpy_version", "numpy_record", "ram"] and row["cpu_frozen_slice_read"] is True and row["model_forward"] is row["compiler_opened"] is row["opencl_opened"] is row["device_opened"] is False

def result_mutations(row: dict, preparation: dict, wheels: dict) -> list[str]:
    cases = {"empty_tx": lambda x: x.__setitem__("transaction_simulation", {}), "missing_tx": lambda x: x["transaction_simulation"].pop("clean_commit"), "extra_tx": lambda x: x["transaction_simulation"].__setitem__("extra", True), "false_tx": lambda x: x["transaction_simulation"].__setitem__("clean_commit", False), "static": lambda x: x["static_contract"].__setitem__("pass", False), "identity": lambda x: x["identity"].__setitem__("base_prefix", "wrong"), "topology": lambda x: x["pre_run_topology"]["temps"].append("stale"), "device": lambda x: x.__setitem__("device_opened", True)}
    out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not result_valid(candidate, preparation, wheels): out.append(name)
    return out

def atomic_output(path: Path, row: dict) -> None: atomic_create(path, canon(row))

def main() -> int:
    live = make_live_identity()
    if not identity_valid(live, EXPECTED_NATIVE, EXPECTED_ARGV): return 3
    live_rejected = live_identity_mutations(live); raw = RESULT.read_bytes(); row = json.loads(raw); preparation = prior.prior.base.independent_preparation(); wheels = {"psutil": prior.prior.base.wheel_record(prior.prior.base.FILES["psutil_record"][0]), "numpy": prior.prior.base.wheel_record(prior.prior.base.FILES["numpy_record"][0])}; own_tx = own_transaction_simulation(); rejected = result_mutations(row, preparation, wheels)
    checks = {"live_dual_identity": True, "live_identity_mutations": live_rejected == ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "direct"], "lock": lock_valid(), "bundle": bundle_valid(), "topology": topology_valid(), "static_contract": static_contract(),
              "r7d1_failure": prior.prior.failure_bundle_valid(prior.prior.FAILURE_ROOT), "runtime": prior.prior.base.runtime_static_valid(prior.prior.base.collect_runtime()), "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899,
              "preparation": sha_bytes(canon(preparation)) == prior.prior.base.PREPARATION_DIGEST, "result": result_valid(row, preparation, wheels), "result_mutations": rejected == ["empty_tx", "missing_tx", "extra_tx", "false_tx", "static", "identity", "topology", "device"], "own_transactions": set(own_tx) == VERIFY_TX_KEYS and all(own_tx.values())}
    output = {"kind": "ph1_intel_execution_r8p3_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "result_sha256": sha_bytes(raw), "manifest_sha256": sha256(MANIFEST), "commit_sha256": sha256(COMMIT), "live_identity_mutations_rejected": live_rejected, "own_transaction_simulation": own_tx, "cpu_frozen_slice_read": True, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
    atomic_output(OUTPUT, output); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3

if __name__ == "__main__": raise SystemExit(main())
