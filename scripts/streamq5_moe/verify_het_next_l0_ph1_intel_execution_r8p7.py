#!/usr/bin/env python3
"""Independent R8P7 verifier; local entry and immutable R8P6 negative."""
from __future__ import annotations

import ast, copy, ctypes as C, hashlib, json, os, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"; sys.path.insert(0, str(S))
import verify_het_next_l0_ph1_intel_execution_r8p6 as prior

SCRIPT = Path(__file__).resolve(); PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r8p7.py"; LOCK = R / "het_next_l0_ph1_intel_execution_r8p7_lock.json"; PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P7_PREREGISTRATION_2026-08-14.md"; DIAGNOSIS = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_DIRECT_ENTRY_FAILURE_DIAGNOSIS_2026-08-14.md"; R8P6_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p7_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p7_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p7_static_preflight.commit.json"; OUTPUT = R / "het_next_l0_ph1_intel_execution_r8p7_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p7_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p7_quarantine"; CORE = (RESULT, MANIFEST, COMMIT)
R8P6_FAILURE_ROOT = prior.FAILED; R8P6_FAILURE = R8P6_FAILURE_ROOT / "attempt_71e198678f004a56a6912d07a4187dfd/failure.json"; R8P6_FAILURE_SHA = "03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372"
ACK = "PH1_INTEL_EXECUTION_R8P7_LOCAL_ENTRY_PROVENANCE_CLOSED"; KIND = "ph1_intel_execution_r8p7_static_preflight"
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; PYVENV = prior.prior.PYVENV; STATIC = dict(prior.STATIC); BASE = prior.BASE
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT)]; EXPECTED_ARGV = [str(SCRIPT)]; PREFLIGHT_NATIVE = [str(ALIAS), "-I", "-B", str(PREFLIGHT.resolve()), "--ack", ACK]; PREFLIGHT_ARGV = [str(PREFLIGHT.resolve()), "--ack", ACK]
CPU_STATES = set(prior.CPU_STATES); TX_KEYS = set(prior.TX_KEYS); VERIFY_TX_KEYS = set(prior.VERIFY_TX_KEYS); FAILURE_KEYS = set(prior.FAILURE_KEYS); FAILURE_SCHEMA = set(prior.FAILURE_SCHEMA); ENTRY_KEYS = {"entry_name", "entry_spec_is_none", "entry_package", "entry_file"}; IDENTITY_KEYS = set(prior.IDENTITY_KEYS) | ENTRY_KEYS
CHECK_NAMES = {"local_entry_identity", "entry_mutations", "hash_bindings", "closed_pending", "runtime_lock", "runtime", "start_ram", "wheel_records", "runtime_mutations", "r7d1_failure", "r8p6_failure", "cpu_preparation", "current_transactions", "failure_simulation", "static_boundary", "topology", "base_clean"}
IDENTITY_MUTATION_NAMES = ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "entry_name", "entry_spec", "entry_package", "entry_file", "direct"]
STATE_MUTATION_NAMES = ["unknown_state", "list_state", "dict_state", "int_state", "null_state", "wrong_started", "wrong_completed", "missing_state", "extra_state"]
CHAIN = {"preflight_sha256": PREFLIGHT, "verifier_sha256": SCRIPT, "prereg_sha256": PREREG, "r8p6_diagnosis_sha256": DIAGNOSIS, "r8p6_audit_sha256": R8P6_AUDIT, "r8p6_failure_sha256": R8P6_FAILURE, "r8p6_lock_sha256": prior.LOCK,
         **{("r8p6_" + k if k in {"preflight_sha256", "verifier_sha256", "prereg_sha256"} else k): v for k, v in prior.CHAIN.items()}}

def sha_bytes(x: bytes) -> str: return hashlib.sha256(x).hexdigest()
def sha256(p: Path) -> str: return sha_bytes(p.read_bytes())
def canon(x: object) -> bytes: return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()
def same(a: object, b: object) -> bool: return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def parse_commandline(raw: str) -> list[str]:
    k = C.WinDLL("kernel32", use_last_error=True); s = C.WinDLL("shell32", use_last_error=True); parse = s.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p); free = k.LocalFree; free.argtypes = (C.c_void_p,); free.restype = C.c_void_p; n = C.c_int(); ptr = parse(raw, C.byref(n))
    if not ptr: raise C.WinError(C.get_last_error())
    try: return [ptr[i] for i in range(n.value)]
    finally:
        if free(C.cast(ptr, C.c_void_p)): raise C.WinError(C.get_last_error())

def live_identity() -> dict:
    k = C.WinDLL("kernel32", use_last_error=True); get = k.GetCommandLineW; get.argtypes = (); get.restype = C.c_wchar_p; raw = get(); cfg = {}
    for line in PYVENV.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1); cfg[key.strip()] = value.strip()
    entry_name = __name__; entry_spec_is_none = __spec__ is None; entry_package = __package__; entry_file = str(Path(__file__).resolve()); direct = entry_name == "__main__" and entry_spec_is_none and entry_package in (None, "") and entry_file == str(SCRIPT)
    return {"native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv), "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix, "venv_launcher_sha256": sha256(VENV_PYTHON), "pyvenv": {"sha256": sha256(PYVENV), "home": cfg.get("home"), "executable": cfg.get("executable"), "version": cfg.get("version")}, "base_binary_sha256": sha256(BASE_BINARY), "base_binary_bytes": BASE_BINARY.stat().st_size, "entry_name": entry_name, "entry_spec_is_none": entry_spec_is_none, "entry_package": entry_package, "entry_file": entry_file, "direct_entry": direct}

def identity_valid(x: object, native: list[str], argv: list[str], script: Path) -> bool:
    if not isinstance(x, dict) or set(x) != IDENTITY_KEYS: return False
    direct = x["entry_name"] == "__main__" and x["entry_spec_is_none"] is True and x["entry_package"] in (None, "") and same(x["entry_file"], str(script))
    return bool(x["native_raw"]) and parse_commandline(x["native_raw"]) == native and x["native_argv"] == native and x["orig_argv"] == native and x["argv"] == argv and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is direct is True

def identity_mutations(row: dict, native: list[str], argv: list[str], script: Path) -> list[str]:
    cases = {"native": lambda x: x["native_argv"].append("x"), "raw": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "orig": lambda x: x["orig_argv"].append("x"), "argv": lambda x: x["argv"].append("x"), "venv": lambda x: x.__setitem__("sys_executable", "wrong"), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "wrong"), "base": lambda x: x.__setitem__("base_executable", "wrong"), "base_prefix": lambda x: x.__setitem__("base_prefix", "wrong"), "flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "script": lambda x: x["argv"].__setitem__(0, "wrong"), "entry_name": lambda x: x.__setitem__("entry_name", "imported.module"), "entry_spec": lambda x: x.__setitem__("entry_spec_is_none", False), "entry_package": lambda x: x.__setitem__("entry_package", "pkg"), "entry_file": lambda x: x.__setitem__("entry_file", "wrong"), "direct": lambda x: x.__setitem__("direct_entry", False)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not identity_valid(candidate, native, argv, script): out.append(name)
    return out

def state_bits(state: object) -> tuple[bool, bool]:
    if not isinstance(state, str) or state not in CPU_STATES: raise ValueError("cpu_slice_state")
    return state != "not_started", state == "completed"
def state_valid(row: dict) -> bool:
    try: started, completed = state_bits(row.get("cpu_slice_state"))
    except (ValueError, TypeError): return False
    return row.get("cpu_frozen_slice_read_started") is started and row.get("cpu_frozen_slice_read_completed") is completed

def atomic_create(path: Path, data: bytes, *, link_fn=os.link, unlink_fn=Path.unlink) -> None:
    if path.exists(): raise FileExistsError(path)
    temp = path.with_name(path.name + ".inprogress." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as h: h.write(data); h.flush(); os.fsync(h.fileno())
        link_fn(temp, path)
    finally:
        if temp.exists(): unlink_fn(temp)
def cleanup_temps(root: Path, stem: str) -> list[str]:
    rows = []
    for path in sorted(root.glob(stem + ".inprogress.*")): path.unlink(); rows.append(path.name)
    return rows
def output_writer_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); target = root / "verification.json"; data = canon({"fixture": True}); atomic_create(target, data); frozen = target.read_bytes()
        try: atomic_create(target, b"overwrite"); repeat = False
        except FileExistsError: repeat = target.read_bytes() == frozen
        stale = root / "stale.json.inprogress.fixed"; stale.write_bytes(b"s"); stale_ok = cleanup_temps(root, "stale.json") == [stale.name]
        def bad_link(_a, _b): raise OSError("prelink")
        try: atomic_create(root / "pre.json", b"x", link_fn=bad_link); pre = False
        except OSError: pre = not (root / "pre.json").exists() and not list(root.glob("pre.json.inprogress.*"))
        def bad_unlink(_p): raise OSError("postlink")
        try: atomic_create(root / "post.json", b"y", unlink_fn=bad_unlink); post = False
        except OSError:
            left = list(root.glob("post.json.inprogress.*")); cleanup_temps(root, "post.json"); post = (root / "post.json").read_bytes() == b"y" and len(left) == 1
        return {"clean_create": target.read_bytes() == data, "repeat_preserved": repeat, "stale_cleaned": stale_ok, "prelink_clean": pre, "postlink_recovered": post, "bytes_immutable": target.read_bytes() == frozen}

def independent_failure(stage: str, exc: BaseException, state: str, root: Path, *, create=atomic_create) -> dict:
    started, completed = state_bits(state); row = {"kind": "ph1_intel_execution_r8p7_failure", "status": "valid_protocol_negative", "stage": stage, "error": f"{type(exc).__name__}:{exc}", "traceback": "independent_fixture", "identity": None, "cpu_slice_state": state, "cpu_frozen_slice_read_started": started, "cpu_frozen_slice_read_completed": completed, "device_opened": False, "compiler_opened": False, "disposition": "bounded_create_new_canonical"}; data = canon(row)
    if set(row) != FAILURE_SCHEMA or len(data) > 65536 or not state_valid(row): raise RuntimeError("schema_cap")
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); target = attempt / "failure.json"
    try: create(target, data)
    except Exception:
        temps = list(attempt.glob(target.name + ".inprogress.*"))
        if target.is_file() and target.read_bytes() == data:
            for p in temps: p.unlink()
            return {"path": target, "row": row, "recovered": True}
        for p in temps:
            try: p.unlink()
            except Exception: pass
        if attempt.exists() and not any(attempt.iterdir()): attempt.rmdir()
        raise
    return {"path": target, "row": row, "recovered": False}
def independent_failure_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); exc = RuntimeError("primary"); first = independent_failure("identity", exc, "not_started", root / "main"); p1 = first["path"]; frozen = p1.read_bytes(); second = independent_failure("post_preparation", exc, "completed", root / "main"); p2 = second["path"]
        def precreate(path, data):
            def fail(_a, _b): raise OSError("prelink")
            return atomic_create(path, data, link_fn=fail)
        pre_root = root / "pre"; secondary = None
        try: independent_failure("identity", exc, "not_started", pre_root, create=precreate)
        except Exception as caught: secondary = f"{type(caught).__name__}:{caught}"
        def postcreate(path, data):
            def fail(_p): raise OSError("postunlink")
            return atomic_create(path, data, unlink_fn=fail)
        post_root = root / "post"; post = independent_failure("preparation", exc, "started_not_completed", post_root, create=postcreate); pp = post["path"]; rows = [first["row"], second["row"], post["row"]]; bad_values = ["unknown", [], {}, 1, None]; rejected = all(not state_valid({**row, "cpu_slice_state": bad}) for row in rows for bad in bad_values)
        return {"baseline_schema_cap": all(set(row) == FAILURE_SCHEMA and state_valid(row) and len(canon(row)) <= 65536 for row in rows), "second_unique_no_overwrite": p1 != p2 and p1.read_bytes() == frozen and len(list((root / "main").rglob("failure.json"))) == 2, "prelink_cleanup": secondary == "OSError:prelink" and (not pre_root.exists() or not list(pre_root.rglob("*"))), "postlink_cleanup_canonical": post["recovered"] is True and pp.is_file() and not list(pp.parent.glob("*.inprogress.*")) and canon(post["row"]) == pp.read_bytes(), "primary_preserved_secondary_recorded": secondary == "OSError:prelink", "cpu_state_provenance_all_three": [r["cpu_slice_state"] for r in rows] == ["not_started", "completed", "started_not_completed"] and rejected}

def r8p6_failure_valid() -> bool:
    files = sorted(p for p in R8P6_FAILURE_ROOT.rglob("*") if p.is_file()); dirs = sorted(p for p in R8P6_FAILURE_ROOT.rglob("*") if p.is_dir())
    if files != [R8P6_FAILURE] or len(dirs) != 1 or R8P6_FAILURE.stat().st_size != 2986 or sha256(R8P6_FAILURE) != R8P6_FAILURE_SHA: return False
    row = json.loads(R8P6_FAILURE.read_text()); ident = row.get("identity", {})
    return set(row) == FAILURE_SCHEMA and row["kind"] == "ph1_intel_execution_r8p6_failure" and row["status"] == "valid_protocol_negative" and row["stage"] == "identity" and row["error"] == "RuntimeError:exact_invocation" and row["cpu_slice_state"] == "not_started" and row["cpu_frozen_slice_read_started"] is row["cpu_frozen_slice_read_completed"] is row["compiler_opened"] is row["device_opened"] is False and isinstance(ident, dict) and set(ident) == prior.IDENTITY_KEYS and ident["direct_entry"] is False

def lock_valid() -> bool:
    lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}
    return set(lock) == {"kind", "execution_open", "audit_token", *STATIC, *observed} and lock["kind"] == "ph1_intel_execution_r8p7_lock" and lock["execution_open"] is False and lock["audit_token"] == "PENDING" and all(lock.get(k) == v for k, v in STATIC.items()) and all(lock.get(k) == v for k, v in observed.items()) and observed["r8p6_diagnosis_sha256"] == "85d59b75a4940dd01df15d5072a0c9a1f4e9faf62260c6f8df07ed6fbfc0cba5"
def bundle_valid() -> bool:
    if not all(p.is_file() for p in CORE): return False
    rb = RESULT.read_bytes(); mb = MANIFEST.read_bytes(); return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": RESULT.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(COMMIT.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}
def topology_valid() -> bool:
    expected = {R / "het_next_l0_ph1_intel_execution_r8_lock.json", *(R / f"het_next_l0_ph1_intel_execution_r8p{i}_lock.json" for i in range(1, 8)), R8P6_FAILURE_ROOT, *CORE}; family = {p for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8")}
    return family == expected and not FAILED.exists() and not QUARANTINE.exists() and not OUTPUT.exists() and not list(R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*")) and r8p6_failure_valid()
def stored_topology_valid(x: object) -> bool:
    expected = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", *(R / f"het_next_l0_ph1_intel_execution_r8p{i}_lock.json" for i in range(1, 8)), R8P6_FAILURE_ROOT))
    return isinstance(x, dict) and set(x) == {"absent", "temps", "family"} and len(x["absent"]) == 47 and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == expected

def static_boundary() -> bool:
    forbidden = {"Popen", "run", "check_call", "check_output", "CDLL", "LoadLibrary", "clBuildProgram", "clCreateContext", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"}; tree = ast.parse(PREFLIGHT.read_text(encoding="utf-8")); funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}; main_calls = {n.func.id for n in ast.walk(funcs["main"]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}; ancestor_identity = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "identity" for n in ast.walk(tree)); entry_names = {n.id for n in ast.walk(funcs["identity"]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    for path in (PREFLIGHT, SCRIPT):
        parsed = ast.parse(path.read_text(encoding="utf-8")); calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in ast.walk(parsed) if isinstance(n, ast.Call)}
        if calls & forbidden: return False
    return "identity" in main_calls and not ancestor_identity and {"__name__", "__spec__", "__package__", "__file__"} <= entry_names and prior.static_boundary()

def result_valid(row: object, preparation: dict, wheels: dict) -> bool:
    if not isinstance(row, dict): return False
    checks = row.get("checks", {}); tx = row.get("transaction_simulation"); failure = row.get("failure_simulation")
    return set(row) == {"kind", "ack", "identity", "pre_run_topology", "checks", "pass", "passed", "total", "runtime", "wheel_records", "preparation", "preparation_digest", "transaction_simulation", "failure_simulation", "identity_mutations_rejected", "runtime_mutations_rejected", "cpu_slice_state", "cpu_frozen_slice_read_started", "cpu_frozen_slice_read_completed", "model_forward", "compiler_opened", "opencl_opened", "device_opened"} and row["kind"] == KIND and row["ack"] == ACK and identity_valid(row["identity"], PREFLIGHT_NATIVE, PREFLIGHT_ARGV, PREFLIGHT) and stored_topology_valid(row["pre_run_topology"]) and set(checks) == CHECK_NAMES and all(v is True for v in checks.values()) and row["pass"] is True and row["passed"] == row["total"] == len(CHECK_NAMES) and BASE.runtime_static_valid(row["runtime"]) and row["runtime"]["available"] >= 16 * 2**30 and row["wheel_records"] == wheels and row["preparation"] == preparation and row["preparation_digest"] == BASE.PREPARATION_DIGEST == sha_bytes(canon(preparation)) and isinstance(tx, dict) and set(tx) == TX_KEYS and all(v is True for v in tx.values()) and isinstance(failure, dict) and set(failure) == FAILURE_KEYS and all(v is True for v in failure.values()) and row["identity_mutations_rejected"] == IDENTITY_MUTATION_NAMES and row["runtime_mutations_rejected"] == ["python_path", "python_hash", "isolation", "bytecode", "pyvenv", "psutil_native", "psutil_record", "numpy_version", "numpy_record", "ram"] and row["cpu_slice_state"] == "completed" and state_valid(row) and row["model_forward"] is row["compiler_opened"] is row["opencl_opened"] is row["device_opened"] is False

def result_mutations(row: dict, preparation: dict, wheels: dict) -> list[str]:
    cases = {"empty_tx": lambda x: x.__setitem__("transaction_simulation", {}), "missing_tx": lambda x: x["transaction_simulation"].pop("clean_commit"), "extra_tx": lambda x: x["transaction_simulation"].__setitem__("extra", True), "false_tx": lambda x: x["transaction_simulation"].__setitem__("clean_commit", False), "entry_name": lambda x: x["identity"].__setitem__("entry_name", "imported"), "entry_spec": lambda x: x["identity"].__setitem__("entry_spec_is_none", False), "entry_package": lambda x: x["identity"].__setitem__("entry_package", "pkg"), "entry_file": lambda x: x["identity"].__setitem__("entry_file", "wrong"), "direct": lambda x: x["identity"].__setitem__("direct_entry", False), "list_state": lambda x: x.__setitem__("cpu_slice_state", []), "dict_state": lambda x: x.__setitem__("cpu_slice_state", {}), "int_state": lambda x: x.__setitem__("cpu_slice_state", 1), "null_state": lambda x: x.__setitem__("cpu_slice_state", None), "device": lambda x: x.__setitem__("device_opened", True)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not result_valid(candidate, preparation, wheels): out.append(name)
    return out

def main() -> int:
    ident = live_identity()
    if not identity_valid(ident, EXPECTED_NATIVE, EXPECTED_ARGV, SCRIPT): return 3
    raw = RESULT.read_bytes(); row = json.loads(raw); preparation = BASE.independent_preparation(); wheels = {"psutil": BASE.wheel_record(BASE.FILES["psutil_record"][0]), "numpy": BASE.wheel_record(BASE.FILES["numpy_record"][0])}; writer = output_writer_simulation(); failure = independent_failure_simulation(); live_rejected = identity_mutations(ident, EXPECTED_NATIVE, EXPECTED_ARGV, SCRIPT); rejected = result_mutations(row, preparation, wheels)
    checks = {"local_live_identity": True, "live_entry_mutations": live_rejected == IDENTITY_MUTATION_NAMES, "lock": lock_valid(), "bundle": bundle_valid(), "topology": topology_valid(), "static_boundary": static_boundary(), "r8p6_failure": r8p6_failure_valid(), "r7d1_failure": prior.prior.prior.prior.prior.prior.failure_bundle_valid(prior.prior.prior.prior.prior.prior.FAILURE_ROOT), "runtime": BASE.runtime_static_valid(BASE.collect_runtime()), "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899, "preparation": sha_bytes(canon(preparation)) == BASE.PREPARATION_DIGEST, "result": result_valid(row, preparation, wheels), "independent_output_transactions": set(writer) == VERIFY_TX_KEYS and all(writer.values()), "independent_failure_simulation": set(failure) == FAILURE_KEYS and all(failure.values()), "entry_state_tx_mutations": rejected == ["empty_tx", "missing_tx", "extra_tx", "false_tx", "entry_name", "entry_spec", "entry_package", "entry_file", "direct", "list_state", "dict_state", "int_state", "null_state", "device"]}
    output = {"kind": "ph1_intel_execution_r8p7_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "result_sha256": sha_bytes(raw), "manifest_sha256": sha256(MANIFEST), "commit_sha256": sha256(COMMIT), "independent_output_transactions": writer, "independent_failure_simulation": failure, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
    atomic_create(OUTPUT, canon(output)); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3

if __name__ == "__main__": raise SystemExit(main())
