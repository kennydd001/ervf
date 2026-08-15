#!/usr/bin/env python3
"""R8P7 closed no-device preflight with current-module entry evidence."""
from __future__ import annotations

import ast, copy, ctypes as C, hashlib, json, os, sys, tempfile, traceback, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"; sys.path.insert(0, str(S))
import preflight_het_next_l0_ph1_intel_execution_r8p6 as prior

SCRIPT = Path(__file__).resolve(); VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8p7.py"; LOCK = R / "het_next_l0_ph1_intel_execution_r8p7_lock.json"; PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P7_PREREGISTRATION_2026-08-14.md"; DIAGNOSIS = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_DIRECT_ENTRY_FAILURE_DIAGNOSIS_2026-08-14.md"; R8P6_AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p7_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p7_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p7_static_preflight.commit.json"; VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p7_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p7_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p7_quarantine"; CORE = (RESULT, MANIFEST, COMMIT)
R8P6_FAILURE_ROOT = prior.FAILED; R8P6_FAILURE = R8P6_FAILURE_ROOT / "attempt_71e198678f004a56a6912d07a4187dfd/failure.json"; R8P6_FAILURE_SHA = "03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372"
ACK = "PH1_INTEL_EXECUTION_R8P7_LOCAL_ENTRY_PROVENANCE_CLOSED"; KIND = "ph1_intel_execution_r8p7_static_preflight"
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; PYVENV = prior.PYVENV; STATIC = dict(prior.STATIC); BASE = prior.BASE
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT), "--ack", ACK]; EXPECTED_ARGV = [str(SCRIPT), "--ack", ACK]
CPU_STATES = set(prior.CPU_STATES); TX_KEYS = set(prior.TX_KEYS); FAILURE_KEYS = set(prior.FAILURE_KEYS)
FAILURE_SCHEMA = set(prior.FAILURE_SCHEMA); ENTRY_KEYS = {"entry_name", "entry_spec_is_none", "entry_package", "entry_file"}; IDENTITY_KEYS = set(prior.IDENTITY_KEYS) | ENTRY_KEYS
CHAIN = {"preflight_sha256": SCRIPT, "verifier_sha256": VERIFIER, "prereg_sha256": PREREG, "r8p6_diagnosis_sha256": DIAGNOSIS, "r8p6_audit_sha256": R8P6_AUDIT, "r8p6_failure_sha256": R8P6_FAILURE, "r8p6_lock_sha256": prior.LOCK,
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

def identity() -> dict:
    k = C.WinDLL("kernel32", use_last_error=True); get = k.GetCommandLineW; get.argtypes = (); get.restype = C.c_wchar_p; raw = get(); cfg = {}
    for line in PYVENV.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1); cfg[key.strip()] = value.strip()
    entry_name = __name__; entry_spec_is_none = __spec__ is None; entry_package = __package__; entry_file = str(Path(__file__).resolve()); direct = entry_name == "__main__" and entry_spec_is_none and entry_package in (None, "") and entry_file == str(SCRIPT)
    return {"native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv), "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix, "venv_launcher_sha256": sha256(VENV_PYTHON), "pyvenv": {"sha256": sha256(PYVENV), "home": cfg.get("home"), "executable": cfg.get("executable"), "version": cfg.get("version")}, "base_binary_sha256": sha256(BASE_BINARY), "base_binary_bytes": BASE_BINARY.stat().st_size, "entry_name": entry_name, "entry_spec_is_none": entry_spec_is_none, "entry_package": entry_package, "entry_file": entry_file, "direct_entry": direct}

def identity_valid(x: object) -> bool:
    if not isinstance(x, dict) or set(x) != IDENTITY_KEYS: return False
    direct = x["entry_name"] == "__main__" and x["entry_spec_is_none"] is True and x["entry_package"] in (None, "") and same(x["entry_file"], str(SCRIPT))
    return bool(x["native_raw"]) and parse_commandline(x["native_raw"]) == EXPECTED_NATIVE and x["native_argv"] == EXPECTED_NATIVE and x["orig_argv"] == EXPECTED_NATIVE and x["argv"] == EXPECTED_ARGV and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is direct is True

IDENTITY_MUTATION_NAMES = ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "entry_name", "entry_spec", "entry_package", "entry_file", "direct"]
def identity_mutations(row: dict) -> list[str]:
    cases = {"native": lambda x: x["native_argv"].append("x"), "raw": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "orig": lambda x: x["orig_argv"].append("x"), "argv": lambda x: x["argv"].append("x"), "venv": lambda x: x.__setitem__("sys_executable", "wrong"), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "wrong"), "base": lambda x: x.__setitem__("base_executable", "wrong"), "base_prefix": lambda x: x.__setitem__("base_prefix", "wrong"), "flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "script": lambda x: x["argv"].__setitem__(0, "wrong"), "entry_name": lambda x: x.__setitem__("entry_name", "imported.module"), "entry_spec": lambda x: x.__setitem__("entry_spec_is_none", False), "entry_package": lambda x: x.__setitem__("entry_package", "pkg"), "entry_file": lambda x: x.__setitem__("entry_file", "wrong"), "direct": lambda x: x.__setitem__("direct_entry", False)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not identity_valid(candidate): out.append(name)
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
def quarantine_core(paths: tuple[Path, ...], root: Path) -> list[dict]:
    existing = [p for p in paths if p.exists()]
    if not existing: return []
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); rows = []
    for path in existing:
        target = attempt / path.name; os.replace(path, target); rows.append({"name": path.name, "sha256": sha256(target), "bytes": target.stat().st_size})
    return rows
def verify_bundle(result: Path, manifest: Path, commit: Path) -> bool:
    if not all(p.is_file() for p in (result, manifest, commit)): return False
    rb = result.read_bytes(); mb = manifest.read_bytes(); return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(commit.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}
def publish(row: dict, result: Path = RESULT, manifest: Path = MANIFEST, commit: Path = COMMIT, quarantine: Path = QUARANTINE, *, create=atomic_create) -> None:
    if any(p.exists() for p in (result, manifest, commit)): raise FileExistsError("bundle_target")
    rb = canon(row); mb = canon({"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]}); cb = canon({"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)})
    try:
        create(result, rb); create(manifest, mb); create(commit, cb)
        if not verify_bundle(result, manifest, commit): raise RuntimeError("bundle")
    except Exception:
        quarantine_core((result, manifest, commit), quarantine); raise

def transaction_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); result, manifest, commit = root / "result.json", root / "manifest.json", root / "commit.json"; row = {"fixture": True}; publish(row, result, manifest, commit, root / "q"); frozen = tuple(p.read_bytes() for p in (result, manifest, commit)); clean = verify_bundle(result, manifest, commit); repeat = False
        try: publish(row, result, manifest, commit, root / "q")
        except FileExistsError: repeat = tuple(p.read_bytes() for p in (result, manifest, commit)) == frozen
        stale = root / "stale.json.inprogress.fixed"; stale.write_bytes(b"s"); stale_ok = cleanup_temps(root, "stale.json") == [stale.name]
        def bad_link(_a, _b): raise OSError("prelink")
        pre = root / "pre.json"
        try: atomic_create(pre, b"x", link_fn=bad_link); pre_ok = False
        except OSError: pre_ok = not pre.exists() and not list(root.glob(pre.name + ".inprogress.*"))
        def bad_unlink(_p): raise OSError("postlink")
        post = root / "post.json"
        try: atomic_create(post, b"y", unlink_fn=bad_unlink); post_ok = False
        except OSError:
            left = list(root.glob(post.name + ".inprogress.*")); cleanup_temps(root, post.name); post_ok = post.read_bytes() == b"y" and len(left) == 1
        p2, m2, c2 = root / "p2.json", root / "m2.json", root / "c2.json"; calls = {"n": 0}
        def fail_second(path, data):
            calls["n"] += 1
            if calls["n"] == 2: raise OSError("interrupt")
            atomic_create(path, data)
        try: publish(row, p2, m2, c2, root / "q2", create=fail_second); partial = False
        except OSError:
            moved = list((root / "q2").rglob("p2.json")); partial = not any(p.exists() for p in (p2, m2, c2)) and len(moved) == 1 and moved[0].read_bytes() == canon(row)
        return {"clean_commit": clean, "repeat_rejected_unchanged": repeat, "stale_temp_cleaned": stale_ok, "prelink_failure_clean": pre_ok, "postlink_interruption_recovered": post_ok, "partial_publish_quarantined": partial, "committed_bytes_immutable": tuple(p.read_bytes() for p in (result, manifest, commit)) == frozen}

def failure_row(stage: str, exc: BaseException, ident: dict | None, state: str) -> dict:
    started, completed = state_bits(state); return {"kind": "ph1_intel_execution_r8p7_failure", "status": "valid_protocol_negative", "stage": stage, "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()[-32768:], "identity": ident, "cpu_slice_state": state, "cpu_frozen_slice_read_started": started, "cpu_frozen_slice_read_completed": completed, "device_opened": False, "compiler_opened": False, "disposition": "bounded_create_new_canonical"}
def atomic_failure(stage: str, exc: BaseException, ident: dict | None, state: str, root: Path = FAILED, *, create=atomic_create) -> dict:
    row = failure_row(stage, exc, ident, state); data = canon(row)
    if len(data) > 65536 or set(row) != FAILURE_SCHEMA or not state_valid(row): raise RuntimeError("failure_schema_cap")
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); target = attempt / "failure.json"
    try: create(target, data)
    except Exception:
        temps = list(attempt.glob(target.name + ".inprogress.*"))
        if target.is_file() and target.read_bytes() == data:
            for p in temps: p.unlink()
            return {"path": str(target), "recovered": True}
        for p in temps:
            try: p.unlink()
            except Exception: pass
        if attempt.exists() and not any(attempt.iterdir()): attempt.rmdir()
        raise
    return {"path": str(target), "recovered": False}
def preserve_primary(stage: str, exc: BaseException, ident: dict | None, state: str, root: Path, *, writer=atomic_failure) -> dict:
    try: return {"primary": f"{type(exc).__name__}:{exc}", "secondary": None, "evidence": writer(stage, exc, ident, state, root)}
    except Exception as secondary: return {"primary": f"{type(exc).__name__}:{exc}", "secondary": f"{type(secondary).__name__}:{secondary}", "evidence": None}

STATE_MUTATION_NAMES = list(prior.STATE_MUTATION_NAMES)
def state_mutations(row: dict) -> list[str]:
    cases = {"unknown_state": lambda x: x.__setitem__("cpu_slice_state", "unknown"), "list_state": lambda x: x.__setitem__("cpu_slice_state", []), "dict_state": lambda x: x.__setitem__("cpu_slice_state", {}), "int_state": lambda x: x.__setitem__("cpu_slice_state", 1), "null_state": lambda x: x.__setitem__("cpu_slice_state", None), "wrong_started": lambda x: x.__setitem__("cpu_frozen_slice_read_started", not x["cpu_frozen_slice_read_started"]), "wrong_completed": lambda x: x.__setitem__("cpu_frozen_slice_read_completed", not x["cpu_frozen_slice_read_completed"]), "missing_state": lambda x: x.pop("cpu_slice_state"), "extra_state": lambda x: x.__setitem__("cpu_slice_state_extra", True)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if set(candidate) != FAILURE_SCHEMA or not state_valid(candidate): out.append(name)
    return out

def failure_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "failed"; primary = RuntimeError("primary")
        try: raise primary
        except RuntimeError as exc: first = atomic_failure("identity", exc, {"fixture": True}, "not_started", root)
        p1 = Path(first["path"]); frozen = p1.read_bytes(); row1 = json.loads(frozen); second = atomic_failure("post_preparation", primary, None, "completed", root); p2 = Path(second["path"]); row2 = json.loads(p2.read_text())
        def precreate(path, data):
            def fail(_a, _b): raise OSError("prelink")
            return atomic_create(path, data, link_fn=fail)
        pre_root = Path(td) / "pre"; pre = preserve_primary("identity", primary, None, "not_started", pre_root, writer=lambda s, e, i, state, r: atomic_failure(s, e, i, state, r, create=precreate))
        def postcreate(path, data):
            def fail(_p): raise OSError("postunlink")
            return atomic_create(path, data, unlink_fn=fail)
        post_root = Path(td) / "post"; post = atomic_failure("preparation", primary, None, "started_not_completed", post_root, create=postcreate); pp = Path(post["path"]); row3 = json.loads(pp.read_text()); states = (row1, row2, row3)
        return {"baseline_schema_cap": set(row1) == FAILURE_SCHEMA and len(frozen) <= 65536 and state_valid(row1), "second_unique_no_overwrite": p1 != p2 and p1.read_bytes() == frozen and len(list(root.rglob("failure.json"))) == 2, "prelink_cleanup": pre["secondary"] == "OSError:prelink" and (not pre_root.exists() or not list(pre_root.rglob("*"))), "postlink_cleanup_canonical": post["recovered"] is True and pp.is_file() and not list(pp.parent.glob("*.inprogress.*")) and canon(row3) == pp.read_bytes(), "primary_preserved_secondary_recorded": pre["primary"] == "RuntimeError:primary" and pre["secondary"] == "OSError:prelink", "cpu_state_provenance_all_three": [r["stage"] for r in states] == ["identity", "post_preparation", "preparation"] and [r["cpu_slice_state"] for r in states] == ["not_started", "completed", "started_not_completed"] and all(state_valid(r) and state_mutations(r) == STATE_MUTATION_NAMES for r in states)}

def r8p6_failure_valid() -> bool:
    files = sorted(p for p in R8P6_FAILURE_ROOT.rglob("*") if p.is_file()); dirs = sorted(p for p in R8P6_FAILURE_ROOT.rglob("*") if p.is_dir())
    if files != [R8P6_FAILURE] or len(dirs) != 1 or R8P6_FAILURE.stat().st_size != 2986 or sha256(R8P6_FAILURE) != R8P6_FAILURE_SHA: return False
    row = json.loads(R8P6_FAILURE.read_text()); ident = row.get("identity", {})
    return set(row) == FAILURE_SCHEMA and row["kind"] == "ph1_intel_execution_r8p6_failure" and row["status"] == "valid_protocol_negative" and row["stage"] == "identity" and row["error"] == "RuntimeError:exact_invocation" and row["cpu_slice_state"] == "not_started" and row["cpu_frozen_slice_read_started"] is row["cpu_frozen_slice_read_completed"] is row["compiler_opened"] is row["device_opened"] is False and isinstance(ident, dict) and set(ident) == prior.IDENTITY_KEYS and ident["direct_entry"] is False

def topology() -> dict:
    ancestors = []
    for module in (prior.prior.prior.prior.prior, prior.prior.prior.prior, prior.prior.prior, prior.prior, prior.prior): ancestors.extend((*module.CORE, module.VERIFY_RESULT, module.QUARANTINE))
    ancestors.extend((*prior.CORE, prior.VERIFY_RESULT, prior.QUARANTINE))
    for module in (prior.prior.prior.prior.prior, prior.prior.prior.prior, prior.prior.prior, prior.prior, prior):
        if module is not prior: ancestors.append(module.FAILED)
    absent = tuple(prior.prior.prior.prior.prior.BASE_R8_PATHS) + tuple(ancestors) + CORE + (VERIFY_RESULT, FAILED, QUARANTINE)
    return {"absent": {str(p): p.exists() for p in absent}, "temps": sorted(str(p) for p in R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*")), "family": sorted(str(p) for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8"))}
def topology_clean(x: dict) -> bool:
    expected = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", *(R / f"het_next_l0_ph1_intel_execution_r8p{i}_lock.json" for i in range(1, 8)), R8P6_FAILURE_ROOT))
    return set(x) == {"absent", "temps", "family"} and len(x["absent"]) == 47 and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == expected and r8p6_failure_valid()

def static_contract() -> bool:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8")); funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}; main_calls = {n.func.id for n in ast.walk(funcs["main"]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}; ancestor_identity = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "identity" for n in ast.walk(tree)); entry_names = {n.id for n in ast.walk(funcs["identity"]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}; all_calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return prior.static_contract() and "identity" in main_calls and not ancestor_identity and {"__name__", "__spec__", "__package__", "__file__"} <= entry_names and not ({"Popen", "run", "CDLL", "LoadLibrary", "clBuildProgram", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"} & all_calls)

def main() -> int:
    if sys.argv != EXPECTED_ARGV: return 3
    ident = None; stage = "identity"; state = "not_started"
    try:
        ident = identity()
        if not identity_valid(ident): raise RuntimeError("exact_invocation")
        pre = topology()
        if not topology_clean(pre): raise RuntimeError("topology")
        stage = "runtime"; runtime = BASE.runner.collect_runtime()
        if runtime["available"] < 16 * 2**30: raise RuntimeError("start_ram")
        wheels = {"psutil": BASE.verify_wheel_record(BASE.runner.RUNTIME_FILES["psutil_record"][0]), "numpy": BASE.verify_wheel_record(BASE.runner.RUNTIME_FILES["numpy_record"][0])}
        stage = "cpu_frozen_slice"; state = "started_not_completed"; preparation = BASE.preparation_summary(); state = "completed"; runtime_ok, runtime_rejected = BASE.runtime_mutations(runtime)
        stage = "simulations"; tx = transaction_simulation(); failure = failure_simulation(); lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}; rejected = identity_mutations(ident)
        checks = {"local_entry_identity": identity_valid(ident), "entry_mutations": rejected == IDENTITY_MUTATION_NAMES, "hash_bindings": all(lock.get(k) == v for k, v in observed.items()), "closed_pending": lock.get("kind") == "ph1_intel_execution_r8p7_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING", "runtime_lock": all(lock.get(k) == v for k, v in STATIC.items()), "runtime": BASE.runner.validate_runtime(runtime), "start_ram": runtime["available"] >= 16 * 2**30, "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899, "runtime_mutations": runtime_ok and len(runtime_rejected) == 10, "r7d1_failure": BASE.runner.prior_failure_valid(), "r8p6_failure": r8p6_failure_valid(), "cpu_preparation": BASE.validate_preparation(preparation), "current_transactions": set(tx) == TX_KEYS and all(tx.values()), "failure_simulation": set(failure) == FAILURE_KEYS and all(failure.values()), "static_boundary": static_contract(), "topology": topology_clean(pre), "base_clean": BASE.runner.clean_now()}
        row = {"kind": KIND, "ack": ACK, "identity": ident, "pre_run_topology": pre, "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "runtime": runtime, "wheel_records": wheels, "preparation": preparation, "preparation_digest": BASE.sha_bytes(BASE.canon(preparation)), "transaction_simulation": tx, "failure_simulation": failure, "identity_mutations_rejected": rejected, "runtime_mutations_rejected": runtime_rejected, "cpu_slice_state": state, "cpu_frozen_slice_read_started": True, "cpu_frozen_slice_read_completed": True, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
        stage = "publish"; publish(row); print(json.dumps(row, indent=2)); return 0 if row["pass"] else 3
    except Exception as primary:
        preserve_primary(stage, primary, ident, state, FAILED); raise

if __name__ == "__main__": raise SystemExit(main())
