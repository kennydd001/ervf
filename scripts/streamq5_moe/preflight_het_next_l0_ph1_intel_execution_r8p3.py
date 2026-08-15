#!/usr/bin/env python3
"""R8P3 closed CPU preflight with current-helper transactions and static boundary."""
from __future__ import annotations

import ast, copy, ctypes as C, hashlib, json, os, sys, tempfile, traceback, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"
sys.path.insert(0, str(S))
import preflight_het_next_l0_ph1_intel_execution_r8p2 as prior

SCRIPT = Path(__file__).resolve(); VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8p3.py"
LOCK = R / "het_next_l0_ph1_intel_execution_r8p3_lock.json"; PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P3_PREREGISTRATION_2026-08-14.md"
AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p3_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p3_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p3_static_preflight.commit.json"
VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p3_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p3_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p3_quarantine"
ACK = "PH1_INTEL_EXECUTION_R8P3_CURRENT_TRANSACTION_STATIC_CPU_PREPARATION_CLOSED"; KIND = "ph1_intel_execution_r8p3_static_preflight"
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; PYVENV = prior.PYVENV
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT), "--ack", ACK]; EXPECTED_ARGV = [str(SCRIPT), "--ack", ACK]
CORE = (RESULT, MANIFEST, COMMIT); STATIC = dict(prior.STATIC)
CHAIN = {"preflight_sha256": SCRIPT, "verifier_sha256": VERIFIER, "prereg_sha256": PREREG, "r8p2_audit_sha256": AUDIT,
         "r8p2_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p2.py", "r8p2_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p2.py",
         "r8p2_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_PREREGISTRATION_2026-08-14.md", "r8p2_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p2_lock.json",
         **{("r8p2_" + k if k in {"preflight_sha256", "verifier_sha256", "prereg_sha256"} else k): v for k, v in prior.CHAIN.items()}}
SOURCE_PATHS = (SCRIPT, VERIFIER, S / "preflight_het_next_l0_ph1_intel_execution_r8p2.py", S / "preflight_het_next_l0_ph1_intel_execution_r8p1.py", S / "preflight_het_next_l0_ph1_intel_execution_r8.py", S / "run_het_next_l0_ph1_intel_execution_r8.py")
TX_KEYS = {"clean_commit", "repeat_rejected_unchanged", "stale_temp_cleaned", "prelink_failure_clean", "postlink_interruption_recovered", "partial_publish_quarantined", "committed_bytes_immutable"}

def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha256(path: Path) -> str: return sha_bytes(path.read_bytes())
def canon(x: object) -> bytes: return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()
def same(a: object, b: object) -> bool: return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def parse_commandline(raw: str) -> list[str]:
    k = C.WinDLL("kernel32", use_last_error=True); s = C.WinDLL("shell32", use_last_error=True)
    parse = s.CommandLineToArgvW; parse.argtypes = (C.c_wchar_p, C.POINTER(C.c_int)); parse.restype = C.POINTER(C.c_wchar_p)
    free = k.LocalFree; free.argtypes = (C.c_void_p,); free.restype = C.c_void_p; n = C.c_int(); ptr = parse(raw, C.byref(n))
    if not ptr: raise C.WinError(C.get_last_error())
    try: return [ptr[i] for i in range(n.value)]
    finally:
        if free(C.cast(ptr, C.c_void_p)): raise C.WinError(C.get_last_error())

def identity() -> dict:
    k = C.WinDLL("kernel32", use_last_error=True); get = k.GetCommandLineW; get.argtypes = (); get.restype = C.c_wchar_p; raw = get()
    values = {}
    for line in PYVENV.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1); values[key.strip()] = value.strip()
    return {"native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv), "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix,
            "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix, "venv_launcher_sha256": sha256(VENV_PYTHON),
            "pyvenv": {"sha256": sha256(PYVENV), "home": values.get("home"), "executable": values.get("executable"), "version": values.get("version")},
            "base_binary_sha256": sha256(BASE_BINARY), "base_binary_bytes": BASE_BINARY.stat().st_size, "direct_entry": __spec__ is None and (__package__ is None or __package__ == "")}

def identity_valid(x: dict) -> bool:
    return set(x) == prior.IDENTITY_KEYS and bool(x["native_raw"]) and parse_commandline(x["native_raw"]) == EXPECTED_NATIVE and x["native_argv"] == EXPECTED_NATIVE and x["orig_argv"] == EXPECTED_NATIVE and x["argv"] == EXPECTED_ARGV and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is True

def identity_mutations(row: dict) -> list[str]:
    cases = {"native": lambda x: x["native_argv"].append("x"), "raw": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "orig": lambda x: x["orig_argv"].append("x"), "argv": lambda x: x["argv"].append("x"),
             "venv": lambda x: x.__setitem__("sys_executable", "wrong"), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "wrong"), "base": lambda x: x.__setitem__("base_executable", "wrong"), "base_prefix": lambda x: x.__setitem__("base_prefix", "wrong"),
             "flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "script": lambda x: x["argv"].__setitem__(0, "wrong"), "direct": lambda x: x.__setitem__("direct_entry", False)}
    out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not identity_valid(candidate): out.append(name)
    return out

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
    rb = result.read_bytes(); mb = manifest.read_bytes()
    return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(commit.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}

def publish(row: dict, result: Path, manifest: Path, commit: Path, quarantine: Path, *, create=atomic_create) -> None:
    if any(p.exists() for p in (result, manifest, commit)): raise FileExistsError("bundle_target")
    rb = canon(row); mb = canon({"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]}); cb = canon({"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)})
    try:
        create(result, rb); create(manifest, mb); create(commit, cb)
        if not verify_bundle(result, manifest, commit): raise RuntimeError("bundle_verify")
    except Exception:
        quarantine_core((result, manifest, commit), quarantine); raise

def transaction_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); result = root / "result.json"; manifest = root / "manifest.json"; commit = root / "commit.json"; quarantine = root / "quarantine"; row = {"fixture": True}
        publish(row, result, manifest, commit, quarantine); frozen = tuple(p.read_bytes() for p in (result, manifest, commit)); clean = verify_bundle(result, manifest, commit)
        repeat = False
        try: publish(row, result, manifest, commit, quarantine)
        except FileExistsError: repeat = tuple(p.read_bytes() for p in (result, manifest, commit)) == frozen
        stale = root / "stale.json.inprogress.fixed"; stale.write_bytes(b"s"); stale_ok = cleanup_temps(root, "stale.json") == [stale.name] and not stale.exists()
        def fail_link(_a, _b): raise OSError("prelink")
        pre = root / "pre.json"; pre_ok = False
        try: atomic_create(pre, b"x", link_fn=fail_link)
        except OSError: pre_ok = not pre.exists() and not list(root.glob(pre.name + ".inprogress.*"))
        def fail_unlink(_p): raise OSError("postlink")
        post = root / "post.json"; post_ok = False
        try: atomic_create(post, b"y", unlink_fn=fail_unlink)
        except OSError:
            leftovers = list(root.glob(post.name + ".inprogress.*")); cleanup_temps(root, post.name); post_ok = post.read_bytes() == b"y" and len(leftovers) == 1
        p2, m2, c2 = root / "p2.json", root / "m2.json", root / "c2.json"; calls = {"n": 0}
        def fail_second(path, data):
            calls["n"] += 1
            if calls["n"] == 2: raise OSError("manifest_interrupt")
            atomic_create(path, data)
        partial = False
        try: publish(row, p2, m2, c2, root / "q2", create=fail_second)
        except OSError:
            moved = list((root / "q2").rglob("p2.json")); partial = not any(p.exists() for p in (p2, m2, c2)) and len(moved) == 1 and moved[0].read_bytes() == canon(row)
        return {"clean_commit": clean, "repeat_rejected_unchanged": repeat, "stale_temp_cleaned": stale_ok, "prelink_failure_clean": pre_ok, "postlink_interruption_recovered": post_ok, "partial_publish_quarantined": partial, "committed_bytes_immutable": tuple(p.read_bytes() for p in (result, manifest, commit)) == frozen}

def static_contract() -> dict:
    forbidden_imports = {"pyopencl", "cupy", "torch", "transformers"}; forbidden_calls = {"Popen", "run", "check_call", "check_output", "system", "startfile", "CDLL", "LoadLibrary", "clBuildProgram", "clCreateContext", "clEnqueueNDRangeKernel", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"}
    scanned = {}; ok = True
    for path in SOURCE_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8")); imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in ([*n.names] if isinstance(n, ast.Import) else [ast.alias(name=n.module or "")])}
        calls = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Call): calls.append(n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else "")
        bad = sorted((imports & forbidden_imports) | (set(calls) & forbidden_calls)); scanned[path.name] = {"sha256": sha256(path), "forbidden": bad}; ok &= not bad
    current = ast.parse(SCRIPT.read_text(encoding="utf-8")); windows = [n for n in ast.walk(current) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "WinDLL"]
    win_literals = sorted({n.args[0].value for n in windows if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)})
    main_node = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    inherited_terminals = sorted({n.func.attr for n in ast.walk(main_node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and any(isinstance(part, ast.Name) and part.id == "prior" for part in ast.walk(n.func))})
    allowed_inherited = {"canon", "clean_now", "collect_runtime", "preparation_summary", "prior_failure_valid", "runtime_mutations", "sha_bytes", "validate_preparation", "validate_runtime", "verify_wheel_record"}
    inherited = prior.prior.base.static_contract()
    return {"pass": bool(ok and inherited and win_literals == ["kernel32", "shell32"] and set(inherited_terminals) <= allowed_inherited), "scanned": scanned, "win_dll_literals": win_literals, "inherited_call_terminals": inherited_terminals, "inherited_r8_cpu_bootstrap": inherited}

def topology() -> dict:
    absent = tuple(prior.prior.BASE_R8_PATHS) + tuple(prior.prior.CORE) + (prior.prior.VERIFY_RESULT, prior.prior.FAILED, prior.prior.QUARANTINE) + tuple(prior.CORE) + (prior.VERIFY_RESULT, prior.FAILED, prior.QUARANTINE) + CORE + (VERIFY_RESULT, FAILED, QUARANTINE)
    return {"absent": {str(p): p.exists() for p in absent}, "temps": sorted(str(p) for p in R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*")), "family": sorted(str(p) for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8"))}

def topology_clean(x: dict) -> bool:
    locks = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", prior.prior.LOCK, prior.LOCK, LOCK))
    return set(x) == {"absent", "temps", "family"} and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == locks

def atomic_failure(stage: str, exc: BaseException, ident: dict | None) -> None:
    row = {"kind": "ph1_intel_execution_r8p3_early_failure", "status": "valid_protocol_negative", "stage": stage, "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()[-32768:], "identity": ident, "device_opened": False, "compiler_opened": False, "cpu_frozen_slice_read": False, "disposition": "bounded_create_new"}
    data = canon(row)
    if len(data) > 65536: raise RuntimeError("failure_cap")
    FAILED.mkdir(parents=True, exist_ok=True); attempt = FAILED / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); atomic_create(attempt / "failure.json", data)

def main() -> int:
    if sys.argv != EXPECTED_ARGV: return 3
    ident = None; stage = "identity"
    try:
        ident = identity()
        if not identity_valid(ident): raise RuntimeError("exact_invocation")
        pre = topology()
        if not topology_clean(pre): raise RuntimeError("topology")
        stage = "runtime"; runtime = prior.prior.base.runner.collect_runtime()
        if runtime["available"] < 16 * 2**30: raise RuntimeError("start_ram")
        wheels = {"psutil": prior.prior.base.verify_wheel_record(prior.prior.base.runner.RUNTIME_FILES["psutil_record"][0]), "numpy": prior.prior.base.verify_wheel_record(prior.prior.base.runner.RUNTIME_FILES["numpy_record"][0])}
        stage = "cpu_frozen_slice"; preparation = prior.prior.base.preparation_summary(); runtime_ok, runtime_rejected = prior.prior.base.runtime_mutations(runtime)
        stage = "static_and_transactions"; transaction = transaction_simulation(); static = static_contract(); lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}
        identity_rejected = identity_mutations(ident)
        checks = {"dual_identity": identity_valid(ident), "identity_mutations": len(identity_rejected) == 12, "hash_bindings": all(lock.get(k) == v for k, v in observed.items()), "closed_pending": lock.get("kind") == "ph1_intel_execution_r8p3_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING",
                  "runtime_lock": all(lock.get(k) == v for k, v in STATIC.items()), "runtime": prior.prior.base.runner.validate_runtime(runtime), "start_ram": runtime["available"] >= 16 * 2**30, "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899,
                  "runtime_mutations": runtime_ok and len(runtime_rejected) == 10, "r7d1_failure": prior.prior.base.runner.prior_failure_valid(), "cpu_preparation": prior.prior.base.validate_preparation(preparation), "current_transactions": set(transaction) == TX_KEYS and all(transaction.values()),
                  "static_no_model_compiler_opencl_device": static["pass"], "topology": topology_clean(pre), "base_clean": prior.prior.base.runner.clean_now()}
        row = {"kind": KIND, "ack": ACK, "identity": ident, "pre_run_topology": pre, "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "runtime": runtime, "wheel_records": wheels,
               "preparation": preparation, "preparation_digest": prior.prior.base.sha_bytes(prior.prior.base.canon(preparation)), "transaction_simulation": transaction, "static_contract": static, "identity_mutations_rejected": identity_rejected, "runtime_mutations_rejected": runtime_rejected,
               "cpu_frozen_slice_read": True, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
        stage = "publish"; publish(row, RESULT, MANIFEST, COMMIT, QUARANTINE); print(json.dumps(row, indent=2)); return 0 if row["pass"] else 3
    except Exception as exc:
        if not FAILED.exists():
            try: atomic_failure(stage, exc, ident)
            except Exception: pass
        raise

if __name__ == "__main__": raise SystemExit(main())
