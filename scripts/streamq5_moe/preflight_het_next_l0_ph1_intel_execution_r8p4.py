#!/usr/bin/env python3
"""R8P4 closed CPU preflight; current bounded early-failure writer is TEMP-tested."""
from __future__ import annotations

import ast, copy, ctypes as C, hashlib, json, os, sys, tempfile, traceback, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"; sys.path.insert(0, str(S))
import preflight_het_next_l0_ph1_intel_execution_r8p3 as prior

SCRIPT = Path(__file__).resolve(); VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8p4.py"; LOCK = R / "het_next_l0_ph1_intel_execution_r8p4_lock.json"; PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P4_PREREGISTRATION_2026-08-14.md"; AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P3_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p4_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p4_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p4_static_preflight.commit.json"; VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p4_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p4_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p4_quarantine"
ACK = "PH1_INTEL_EXECUTION_R8P4_BOUNDED_FAILURE_CPU_PREPARATION_CLOSED"; KIND = "ph1_intel_execution_r8p4_static_preflight"; CORE = (RESULT, MANIFEST, COMMIT)
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; PYVENV = prior.PYVENV; STATIC = dict(prior.STATIC)
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT), "--ack", ACK]; EXPECTED_ARGV = [str(SCRIPT), "--ack", ACK]
TX_KEYS = prior.TX_KEYS; FAILURE_KEYS = {"baseline_schema_cap", "second_unique_no_overwrite", "prelink_cleanup", "postlink_cleanup_canonical", "primary_preserved_secondary_recorded"}
CHAIN = {"preflight_sha256": SCRIPT, "verifier_sha256": VERIFIER, "prereg_sha256": PREREG, "r8p3_audit_sha256": AUDIT,
         "r8p3_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p3.py", "r8p3_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p3.py", "r8p3_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P3_PREREGISTRATION_2026-08-14.md", "r8p3_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p3_lock.json",
         **{("r8p3_" + k if k in {"preflight_sha256", "verifier_sha256", "prereg_sha256"} else k): v for k, v in prior.CHAIN.items()}}
FAILURE_SCHEMA = {"kind", "status", "stage", "error", "traceback", "identity", "device_opened", "compiler_opened", "cpu_frozen_slice_read", "disposition"}

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
    return {"native_raw": raw, "native_argv": parse_commandline(raw), "orig_argv": list(sys.orig_argv), "argv": list(sys.argv), "sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_executable": getattr(sys, "_base_executable", None), "base_prefix": sys.base_prefix, "venv_launcher_sha256": sha256(VENV_PYTHON), "pyvenv": {"sha256": sha256(PYVENV), "home": cfg.get("home"), "executable": cfg.get("executable"), "version": cfg.get("version")}, "base_binary_sha256": sha256(BASE_BINARY), "base_binary_bytes": BASE_BINARY.stat().st_size, "direct_entry": __spec__ is None and (__package__ is None or __package__ == "")}

def identity_valid(x: dict) -> bool:
    return set(x) == prior.prior.IDENTITY_KEYS and bool(x["native_raw"]) and parse_commandline(x["native_raw"]) == EXPECTED_NATIVE and x["native_argv"] == EXPECTED_NATIVE and x["orig_argv"] == EXPECTED_NATIVE and x["argv"] == EXPECTED_ARGV and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is True

def identity_mutations(row: dict) -> list[str]:
    cases = {"native": lambda x: x["native_argv"].append("x"), "raw": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "orig": lambda x: x["orig_argv"].append("x"), "argv": lambda x: x["argv"].append("x"), "venv": lambda x: x.__setitem__("sys_executable", "wrong"), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "wrong"), "base": lambda x: x.__setitem__("base_executable", "wrong"), "base_prefix": lambda x: x.__setitem__("base_prefix", "wrong"), "flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "script": lambda x: x["argv"].__setitem__(0, "wrong"), "direct": lambda x: x.__setitem__("direct_entry", False)}; out = []
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
    for p in sorted(root.glob(stem + ".inprogress.*")): p.unlink(); rows.append(p.name)
    return rows

def quarantine_core(paths: tuple[Path, ...], root: Path) -> list[dict]:
    existing = [p for p in paths if p.exists()]
    if not existing: return []
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); rows = []
    for p in existing:
        target = attempt / p.name; os.replace(p, target); rows.append({"name": p.name, "sha256": sha256(target), "bytes": target.stat().st_size})
    return rows

def verify_bundle(result: Path, manifest: Path, commit: Path) -> bool:
    if not all(p.is_file() for p in (result, manifest, commit)): return False
    rb = result.read_bytes(); mb = manifest.read_bytes(); return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(commit.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}

def publish(row: dict, result: Path, manifest: Path, commit: Path, quarantine: Path) -> None:
    if any(p.exists() for p in (result, manifest, commit)): raise FileExistsError("bundle_target")
    rb = canon(row); mb = canon({"kind": KIND + "_manifest", "files": [{"name": result.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]}); cb = canon({"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)})
    try:
        atomic_create(result, rb); atomic_create(manifest, mb); atomic_create(commit, cb)
        if not verify_bundle(result, manifest, commit): raise RuntimeError("bundle")
    except Exception:
        quarantine_core((result, manifest, commit), quarantine); raise

def transaction_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); result = root / "result.json"; manifest = root / "manifest.json"; commit = root / "commit.json"; quarantine = root / "quarantine"; row = {"fixture": True}
        publish(row, result, manifest, commit, quarantine); frozen = tuple(p.read_bytes() for p in (result, manifest, commit)); clean = verify_bundle(result, manifest, commit); repeat = False
        try: publish(row, result, manifest, commit, quarantine)
        except FileExistsError: repeat = tuple(p.read_bytes() for p in (result, manifest, commit)) == frozen
        stale = root / "stale.json.inprogress.fixed"; stale.write_bytes(b"s"); stale_ok = cleanup_temps(root, "stale.json") == [stale.name]
        def bad_link(_a, _b): raise OSError("prelink")
        pre = root / "pre.json"; pre_ok = False
        try: atomic_create(pre, b"x", link_fn=bad_link)
        except OSError: pre_ok = not pre.exists() and not list(root.glob(pre.name + ".inprogress.*"))
        def bad_unlink(_p): raise OSError("postlink")
        post = root / "post.json"; post_ok = False
        try: atomic_create(post, b"y", unlink_fn=bad_unlink)
        except OSError:
            leftovers = list(root.glob(post.name + ".inprogress.*")); cleanup_temps(root, post.name); post_ok = post.read_bytes() == b"y" and len(leftovers) == 1
        p2, m2, c2 = root / "p2.json", root / "m2.json", root / "c2.json"; count = {"n": 0}
        def fail_second(path, data):
            count["n"] += 1
            if count["n"] == 2: raise OSError("interrupt")
            atomic_create(path, data)
        partial = False
        try:
            rb = canon(row); mb = canon({"kind": KIND + "_manifest", "files": [{"name": p2.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]}); cb = canon({"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)})
            try:
                fail_second(p2, rb); fail_second(m2, mb); fail_second(c2, cb)
            except Exception:
                quarantine_core((p2, m2, c2), root / "q2"); raise
        except OSError:
            moved = list((root / "q2").rglob("p2.json")); partial = not any(p.exists() for p in (p2, m2, c2)) and len(moved) == 1 and moved[0].read_bytes() == canon(row)
        return {"clean_commit": clean, "repeat_rejected_unchanged": repeat, "stale_temp_cleaned": stale_ok, "prelink_failure_clean": pre_ok, "postlink_interruption_recovered": post_ok, "partial_publish_quarantined": partial, "committed_bytes_immutable": tuple(p.read_bytes() for p in (result, manifest, commit)) == frozen}

def failure_row(stage: str, exc: BaseException, ident: dict | None) -> dict:
    return {"kind": "ph1_intel_execution_r8p4_early_failure", "status": "valid_protocol_negative", "stage": stage, "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()[-32768:], "identity": ident, "device_opened": False, "compiler_opened": False, "cpu_frozen_slice_read": False, "disposition": "bounded_create_new_canonical"}

def atomic_failure(stage: str, exc: BaseException, ident: dict | None, root: Path = FAILED, *, create=atomic_create) -> dict:
    row = failure_row(stage, exc, ident); data = canon(row)
    if len(data) > 65536: raise RuntimeError("failure_cap")
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); target = attempt / "failure.json"
    try: create(target, data)
    except Exception:
        temps = list(attempt.glob(target.name + ".inprogress.*"))
        if target.is_file() and target.read_bytes() == data:
            for temp in temps: temp.unlink()
            return {"path": str(target), "sha256": sha256(target), "bytes": len(data), "recovered_postlink": True}
        for temp in temps:
            try: temp.unlink()
            except Exception: pass
        if attempt.exists() and not any(attempt.iterdir()): attempt.rmdir()
        raise
    return {"path": str(target), "sha256": sha256(target), "bytes": len(data), "recovered_postlink": False}

def preserve_primary(stage: str, primary: BaseException, ident: dict | None, root: Path, *, writer=atomic_failure) -> dict:
    try: evidence = writer(stage, primary, ident, root); return {"primary": f"{type(primary).__name__}:{primary}", "secondary": None, "evidence": evidence}
    except Exception as secondary: return {"primary": f"{type(primary).__name__}:{primary}", "secondary": f"{type(secondary).__name__}:{secondary}", "evidence": None}

def failure_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "failed"; primary = RuntimeError("primary")
        try: raise primary
        except RuntimeError as exc: first = atomic_failure("identity", exc, {"fixture": True}, root)
        path1 = Path(first["path"]); row = json.loads(path1.read_text()); frozen = path1.read_bytes(); second = atomic_failure("runtime", primary, None, root); path2 = Path(second["path"])
        baseline = set(row) == FAILURE_SCHEMA and len(frozen) <= 65536 and row["disposition"] == "bounded_create_new_canonical" and first["sha256"] == sha_bytes(frozen)
        unique = path1 != path2 and path1.read_bytes() == frozen and len(list(root.rglob("failure.json"))) == 2
        pre_root = Path(td) / "pre"
        def precreate(path, data):
            def fail(_a, _b): raise OSError("prelink")
            return atomic_create(path, data, link_fn=fail)
        pre = preserve_primary("identity", primary, None, pre_root, writer=lambda s, e, i, r: atomic_failure(s, e, i, r, create=precreate))
        preclean = pre["primary"] == "RuntimeError:primary" and pre["secondary"] == "OSError:prelink" and (not pre_root.exists() or not list(pre_root.rglob("*")))
        post_root = Path(td) / "post"
        def postcreate(path, data):
            def fail(_p): raise OSError("postunlink")
            return atomic_create(path, data, unlink_fn=fail)
        post = atomic_failure("runtime", primary, None, post_root, create=postcreate); postpath = Path(post["path"]); postok = post["recovered_postlink"] is True and postpath.is_file() and postpath.stat().st_size <= 65536 and not list(postpath.parent.glob("*.inprogress.*")) and canon(json.loads(postpath.read_text())) == postpath.read_bytes()
        return {"baseline_schema_cap": baseline, "second_unique_no_overwrite": unique, "prelink_cleanup": preclean, "postlink_cleanup_canonical": postok, "primary_preserved_secondary_recorded": pre["primary"] == "RuntimeError:primary" and pre["secondary"] == "OSError:prelink"}

def topology() -> dict:
    absent = tuple(prior.prior.prior.BASE_R8_PATHS) + tuple(prior.prior.prior.CORE) + (prior.prior.prior.VERIFY_RESULT, prior.prior.prior.FAILED, prior.prior.prior.QUARANTINE) + tuple(prior.prior.CORE) + (prior.prior.VERIFY_RESULT, prior.prior.FAILED, prior.prior.QUARANTINE) + tuple(prior.CORE) + (prior.VERIFY_RESULT, prior.FAILED, prior.QUARANTINE) + CORE + (VERIFY_RESULT, FAILED, QUARANTINE)
    return {"absent": {str(p): p.exists() for p in absent}, "temps": sorted(str(p) for p in R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*")), "family": sorted(str(p) for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8"))}

def topology_clean(x: dict) -> bool:
    locks = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", prior.prior.prior.LOCK, prior.prior.LOCK, prior.LOCK, LOCK)); return set(x) == {"absent", "temps", "family"} and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == locks

def static_contract() -> bool:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8")); imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}; calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return prior.static_contract()["pass"] and not ({"pyopencl", "cupy", "torch", "transformers"} & imports) and not ({"Popen", "run", "CDLL", "LoadLibrary", "clBuildProgram", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"} & calls)

def main() -> int:
    if sys.argv != EXPECTED_ARGV: return 3
    ident = None; stage = "identity"
    try:
        ident = identity()
        if not identity_valid(ident): raise RuntimeError("exact_invocation")
        pre = topology()
        if not topology_clean(pre): raise RuntimeError("topology")
        stage = "runtime"; runtime = prior.prior.prior.base.runner.collect_runtime()
        if runtime["available"] < 16 * 2**30: raise RuntimeError("start_ram")
        wheels = {"psutil": prior.prior.prior.base.verify_wheel_record(prior.prior.prior.base.runner.RUNTIME_FILES["psutil_record"][0]), "numpy": prior.prior.prior.base.verify_wheel_record(prior.prior.prior.base.runner.RUNTIME_FILES["numpy_record"][0])}
        stage = "cpu_frozen_slice"; preparation = prior.prior.prior.base.preparation_summary(); runtime_ok, runtime_rejected = prior.prior.prior.base.runtime_mutations(runtime)
        stage = "simulations"; transaction = transaction_simulation(); failure = failure_simulation(); lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}; rejected = identity_mutations(ident)
        checks = {"dual_identity": identity_valid(ident), "identity_mutations": len(rejected) == 12, "hash_bindings": all(lock.get(k) == v for k, v in observed.items()), "closed_pending": lock.get("kind") == "ph1_intel_execution_r8p4_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING", "runtime_lock": all(lock.get(k) == v for k, v in STATIC.items()), "runtime": prior.prior.prior.base.runner.validate_runtime(runtime), "start_ram": runtime["available"] >= 16 * 2**30, "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899, "runtime_mutations": runtime_ok and len(runtime_rejected) == 10, "r7d1_failure": prior.prior.prior.base.runner.prior_failure_valid(), "cpu_preparation": prior.prior.prior.base.validate_preparation(preparation), "current_transactions": set(transaction) == TX_KEYS and all(transaction.values()), "failure_simulation": set(failure) == FAILURE_KEYS and all(failure.values()), "static_boundary": static_contract(), "topology": topology_clean(pre), "base_clean": prior.prior.prior.base.runner.clean_now()}
        row = {"kind": KIND, "ack": ACK, "identity": ident, "pre_run_topology": pre, "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "runtime": runtime, "wheel_records": wheels, "preparation": preparation, "preparation_digest": prior.prior.prior.base.sha_bytes(prior.prior.prior.base.canon(preparation)), "transaction_simulation": transaction, "failure_simulation": failure, "identity_mutations_rejected": rejected, "runtime_mutations_rejected": runtime_rejected, "cpu_frozen_slice_read": True, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
        stage = "publish"; publish(row, RESULT, MANIFEST, COMMIT, QUARANTINE); print(json.dumps(row, indent=2)); return 0 if row["pass"] else 3
    except Exception as primary:
        preserve_primary(stage, primary, ident, FAILED)
        raise

if __name__ == "__main__": raise SystemExit(main())
