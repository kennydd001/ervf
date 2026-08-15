#!/usr/bin/env python3
"""R8P6 consolidated closed preflight: current writers, typed CPU-slice state."""
from __future__ import annotations

import ast, copy, hashlib, json, os, sys, tempfile, traceback, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"; sys.path.insert(0, str(S))
import preflight_het_next_l0_ph1_intel_execution_r8p5 as prior

SCRIPT = Path(__file__).resolve(); VERIFIER = S / "verify_het_next_l0_ph1_intel_execution_r8p6.py"; LOCK = R / "het_next_l0_ph1_intel_execution_r8p6_lock.json"; PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_PREREGISTRATION_2026-08-14.md"; AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P5_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p6_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p6_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p6_static_preflight.commit.json"; VERIFY_RESULT = R / "het_next_l0_ph1_intel_execution_r8p6_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p6_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p6_quarantine"
ACK = "PH1_INTEL_EXECUTION_R8P6_CONSOLIDATED_CURRENT_WRITER_CLOSED"; KIND = "ph1_intel_execution_r8p6_static_preflight"; CORE = (RESULT, MANIFEST, COMMIT)
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; PYVENV = prior.PYVENV; STATIC = dict(prior.STATIC); BASE = prior.BASE
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT), "--ack", ACK]; EXPECTED_ARGV = [str(SCRIPT), "--ack", ACK]
CPU_STATES = {"not_started", "started_not_completed", "completed"}; TX_KEYS = {"clean_commit", "repeat_rejected_unchanged", "stale_temp_cleaned", "prelink_failure_clean", "postlink_interruption_recovered", "partial_publish_quarantined", "committed_bytes_immutable"}; FAILURE_KEYS = set(prior.FAILURE_KEYS)
FAILURE_SCHEMA = {"kind", "status", "stage", "error", "traceback", "identity", "cpu_slice_state", "cpu_frozen_slice_read_started", "cpu_frozen_slice_read_completed", "device_opened", "compiler_opened", "disposition"}
CHAIN = {"preflight_sha256": SCRIPT, "verifier_sha256": VERIFIER, "prereg_sha256": PREREG, "r8p5_audit_sha256": AUDIT,
         "r8p5_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p5.py", "r8p5_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p5.py", "r8p5_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P5_PREREGISTRATION_2026-08-14.md", "r8p5_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p5_lock.json",
         **{("r8p5_" + k if k in {"preflight_sha256", "verifier_sha256", "prereg_sha256"} else k): v for k, v in prior.CHAIN.items()}}

def sha_bytes(x: bytes) -> str: return hashlib.sha256(x).hexdigest()
def sha256(p: Path) -> str: return sha_bytes(p.read_bytes())
def canon(x: object) -> bytes: return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()
def same(a: object, b: object) -> bool: return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def state_bits(state: object) -> tuple[bool, bool]:
    if not isinstance(state, str) or state not in CPU_STATES: raise ValueError("cpu_slice_state")
    return state != "not_started", state == "completed"

def state_valid(row: dict) -> bool:
    try: started, completed = state_bits(row.get("cpu_slice_state"))
    except (ValueError, TypeError): return False
    return row.get("cpu_frozen_slice_read_started") is started and row.get("cpu_frozen_slice_read_completed") is completed

IDENTITY_KEYS = prior.prior.prior.prior.IDENTITY_KEYS
def identity_valid(x: dict) -> bool:
    return set(x) == IDENTITY_KEYS and bool(x["native_raw"]) and prior.prior.parse_commandline(x["native_raw"]) == EXPECTED_NATIVE and x["native_argv"] == EXPECTED_NATIVE and x["orig_argv"] == EXPECTED_NATIVE and x["argv"] == EXPECTED_ARGV and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is True

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
        p2, m2, c2 = root / "p2.json", root / "m2.json", root / "c2.json"; calls = {"n": 0}
        def fail_second(path, data):
            calls["n"] += 1
            if calls["n"] == 2: raise OSError("interrupt")
            atomic_create(path, data)
        partial = False
        try: publish(row, p2, m2, c2, root / "q2", create=fail_second)
        except OSError:
            moved = list((root / "q2").rglob("p2.json")); partial = not any(p.exists() for p in (p2, m2, c2)) and len(moved) == 1 and moved[0].read_bytes() == canon(row)
        return {"clean_commit": clean, "repeat_rejected_unchanged": repeat, "stale_temp_cleaned": stale_ok, "prelink_failure_clean": pre_ok, "postlink_interruption_recovered": post_ok, "partial_publish_quarantined": partial, "committed_bytes_immutable": tuple(p.read_bytes() for p in (result, manifest, commit)) == frozen}

def failure_row(stage: str, exc: BaseException, ident: dict | None, cpu_state: str) -> dict:
    started, completed = state_bits(cpu_state); return {"kind": "ph1_intel_execution_r8p6_failure", "status": "valid_protocol_negative", "stage": stage, "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()[-32768:], "identity": ident, "cpu_slice_state": cpu_state, "cpu_frozen_slice_read_started": started, "cpu_frozen_slice_read_completed": completed, "device_opened": False, "compiler_opened": False, "disposition": "bounded_create_new_canonical"}

def atomic_failure(stage: str, exc: BaseException, ident: dict | None, cpu_state: str, root: Path = FAILED, *, create=atomic_create) -> dict:
    row = failure_row(stage, exc, ident, cpu_state); data = canon(row)
    if len(data) > 65536 or set(row) != FAILURE_SCHEMA or not state_valid(row): raise RuntimeError("failure_schema_cap")
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

def preserve_primary(stage: str, primary: BaseException, ident: dict | None, cpu_state: str, root: Path, *, writer=atomic_failure) -> dict:
    try: return {"primary": f"{type(primary).__name__}:{primary}", "secondary": None, "evidence": writer(stage, primary, ident, cpu_state, root)}
    except Exception as secondary: return {"primary": f"{type(primary).__name__}:{primary}", "secondary": f"{type(secondary).__name__}:{secondary}", "evidence": None}

STATE_MUTATION_NAMES = ["unknown_state", "list_state", "dict_state", "int_state", "null_state", "wrong_started", "wrong_completed", "missing_state", "extra_state"]
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
        return {"baseline_schema_cap": set(row1) == FAILURE_SCHEMA and len(frozen) <= 65536 and state_valid(row1), "second_unique_no_overwrite": p1 != p2 and p1.read_bytes() == frozen and len(list(root.rglob("failure.json"))) == 2, "prelink_cleanup": pre["secondary"] == "OSError:prelink" and (not pre_root.exists() or not list(pre_root.rglob("*"))), "postlink_cleanup_canonical": post["recovered_postlink"] is True and pp.is_file() and not list(pp.parent.glob("*.inprogress.*")) and canon(row3) == pp.read_bytes(), "primary_preserved_secondary_recorded": pre["primary"] == "RuntimeError:primary" and pre["secondary"] == "OSError:prelink", "cpu_state_provenance_all_three": [r["stage"] for r in states] == ["identity", "post_preparation", "preparation"] and [r["cpu_slice_state"] for r in states] == ["not_started", "completed", "started_not_completed"] and all(state_valid(r) and state_mutations(r) == STATE_MUTATION_NAMES for r in states)}

def topology() -> dict:
    ancestors = []
    for module in (prior.prior.prior.prior.prior, prior.prior.prior.prior, prior.prior.prior, prior.prior, prior): ancestors.extend((*module.CORE, module.VERIFY_RESULT, module.FAILED, module.QUARANTINE))
    absent = tuple(prior.prior.prior.prior.prior.BASE_R8_PATHS) + tuple(ancestors) + CORE + (VERIFY_RESULT, FAILED, QUARANTINE)
    return {"absent": {str(p): p.exists() for p in absent}, "temps": sorted(str(p) for p in R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*")), "family": sorted(str(p) for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8"))}

def topology_clean(x: dict) -> bool:
    locks = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", prior.prior.prior.prior.prior.LOCK, prior.prior.prior.prior.LOCK, prior.prior.prior.LOCK, prior.prior.LOCK, prior.LOCK, LOCK)); return set(x) == {"absent", "temps", "family"} and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == locks

def static_contract() -> bool:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8")); calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in ast.walk(tree) if isinstance(n, ast.Call)}; return prior.static_contract() and not ({"Popen", "run", "CDLL", "LoadLibrary", "clBuildProgram", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"} & calls)

def main() -> int:
    if sys.argv != EXPECTED_ARGV: return 3
    ident = None; stage = "identity"; cpu_state = "not_started"
    try:
        ident = prior.prior.identity()
        if not identity_valid(ident): raise RuntimeError("exact_invocation")
        pre = topology()
        if not topology_clean(pre): raise RuntimeError("topology")
        stage = "runtime"; runtime = BASE.runner.collect_runtime()
        if runtime["available"] < 16 * 2**30: raise RuntimeError("start_ram")
        wheels = {"psutil": BASE.verify_wheel_record(BASE.runner.RUNTIME_FILES["psutil_record"][0]), "numpy": BASE.verify_wheel_record(BASE.runner.RUNTIME_FILES["numpy_record"][0])}
        stage = "cpu_frozen_slice"; cpu_state = "started_not_completed"; preparation = BASE.preparation_summary(); cpu_state = "completed"; runtime_ok, runtime_rejected = BASE.runtime_mutations(runtime)
        stage = "simulations"; transaction = transaction_simulation(); failure = failure_simulation(); lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}; rejected = identity_mutations(ident)
        checks = {"dual_identity": identity_valid(ident), "identity_mutations": len(rejected) == 12, "hash_bindings": all(lock.get(k) == v for k, v in observed.items()), "closed_pending": lock.get("kind") == "ph1_intel_execution_r8p6_lock" and lock.get("execution_open") is False and lock.get("audit_token") == "PENDING", "runtime_lock": all(lock.get(k) == v for k, v in STATIC.items()), "runtime": BASE.runner.validate_runtime(runtime), "start_ram": runtime["available"] >= 16 * 2**30, "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899, "runtime_mutations": runtime_ok and len(runtime_rejected) == 10, "r7d1_failure": BASE.runner.prior_failure_valid(), "cpu_preparation": BASE.validate_preparation(preparation), "current_transactions": set(transaction) == TX_KEYS and all(transaction.values()), "failure_simulation": set(failure) == FAILURE_KEYS and all(failure.values()), "static_boundary": static_contract(), "topology": topology_clean(pre), "base_clean": BASE.runner.clean_now()}
        row = {"kind": KIND, "ack": ACK, "identity": ident, "pre_run_topology": pre, "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "runtime": runtime, "wheel_records": wheels, "preparation": preparation, "preparation_digest": BASE.sha_bytes(BASE.canon(preparation)), "transaction_simulation": transaction, "failure_simulation": failure, "identity_mutations_rejected": rejected, "runtime_mutations_rejected": runtime_rejected, "cpu_slice_state": cpu_state, "cpu_frozen_slice_read_started": True, "cpu_frozen_slice_read_completed": True, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
        stage = "publish"; publish(row); print(json.dumps(row, indent=2)); return 0 if row["pass"] else 3
    except Exception as primary:
        preserve_primary(stage, primary, ident, cpu_state, FAILED)
        raise

if __name__ == "__main__": raise SystemExit(main())
