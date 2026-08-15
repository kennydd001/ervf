#!/usr/bin/env python3
"""Independent verifier for the R8P6 closed, no-device preflight."""
from __future__ import annotations

import ast, copy, hashlib, json, os, sys, tempfile, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"; sys.path.insert(0, str(S))
import verify_het_next_l0_ph1_intel_execution_r8p5 as prior

SCRIPT = Path(__file__).resolve(); PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r8p6.py"; LOCK = R / "het_next_l0_ph1_intel_execution_r8p6_lock.json"; PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P6_PREREGISTRATION_2026-08-14.md"; AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P5_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p6_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p6_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p6_static_preflight.commit.json"; OUTPUT = R / "het_next_l0_ph1_intel_execution_r8p6_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p6_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p6_quarantine"
ACK = "PH1_INTEL_EXECUTION_R8P6_CONSOLIDATED_CURRENT_WRITER_CLOSED"; KIND = "ph1_intel_execution_r8p6_static_preflight"; CORE = (RESULT, MANIFEST, COMMIT)
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; STATIC = dict(prior.STATIC); BASE = prior.BASE
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT)]; EXPECTED_ARGV = [str(SCRIPT)]; PREFLIGHT_NATIVE = [str(ALIAS), "-I", "-B", str(PREFLIGHT.resolve()), "--ack", ACK]; PREFLIGHT_ARGV = [str(PREFLIGHT.resolve()), "--ack", ACK]
CPU_STATES = {"not_started", "started_not_completed", "completed"}
TX_KEYS = {"clean_commit", "repeat_rejected_unchanged", "stale_temp_cleaned", "prelink_failure_clean", "postlink_interruption_recovered", "partial_publish_quarantined", "committed_bytes_immutable"}
VERIFY_TX_KEYS = {"clean_create", "repeat_preserved", "stale_cleaned", "prelink_clean", "postlink_recovered", "bytes_immutable"}
FAILURE_KEYS = {"baseline_schema_cap", "second_unique_no_overwrite", "prelink_cleanup", "postlink_cleanup_canonical", "primary_preserved_secondary_recorded", "cpu_state_provenance_all_three"}
CHECK_NAMES = {"dual_identity", "identity_mutations", "hash_bindings", "closed_pending", "runtime_lock", "runtime", "start_ram", "wheel_records", "runtime_mutations", "r7d1_failure", "cpu_preparation", "current_transactions", "failure_simulation", "static_boundary", "topology", "base_clean"}
FAILURE_SCHEMA = {"kind", "status", "stage", "error", "traceback", "identity", "cpu_slice_state", "cpu_frozen_slice_read_started", "cpu_frozen_slice_read_completed", "device_opened", "compiler_opened", "disposition"}
CHAIN = {"preflight_sha256": PREFLIGHT, "verifier_sha256": SCRIPT, "prereg_sha256": PREREG, "r8p5_audit_sha256": AUDIT,
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

IDENTITY_KEYS = prior.prior.prior.IDENTITY_KEYS
def identity_valid(x: dict, native: list[str], argv: list[str]) -> bool:
    return isinstance(x, dict) and set(x) == IDENTITY_KEYS and bool(x["native_raw"]) and prior.prior.parse_commandline(x["native_raw"]) == native and x["native_argv"] == native and x["orig_argv"] == native and x["argv"] == argv and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is True

def live_identity_mutations(row: dict) -> list[str]:
    cases = {"native": lambda x: x["native_argv"].append("x"), "raw": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "orig": lambda x: x["orig_argv"].append("x"), "argv": lambda x: x["argv"].append("x"), "venv": lambda x: x.__setitem__("sys_executable", "wrong"), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "wrong"), "base": lambda x: x.__setitem__("base_executable", "wrong"), "base_prefix": lambda x: x.__setitem__("base_prefix", "wrong"), "flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "script": lambda x: x["argv"].__setitem__(0, "wrong"), "direct": lambda x: x.__setitem__("direct_entry", False)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not identity_valid(candidate, EXPECTED_NATIVE, EXPECTED_ARGV): out.append(name)
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

def output_writer_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); target = root / "verification.json"; data = canon({"fixture": True})
        atomic_create(target, data); frozen = target.read_bytes(); repeat = False
        try: atomic_create(target, b"overwrite")
        except FileExistsError: repeat = target.read_bytes() == frozen
        stale = root / "stale.json.inprogress.fixed"; stale.write_bytes(b"stale"); stale_ok = cleanup_temps(root, "stale.json") == [stale.name]
        def bad_link(_a, _b): raise OSError("prelink")
        pre = root / "pre.json"; pre_ok = False
        try: atomic_create(pre, b"x", link_fn=bad_link)
        except OSError: pre_ok = not pre.exists() and not list(root.glob(pre.name + ".inprogress.*"))
        def bad_unlink(_p): raise OSError("postlink")
        post = root / "post.json"; post_ok = False
        try: atomic_create(post, b"y", unlink_fn=bad_unlink)
        except OSError:
            leftovers = list(root.glob(post.name + ".inprogress.*")); cleanup_temps(root, post.name); post_ok = post.read_bytes() == b"y" and len(leftovers) == 1
        return {"clean_create": target.read_bytes() == data, "repeat_preserved": repeat, "stale_cleaned": stale_ok, "prelink_clean": pre_ok, "postlink_recovered": post_ok, "bytes_immutable": target.read_bytes() == frozen}

def failure_row(stage: str, exc: BaseException, state: object) -> dict:
    started, completed = state_bits(state); return {"kind": "ph1_intel_execution_r8p6_failure", "status": "valid_protocol_negative", "stage": stage, "error": f"{type(exc).__name__}:{exc}", "traceback": "independent_fixture", "identity": None, "cpu_slice_state": state, "cpu_frozen_slice_read_started": started, "cpu_frozen_slice_read_completed": completed, "device_opened": False, "compiler_opened": False, "disposition": "bounded_create_new_canonical"}

def independent_failure(stage: str, exc: BaseException, state: object, root: Path, *, create=atomic_create) -> dict:
    row = failure_row(stage, exc, state); data = canon(row)
    if len(data) > 65536 or set(row) != FAILURE_SCHEMA or not state_valid(row): raise RuntimeError("schema_cap")
    root.mkdir(parents=True, exist_ok=True); attempt = root / ("attempt_" + uuid.uuid4().hex); attempt.mkdir(); target = attempt / "failure.json"
    try: create(target, data)
    except Exception:
        temps = list(attempt.glob(target.name + ".inprogress.*"))
        if target.is_file() and target.read_bytes() == data:
            for p in temps: p.unlink()
            return {"path": target, "recovered": True}
        for p in temps:
            try: p.unlink()
            except Exception: pass
        if attempt.exists() and not any(attempt.iterdir()): attempt.rmdir()
        raise
    return {"path": target, "recovered": False}

STATE_MUTATION_NAMES = ["unknown_state", "list_state", "dict_state", "int_state", "null_state", "wrong_started", "wrong_completed", "missing_state", "extra_state"]
def state_mutations(row: dict) -> list[str]:
    cases = {"unknown_state": lambda x: x.__setitem__("cpu_slice_state", "unknown"), "list_state": lambda x: x.__setitem__("cpu_slice_state", []), "dict_state": lambda x: x.__setitem__("cpu_slice_state", {}), "int_state": lambda x: x.__setitem__("cpu_slice_state", 1), "null_state": lambda x: x.__setitem__("cpu_slice_state", None), "wrong_started": lambda x: x.__setitem__("cpu_frozen_slice_read_started", not x["cpu_frozen_slice_read_started"]), "wrong_completed": lambda x: x.__setitem__("cpu_frozen_slice_read_completed", not x["cpu_frozen_slice_read_completed"]), "missing_state": lambda x: x.pop("cpu_slice_state"), "extra_state": lambda x: x.__setitem__("cpu_slice_state_extra", True)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if set(candidate) != FAILURE_SCHEMA or not state_valid(candidate): out.append(name)
    return out

def independent_failure_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "failed"; exc = RuntimeError("primary"); first = independent_failure("identity", exc, "not_started", root); p1 = first["path"]; frozen = p1.read_bytes(); row1 = json.loads(frozen); second = independent_failure("post_preparation", exc, "completed", root); p2 = second["path"]; row2 = json.loads(p2.read_text())
        def precreate(path, data):
            def fail(_a, _b): raise OSError("prelink")
            return atomic_create(path, data, link_fn=fail)
        pre_root = Path(td) / "pre"; secondary = None
        try: independent_failure("identity", exc, "not_started", pre_root, create=precreate)
        except Exception as caught: secondary = f"{type(caught).__name__}:{caught}"
        def postcreate(path, data):
            def fail(_p): raise OSError("postunlink")
            return atomic_create(path, data, unlink_fn=fail)
        post_root = Path(td) / "post"; post = independent_failure("preparation", exc, "started_not_completed", post_root, create=postcreate); pp = post["path"]; row3 = json.loads(pp.read_text()); states = [row1, row2, row3]
        return {"baseline_schema_cap": set(row1) == FAILURE_SCHEMA and len(frozen) <= 65536 and state_valid(row1), "second_unique_no_overwrite": p1 != p2 and p1.read_bytes() == frozen and len(list(root.rglob("failure.json"))) == 2, "prelink_cleanup": secondary == "OSError:prelink" and (not pre_root.exists() or not list(pre_root.rglob("*"))), "postlink_cleanup_canonical": post["recovered"] is True and pp.is_file() and not list(pp.parent.glob("*.inprogress.*")) and canon(row3) == pp.read_bytes(), "primary_preserved_secondary_recorded": secondary == "OSError:prelink", "cpu_state_provenance_all_three": [r["stage"] for r in states] == ["identity", "post_preparation", "preparation"] and [r["cpu_slice_state"] for r in states] == ["not_started", "completed", "started_not_completed"] and all(state_valid(r) and state_mutations(r) == STATE_MUTATION_NAMES for r in states)}

def lock_valid() -> bool:
    lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}
    return set(lock) == {"kind", "execution_open", "audit_token", *STATIC, *observed} and lock["kind"] == "ph1_intel_execution_r8p6_lock" and lock["execution_open"] is False and lock["audit_token"] == "PENDING" and all(lock.get(k) == v for k, v in STATIC.items()) and all(lock.get(k) == v for k, v in observed.items()) and observed["r8p5_audit_sha256"] == "c431578cc6a1edefa0d3843ca0fdd26ec5d07b9e592d5a758a4ed0f40e36d608"

def bundle_valid() -> bool:
    if not all(p.is_file() for p in CORE): return False
    rb = RESULT.read_bytes(); mb = MANIFEST.read_bytes()
    return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": RESULT.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(COMMIT.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}

def topology_valid() -> bool:
    expected = {R / "het_next_l0_ph1_intel_execution_r8_lock.json", *(R / f"het_next_l0_ph1_intel_execution_r8p{i}_lock.json" for i in range(1, 7)), *CORE}
    family = {p for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8")}
    return family == expected and not FAILED.exists() and not QUARANTINE.exists() and not OUTPUT.exists() and not list(R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*"))

def stored_topology_valid(x: dict) -> bool:
    locks = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", *(R / f"het_next_l0_ph1_intel_execution_r8p{i}_lock.json" for i in range(1, 7))))
    return isinstance(x, dict) and set(x) == {"absent", "temps", "family"} and len(x["absent"]) == 42 and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == locks

def static_boundary() -> bool:
    forbidden = {"Popen", "run", "check_call", "check_output", "CDLL", "LoadLibrary", "clBuildProgram", "clCreateContext", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"}
    for path in (PREFLIGHT, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8")); calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if calls & forbidden: return False
    return prior.static_boundary()

def result_valid(row: dict, preparation: dict, wheels: dict) -> bool:
    checks = row.get("checks", {}); tx = row.get("transaction_simulation"); failure = row.get("failure_simulation")
    return set(row) == {"kind", "ack", "identity", "pre_run_topology", "checks", "pass", "passed", "total", "runtime", "wheel_records", "preparation", "preparation_digest", "transaction_simulation", "failure_simulation", "identity_mutations_rejected", "runtime_mutations_rejected", "cpu_slice_state", "cpu_frozen_slice_read_started", "cpu_frozen_slice_read_completed", "model_forward", "compiler_opened", "opencl_opened", "device_opened"} and row["kind"] == KIND and row["ack"] == ACK and identity_valid(row["identity"], PREFLIGHT_NATIVE, PREFLIGHT_ARGV) and stored_topology_valid(row["pre_run_topology"]) and set(checks) == CHECK_NAMES and all(v is True for v in checks.values()) and row["pass"] is True and row["passed"] == row["total"] == len(CHECK_NAMES) and BASE.runtime_static_valid(row["runtime"]) and row["runtime"]["available"] >= 16 * 2**30 and row["wheel_records"] == wheels and row["preparation"] == preparation and row["preparation_digest"] == BASE.PREPARATION_DIGEST == sha_bytes(canon(preparation)) and isinstance(tx, dict) and set(tx) == TX_KEYS and all(v is True for v in tx.values()) and isinstance(failure, dict) and set(failure) == FAILURE_KEYS and all(v is True for v in failure.values()) and row["identity_mutations_rejected"] == ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "direct"] and row["runtime_mutations_rejected"] == ["python_path", "python_hash", "isolation", "bytecode", "pyvenv", "psutil_native", "psutil_record", "numpy_version", "numpy_record", "ram"] and row["cpu_slice_state"] == "completed" and state_valid(row) and row["model_forward"] is row["compiler_opened"] is row["opencl_opened"] is row["device_opened"] is False

def result_mutations(row: dict, preparation: dict, wheels: dict) -> list[str]:
    cases = {"empty_tx": lambda x: x.__setitem__("transaction_simulation", {}), "missing_tx": lambda x: x["transaction_simulation"].pop("clean_commit"), "extra_tx": lambda x: x["transaction_simulation"].__setitem__("extra", True), "false_tx": lambda x: x["transaction_simulation"].__setitem__("clean_commit", False), "empty_failure": lambda x: x.__setitem__("failure_simulation", {}), "list_state": lambda x: x.__setitem__("cpu_slice_state", []), "dict_state": lambda x: x.__setitem__("cpu_slice_state", {}), "int_state": lambda x: x.__setitem__("cpu_slice_state", 1), "null_state": lambda x: x.__setitem__("cpu_slice_state", None), "wrong_started": lambda x: x.__setitem__("cpu_frozen_slice_read_started", False), "wrong_completed": lambda x: x.__setitem__("cpu_frozen_slice_read_completed", False), "device": lambda x: x.__setitem__("device_opened", True)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not result_valid(candidate, preparation, wheels): out.append(name)
    return out

def main() -> int:
    ident = prior.prior.live_identity()
    if not identity_valid(ident, EXPECTED_NATIVE, EXPECTED_ARGV): return 3
    raw = RESULT.read_bytes(); row = json.loads(raw); preparation = BASE.independent_preparation(); wheels = {"psutil": BASE.wheel_record(BASE.FILES["psutil_record"][0]), "numpy": BASE.wheel_record(BASE.FILES["numpy_record"][0])}; failure = independent_failure_simulation(); writer = output_writer_simulation(); live_rejected = live_identity_mutations(ident); rejected = result_mutations(row, preparation, wheels)
    checks = {"live_identity": True, "live_identity_mutations": live_rejected == ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "direct"], "lock": lock_valid(), "bundle": bundle_valid(), "topology": topology_valid(), "static_boundary": static_boundary(), "r7d1_failure": prior.prior.prior.prior.prior.failure_bundle_valid(prior.prior.prior.prior.prior.FAILURE_ROOT), "runtime": BASE.runtime_static_valid(BASE.collect_runtime()), "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899, "preparation": sha_bytes(canon(preparation)) == BASE.PREPARATION_DIGEST, "result": result_valid(row, preparation, wheels), "independent_output_transactions": set(writer) == VERIFY_TX_KEYS and all(writer.values()), "independent_failure_simulation": set(failure) == FAILURE_KEYS and all(failure.values()), "typed_state_and_tx_mutations": rejected == ["empty_tx", "missing_tx", "extra_tx", "false_tx", "empty_failure", "list_state", "dict_state", "int_state", "null_state", "wrong_started", "wrong_completed", "device"]}
    output = {"kind": "ph1_intel_execution_r8p6_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "result_sha256": sha_bytes(raw), "manifest_sha256": sha256(MANIFEST), "commit_sha256": sha256(COMMIT), "independent_output_transactions": writer, "independent_failure_simulation": failure, "cpu_slice_state": "completed", "cpu_frozen_slice_read_started": True, "cpu_frozen_slice_read_completed": True, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
    atomic_create(OUTPUT, canon(output)); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3

if __name__ == "__main__": raise SystemExit(main())
