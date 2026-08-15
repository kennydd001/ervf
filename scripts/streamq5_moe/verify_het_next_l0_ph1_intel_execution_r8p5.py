#!/usr/bin/env python3
"""Independent R8P5 verifier for three-state CPU-slice failure provenance."""
from __future__ import annotations

import ast, copy, hashlib, json, os, sys, tempfile, traceback, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; S = ROOT / "scripts/streamq5_moe"; R = ROOT / "reports/streamq5_moe"; sys.path.insert(0, str(S))
import verify_het_next_l0_ph1_intel_execution_r8p4 as prior

SCRIPT = Path(__file__).resolve(); PREFLIGHT = S / "preflight_het_next_l0_ph1_intel_execution_r8p5.py"; LOCK = R / "het_next_l0_ph1_intel_execution_r8p5_lock.json"; PREREG = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P5_PREREGISTRATION_2026-08-14.md"; AUDIT = R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P4_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md"
RESULT = R / "het_next_l0_ph1_intel_execution_r8p5_static_preflight.json"; MANIFEST = R / "het_next_l0_ph1_intel_execution_r8p5_static_preflight.manifest.json"; COMMIT = R / "het_next_l0_ph1_intel_execution_r8p5_static_preflight.commit.json"; OUTPUT = R / "het_next_l0_ph1_intel_execution_r8p5_independent_verification.json"; FAILED = R / "het_next_l0_ph1_intel_execution_r8p5_failed_attempts"; QUARANTINE = R / "het_next_l0_ph1_intel_execution_r8p5_quarantine"
ACK = "PH1_INTEL_EXECUTION_R8P5_CPU_SLICE_STAGE_PROVENANCE_CLOSED"; KIND = "ph1_intel_execution_r8p5_static_preflight"
ALIAS = prior.ALIAS; BASE_PREFIX = prior.BASE_PREFIX; BASE_BINARY = prior.BASE_BINARY; VENV = prior.VENV; VENV_PYTHON = prior.VENV_PYTHON; PYVENV = prior.PYVENV; STATIC = dict(prior.STATIC); BASE = prior.prior.prior.prior.base
EXPECTED_NATIVE = [str(ALIAS), "-I", "-B", str(SCRIPT)]; EXPECTED_ARGV = [str(SCRIPT)]; PREFLIGHT_NATIVE = [str(ALIAS), "-I", "-B", str(PREFLIGHT.resolve()), "--ack", ACK]; PREFLIGHT_ARGV = [str(PREFLIGHT.resolve()), "--ack", ACK]
CPU_STATES = {"not_started", "started_not_completed", "completed"}; TX_KEYS = prior.TX_KEYS; FAILURE_KEYS = set(prior.FAILURE_KEYS) | {"cpu_state_provenance_all_three"}; CHECK_NAMES = prior.CHECK_NAMES
FAILURE_SCHEMA = {"kind", "status", "stage", "error", "traceback", "identity", "cpu_slice_state", "cpu_frozen_slice_read_started", "cpu_frozen_slice_read_completed", "device_opened", "compiler_opened", "disposition"}
CHAIN = {"preflight_sha256": PREFLIGHT, "verifier_sha256": SCRIPT, "prereg_sha256": PREREG, "r8p4_audit_sha256": AUDIT,
         "r8p4_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p4.py", "r8p4_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p4.py", "r8p4_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P4_PREREGISTRATION_2026-08-14.md", "r8p4_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p4_lock.json",
         "r8p3_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P3_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r8p3_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p3.py", "r8p3_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p3.py", "r8p3_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P3_PREREGISTRATION_2026-08-14.md", "r8p3_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p3_lock.json",
         "r8p2_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r8p2_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p2.py", "r8p2_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p2.py", "r8p2_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P2_PREREGISTRATION_2026-08-14.md", "r8p2_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p2_lock.json",
         "r8p1_diagnosis_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_EXACT_INVOCATION_FAILURE_DIAGNOSIS_2026-08-14.md", "r8p1_source_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r8p_source_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md", "r8p1_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8p1.py", "r8p1_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p1.py", "r8p1_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8P1_PREREGISTRATION_2026-08-14.md", "r8p1_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8p1_lock.json",
         "r8_runner_sha256": S / "run_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_sha256": S / "preflight_het_next_l0_ph1_intel_execution_r8.py", "r8_preflight_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8p.py", "r8_physical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r8.py", "r8_prereg_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R8_PREREGISTRATION_2026-08-14.md", "r8_lock_sha256": R / "het_next_l0_ph1_intel_execution_r8_lock.json", "runtime_audit_sha256": R / "HET_NEXT_L0_PH1_INTEL_EXECUTION_R7D1_PSUTIL_FAILURE_AND_R8_RUNTIME_REPAIR_AUDIT_2026-08-14.md", "r7d1_failure_sha256": prior.prior.prior.prior.FAILURE, "common_sha256": S / "het_next_l0_ph1_intel_execution_r6_common.py", "numerical_verifier_sha256": S / "verify_het_next_l0_ph1_intel_execution_r7a.py", "cpu_result_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.json", "cpu_raw_sha256": R / "het_next_l0_ph1_cpu_freeze_r2/cpu_stage_freeze.safetensors"}

def sha_bytes(x: bytes) -> str: return hashlib.sha256(x).hexdigest()
def sha256(p: Path) -> str: return sha_bytes(p.read_bytes())
def canon(x: object) -> bytes: return (json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n").encode()
def same(a: object, b: object) -> bool: return isinstance(a, str) and isinstance(b, str) and a.casefold() == b.casefold()

def state_bits(state: str) -> tuple[bool, bool]:
    if state not in CPU_STATES: raise ValueError("cpu_slice_state")
    return state != "not_started", state == "completed"

def state_valid(row: dict) -> bool:
    try: started, completed = state_bits(row.get("cpu_slice_state"))
    except ValueError: return False
    return row.get("cpu_frozen_slice_read_started") is started and row.get("cpu_frozen_slice_read_completed") is completed

def identity_valid(x: dict, native: list[str], argv: list[str]) -> bool:
    return set(x) == prior.prior.IDENTITY_KEYS and bool(x["native_raw"]) and prior.parse_commandline(x["native_raw"]) == native and x["native_argv"] == native and x["orig_argv"] == native and x["argv"] == argv and same(x["sys_executable"], str(VENV_PYTHON.resolve())) and same(x["sys_prefix"], str(VENV.resolve())) and same(x["base_executable"], str(ALIAS)) and same(x["base_prefix"], str(BASE_PREFIX)) and x["venv_launcher_sha256"] == STATIC["python_sha256"] and x["pyvenv"] == {"sha256": STATIC["pyvenv_sha256"], "home": str(ALIAS.parent), "executable": str(ALIAS), "version": "3.12.10"} and x["base_binary_sha256"] == STATIC["base_binary_sha256"] and x["base_binary_bytes"] == STATIC["base_binary_bytes"] and x["direct_entry"] is True

def live_identity_mutations(row: dict) -> list[str]:
    cases = {"native": lambda x: x["native_argv"].append("x"), "raw": lambda x: x.__setitem__("native_raw", x["native_raw"] + " --extra"), "orig": lambda x: x["orig_argv"].append("x"), "argv": lambda x: x["argv"].append("x"), "venv": lambda x: x.__setitem__("sys_executable", "wrong"), "venv_prefix": lambda x: x.__setitem__("sys_prefix", "wrong"), "base": lambda x: x.__setitem__("base_executable", "wrong"), "base_prefix": lambda x: x.__setitem__("base_prefix", "wrong"), "flags": lambda x: x["orig_argv"].__setitem__(slice(1, 3), ["-B", "-I"]), "trampoline": lambda x: x["orig_argv"].__setitem__(3, "-c"), "script": lambda x: x["argv"].__setitem__(0, "wrong"), "direct": lambda x: x.__setitem__("direct_entry", False)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not identity_valid(candidate, EXPECTED_NATIVE, EXPECTED_ARGV): out.append(name)
    return out

def atomic_create(path: Path, data: bytes, *, link_fn=os.link, unlink_fn=Path.unlink) -> None: prior.atomic_create(path, data, link_fn=link_fn, unlink_fn=unlink_fn)

def failure_row(stage: str, exc: BaseException, state: str) -> dict:
    started, completed = state_bits(state); return {"kind": "ph1_intel_execution_r8p5_failure", "status": "valid_protocol_negative", "stage": stage, "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc()[-32768:], "identity": None, "cpu_slice_state": state, "cpu_frozen_slice_read_started": started, "cpu_frozen_slice_read_completed": completed, "device_opened": False, "compiler_opened": False, "disposition": "bounded_create_new_canonical"}

def independent_failure(stage: str, exc: BaseException, state: str, root: Path, *, create=atomic_create) -> dict:
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

def state_mutations(row: dict) -> list[str]:
    cases = {"unknown_state": lambda x: x.__setitem__("cpu_slice_state", "unknown"), "wrong_started": lambda x: x.__setitem__("cpu_frozen_slice_read_started", not x["cpu_frozen_slice_read_started"]), "wrong_completed": lambda x: x.__setitem__("cpu_frozen_slice_read_completed", not x["cpu_frozen_slice_read_completed"]), "missing_state": lambda x: x.pop("cpu_slice_state"), "extra_state": lambda x: x.__setitem__("cpu_slice_state_extra", True)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if set(candidate) != FAILURE_SCHEMA or not state_valid(candidate): out.append(name)
    return out

def independent_failure_simulation() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "failed"; primary = RuntimeError("primary")
        try: raise primary
        except RuntimeError as exc: first = independent_failure("identity", exc, "not_started", root)
        p1 = first["path"]; frozen = p1.read_bytes(); row1 = json.loads(frozen); second = independent_failure("post", primary, "completed", root); p2 = second["path"]; row2 = json.loads(p2.read_text())
        def precreate(path, data):
            def fail(_a, _b): raise OSError("prelink")
            return atomic_create(path, data, link_fn=fail)
        pre_root = Path(td) / "pre"; secondary = None
        try: independent_failure("identity", primary, "not_started", pre_root, create=precreate)
        except Exception as exc: secondary = f"{type(exc).__name__}:{exc}"
        def postcreate(path, data):
            def fail(_p): raise OSError("postunlink")
            return atomic_create(path, data, unlink_fn=fail)
        post_root = Path(td) / "post"; post = independent_failure("preparation", primary, "started_not_completed", post_root, create=postcreate); pp = post["path"]; row3 = json.loads(pp.read_text())
        states = [row1, row2, row3]; mutations = all(state_mutations(row) == ["unknown_state", "wrong_started", "wrong_completed", "missing_state", "extra_state"] for row in states)
        return {"baseline_schema_cap": set(row1) == FAILURE_SCHEMA and len(frozen) <= 65536 and state_valid(row1), "second_unique_no_overwrite": p1 != p2 and p1.read_bytes() == frozen and len(list(root.rglob("failure.json"))) == 2, "prelink_cleanup": secondary == "OSError:prelink" and (not pre_root.exists() or not list(pre_root.rglob("*"))), "postlink_cleanup_canonical": post["recovered"] is True and pp.is_file() and not list(pp.parent.glob("*.inprogress.*")) and canon(row3) == pp.read_bytes(), "primary_preserved_secondary_recorded": secondary == "OSError:prelink", "cpu_state_provenance_all_three": [r["stage"] for r in states] == ["identity", "post", "preparation"] and [r["cpu_slice_state"] for r in states] == ["not_started", "completed", "started_not_completed"] and all(state_valid(r) for r in states) and mutations}

def lock_valid() -> bool:
    lock = json.loads(LOCK.read_text()); observed = {k: sha256(v) for k, v in CHAIN.items()}; return set(lock) == {"kind", "execution_open", "audit_token", *STATIC, *observed} and lock["kind"] == "ph1_intel_execution_r8p5_lock" and lock["execution_open"] is False and lock["audit_token"] == "PENDING" and all(lock.get(k) == v for k, v in STATIC.items()) and all(lock.get(k) == v for k, v in observed.items()) and observed["r8p4_audit_sha256"] == "a02e2e619c155d36d5c21fb010504e77fbe4526ca9e0cd846918c1156a9d9815"

def bundle_valid() -> bool:
    if not all(p.is_file() for p in (RESULT, MANIFEST, COMMIT)): return False
    rb = RESULT.read_bytes(); mb = MANIFEST.read_bytes(); return json.loads(mb) == {"kind": KIND + "_manifest", "files": [{"name": RESULT.name, "bytes": len(rb), "sha256": sha_bytes(rb)}]} and json.loads(COMMIT.read_text()) == {"kind": KIND + "_commit", "result_sha256": sha_bytes(rb), "manifest_sha256": sha_bytes(mb)}

def topology_valid() -> bool:
    expected = {R / "het_next_l0_ph1_intel_execution_r8_lock.json", R / "het_next_l0_ph1_intel_execution_r8p1_lock.json", R / "het_next_l0_ph1_intel_execution_r8p2_lock.json", R / "het_next_l0_ph1_intel_execution_r8p3_lock.json", R / "het_next_l0_ph1_intel_execution_r8p4_lock.json", LOCK, RESULT, MANIFEST, COMMIT}; family = {p for p in R.iterdir() if p.name.startswith("het_next_l0_ph1_intel_execution_r8")}; return family == expected and not FAILED.exists() and not QUARANTINE.exists() and not OUTPUT.exists() and not list(R.glob("het_next_l0_ph1_intel_execution_r8*.inprogress.*"))

def stored_topology_valid(x: dict) -> bool:
    locks = sorted(str(p) for p in (R / "het_next_l0_ph1_intel_execution_r8_lock.json", R / "het_next_l0_ph1_intel_execution_r8p1_lock.json", R / "het_next_l0_ph1_intel_execution_r8p2_lock.json", R / "het_next_l0_ph1_intel_execution_r8p3_lock.json", R / "het_next_l0_ph1_intel_execution_r8p4_lock.json", LOCK)); return set(x) == {"absent", "temps", "family"} and len(x["absent"]) == 36 and all(v is False for v in x["absent"].values()) and x["temps"] == [] and x["family"] == locks

def static_boundary() -> bool:
    forbidden = {"Popen", "run", "check_call", "check_output", "CDLL", "LoadLibrary", "clBuildProgram", "clCreateContext", "cuLaunchKernel", "nvrtcCompileProgram", "from_pretrained"}
    for path in (PREFLIGHT, SCRIPT):
        tree = ast.parse(path.read_text(encoding="utf-8")); calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in ast.walk(tree) if isinstance(n, ast.Call)}
        if calls & forbidden: return False
    return prior.static_boundary()

def result_valid(row: dict, preparation: dict, wheels: dict) -> bool:
    checks = row.get("checks", {}); tx = row.get("transaction_simulation"); failure = row.get("failure_simulation")
    return set(row) == {"kind", "ack", "identity", "pre_run_topology", "checks", "pass", "passed", "total", "runtime", "wheel_records", "preparation", "preparation_digest", "transaction_simulation", "failure_simulation", "identity_mutations_rejected", "runtime_mutations_rejected", "cpu_slice_state", "cpu_frozen_slice_read_started", "cpu_frozen_slice_read_completed", "model_forward", "compiler_opened", "opencl_opened", "device_opened"} and row["kind"] == KIND and row["ack"] == ACK and identity_valid(row["identity"], PREFLIGHT_NATIVE, PREFLIGHT_ARGV) and stored_topology_valid(row["pre_run_topology"]) and set(checks) == CHECK_NAMES and all(v is True for v in checks.values()) and row["pass"] is True and row["passed"] == row["total"] == len(CHECK_NAMES) and BASE.runtime_static_valid(row["runtime"]) and row["runtime"]["available"] >= 16 * 2**30 and row["wheel_records"] == wheels and row["preparation"] == preparation and row["preparation_digest"] == BASE.PREPARATION_DIGEST == sha_bytes(canon(preparation)) and isinstance(tx, dict) and set(tx) == TX_KEYS and all(v is True for v in tx.values()) and isinstance(failure, dict) and set(failure) == FAILURE_KEYS and all(v is True for v in failure.values()) and row["identity_mutations_rejected"] == ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "direct"] and row["runtime_mutations_rejected"] == ["python_path", "python_hash", "isolation", "bytecode", "pyvenv", "psutil_native", "psutil_record", "numpy_version", "numpy_record", "ram"] and row["cpu_slice_state"] == "completed" and row["cpu_frozen_slice_read_started"] is row["cpu_frozen_slice_read_completed"] is True and row["model_forward"] is row["compiler_opened"] is row["opencl_opened"] is row["device_opened"] is False

def mutations(row: dict, preparation: dict, wheels: dict) -> list[str]:
    cases = {"empty_failure": lambda x: x.__setitem__("failure_simulation", {}), "missing_failure": lambda x: x["failure_simulation"].pop("cpu_state_provenance_all_three"), "extra_failure": lambda x: x["failure_simulation"].__setitem__("extra", True), "false_failure": lambda x: x["failure_simulation"].__setitem__("cpu_state_provenance_all_three", False), "wrong_state": lambda x: x.__setitem__("cpu_slice_state", "started_not_completed"), "wrong_started": lambda x: x.__setitem__("cpu_frozen_slice_read_started", False), "wrong_completed": lambda x: x.__setitem__("cpu_frozen_slice_read_completed", False), "device": lambda x: x.__setitem__("device_opened", True)}; out = []
    for name, fn in cases.items():
        candidate = copy.deepcopy(row); fn(candidate)
        if not result_valid(candidate, preparation, wheels): out.append(name)
    return out

def main() -> int:
    ident = prior.live_identity()
    if not identity_valid(ident, EXPECTED_NATIVE, EXPECTED_ARGV): return 3
    live_rejected = live_identity_mutations(ident); raw = RESULT.read_bytes(); row = json.loads(raw); preparation = BASE.independent_preparation(); wheels = {"psutil": BASE.wheel_record(BASE.FILES["psutil_record"][0]), "numpy": BASE.wheel_record(BASE.FILES["numpy_record"][0])}; failure = independent_failure_simulation(); rejected = mutations(row, preparation, wheels)
    checks = {"live_identity": True, "live_identity_mutations": live_rejected == ["native", "raw", "orig", "argv", "venv", "venv_prefix", "base", "base_prefix", "flags", "trampoline", "script", "direct"], "lock": lock_valid(), "bundle": bundle_valid(), "topology": topology_valid(), "static_boundary": static_boundary(), "r7d1_failure": prior.prior.prior.prior.failure_bundle_valid(prior.prior.prior.prior.FAILURE_ROOT), "runtime": BASE.runtime_static_valid(BASE.collect_runtime()), "wheel_records": wheels["psutil"]["hashed_files_verified"] == 17 and wheels["numpy"]["hashed_files_verified"] == 899, "preparation": sha_bytes(canon(preparation)) == BASE.PREPARATION_DIGEST, "result": result_valid(row, preparation, wheels), "failure_simulation": set(failure) == FAILURE_KEYS and all(failure.values()), "state_and_result_mutations": rejected == ["empty_failure", "missing_failure", "extra_failure", "false_failure", "wrong_state", "wrong_started", "wrong_completed", "device"]}
    output = {"kind": "ph1_intel_execution_r8p5_independent_verification", "checks": checks, "pass": all(checks.values()), "passed": sum(v is True for v in checks.values()), "total": len(checks), "result_sha256": sha_bytes(raw), "manifest_sha256": sha256(MANIFEST), "commit_sha256": sha256(COMMIT), "independent_failure_simulation": failure, "cpu_slice_state": "completed", "cpu_frozen_slice_read_started": True, "cpu_frozen_slice_read_completed": True, "model_forward": False, "compiler_opened": False, "opencl_opened": False, "device_opened": False}
    atomic_create(OUTPUT, canon(output)); print(json.dumps(output, indent=2)); return 0 if output["pass"] else 3

if __name__ == "__main__": raise SystemExit(main())
