#!/usr/bin/env python3
"""NC19I0 import-inert, stdlib-only compile contract.

Runner and static preflight call these exact functions.  This module performs no
I/O, environment mutation, DLL load, compilation, or device action at import.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

REVISION = "NC19I0"
NVRTC_OPS = (
    "nvrtcVersion", "nvrtcCreateProgram", "nvrtcCompileProgram",
    "nvrtcGetProgramLogSize", "nvrtcGetProgramLog", "nvrtcGetPTXSize",
    "nvrtcGetPTX", "nvrtcGetCUBINSize", "nvrtcGetCUBIN",
    "nvrtcDestroyProgram",
)
OPTIONS = (
    "--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true",
    "--ftz=false", "--gpu-architecture=sm_120",
    "--device-as-default-execution-space",
)
PROGRAM_NAME = b"het_next_l0_ph1_nvidia_nc19i0.cu"
CAPS = {
    "fixture_manifest": 8 * 2**20, "json_each": 4 * 2**20,
    "source": 65536, "log": 4 * 2**20, "ptx": 16 * 2**20,
    "cubin": 32 * 2**20, "bundle": 64 * 2**20, "prefix": 4096,
}
ENV_KEYS = (
    "CUDA_CACHE_DISABLE", "CUDA_CACHE_MAXSIZE", "CUDA_CACHE_PATH",
    "TMP", "TEMP", "NVRTC_CACHE_PATH",
)
ENV_SUBDIRS = {
    "CUDA_CACHE_PATH": "cuda_cache", "TMP": "tmp", "TEMP": "temp",
    "NVRTC_CACHE_PATH": "nvrtc_cache",
}
TERMINAL_STATES = {
    "compile_positive", "compile_valid_negative", "incidental_failure",
    "postcommit_incident", "verifier_protocol_negative", "transaction_debris",
}
INPROGRESS = re.compile(r"^[a-z0-9_]+\.inprogress\.[0-9]+\.[0-9a-f]{16}$")


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_digest(function) -> str:
    code = function.__code__
    payload = repr((code.co_code, code.co_consts, code.co_names,
                    code.co_varnames)).encode("utf-8")
    return sha256(payload)


def paths_for_revision(descriptor: dict) -> tuple[str, ...]:
    """Return the exact normalized, unique descriptor path universe."""
    required = {"revision", "prefix", "roots", "patterns", "caps",
                "required_present_by_stage", "expected_absent_by_stage"}
    if set(descriptor) != required or not isinstance(descriptor["revision"], str):
        raise ValueError("descriptor_schema")
    paths = []
    for root in descriptor["roots"]:
        if not isinstance(root, dict) or not isinstance(root.get("path"), str):
            raise ValueError("root_schema")
        path = root["path"].replace("\\", "/")
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError("root_path")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("root_order_or_duplicate")
    return tuple(paths)


def _schema_ok(rule: dict, observed: dict | None) -> bool:
    if not rule:
        return observed in (None, {})
    if not isinstance(observed, dict) or set(observed) != set(rule):
        return False
    for key, spec in rule.items():
        value = observed[key]
        type_name = spec.get("type")
        if type_name == "bool" and type(value) is not bool:
            return False
        if type_name == "int" and (type(value) is not int or value < spec.get("minimum", -2**63)):
            return False
        if type_name == "str" and not isinstance(value, str):
            return False
        if type_name == "array" and not isinstance(value, list):
            return False
        if type_name == "object" and not isinstance(value, dict):
            return False
        if "exact" in spec and value != spec["exact"]:
            return False
        if "enum" in spec and value not in spec["enum"]:
            return False
    return True


def classify_topology(descriptor: dict, observed_entries: list[dict], stage: str,
                      terminal_id: str | None = None) -> dict:
    """Pure semantic topology classifier used unchanged by runner/preflight."""
    universe = set(paths_for_revision(descriptor))
    if stage not in descriptor["required_present_by_stage"]:
        return {"classification": "invalid_stage", "valid": False, "dispositions": []}
    expected = descriptor["expected_absent_by_stage"][stage]
    if stage == "runtime":
        absent = set(expected["no_terminal"] if terminal_id is None
                     else expected["terminal_choices"].get(terminal_id, []))
    else:
        absent = set(expected["paths"])
    required = set(descriptor["required_present_by_stage"][stage])
    if required & absent:
        return {"classification": "descriptor_intersection", "valid": False, "dispositions": []}
    seen = []
    root_by_path = {row["path"]: row for row in descriptor["roots"]}
    for row in observed_entries:
        if not isinstance(row, dict) or set(row) != {
            "path", "node_type", "size", "sha256_or_null", "children",
            "schema_key_values", "parse_status", "content_base64_or_null",
        }:
            return {"classification": "observed_schema", "valid": False, "dispositions": []}
        path = row["path"]
        if path in seen or path not in universe or path in absent:
            return {"classification": "collision_or_absent", "valid": False, "dispositions": []}
        seen.append(path)
        spec = root_by_path[path]
        if row["node_type"] != spec["allowed_node_type"] or row["size"] > spec["cap_bytes"]:
            return {"classification": "node_or_cap", "valid": False, "dispositions": []}
        if spec["allowed_node_type"] == "file" and spec.get("schema_rule"):
            if row["parse_status"] != "valid_json" or not _schema_ok(spec["schema_rule"], row["schema_key_values"]):
                return {"classification": "invalid_schema", "valid": False, "dispositions": []}
    if set(seen) != required:
        return {"classification": "required_set", "valid": False, "dispositions": []}
    return {"classification": f"{stage}_valid", "valid": True, "dispositions": []}


def capture_environment(mapping) -> dict:
    return {key: {"present": key in mapping, "value": mapping.get(key)} for key in ENV_KEYS}


def apply_private_environment(mapping, private_root: Path) -> dict:
    private_root = Path(private_root).resolve()
    replacements = {
        "CUDA_CACHE_DISABLE": "1", "CUDA_CACHE_MAXSIZE": "0",
        **{key: str((private_root / subdir).resolve()) for key, subdir in ENV_SUBDIRS.items()},
    }
    if len(set(replacements[key] for key in ENV_SUBDIRS)) != 4:
        raise ValueError("cache_alias")
    for key in ENV_KEYS:
        mapping[key] = replacements[key]
    return replacements


def restore_environment(mapping, captured: dict) -> list[dict]:
    rows = []
    for key in reversed(ENV_KEYS):
        try:
            if captured[key]["present"]:
                mapping[key] = captured[key]["value"]
            else:
                mapping.pop(key, None)
            rows.append({"key": key, "attempted": True, "code": 0})
        except Exception as exc:
            rows.append({"key": key, "attempted": True, "code": "exception",
                         "error": f"{type(exc).__name__}:{exc}"})
    return rows


def cache_entries(private_root: Path) -> list[dict]:
    root = Path(private_root).resolve()
    rows = []
    for path in sorted([root, *root.rglob("*")], key=lambda p: p.relative_to(root).as_posix() if p != root else "."):
        rel = "." if path == root else path.relative_to(root).as_posix()
        if path.is_symlink() or ".." in Path(rel).parts:
            raise ValueError("cache_reparse_or_traversal")
        stat = path.stat()
        if path.is_dir():
            rows.append({"path": rel, "type": "dir", "size": 0,
                         "mtime_ns": stat.st_mtime_ns, "sha256": None})
        elif path.is_file():
            raw = path.read_bytes()
            rows.append({"path": rel, "type": "file", "size": len(raw),
                         "mtime_ns": stat.st_mtime_ns, "sha256": sha256(raw)})
        else:
            raise ValueError("cache_node")
    return rows


def cache_tree_digest(entries: list[dict]) -> str:
    raw = b"".join((f"{x['path']}\0{x['type']}\0{x['size']}\0{x['mtime_ns']}\0{x['sha256'] or ''}\n").encode()
                   for x in entries)
    return sha256(raw)


def cache_history_digest(history: list[dict]) -> str:
    return sha256(json.dumps(history, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode())


def not_attempted_ledger() -> list[dict]:
    return [{"sequence": i, "op": op, "attempted": False,
             "code": "not_attempted", "handle_before": 0, "handle_after": 0}
            for i, op in enumerate(NVRTC_OPS)]


def compile_with_adapter(adapter, source: bytes, snapshot=None) -> dict:
    """Exact ten-operation state machine; adapter is physical or fake."""
    ledger, artifacts, primary, secondary = [], {}, None, []
    handle = 0
    for index, op in enumerate(NVRTC_OPS[:-1]):
        before = handle
        collect_compile_log = (primary is not None and
                               str(primary.get("value", "")).startswith("nvrtcCompileProgram:") and
                               op in {"nvrtcGetProgramLogSize", "nvrtcGetProgramLog"})
        if primary is not None and not collect_compile_log:
            ledger.append({"sequence": index, "op": op, "attempted": False,
                           "code": "not_attempted", "handle_before": before,
                           "handle_after": handle})
            continue
        try:
            reply = adapter.call(op, handle=handle, source=source,
                                 program_name=PROGRAM_NAME, options=OPTIONS)
            code = int(reply.get("code", 0))
            if op == "nvrtcCreateProgram":
                handle = int(reply.get("handle", 0))
            ledger.append({"sequence": index, "op": op, "attempted": True,
                           "code": code, "handle_before": before,
                           "handle_after": handle})
            for name in ("log", "ptx", "cubin"):
                if name in reply:
                    artifacts[name] = reply[name]
            if snapshot is not None:
                snapshot(op)
            if code != 0 or (op == "nvrtcCreateProgram" and not handle):
                if primary is None:
                    primary = {"state": "failure", "value": f"{op}:{code}"}
                else:
                    secondary.append(f"{op}:{code}")
        except Exception as exc:
            ledger.append({"sequence": index, "op": op, "attempted": True,
                           "code": "ctypes_exception", "handle_before": before,
                           "handle_after": handle, "error": f"{type(exc).__name__}:{exc}"})
            if primary is None:
                primary = {"state": "failure", "value": f"{op}:ctypes_exception"}
            else:
                secondary.append(f"{op}:ctypes_exception")
    destroy_index = len(NVRTC_OPS) - 1
    if handle:
        try:
            reply = adapter.call("nvrtcDestroyProgram", handle=handle, source=source,
                                 program_name=PROGRAM_NAME, options=OPTIONS)
            code = int(reply.get("code", 0))
            ledger.append({"sequence": destroy_index, "op": "nvrtcDestroyProgram",
                           "attempted": True, "code": code,
                           "handle_before": handle, "handle_after": int(reply.get("handle", 0))})
            if snapshot is not None:
                snapshot("nvrtcDestroyProgram")
            if code != 0:
                secondary.append(f"nvrtcDestroyProgram:{code}")
        except Exception as exc:
            ledger.append({"sequence": destroy_index, "op": "nvrtcDestroyProgram",
                           "attempted": True, "code": "ctypes_exception",
                           "handle_before": handle, "handle_after": handle,
                           "error": f"{type(exc).__name__}:{exc}"})
            secondary.append("nvrtcDestroyProgram:ctypes_exception")
    else:
        ledger.append({"sequence": destroy_index, "op": "nvrtcDestroyProgram",
                       "attempted": False, "code": "not_attempted",
                       "handle_before": 0, "handle_after": 0})
    if primary is None:
        primary = {"state": "none", "value": None}
    return {"ledger": ledger, "artifacts": artifacts, "primary": primary,
            "secondary": secondary}


def validate_compile_evidence(evidence: dict) -> dict:
    checks = {}
    ledger = evidence.get("ledger", [])
    checks["ledger_shape"] = len(ledger) == 10 and [x.get("sequence") for x in ledger] == list(range(10)) and [x.get("op") for x in ledger] == list(NVRTC_OPS)
    attempted = [x for x in ledger if x.get("attempted")]
    handles = {x.get("handle_after") for x in attempted if x.get("handle_after")}
    checks["single_program"] = len(handles) <= 1
    checks["destroy_suffix"] = bool(ledger) and ledger[-1]["op"] == "nvrtcDestroyProgram"
    artifacts = evidence.get("artifacts", {})
    checks["artifact_keys"] = set(artifacts) >= {"log", "ptx", "cubin"} or evidence.get("primary", {}).get("state") == "failure"
    if set(artifacts) >= {"log", "ptx", "cubin"}:
        log, ptx, cubin = artifacts["log"], artifacts["ptx"], artifacts["cubin"]
        checks["log_canonical"] = 1 <= len(log) <= CAPS["log"] and log.endswith(b"\0") and b"\0" not in log[:-1]
        checks["ptx_canonical"] = 1 < len(ptx) <= CAPS["ptx"] and ptx.endswith(b"\0") and b"\0" not in ptx[:-1]
        logical = ptx[:-1].decode("utf-8", "strict") if checks["ptx_canonical"] else ""
        checks["ptx_contract"] = all(token in logical for token in (".version", ".target sm_120", ".address_size 64")) and logical.count(".entry q5_linear") == 1 and logical.count(".entry bf16_lut_activation") == 1
        checks["cubin_contract"] = 1 < len(cubin) <= CAPS["cubin"] and cubin.startswith(b"\x7fELF") and cubin.count(b"q5_linear") >= 1 and cubin.count(b"bf16_lut_activation") >= 1
    return checks


def atomic_create(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".inprogress.{os.getpid()}.{uuid.uuid4().hex[:16]}")
    if path.exists():
        raise FileExistsError(path)
    try:
        with temp.open("xb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        flush_directory(path.parent)
        temp.unlink()
        flush_directory(path.parent)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def flush_directory(path: Path) -> None:
    """Flush a Windows directory handle; local ctypes import keeps import inert."""
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return
    import ctypes as c
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [c.c_wchar_p,c.c_uint32,c.c_uint32,c.c_void_p,c.c_uint32,c.c_uint32,c.c_void_p]
    kernel.CreateFileW.restype = c.c_void_p
    kernel.FlushFileBuffers.argtypes = [c.c_void_p]; kernel.FlushFileBuffers.restype = c.c_int32
    kernel.CloseHandle.argtypes = [c.c_void_p]; kernel.CloseHandle.restype = c.c_int32
    handle = kernel.CreateFileW(str(Path(path).resolve()), 0x80000000,
                                0x1|0x2|0x4, None, 3, 0x02000000, None)
    if handle in (None, c.c_void_p(-1).value):
        raise OSError(c.get_last_error(), "CreateFileW(directory)")
    try:
        if not kernel.FlushFileBuffers(handle):
            raise OSError(c.get_last_error(), "FlushFileBuffers(directory)")
    finally:
        if not kernel.CloseHandle(handle):
            raise OSError(c.get_last_error(), "CloseHandle(directory)")


def build_bundle(result: dict, files: dict[str, bytes], kind: str) -> dict[str, bytes]:
    if set(files) & {"result.json", "manifest.json", "commit.json"}:
        raise ValueError("reserved_name")
    payload = {"result.json": canonical(result), **files}
    manifest = {"kind": kind + "_manifest", "revision": REVISION,
                "files": [{"name": name, "bytes": len(raw), "sha256": sha256(raw)}
                          for name, raw in sorted(payload.items())]}
    manifest_raw = canonical(manifest)
    commit = {"kind": kind + "_commit", "revision": REVISION, "state": "complete",
              "result_sha256": sha256(payload["result.json"]),
              "manifest_sha256": sha256(manifest_raw)}
    bundle = {**payload, "manifest.json": manifest_raw, "commit.json": canonical(commit)}
    if sum(map(len, bundle.values())) > CAPS["bundle"]:
        raise ValueError("bundle_cap")
    return bundle


def verify_bundle_bytes(bundle: dict[str, bytes], kind: str) -> bool:
    try:
        if sum(map(len, bundle.values())) > CAPS["bundle"]:
            return False
        manifest = json.loads(bundle["manifest.json"])
        commit = json.loads(bundle["commit.json"])
        payload = {k: v for k, v in bundle.items() if k not in {"manifest.json", "commit.json"}}
        expected = {"kind": kind + "_manifest", "revision": REVISION,
                    "files": [{"name": n, "bytes": len(v), "sha256": sha256(v)}
                              for n, v in sorted(payload.items())]}
        return manifest == expected and commit == {
            "kind": kind + "_commit", "revision": REVISION, "state": "complete",
            "result_sha256": sha256(payload["result.json"]),
            "manifest_sha256": sha256(bundle["manifest.json"]),
        }
    except Exception:
        return False


def publish_transaction(output: Path, bundle: dict[str, bytes], kind: str, verifier) -> None:
    output = Path(output)
    stage = output.with_name(output.name + f".inprogress.{os.getpid()}.{uuid.uuid4().hex[:16]}")
    if output.exists() or stage.exists() or not verify_bundle_bytes(bundle, kind):
        raise FileExistsError("nonclean_or_invalid")
    stage.mkdir(parents=False)
    try:
        order = [x for x in sorted(bundle) if x not in {"manifest.json", "commit.json"}] + ["manifest.json", "commit.json"]
        for name in order:
            atomic_create(stage / name, bundle[name])
        flush_directory(stage)
        if not verifier(stage):
            raise RuntimeError("precommit_verifier")
        os.rename(stage, output)
        flush_directory(output.parent)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def recover_inprogress(parent: Path, prefix: str, quarantine: Path) -> dict:
    parent, quarantine = Path(parent), Path(quarantine)
    debris = [p for p in parent.iterdir() if p.name.startswith(prefix + ".inprogress.")]
    if len(debris) != 1 or not INPROGRESS.fullmatch(debris[0].name):
        return {"classification": "not_single_debris", "recovered": False}
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / (debris[0].name + ".quarantined")
    if target.exists():
        return {"classification": "collision", "recovered": False}
    debris[0].rename(target)
    return {"classification": "transaction_debris", "recovered": True,
            "attempt_consumed": False, "next_invocation_allowed": True,
            "source": str(debris[0]), "target": str(target)}


def write_incidental_failure(root: Path, row: dict) -> Path:
    raw = canonical(row)
    if len(raw) > CAPS["json_each"]:
        prefix = raw[:CAPS["prefix"]]
        row = {"kind": "het_next_l0_ph1_nvidia_nc19i0_bounded_failure",
               "revision": REVISION, "status": "incidental_failure",
               "full_bytes": len(raw), "full_sha256": sha256(raw),
               "prefix_offset": 0, "prefix_length": len(prefix),
               "prefix_base64": base64.b64encode(prefix).decode(),
               "prefix_sha256": sha256(prefix)}
        raw = canonical(row)
    attempt = Path(root) / f"attempt.{time.time_ns()}.{uuid.uuid4().hex[:16]}"
    attempt.mkdir(parents=True, exist_ok=False)
    atomic_create(attempt / "failure.json", raw)
    return attempt


def adjudicate_terminal(result: dict, cleanup_ok: bool) -> dict:
    status = result.get("status")
    valid = status in {"compile_positive", "compile_valid_negative"} and cleanup_ok
    return {"terminal": status, "terminal_valid": valid,
            "next_invocation_allowed": status == "transaction_debris",
            "attempt_consumed": status != "transaction_debris"}


EXPORTS = (
    paths_for_revision, classify_topology, capture_environment,
    apply_private_environment, restore_environment, compile_with_adapter,
    validate_compile_evidence, recover_inprogress, publish_transaction,
    write_incidental_failure, adjudicate_terminal,
)
