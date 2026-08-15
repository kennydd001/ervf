#!/usr/bin/env python3
"""NC19I1 import-inert, stdlib-only compile contract.

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

REVISION = "NC19I1"
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
PROGRAM_NAME = b"het_next_l0_ph1_nvidia_nc19i1.cu"
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


OBSERVED_FIELDS = {
    "path", "node_type", "size", "sha256_or_null", "children",
    "schema_key_values", "parse_status", "content_base64_or_null",
}


def _schema_ok(rule: dict, observed: dict | None) -> bool:
    if not rule:
        return observed in (None, {})
    if not isinstance(observed, dict) or set(observed) != set(rule):
        return False
    for key, spec in rule.items():
        value = observed[key]
        type_name = spec.get("type")
        if type_name == "boolean" and type(value) is not bool:
            return False
        if type_name == "integer" and type(value) is not int:
            return False
        if type_name == "string" and not isinstance(value, str):
            return False
        if type_name == "array" and not isinstance(value, list):
            return False
        if type_name == "object" and not isinstance(value, dict):
            return False
        constraint = spec.get("constraint")
        operand = spec.get("value")
        if constraint == "exact" and value != operand:
            return False
        if constraint == "enum" and value not in operand:
            return False
        if constraint == "predicate":
            predicates = {
                "nonempty_object": lambda x: isinstance(x, dict) and bool(x),
                "nonempty_array": lambda x: isinstance(x, list) and bool(x),
                "lowerhex64": lambda x: isinstance(x, str) and bool(re.fullmatch(r"[0-9a-f]{64}", x)),
                "integer_gt_zero": lambda x: type(x) is int and x > 0,
                "nonempty_string": lambda x: isinstance(x, str) and bool(x),
                "nonempty_string_keys_boolean_values_all_true": lambda x: isinstance(x, dict) and bool(x) and all(isinstance(k,str) and type(v) is bool and v for k,v in x.items()),
                "nonempty_unique_normalized_relative_strings": lambda x: isinstance(x,list) and bool(x) and len(x)==len(set(x)) and x==sorted(x) and all(isinstance(v,str) and not v.startswith(("/","\\")) and ".." not in Path(v).parts for v in x),
            }
            if operand not in predicates or not predicates[operand](value):
                return False
    return True


def _decode_observed_file(row: dict) -> tuple[bytes | None, dict | None, str | None]:
    encoded = row.get("content_base64_or_null")
    if encoded is None:
        return None, None, None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        return None, None, "invalid_base64"
    if row.get("size") != len(raw) or row.get("sha256_or_null") != sha256(raw):
        return raw, None, "identity_mismatch"
    if row.get("parse_status") == "valid_json":
        try:
            parsed = json.loads(raw.decode("utf-8-sig"))
        except Exception:
            return raw, None, "invalid_json"
        if parsed != row.get("schema_key_values"):
            return raw, parsed, "raw_schema_mismatch"
        return raw, parsed, None
    if row.get("schema_key_values") is not None:
        return raw, None, "unexpected_schema_projection"
    return raw, None, None


def classify_topology(descriptor: dict, observed_entries: list[dict], stage: str,
                      terminal_id: str | None = None) -> dict:
    """Pure semantic topology classifier used unchanged by runner/preflight."""
    universe = set(paths_for_revision(descriptor))
    if stage not in descriptor["required_present_by_stage"]:
        return {"classification": "invalid_stage", "valid": False, "dispositions": []}
    expected = descriptor["expected_absent_by_stage"][stage]
    if stage == "runtime":
        choices = expected["terminal_choices"]
        if terminal_id is not None and terminal_id not in choices:
            return {"classification": "invalid_terminal", "valid": False,
                    "terminal": False, "recoverable": False,
                    "disposition": "reject", "dispositions": []}
        absent = set(expected["no_terminal"] if terminal_id is None else choices[terminal_id])
    else:
        absent = set(expected["paths"])
    required = set(descriptor["required_present_by_stage"][stage])
    if required & absent:
        return {"classification": "descriptor_intersection", "valid": False, "dispositions": []}
    seen = []
    root_by_path = {row["path"]: row for row in descriptor["roots"]}
    for row in observed_entries:
        if not isinstance(row, dict) or set(row) != OBSERVED_FIELDS:
            return {"classification": "observed_schema", "valid": False, "dispositions": []}
        path = row["path"]
        if path in seen:
            return {"classification": "collision", "valid": False, "terminal": False,
                    "recoverable": False, "disposition": "reject", "dispositions": []}
        if path not in universe:
            matches=[pattern for pattern in descriptor["patterns"] if re.fullmatch(pattern["regex"],path)]
            if len(matches)==1 and len(observed_entries)==1 and row["node_type"]=="directory" and row["size"]<=matches[0]["cap_bytes"]:
                return {"classification":"recoverable_debris","valid":True,"terminal":False,
                        "recoverable":True,"disposition":matches[0]["disposition"],"dispositions":[matches[0]["disposition"]]}
            if path.startswith(("/","\\")) or ".." in Path(path).parts or len(matches)>1:
                return {"classification":"invalid_path","valid":False,"terminal":False,
                        "recoverable":False,"disposition":"reject","dispositions":[]}
            return {"classification": "orphan", "valid": False, "terminal": False,
                    "recoverable": False, "disposition": "reject", "dispositions": []}
        if path in absent:
            return {"classification": "stage_policy_violation", "valid": False, "terminal": False,
                    "recoverable": False, "disposition": "reject", "dispositions": []}
        seen.append(path)
        spec = root_by_path[path]
        if row["node_type"] != spec["allowed_node_type"]:
            return {"classification": "invalid_schema", "valid": False, "terminal": False,
                    "recoverable": False, "disposition": "reject", "dispositions": []}
        if type(row["size"]) is not int or row["size"] < 0 or row["size"] > spec["cap_bytes"]:
            return {"classification": "oversize", "valid": False, "terminal": False,
                    "recoverable": False, "disposition": "reject", "dispositions": []}
        raw, parsed, raw_error = _decode_observed_file(row)
        if raw_error:
            return {"classification": "invalid_schema", "valid": False, "terminal": False,
                    "recoverable": False, "disposition": "reject", "dispositions": []}
        if spec["allowed_node_type"] == "file" and spec["required_schema_spec"]:
            if row["parse_status"] != "valid_json" or not _schema_ok(spec["required_schema_spec"], parsed):
                return {"classification": "invalid_schema", "valid": False, "terminal": False,
                        "recoverable": False, "disposition": "reject", "dispositions": []}
        policy = spec["identity_policy"]
        if policy == "bound_size_sha256":
            if raw is None or row["size"] != spec["expected_bytes"] or row["sha256_or_null"] != spec["expected_sha256"]:
                return {"classification": "invalid_source_lock", "valid": False,
                        "terminal": False, "recoverable": False,
                        "disposition": "reject", "dispositions": []}
        if path.endswith("/het_next_l0_ph1_nvidia_nc19_source_lock.json"):
            expected_absent=descriptor["expected_absent_by_stage"]["implementation_freeze"]["paths"]
            required_freeze=descriptor["required_present_by_stage"]["implementation_freeze"]
            observed_absent=parsed.get("expected_absent") if isinstance(parsed,dict) else None
            if (observed_absent!=expected_absent or len(observed_absent or [])!=100
                    or set(observed_absent or [])&set(required_freeze) or path in set(observed_absent or [])):
                return {"classification":"invalid_source_lock_absent_set","valid":False,
                        "terminal":False,"recoverable":False,"disposition":"reject","dispositions":[]}
        if path.endswith("_source_lock.json") and isinstance(parsed,dict):
            entries=parsed.get("bindings",{}).get("source_identity_entries")
            expected_entries=[{"path":item["path"],"bytes":item["expected_bytes"],"sha256":item["expected_sha256"]}
                              for item in descriptor["roots"] if item["identity_policy"]=="bound_size_sha256"]
            if (not isinstance(entries,list) or entries!=expected_entries
                    or len(entries)!=len({item.get("path") for item in entries})
                    or any(item.get("path")==path for item in entries)):
                return {"classification":"invalid_source_lock","valid":False,"terminal":False,
                        "recoverable":False,"disposition":"reject","dispositions":[]}
        if row["node_type"] == "directory":
            if row["content_base64_or_null"] is not None or row["schema_key_values"] is not None:
                return {"classification": "invalid_schema", "valid": False,
                        "terminal": False, "recoverable": False,
                        "disposition": "reject", "dispositions": []}
    required_seen = set(required)
    selected = set()
    if stage == "runtime" and terminal_id is not None:
        selected = set(expected["no_terminal"]) - absent
        if len(selected) != 1:
            return {"classification": "invalid_terminal", "valid": False,
                    "terminal": False, "recoverable": False,
                    "disposition": "reject", "dispositions": []}
        required_seen |= selected
    if set(seen) != required_seen:
        return {"classification": "missing_required_present", "valid": False,
                "terminal": False, "recoverable": False,
                "disposition": "reject", "dispositions": []}
    if stage == "design":
        classification, terminal, disposition = "fresh", False, "proceed"
    elif stage == "implementation_freeze":
        classification, terminal, disposition = "implementation_freeze_valid", False, "freeze"
    else:
        classification = root_by_path[next(iter(selected))]["disposition"] if selected else "fresh"
        terminal = bool(selected)
        disposition = root_by_path[next(iter(selected))]["disposition"] if selected else "proceed"
    return {"classification": classification, "valid": True, "terminal": terminal,
            "recoverable": disposition == "quarantine_then_retry",
            "disposition": disposition, "dispositions": []}


def evaluate_fixture_case(case: dict, fixture: dict) -> dict:
    """Evaluate a frozen case with the production classifier, never its expected row."""
    descriptors={row["revision"]:row for row in fixture["descriptors"]}
    rows = case["observed_entries"]
    if case["observed_total_bytes"] != sum(row["size"] for row in rows):
        return {"classification": "tree_bytes", "valid": False, "terminal": False,
                "recoverable": False, "disposition": "reject", "dispositions": []}
    if case["observed_tree_digest"] != sha256(json.dumps(rows,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")):
        return {"classification": "tree_digest", "valid": False, "terminal": False,
                "recoverable": False, "disposition": "reject", "dispositions": []}
    descriptor = descriptors[case["descriptor"]]
    if case["mutation"] == "descriptor_exact":
        revision=case["name"].split("_",1)[0].upper()
        expected=next(row for row in fixture["historical_lock_expectations"] if row["revision"]==revision)
        actual=descriptors[revision]
        paths=list(paths_for_revision(actual)); digest=sha256(json.dumps(paths,separators=(",",":"),ensure_ascii=False).encode("utf-8"))
        valid=len(paths)==expected["path_count"] and digest==expected["pathset_sha256"]
        return {"classification":"historical_lockset_equal" if valid else "invalid_descriptor",
                "valid":valid,"terminal":False,"recoverable":False,
                "disposition":"proven" if valid else "reject","dispositions":[]}
    if case["mutation"].startswith("descriptor_"):
        import copy
        revision=case["name"].split("_",1)[0].upper(); mutated=copy.deepcopy(descriptors[revision])
        operand=case["mutation_operand"]; pointer=operand["json_pointer"].split("/")[3:]
        target=mutated
        for component in pointer[:-1]: target=target[int(component)] if isinstance(target,list) else target[component]
        leaf=pointer[-1]
        try:
            if operand["operation"]=="remove": target.pop(int(leaf) if isinstance(target,list) else leaf)
            elif operand["operation"]=="add": target.append(operand["new"])
            else: target[int(leaf) if isinstance(target,list) else leaf]=operand["new"]
            paths_for_revision(mutated)
            invalid=False
        except (KeyError,IndexError,TypeError,ValueError): invalid=True
        if not invalid:
            invalid=mutated!=descriptors[revision]
        return {"classification":"invalid_descriptor" if invalid else "descriptor_mutation_ineffective",
                "valid":False,"terminal":False,"recoverable":False,"disposition":"reject","dispositions":[]}
    terminal_id = None
    if case["evaluation_stage"] == "runtime":
        required = set(descriptor["required_present_by_stage"]["runtime"])
        extras = {row["path"] for row in rows} - required
        if len(extras) == 1:
            no_terminal = set(descriptor["expected_absent_by_stage"]["runtime"]["no_terminal"])
            for candidate, absent in descriptor["expected_absent_by_stage"]["runtime"]["terminal_choices"].items():
                if no_terminal - set(absent) == extras:
                    terminal_id = candidate
                    break
    return classify_topology(descriptor, rows, case["evaluation_stage"], terminal_id)


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
    def take_snapshot(label):
        nonlocal primary
        if snapshot is None:
            return
        try:
            snapshot(label)
        except Exception as exc:
            value=f"snapshot:{label}:{type(exc).__name__}"
            if primary is None: primary={"state":"failure","value":value}
            else: secondary.append(value)
    for index, op in enumerate(NVRTC_OPS[:-1]):
        before = handle
        collect_compile_log = (primary is not None and
                               str(primary.get("value", "")).startswith("nvrtcCompileProgram:") and
                               op in {"nvrtcGetProgramLogSize", "nvrtcGetProgramLog"})
        if primary is not None and not collect_compile_log:
            ledger.append({"sequence": index, "op": op, "attempted": False,
                           "code": "not_attempted", "handle_before": before,
                           "handle_after": handle})
            take_snapshot(op)
            continue
        try:
            reply = adapter.call(op, handle=handle, source=source,
                                 program_name=PROGRAM_NAME, options=OPTIONS)
            code = int(reply.get("code", 0))
            if op == "nvrtcCreateProgram":
                handle = int(reply.get("handle", 0))
            row = {"sequence": index, "op": op, "attempted": True,
                           "code": code, "handle_before": before,
                           "handle_after": handle}
            if "size" in reply:
                row["returned_size"] = int(reply["size"])
            ledger.append(row)
            for name in ("log", "ptx", "cubin"):
                if name in reply:
                    artifacts[name] = reply[name]
            take_snapshot(op)
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
            take_snapshot(op)
    destroy_index = len(NVRTC_OPS) - 1
    if handle:
        try:
            reply = adapter.call("nvrtcDestroyProgram", handle=handle, source=source,
                                 program_name=PROGRAM_NAME, options=OPTIONS)
            code = int(reply.get("code", 0))
            ledger.append({"sequence": destroy_index, "op": "nvrtcDestroyProgram",
                           "attempted": True, "code": code,
                           "handle_before": handle, "handle_after": int(reply.get("handle", 0))})
            take_snapshot("nvrtcDestroyProgram")
            if code != 0:
                if primary is None:
                    primary = {"state": "failure", "value": f"nvrtcDestroyProgram:{code}"}
                else:
                    secondary.append(f"nvrtcDestroyProgram:{code}")
        except Exception as exc:
            ledger.append({"sequence": destroy_index, "op": "nvrtcDestroyProgram",
                           "attempted": True, "code": "ctypes_exception",
                           "handle_before": handle, "handle_after": handle,
                           "error": f"{type(exc).__name__}:{exc}"})
            if primary is None:
                primary = {"state": "failure", "value": "nvrtcDestroyProgram:ctypes_exception"}
            else:
                secondary.append("nvrtcDestroyProgram:ctypes_exception")
            take_snapshot("nvrtcDestroyProgram")
    else:
        ledger.append({"sequence": destroy_index, "op": "nvrtcDestroyProgram",
                       "attempted": False, "code": "not_attempted",
                       "handle_before": 0, "handle_after": 0})
        take_snapshot("nvrtcDestroyProgram")
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
    checks["destroy_suffix"] = (bool(ledger) and ledger[-1]["op"] == "nvrtcDestroyProgram"
                                and (not ledger[-1]["attempted"] or ledger[-1]["handle_before"] != 0))
    checks["size_rows"] = all(
        (row["op"] not in {"nvrtcGetProgramLogSize", "nvrtcGetPTXSize", "nvrtcGetCUBINSize"}
         or not row["attempted"] or (type(row.get("returned_size")) is int and row["returned_size"] >= 0))
        for row in ledger)
    artifacts = evidence.get("artifacts", {})
    checks["artifact_keys"] = set(artifacts) >= {"log", "ptx", "cubin"} or evidence.get("primary", {}).get("state") == "failure"
    if set(artifacts) >= {"log", "ptx", "cubin"}:
        log, ptx, cubin = artifacts["log"], artifacts["ptx"], artifacts["cubin"]
        checks["log_canonical"] = 1 <= len(log) <= CAPS["log"] and log.endswith(b"\0") and b"\0" not in log[:-1]
        checks["ptx_canonical"] = 1 < len(ptx) <= CAPS["ptx"] and ptx.endswith(b"\0") and b"\0" not in ptx[:-1]
        logical = ptx[:-1].decode("utf-8", "strict") if checks["ptx_canonical"] else ""
        checks["ptx_contract"] = (logical.count(".version") == 1
                                  and logical.count(".target sm_120") == 1
                                  and logical.count(".address_size 64") == 1
                                  and len(re.findall(r"(?m)^\s*\.visible\s+\.entry\s+q5_linear\b", logical)) == 1
                                  and len(re.findall(r"(?m)^\s*\.visible\s+\.entry\s+bf16_lut_activation\b", logical)) == 1
                                  and len(re.findall(r"(?m)^\s*(?:\.visible\s+)?\.entry\s+", logical)) == 2)
        checks["cubin_contract"] = _elf_kernel_symbols(cubin) == {"q5_linear", "bf16_lut_activation"}
    return checks


def _elf_kernel_symbols(raw: bytes) -> set[str] | None:
    """Bounded ELF64 little-endian symbol parser; no substring-only acceptance."""
    import struct
    try:
        if not (1 < len(raw) <= CAPS["cubin"] and raw[:6] == b"\x7fELF\x02\x01"):
            return None
        e_shoff = struct.unpack_from("<Q", raw, 40)[0]
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", raw, 58)
        if e_shentsize != 64 or not (0 < e_shnum <= 4096) or e_shstrndx >= e_shnum:
            return None
        if e_shoff + e_shentsize * e_shnum > len(raw):
            return None
        sections = [struct.unpack_from("<IIQQQQIIQQ", raw, e_shoff + i * 64)
                    for i in range(e_shnum)]
        names_hdr = sections[e_shstrndx]
        names = raw[names_hdr[4]:names_hdr[4] + names_hdr[5]]
        if len(names) != names_hdr[5]:
            return None
        def cstr(blob: bytes, offset: int) -> str:
            end = blob.index(0, offset)
            return blob[offset:end].decode("utf-8", "strict")
        symbols = set()
        for section in sections:
            if section[1] not in (2, 11):
                continue
            link, offset, size, entsize = section[6], section[4], section[5], section[9]
            if link >= e_shnum or entsize != 24 or offset + size > len(raw):
                return None
            strings_hdr = sections[link]
            strings = raw[strings_hdr[4]:strings_hdr[4] + strings_hdr[5]]
            if len(strings) != strings_hdr[5]:
                return None
            for pos in range(offset, offset + size, entsize):
                name_offset, info = struct.unpack_from("<IB", raw, pos)
                if name_offset and (info & 0x0F) == 2:
                    symbols.add(cstr(strings, name_offset))
        return symbols
    except (IndexError, UnicodeError, ValueError, struct.error):
        return None


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
        with path.open("r+b") as handle:
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
    flush_directory(quarantine)
    flush_directory(parent)
    return {"classification": "transaction_debris", "recovered": True,
            "attempt_consumed": False, "next_invocation_allowed": True,
            "source": str(debris[0]), "target": str(target)}


def write_incidental_failure(root: Path, row: dict) -> Path:
    raw = canonical(row)
    if len(raw) > CAPS["json_each"]:
        prefix = raw[:CAPS["prefix"]]
        row = {"kind": "het_next_l0_ph1_nvidia_nc19i1_bounded_failure",
               "revision": REVISION, "status": "incidental_failure",
               "full_bytes": len(raw), "full_sha256": sha256(raw),
               "prefix_offset": 0, "prefix_length": len(prefix),
               "prefix_base64": base64.b64encode(prefix).decode(),
               "prefix_sha256": sha256(prefix)}
        raw = canonical(row)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    nonce = f"{time.time_ns()}.{uuid.uuid4().hex[:16]}"
    stage = root / f"attempt.inprogress.{nonce}"
    attempt = root / f"attempt.{nonce}"
    stage.mkdir(parents=False, exist_ok=False)
    try:
        atomic_create(stage / "failure.json", raw)
        flush_directory(stage)
        os.rename(stage, attempt)
        flush_directory(root)
        return attempt
    except Exception as primary:
        secondary = {"kind": "het_next_l0_ph1_nvidia_nc19i1_failure_writer_secondary",
                     "revision": REVISION, "status": "incidental_failure",
                     "primary": f"{type(primary).__name__}:{primary}"}
        secondary_path = root / f"writer_failure.{nonce}.json"
        try:
            atomic_create(secondary_path, canonical(secondary))
        except Exception:
            pass
        raise


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
