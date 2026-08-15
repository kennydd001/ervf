#!/usr/bin/env python3
"""Device-free PH1-R2 source revision derived from the immutable R1 source."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R0_BACKEND = ROOT / "scripts/streamq5_moe/het_next_l0_ph1_intel_backend.py"
R0_BACKEND_SHA256 = "1c70d4248bdf64404589916a6be624594e8343442a64c57e926e52926f51ceac"
R1_SOURCE_SHA256 = "06be3a9ba863d5f01d4025dc8d5e5679cdcc9827c13db5663c149227f3254528"
R2_SOURCE_SHA256 = "f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21"
R2_SOURCE_BYTES = 7_852
OPTIONS = "-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _r0_literal() -> str:
    if file_sha256(R0_BACKEND) != R0_BACKEND_SHA256:
        raise RuntimeError("r0_backend_hash_drift")
    tree = ast.parse(R0_BACKEND.read_text(encoding="utf-8"), filename=str(R0_BACKEND))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SRC" for target in node.targets):
            source = ast.literal_eval(node.value)
            if isinstance(source, str):
                return source
    raise RuntimeError("r0_source_literal_missing")


def r1_source() -> str:
    original = _r0_literal()
    old_pragmas = (
        "#pragma OPENCL EXTENSION cl_intel_required_sub_group_size : enable\n"
        "#pragma OPENCL EXTENSION cl_khr_int64 : enable\n"
    )
    if original.count(old_pragmas) != 1:
        raise RuntimeError("r0_pragma_contract")
    source = original.replace(
        old_pragmas,
        "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n",
    )
    if sha256_bytes(source.encode()) != R1_SOURCE_SHA256:
        raise RuntimeError("r1_source_hash_drift")
    return source


def r2_source() -> str:
    """Apply exactly the two post-failure repairs and reject any ambiguous occurrence count."""
    source = r1_source()
    subgroup_pragma = "#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n"
    if source.count(subgroup_pragma) != 1:
        raise RuntimeError("r1_required_subgroup_pragma_contract")
    if source.count("ulong half =") != 1:
        raise RuntimeError("r1_half_declaration_contract")
    if source.count("remainder > half || (remainder == half") != 1:
        raise RuntimeError("r1_half_use_contract")
    source = source.replace(subgroup_pragma, "")
    source = source.replace("ulong half =", "ulong halfway =", 1)
    source = source.replace(
        "remainder > half || (remainder == half",
        "remainder > halfway || (remainder == halfway",
        1,
    )
    encoded = source.encode()
    if len(encoded) != R2_SOURCE_BYTES or sha256_bytes(encoded) != R2_SOURCE_SHA256:
        raise RuntimeError("r2_source_hash_drift")
    return source


SRC = r2_source()


def source_contract() -> dict:
    """Static evidence only; this function has no compiler or device surface."""
    return {
        "source_bytes": len(SRC.encode()),
        "source_sha256": sha256_bytes(SRC.encode()),
        "options": OPTIONS,
        "required_subgroup_extension_pragma_removed": "cl_intel_required_subgroup_size : enable" not in SRC,
        "subgroups_extension_pragma_retained": SRC.count("cl_intel_subgroups : enable") == 1,
        "required_subgroup_attributes_retained": SRC.count("intel_reqd_sub_group_size(8)") == 3,
        "reserved_half_identifier_absent": "ulong half" not in SRC,
        "halfway_identifier_count": SRC.count("halfway") == 3,
        "entrypoints": {
            name: SRC.count("void " + name + "(")
            for name in ("gate_linear", "up_linear", "activation", "down_linear")
        },
        "compiler_calls": 0,
        "device_calls": 0,
        "payload_reads": 0,
    }
