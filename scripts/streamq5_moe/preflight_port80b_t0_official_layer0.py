#!/usr/bin/env python3
"""CPU-only, read-only preflight for PORT80B T0-R/T0-P.

It intentionally does not download, initialize CUDA, build a bank or run the
reference.  A missing shard/config/tokenizer/complete hash is a truthful
blocked result, not an error to bypass.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
T0R = REPORTS / "PORT80B_T0R_OFFICIAL_LAYER0_REFERENCE_PREREGISTRATION_2026-08-13.md"
T0P = REPORTS / "PORT80B_T0P_OFFICIAL_LAYER0_PHYSICAL_PREREGISTRATION_2026-08-13.md"
OUT = REPORTS / "port80b_t0_official_layer0_cpu_preflight.json"
SNAPSHOT = (Path.home() / ".cache" / "huggingface" / "hub" /
            "models--Qwen--Qwen3-Coder-Next" / "snapshots" /
            "a19358a7659bd1f564300250ee189120c49a562f")
INDEX = SNAPSHOT / "model.safetensors.index.json"
SHARD = SNAPSHOT / "model-00001-of-00040.safetensors"
REF_PYTHON = ROOT / ".venv-next-ref" / "Scripts" / "python.exe"
REF_MODELING = (ROOT / ".venv-next-ref" / "Lib" / "site-packages" /
                "transformers" / "models" / "qwen3_next" / "modeling_qwen3_next.py")
REF_CONFIG = REF_MODELING.with_name("configuration_qwen3_next.py")
LLAMA_MODEL = ROOT / "third_party" / "llama.cpp" / "src" / "models" / "qwen3next.cpp"
LLAMA_GDN = ROOT / "third_party" / "llama.cpp" / "ggml" / "src" / "ggml-cuda" / "gated_delta_net.cu"

REVISION = "a19358a7659bd1f564300250ee189120c49a562f"
INDEX_BYTES = 6_759_619
INDEX_SHA256 = "e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b"
SHARD_BYTES = 3_999_619_288
# Metadata supplied to the research task is abbreviated. A full digest is a
# hard blocker and must be filled by a new immutable revision, never guessed.
SHARD_SHA256_METADATA = "8e9a5171..."
MODELING_SHA256 = "de40823607becdd616436e3b332f14e0c92df5495ac72ef8af027c4488b9afca"
CONFIGURATION_SHA256 = "931f04df4c8631942e12e1e045c3123d7a0a9336792aef902a7a5f899d0f4049"
LLAMA_MODEL_SHA256 = "651c74364d25a65be5d3b96fb5f9ff1675849a3970fdbf545cfdccac87bb23ab"
LLAMA_GDN_SHA256 = "6c95caa9dff67279b23b39058a74ddb4ab6d634f651716f82482e06e53f027d8"

EXPERT_BYTES = 2_027_520
LAYER0_BANK_BYTES = 513 * EXPERT_BYTES
PREFIX_BYTES = 499 * EXPERT_BYTES
COLD_BYTES = 13 * EXPERT_BYTES
MIN_DISK = 20 * 2**30
MIN_RAM = 8 * 2**30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def main() -> int:
    checks: list[dict[str, Any]] = []
    checks.append(check("preregistrations_exist", T0R.is_file() and T0P.is_file()))
    checks.append(check("index_exact", INDEX.is_file() and INDEX.stat().st_size == INDEX_BYTES
                        and sha256(INDEX) == INDEX_SHA256))
    index = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.is_file() else {"weight_map": {}}
    weight_map = index.get("weight_map", {})
    layer0 = [key for key, shard in weight_map.items()
              if key.startswith("model.layers.0.") and shard == SHARD.name]
    shard1 = [key for key, shard in weight_map.items() if shard == SHARD.name]
    checks.append(check("index_maps_complete_layer0_and_embedding_to_shard1",
                        len(layer0) == 1550 and len(shard1) == 1567
                        and weight_map.get("model.embed_tokens.weight") == SHARD.name,
                        {"layer0_keys": len(layer0), "shard1_keys": len(shard1)}))
    complete_shard_hash = (len(SHARD_SHA256_METADATA) == 64
                           and all(c in "0123456789abcdef" for c in SHARD_SHA256_METADATA))
    checks.append(check("official_shard1_full_sha256_locked", complete_shard_hash,
                        SHARD_SHA256_METADATA))
    shard_ok = SHARD.is_file() and SHARD.stat().st_size == SHARD_BYTES
    # Do not hash a 4-GB file until its complete expected digest is frozen.
    if shard_ok and complete_shard_hash:
        shard_ok = sha256(SHARD) == SHARD_SHA256_METADATA
    checks.append(check("official_shard1_present_size_and_hash", shard_ok,
                        {"present": SHARD.is_file(), "expected_bytes": SHARD_BYTES,
                         "observed_bytes": SHARD.stat().st_size if SHARD.is_file() else None}))
    support = [SNAPSHOT / name for name in
               ("config.json", "tokenizer.json", "tokenizer_config.json")]
    checks.append(check("official_config_and_tokenizer_present", all(path.is_file() for path in support),
                        [str(path) for path in support if not path.is_file()]))
    ref_files = {
        REF_MODELING: MODELING_SHA256,
        REF_CONFIG: CONFIGURATION_SHA256,
        LLAMA_MODEL: LLAMA_MODEL_SHA256,
        LLAMA_GDN: LLAMA_GDN_SHA256,
    }
    checks.append(check("reference_sources_exact", all(path.is_file() and sha256(path) == expected
                                                        for path, expected in ref_files.items())))
    ref_probe: dict[str, Any] = {"executed": False}
    if REF_PYTHON.is_file():
        command = [str(REF_PYTHON), "-c",
                   "import json,torch,transformers; import transformers.models.qwen3_next; "
                   "print(json.dumps({'transformers':transformers.__version__,'torch':torch.__version__}))"]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        ref_probe = {"executed": True, "returncode": proc.returncode,
                     "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        if proc.returncode == 0:
            try:
                versions = json.loads(proc.stdout)
            except json.JSONDecodeError:
                versions = {}
        else:
            versions = {}
    else:
        versions = {}
    checks.append(check("reference_environment_versions",
                        versions.get("transformers") == "5.15.0"
                        and versions.get("torch") == "2.12.1+cu132", ref_probe))
    disk = shutil.disk_usage(ROOT)
    memory = psutil.virtual_memory()
    checks.append(check("resource_headroom", disk.free >= MIN_DISK and memory.available >= MIN_RAM,
                        {"free_disk_bytes": disk.free, "available_ram_bytes": memory.available,
                         "minimum_disk_bytes": MIN_DISK, "minimum_ram_bytes": MIN_RAM}))
    checks.append(check("layer0_bank_byte_math",
                        LAYER0_BANK_BYTES == 1_040_117_760
                        and PREFIX_BYTES == 1_011_732_480 and COLD_BYTES == 26_357_760,
                        {"bank": LAYER0_BANK_BYTES, "prefix": PREFIX_BYTES, "cold": COLD_BYTES}))

    failures = [row for row in checks if not row["pass"]]
    blocked_reasons = [row["name"] for row in failures]
    result = {
        "kind": "port80b_t0_official_layer0_cpu_preflight",
        "status": "cpu_preflight_pass_payload_actions_still_closed" if not failures
                  else "blocked_before_download_or_gpu",
        "pass": not failures,
        "inputs": {
            "revision": REVISION,
            "t0r_preregistration_sha256": sha256(T0R) if T0R.is_file() else None,
            "t0p_preregistration_sha256": sha256(T0P) if T0P.is_file() else None,
            "runner_sha256": sha256(Path(__file__)),
            "index_sha256": sha256(INDEX) if INDEX.is_file() else None,
        },
        "checks": checks,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "blocked_reasons": blocked_reasons,
        "physical_actions": {
            "network": False, "download": False, "cuda_initialized": False,
            "gpu_allocation": False, "kernel_launch": False,
            "host_registration": False, "bank_build": False, "registry_edit": False,
        },
        "claim_boundary": (
            "CPU/read-only provenance, toolchain and resource preflight only; "
            "no checkpoint payload acquired, reference executed, bank built or GPU used."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("status", "pass", "checks_passed", "checks_total", "blocked_reasons")}, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

