#!/usr/bin/env python3
"""CPU-only immutable-input preflight for PORT80B T0-R1/T0-P1.

This script performs no network access, checkpoint download, CUDA call, bank
build, host registration or registry edit. The expected pre-download outcome
is blocked solely because the pinned official shard payload is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
T0R1 = REPORTS / "PORT80B_T0R1_OFFICIAL_LAYER0_REFERENCE_PREREGISTRATION_2026-08-13.md"
T0P1 = REPORTS / "PORT80B_T0P1_OFFICIAL_LAYER0_PHYSICAL_PREREGISTRATION_2026-08-13.md"
PROMPT_LOCK = REPORTS / "port80b_t0r1_prompt_lock.json"
ENV_LOCK = REPORTS / "port80b_t0r1_reference_environment_lock.json"
OUT = REPORTS / "port80b_t0r1_t0p1_cpu_preflight.json"

SNAPSHOT = (Path.home() / ".cache" / "huggingface" / "hub" /
            "models--Qwen--Qwen3-Coder-Next" / "snapshots" /
            "a19358a7659bd1f564300250ee189120c49a562f")
INDEX = SNAPSHOT / "model.safetensors.index.json"
SHARD = SNAPSHOT / "model-00001-of-00040.safetensors"
REF_PYTHON = ROOT / ".venv-next-ref" / "Scripts" / "python.exe"
REF_QWEN_DIR = (ROOT / ".venv-next-ref" / "Lib" / "site-packages" /
                "transformers" / "models" / "qwen3_next")
BASE_SITE = ROOT / ".venv" / "Lib" / "site-packages"
P4D_LOCK = REPORTS / "p4d_route_input_lock.json"

REVISION = "a19358a7659bd1f564300250ee189120c49a562f"
SHARD_BYTES = 3_999_619_288
SHARD_SHA256 = "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a"
T0R1_SHA256 = "9993828c2eb8048c282f614fb642f3e9a78a16686da0d2ed96e35bfa00d9c801"
T0P1_SHA256 = "0469cf58499a472d7ae982c75a93bf2a2dcef7ac247e1b82f133ca39bfcb5043"
PROMPT_LOCK_SHA256 = "f283da7e86adf915431459b08aac967d9c18c3de155699c369f5a55be20e5f34"
ENV_LOCK_SHA256 = "eb31d4e0c1f6a806434ea8a20b6b00200781a89ed9f91e485aad0e3583c0f455"

SUPPORT = {
    "config.json": (1_178, "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8"),
    "generation_config.json": (214, "37a3c1ef63516ca489c575f0db1c0405ddc0c3dbaca9ed987344c966c37aeef5"),
    "model.safetensors.index.json": (6_759_619, "e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b"),
    "tokenizer.json": (7_032_399, "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d"),
    "tokenizer_config.json": (11_681, "fc76878832c668e3f0f8be66e6239a475b9093d2fe5cef97c242369779e6c6e6"),
}
QWEN_SOURCES = {
    "__init__.py": "3ee3ef179c5c11f07150a8ea8996960a21de0de2ae4e635d7ee23ac2f6fe71b2",
    "configuration_qwen3_next.py": "931f04df4c8631942e12e1e045c3123d7a0a9336792aef902a7a5f899d0f4049",
    "modeling_qwen3_next.py": "de40823607becdd616436e3b332f14e0c92df5495ac72ef8af027c4488b9afca",
    "modular_qwen3_next.py": "d3da9b3e3548bdea4de092e79946ebbead6c17326759e291b1c8ab0ac465f4d2",
}

EXPERT_BYTES = 2_027_520
BANK_BYTES = 513 * EXPERT_BYTES
PREFIX_BYTES = 499 * EXPERT_BYTES
COLD_BYTES = 13 * EXPERT_BYTES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def row(name: str, passed: bool, detail: Any = None, blocker: bool = True) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "blocker": blocker, "detail": detail}


def exact_file(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256(path) == digest


def reference_probe(prompt_lock: Path) -> dict[str, Any]:
    code = r'''import hashlib,json,os,sys
import numpy, safetensors, tokenizers, torch, transformers
from transformers import AutoTokenizer
from transformers.models.qwen3_next import Qwen3NextConfig
lock=json.load(open(sys.argv[1],encoding="utf-8"))
tok=AutoTokenizer.from_pretrained(sys.argv[2],local_files_only=True,trust_remote_code=False)
got=[]
for p in lock["prompts"]:
    ids=tok(p["utf8_text"],add_special_tokens=False)["input_ids"][:16]
    raw=b"".join(int(x).to_bytes(4,"little",signed=False) for x in ids)
    got.append({"domain":p["domain"],"token_ids":ids,"token_ids_le_u32_sha256":hashlib.sha256(raw).hexdigest()})
cfg=Qwen3NextConfig.from_pretrained(sys.argv[2],local_files_only=True,trust_remote_code=False)
print(json.dumps({
 "python":sys.version.split()[0],"executable":sys.executable,
 "transformers":transformers.__version__,"torch":torch.__version__,
 "safetensors":safetensors.__version__,"tokenizers":tokenizers.__version__,"numpy":numpy.__version__,
 "torch_path":torch.__file__,"numpy_path":numpy.__file__,"safetensors_path":safetensors.__file__,
 "cuda_initialized":torch.cuda.is_initialized(),"token_rows":got,
 "config":{"model_type":cfg.model_type,"layers":cfg.num_hidden_layers,"experts":cfg.num_experts,
           "top_k":cfg.num_experts_per_tok,"hidden":cfg.hidden_size,"moe_intermediate":cfg.moe_intermediate_size}
},sort_keys=True))'''
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    proc = subprocess.run([str(REF_PYTHON), "-c", code, str(prompt_lock), str(SNAPSHOT)],
                          capture_output=True, text=True, timeout=60, env=env, check=False)
    result: dict[str, Any] = {"returncode": proc.returncode,
                              "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    if proc.returncode == 0:
        try:
            result["parsed"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass
    return result


def main() -> int:
    checks: list[dict[str, Any]] = []
    checks.append(row("immutable_preregistrations_exact",
                      T0R1.is_file() and T0P1.is_file()
                      and sha256(T0R1) == T0R1_SHA256 and sha256(T0P1) == T0P1_SHA256,
                      {"t0r1": sha256(T0R1) if T0R1.is_file() else None,
                       "t0p1": sha256(T0P1) if T0P1.is_file() else None}))
    checks.append(row("prompt_and_environment_locks_exact",
                      PROMPT_LOCK.is_file() and ENV_LOCK.is_file()
                      and sha256(PROMPT_LOCK) == PROMPT_LOCK_SHA256
                      and sha256(ENV_LOCK) == ENV_LOCK_SHA256,
                      {"prompt": sha256(PROMPT_LOCK) if PROMPT_LOCK.is_file() else None,
                       "environment": sha256(ENV_LOCK) if ENV_LOCK.is_file() else None}))

    support_detail = {}
    support_ok = True
    for name, (size, digest) in SUPPORT.items():
        path = SNAPSHOT / name
        observed = {"present": path.is_file(), "expected_bytes": size, "expected_sha256": digest}
        if path.is_file():
            observed.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        observed["pass"] = exact_file(path, size, digest)
        support_ok &= observed["pass"]
        support_detail[name] = observed
    checks.append(row("official_support_files_exact", support_ok, support_detail))

    weight_map: dict[str, str] = {}
    if exact_file(INDEX, *SUPPORT["model.safetensors.index.json"]):
        weight_map = json.loads(INDEX.read_text(encoding="utf-8"))["weight_map"]
    layer0 = [key for key, value in weight_map.items()
              if key.startswith("model.layers.0.") and value == SHARD.name]
    shard1 = [key for key, value in weight_map.items() if value == SHARD.name]
    checks.append(row("index_layer0_shard_locality_exact",
                      len(layer0) == 1_550 and len(shard1) == 1_567
                      and weight_map.get("model.embed_tokens.weight") == SHARD.name,
                      {"layer0_keys": len(layer0), "shard1_keys": len(shard1),
                       "embedding_shard": weight_map.get("model.embed_tokens.weight")}))

    source_detail = {name: sha256(REF_QWEN_DIR / name) if (REF_QWEN_DIR / name).is_file() else None
                     for name in QWEN_SOURCES}
    checks.append(row("qwen3_next_reference_sources_exact",
                      all(source_detail[name] == digest for name, digest in QWEN_SOURCES.items()),
                      source_detail))

    prompts = json.loads(PROMPT_LOCK.read_text(encoding="utf-8")) if PROMPT_LOCK.is_file() else {}
    prompt_rows = prompts.get("prompts", [])
    prompt_internal = len(prompt_rows) == 4 and all(len(p.get("token_ids", [])) == 16 for p in prompt_rows)
    for prompt in prompt_rows:
        text_digest = hashlib.sha256(prompt["utf8_text"].encode("utf-8")).hexdigest()
        packed = b"".join(int(x).to_bytes(4, "little", signed=False) for x in prompt["token_ids"])
        prompt_internal &= text_digest == prompt["utf8_sha256"]
        prompt_internal &= hashlib.sha256(packed).hexdigest() == prompt["token_ids_le_u32_sha256"]
    checks.append(row("prompt_lock_internal_hashes_and_shape", prompt_internal,
                      {"domains": [p.get("domain") for p in prompt_rows],
                       "rows": len(prompt_rows), "tokens_each": [len(p.get("token_ids", [])) for p in prompt_rows]}))

    probe = reference_probe(PROMPT_LOCK) if REF_PYTHON.is_file() and prompt_internal else {"returncode": None}
    parsed = probe.get("parsed", {})
    expected_tokens = [{"domain": p["domain"], "token_ids": p["token_ids"],
                        "token_ids_le_u32_sha256": p["token_ids_le_u32_sha256"]} for p in prompt_rows]
    env_ok = (probe.get("returncode") == 0 and parsed.get("python") == "3.12.10"
              and parsed.get("transformers") == "5.15.0"
              and parsed.get("torch") == "2.12.1+cu132"
              and parsed.get("safetensors") == "0.8.0"
              and parsed.get("tokenizers") == "0.22.2" and parsed.get("numpy") == "2.2.6"
              and parsed.get("cuda_initialized") is False
              and Path(parsed.get("torch_path", "")).resolve() == (BASE_SITE / "torch" / "__init__.py").resolve()
              and Path(parsed.get("numpy_path", "")).resolve() == (BASE_SITE / "numpy" / "__init__.py").resolve()
              and Path(parsed.get("safetensors_path", "")).resolve() == (BASE_SITE / "safetensors" / "__init__.py").resolve())
    checks.append(row("reference_environment_and_cpu_policy_exact", env_ok, probe))
    checks.append(row("pinned_tokenizer_reproduces_prompt_lock",
                      parsed.get("token_rows") == expected_tokens,
                      {"expected": expected_tokens, "observed": parsed.get("token_rows")}))
    checks.append(row("official_config_architecture_exact",
                      parsed.get("config") == {"model_type": "qwen3_next", "layers": 48, "experts": 512,
                                                "top_k": 10, "hidden": 2048, "moe_intermediate": 512},
                      parsed.get("config")))

    prior_digest_values: set[str] = set()
    if P4D_LOCK.is_file():
        prior = json.loads(P4D_LOCK.read_text(encoding="utf-8"))
        prior_digest_values.update(prior.get("input_ids_sha256", {}).values())
        prior_digest_values.update(prior.get("prior_input_sha256", {}).values())
    current_digests = {p.get("token_ids_le_u32_sha256") for p in prompt_rows}
    checks.append(row("prompt_hashes_disjoint_from_prior_locked_inputs",
                      len(current_digests) == 4 and current_digests.isdisjoint(prior_digest_values),
                      {"current": sorted(current_digests), "prior_digest_count": len(prior_digest_values)}))

    checks.append(row("layer0_bank_byte_math_exact",
                      BANK_BYTES == 1_040_117_760 and PREFIX_BYTES == 1_011_732_480
                      and COLD_BYTES == 26_357_760,
                      {"expert_record_bytes": EXPERT_BYTES, "bank": BANK_BYTES,
                       "prefix_499": PREFIX_BYTES, "cold_13": COLD_BYTES,
                       "selected_top10_bytes": 10 * EXPERT_BYTES}))
    disk = shutil.disk_usage(ROOT)
    memory = psutil.virtual_memory()
    checks.append(row("pre_download_resource_headroom",
                      disk.free >= 20 * 2**30 and memory.available >= 8 * 2**30,
                      {"free_disk_bytes": disk.free, "available_ram_bytes": memory.available,
                       "minimum_disk_bytes": 20 * 2**30, "minimum_ram_bytes": 8 * 2**30}))

    shard_detail: dict[str, Any] = {"path": str(SHARD), "present": SHARD.is_file(),
                                    "expected_bytes": SHARD_BYTES, "expected_sha256": SHARD_SHA256}
    shard_ok = False
    if SHARD.is_file():
        shard_detail["observed_bytes"] = SHARD.stat().st_size
        if SHARD.stat().st_size == SHARD_BYTES:
            shard_detail["observed_sha256"] = sha256(SHARD)
            shard_ok = shard_detail["observed_sha256"] == SHARD_SHA256
    checks.append(row("official_shard1_payload_present_size_hash", shard_ok, shard_detail))

    failures = [item for item in checks if item["blocker"] and not item["pass"]]
    result = {
        "kind": "port80b_t0r1_t0p1_cpu_preflight",
        "status": "cpu_preflight_pass_payload_actions_still_closed" if not failures
                  else "blocked_before_download_or_gpu",
        "pass": not failures,
        "checks_passed": len(checks) - len(failures),
        "checks_total": len(checks),
        "blocked_reasons": [item["name"] for item in failures],
        "inputs": {"revision": REVISION, "t0r1_sha256": T0R1_SHA256,
                   "t0p1_sha256": T0P1_SHA256, "prompt_lock_sha256": PROMPT_LOCK_SHA256,
                   "environment_lock_sha256": ENV_LOCK_SHA256,
                   "preflight_script_sha256": sha256(Path(__file__))},
        "checks": checks,
        "physical_actions": {"network": False, "download": False, "cuda_initialized": False,
                             "gpu_allocation": False, "kernel_launch": False,
                             "host_registration": False, "bank_build": False,
                             "registry_edit": False},
        "claim_boundary": ("CPU/offline immutable-input preflight only. No official shard payload "
                           "was acquired or read; no model forward, bank build or physical test ran."),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("status", "pass", "checks_passed", "checks_total", "blocked_reasons")}, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

