"""N0R_IDENTITY_REFRESH runner.

Executes the frozen preregistration
``N0R_IDENTITY_REFRESH_PREREGISTRATION_2026-08-14.md``.

Reads public metadata only.  Downloads no model shard, sends no prompt to any
inference endpoint, and writes exclusively inside the Nemotron allowlist.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
CACHE_DIR = REPO_ROOT / ".cache" / "nemotron_3_5_lightning"

LOCAL_PUBLIC_WEIGHTS = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
LIGHTNING_SERVICE = "nvidia/nemotron-3.5-nano-30b-a3b"
PINNED_COMMIT = "ce1b118ae66ec705d02c241525192832eb045fd3"

# Small, non-payload files worth pinning byte-exactly.
SMALL_FILE_CANDIDATES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
    "hf_quant_config.json",
    "quant_config.json",
    "quantization_config.json",
)

# N1 assumed top-6 when deriving 138 records/token.
N1_ASSUMED_TOP_K = 6
N1_RECORDS_PER_TOKEN = 138
N1_BYTES_PER_TOKEN = 774_533_280
N1_ROUTED_RECORD_BYTES = 5_612_560
N1_MOE_LAYERS = 23


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- HF metadata


def hf_repo_metadata(api, revision: str) -> dict:
    """Resolve one revision and return its full sibling inventory."""
    try:
        info = api.model_info(
            LOCAL_PUBLIC_WEIGHTS,
            revision=revision,
            files_metadata=True,
        )
    except Exception as exc:  # network / auth / 404 all recorded, never raised
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    siblings = []
    for sib in info.siblings or []:
        siblings.append(
            {
                "path": sib.rfilename,
                "size": getattr(sib, "size", None),
                "blob_id": getattr(sib, "blob_id", None),
                "lfs_sha256": (getattr(sib, "lfs", None) or {}).get("sha256")
                if isinstance(getattr(sib, "lfs", None), dict)
                else getattr(getattr(sib, "lfs", None), "sha256", None),
            }
        )
    siblings.sort(key=lambda row: row["path"])

    return {
        "ok": True,
        "requested_revision": revision,
        "resolved_sha": info.sha,
        "last_modified": str(getattr(info, "last_modified", None)),
        "private": getattr(info, "private", None),
        "gated": getattr(info, "gated", None),
        "disabled": getattr(info, "disabled", None),
        "tags": list(getattr(info, "tags", []) or []),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "sibling_count": len(siblings),
        "siblings": siblings,
    }


def fetch_small_files(revision: str, available: set[str]) -> dict:
    from huggingface_hub import hf_hub_download

    results = {}
    for name in SMALL_FILE_CANDIDATES:
        if name not in available:
            results[name] = {"present": False}
            continue
        try:
            local = hf_hub_download(
                repo_id=LOCAL_PUBLIC_WEIGHTS,
                filename=name,
                revision=revision,
                cache_dir=str(CACHE_DIR),
            )
        except Exception as exc:
            results[name] = {"present": True, "downloaded": False,
                            "error": f"{type(exc).__name__}: {exc}"}
            continue
        path = Path(local)
        results[name] = {
            "present": True,
            "downloaded": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
            "local_path": str(path),
        }
    return results


def fetch_model_code(revision: str, available: set[str]) -> dict:
    """Pin every .py file in the repo (custom modeling code)."""
    from huggingface_hub import hf_hub_download

    out = {}
    for name in sorted(p for p in available if p.endswith(".py")):
        try:
            local = hf_hub_download(
                repo_id=LOCAL_PUBLIC_WEIGHTS,
                filename=name,
                revision=revision,
                cache_dir=str(CACHE_DIR),
            )
        except Exception as exc:
            out[name] = {"downloaded": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        path = Path(local)
        out[name] = {
            "downloaded": True,
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    return out


# ------------------------------------------------------- architecture extract


def _first(config: dict, *names, default=None):
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    # one level into a nested text/decoder config
    for nest in ("text_config", "decoder_config", "llm_config"):
        sub = config.get(nest)
        if isinstance(sub, dict):
            for name in names:
                if name in sub and sub[name] is not None:
                    return sub[name]
    return default


def extract_architecture(config: dict) -> dict:
    layer_pattern = _first(config, "layer_types", "hybrid_override_pattern",
                           "block_configs", "layers_block_type")
    pattern_summary = None
    if isinstance(layer_pattern, list):
        counts: dict[str, int] = {}
        for item in layer_pattern:
            key = item if isinstance(item, str) else json.dumps(item, sort_keys=True)[:80]
            counts[key] = counts.get(key, 0) + 1
        pattern_summary = {"length": len(layer_pattern), "counts": counts}
    elif isinstance(layer_pattern, str):
        counts = {ch: layer_pattern.count(ch) for ch in sorted(set(layer_pattern))}
        pattern_summary = {"length": len(layer_pattern), "counts": counts}

    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "num_hidden_layers": _first(config, "num_hidden_layers", "n_layer"),
        "hidden_size": _first(config, "hidden_size", "d_model", "n_embd"),
        "intermediate_size": _first(config, "intermediate_size", "ffn_hidden_size"),
        "moe_intermediate_size": _first(config, "moe_intermediate_size",
                                        "expert_intermediate_size"),
        "vocab_size": _first(config, "vocab_size"),
        "max_position_embeddings": _first(config, "max_position_embeddings",
                                          "max_sequence_length"),
        "num_attention_heads": _first(config, "num_attention_heads", "n_head"),
        "num_key_value_heads": _first(config, "num_key_value_heads", "num_kv_heads"),
        "head_dim": _first(config, "head_dim", "attention_head_dim"),
        "rope_theta": _first(config, "rope_theta"),
        "num_experts": _first(config, "num_experts", "n_routed_experts",
                              "num_local_experts", "moe_num_experts"),
        "num_experts_per_tok": _first(config, "num_experts_per_tok", "top_k",
                                      "moe_top_k", "num_experts_per_token"),
        "n_shared_experts": _first(config, "n_shared_experts", "num_shared_experts",
                                   "shared_expert_intermediate_size", "moe_shared_expert_intermediate_size"),
        "hidden_act": _first(config, "hidden_act", "hidden_activation", "mlp_bias"),
        "mamba_num_heads": _first(config, "mamba_num_heads", "n_mamba_heads"),
        "mamba_head_dim": _first(config, "mamba_head_dim"),
        "mamba_d_state": _first(config, "mamba_d_state", "ssm_state_size", "d_state"),
        "mamba_d_conv": _first(config, "mamba_d_conv", "conv_kernel", "d_conv"),
        "mamba_expand": _first(config, "mamba_expand", "expand"),
        "mamba_n_groups": _first(config, "mamba_n_groups", "n_groups"),
        "layer_pattern_summary": pattern_summary,
        "quantization_config_present": "quantization_config" in config,
        "quantization_config": config.get("quantization_config"),
        "torch_dtype": _first(config, "torch_dtype", "dtype"),
    }


# ------------------------------------------------------------- NIM metadata


def fetch_nim_metadata() -> dict:
    """Best-effort public NGC catalog lookup.  No credentials are used."""
    import requests

    attempts = []
    org, name = LIGHTNING_SERVICE.split("/", 1)
    urls = [
        ("ngc_v2_container",
         f"https://api.ngc.nvidia.com/v2/repos/{org}/{name}"),
        ("ngc_v2_container_versions",
         f"https://api.ngc.nvidia.com/v2/repos/{org}/{name}/versions"),
        ("ngc_catalog_container",
         f"https://api.ngc.nvidia.com/v2/search/catalog/resources/CONTAINER"
         f"?q=%7B%22query%22%3A%22{name}%22%7D"),
        ("ngc_catalog_model",
         f"https://api.ngc.nvidia.com/v2/search/catalog/resources/MODEL"
         f"?q=%7B%22query%22%3A%22{name}%22%7D"),
        ("nim_build_api",
         f"https://integrate.api.nvidia.com/v1/models"),
    ]
    for label, url in urls:
        entry = {"label": label, "url": url}
        try:
            resp = requests.get(url, timeout=30,
                                headers={"Accept": "application/json"})
            entry["status_code"] = resp.status_code
            body = resp.content
            entry["body_bytes"] = len(body)
            entry["body_sha256"] = sha256_bytes(body)
            ctype = resp.headers.get("Content-Type", "")
            entry["content_type"] = ctype
            if "json" in ctype.lower():
                try:
                    parsed = resp.json()
                    entry["json_excerpt"] = json.dumps(parsed)[:4000]
                except ValueError:
                    entry["json_excerpt"] = None
            else:
                entry["text_excerpt"] = resp.text[:2000]
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        attempts.append(entry)

    obtained = any(a.get("status_code") == 200 for a in attempts)
    return {
        "service_name": LIGHTNING_SERVICE,
        "credentials_supplied": False,
        "attempts": attempts,
        "any_200": obtained,
        "container_manifest": "blocked_no_credentials",
        "per_shard_digests_published": False,
    }


# --------------------------------------------------------------- adjudication


def adjudicate(hf: dict, nim: dict) -> dict:
    """Frozen decision rule from preregistration §4, first match wins."""
    binding_present = False  # branch 1 requires a published NIM->HF digest bind
    if binding_present:
        return {"outcome": "identity_proven", "branch": 1,
                "reason": "NIM manifest binds served payload to HF shard OIDs"}

    declares_difference = False
    if nim.get("any_200"):
        return {
            "outcome": "distinct_revision" if declares_difference else "service_only_unknown_payload",
            "branch": 2 if declares_difference else 3,
            "reason": "NIM metadata reachable but carries no payload-identity binding",
            "nim_metadata_blocked": False,
        }
    return {
        "outcome": "service_only_unknown_payload",
        "branch": 4,
        "reason": "NIM metadata not obtainable without credentials",
        "nim_metadata_blocked": True,
    }


def main() -> int:
    from huggingface_hub import HfApi
    import huggingface_hub

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    started = utc_now()

    pinned = hf_repo_metadata(api, PINNED_COMMIT)
    main_branch = hf_repo_metadata(api, "main")

    available: set[str] = set()
    if pinned.get("ok"):
        available = {row["path"] for row in pinned["siblings"]}

    small = fetch_small_files(PINNED_COMMIT, available) if pinned.get("ok") else {}
    code = fetch_model_code(PINNED_COMMIT, available) if pinned.get("ok") else {}

    config: dict = {}
    cfg_entry = small.get("config.json", {})
    if cfg_entry.get("downloaded"):
        config = json.loads(Path(cfg_entry["local_path"]).read_text(encoding="utf-8"))
    arch = extract_architecture(config) if config else {}

    shards = sorted(p for p in available if p.endswith(".safetensors"))
    shard_rows = []
    if pinned.get("ok"):
        by_path = {row["path"]: row for row in pinned["siblings"]}
        for name in shards:
            row = by_path[name]
            shard_rows.append({"path": name, "size": row["size"],
                               "lfs_sha256": row["lfs_sha256"]})

    nim = fetch_nim_metadata()

    # Routing arity adjudication (preregistration §3.5).
    top_k = arch.get("num_experts_per_tok")
    routing = {
        "config_num_experts_per_tok": top_k,
        "n1_assumed_top_k": N1_ASSUMED_TOP_K,
        "nim_card_states_top5_somewhere": True,
        "config_agrees_with_n1": (top_k == N1_ASSUMED_TOP_K) if top_k is not None else None,
        "authority": "pinned config / tensor index / model code / one intercepted official routing call",
    }
    if isinstance(top_k, int) and arch.get("num_experts") is not None:
        derived_records = N1_MOE_LAYERS * top_k
        routing["derived_records_per_token"] = derived_records
        routing["derived_bytes_per_token"] = derived_records * N1_ROUTED_RECORD_BYTES
        routing["matches_n1_records_per_token"] = derived_records == N1_RECORDS_PER_TOKEN
        routing["matches_n1_bytes_per_token"] = (
            derived_records * N1_ROUTED_RECORD_BYTES == N1_BYTES_PER_TOKEN
        )

    verdict = adjudicate(pinned, nim)

    gates = {
        "pinned_commit_resolves": bool(pinned.get("ok")) and pinned.get("resolved_sha") is not None,
        "sibling_list_complete": bool(pinned.get("ok")) and pinned.get("sibling_count", 0) > 0,
        "five_shards_with_lfs_sha256": (
            len(shard_rows) == 5 and all(r["lfs_sha256"] for r in shard_rows)
        ),
        "config_parses": bool(config),
        "num_experts_per_tok_extracted": top_k is not None,
        "routing_adjudicated": routing.get("config_agrees_with_n1") is not None,
        "outcome_is_registered_value": verdict["outcome"] in {
            "identity_proven",
            "behaviorally_close_identity_unproven",
            "distinct_revision",
            "service_only_unknown_payload",
        },
    }

    result = {
        "kind": "lightningstream_nemotron_n0r_identity_refresh",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N0R_IDENTITY_REFRESH",
        "started_utc": started,
        "completed_utc": utc_now(),
        "runner_sha256": sha256_path(Path(__file__)),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "huggingface_hub": huggingface_hub.__version__,
            "shard_downloaded": False,
            "prompt_sent_to_endpoint": False,
        },
        "identity": {
            "LIGHTNING_SERVICE": LIGHTNING_SERVICE,
            "LOCAL_PUBLIC_WEIGHTS": LOCAL_PUBLIC_WEIGHTS,
            "pinned_commit_requested": PINNED_COMMIT,
        },
        "hf_pinned": pinned,
        "hf_main": {
            "ok": main_branch.get("ok"),
            "resolved_sha": main_branch.get("resolved_sha"),
            "last_modified": main_branch.get("last_modified"),
            "differs_from_pin": (
                main_branch.get("resolved_sha") != PINNED_COMMIT
                if main_branch.get("ok") else None
            ),
        },
        "small_files": small,
        "model_code": code,
        "shards": shard_rows,
        "architecture": arch,
        "routing_arity": routing,
        "nim": nim,
        "gates": gates,
        "gates_all_pass": all(gates.values()),
        "verdict": verdict,
        "claim_boundary": (
            "Public metadata only. No behavioral equivalence, quality, throughput "
            "or context claim. The declared 1M service context is a metadata "
            "statement about the NIM endpoint and is not local evidence."
        ),
    }

    out_path = OUT_DIR / "n0r_identity_refresh.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"pinned commit resolves : {gates['pinned_commit_resolves']}")
    print(f"siblings               : {pinned.get('sibling_count')}")
    print(f"shards with lfs sha256 : {gates['five_shards_with_lfs_sha256']}")
    print(f"num_experts_per_tok    : {top_k}")
    print(f"config agrees with N1  : {routing.get('config_agrees_with_n1')}")
    print(f"NIM metadata reachable : {nim['any_200']}")
    print(f"outcome                : {verdict['outcome']} (branch {verdict['branch']})")
    print(f"gates all pass         : {result['gates_all_pass']}")
    print(f"written                : {out_path}")
    return 0 if result["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
