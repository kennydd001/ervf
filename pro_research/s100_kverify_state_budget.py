"""S100-KVERIFY K0: compute Mamba/KV rollback byte budgets from config only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from common import REPO, require_model_dir, sha256_file, utc_now

sys.path.insert(0, str(REPO / "src"))
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

RESULT_DIR = REPO / "pro_research" / "results" / "s100_kverify"
OUT = RESULT_DIR / "PRO_S100_KVERIFY_K0_STATE_BUDGET.json"
PREREG = REPO / "pro_research" / "S100_KVERIFY_PREREGISTRATION.md"


def _write(payload):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def main() -> int:
    model = require_model_dir()
    idx = ShardIndex(model)
    c = idx.config
    pattern = idx.pattern_string()
    n_mamba = pattern.count("M")
    n_attn = pattern.count("*")

    mh = int(c["mamba_num_heads"])
    hd = int(c["mamba_head_dim"])
    ns = int(c["ssm_state_size"])
    ng = int(c["n_groups"])
    ck = int(c["conv_kernel"])
    nkv = int(c["num_key_value_heads"])
    ahd = int(c["head_dim"])
    hidden = int(c["hidden_size"])

    d_inner = mh * hd
    conv_dim = d_inner + 2 * ng * ns
    proj_width = d_inner + conv_dim + mh

    f32 = 4
    fp8 = 1
    ssm_per_layer = mh * hd * ns * f32
    conv_per_layer = conv_dim * ck * f32
    mutable_per_layer = ssm_per_layer + conv_per_layer
    mutable_all = n_mamba * mutable_per_layer
    proj_per_token_all_layers = n_mamba * proj_width * f32
    kv_append_per_token = n_attn * 2 * nkv * ahd * fp8

    by_k = {}
    for k in (2, 4, 8):
        stored_proj = k * proj_per_token_all_layers
        one_snapshot_plus_proj = mutable_all + stored_proj
        # Full per-token Mamba snapshots are recorded only as the rejected
        # naive alternative. KVERIFY K1 attempts to avoid this K multiplier.
        naive_k_snapshots = k * mutable_all
        by_k[str(k)] = {
            "stored_mamba_in_proj_bytes": stored_proj,
            "stored_mamba_in_proj_mib": stored_proj / (1024 ** 2),
            "one_initial_mamba_snapshot_plus_proj_bytes": one_snapshot_plus_proj,
            "one_initial_mamba_snapshot_plus_proj_mib": one_snapshot_plus_proj / (1024 ** 2),
            "naive_k_full_mamba_snapshots_bytes": naive_k_snapshots,
            "naive_k_full_mamba_snapshots_mib": naive_k_snapshots / (1024 ** 2),
            "fp8_kv_append_bytes": k * kv_append_per_token,
            "fp8_kv_append_kib": k * kv_append_per_token / 1024,
        }

    payload = {
        "kind": "pro_s100_kverify_k0_state_budget",
        "status": "measured_metadata_budget",
        "created_utc": utc_now(),
        "claim_boundary": "config-derived byte budget only; no GPU allocation, speed or rollback correctness claim",
        "preregistration": str(PREREG.relative_to(REPO)),
        "model_dir": str(model),
        "metadata_hashes": {
            "config_json_sha256": sha256_file(model / "config.json"),
            "safetensors_index_sha256": sha256_file(model / "model.safetensors.index.json"),
        },
        "architecture": {
            "pattern": pattern,
            "mamba_layers": n_mamba,
            "attention_layers": n_attn,
            "hidden": hidden,
            "mamba_num_heads": mh,
            "mamba_head_dim": hd,
            "ssm_state_size": ns,
            "n_groups": ng,
            "conv_kernel": ck,
            "d_inner": d_inner,
            "conv_dim": conv_dim,
            "mamba_in_proj_width": proj_width,
            "num_key_value_heads": nkv,
            "attention_head_dim": ahd,
        },
        "bytes": {
            "ssm_state_per_mamba_layer": ssm_per_layer,
            "conv_state_per_mamba_layer": conv_per_layer,
            "mutable_mamba_state_per_layer": mutable_per_layer,
            "mutable_mamba_state_all_layers": mutable_all,
            "mutable_mamba_state_all_layers_mib": mutable_all / (1024 ** 2),
            "stored_in_proj_per_token_all_mamba_layers": proj_per_token_all_layers,
            "stored_in_proj_per_token_all_mamba_layers_mib": proj_per_token_all_layers / (1024 ** 2),
            "fp8_kv_append_per_token": kv_append_per_token,
        },
        "budget_by_k": by_k,
        "interpretation": {
            "one_snapshot_strategy": "one initial exact copy of all mutable Mamba state plus K stored in_proj outputs; partial commit would restore the initial copy and replay state-only transitions",
            "naive_strategy": "K full Mamba state snapshots; reported only for comparison and not adopted",
            "kv_rollback": "KV append bytes are diagnostic; rollback can ignore later positions by restoring logical position, subject to K2 exact proof",
        },
    }
    _write(payload)
    print(json.dumps({
        "status": payload["status"],
        "output": str(OUT),
        "mutable_mamba_state_all_layers_mib": payload["bytes"]["mutable_mamba_state_all_layers_mib"],
        "stored_in_proj_per_token_all_mamba_layers_mib": payload["bytes"]["stored_in_proj_per_token_all_mamba_layers_mib"],
        "budget_by_k": by_k,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
