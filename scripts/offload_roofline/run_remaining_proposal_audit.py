from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/offload_roofline/P_D_UNION_PREREGISTRATION.md"
HERA = ROOT / "reports/hera_moe/p0_multidomain_tier_result.json"
QWEN_ACQUISITION = ROOT / "reports/rsiv_moe/qwen_checkpoint_acquisition.json"
CORETAIL = ROOT / "reports/coretail_moe/p0a_locked16_format_result.json"
ATOMIC_FULL = ROOT / "reports/craft_moe/atomic_full_depth_oracle.json"
OUT_JSON = ROOT / "reports/offload_roofline/remaining_proposal_audit.json"
OUT_MD = ROOT / "reports/offload_roofline/REMAINING_PROPOSAL_AUDIT.md"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
DEPTHS = (1, 2, 4, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: np.ndarray):
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95, method="higher")),
        "maximum": int(values.max()), "samples": int(values.size),
    }


def union_curve(routes: np.ndarray):
    # routes: [tokens, layers, top_k]
    rows = {}
    for depth in DEPTHS:
        windows = routes.shape[0] - depth + 1
        combined = np.concatenate(
            [routes[offset : offset + windows] for offset in range(depth)], axis=2
        )
        combined.sort(axis=2)
        counts = 1 + np.count_nonzero(np.diff(combined, axis=2), axis=2)
        flat = counts.reshape(-1)
        row = summary(flat)
        row["mean_fraction_of_naive_k_times_s"] = row["mean"] / (8 * depth)
        row["mean_naive_over_unique_factor"] = 8 * depth / row["mean"]
        row["uniform_independent_expectation"] = 128 * (1 - (1 - 8 / 128) ** depth)
        rows[str(depth)] = row
    return rows


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite remaining-proposal audit")
    hera = json.loads(HERA.read_text(encoding="utf-8"))
    qwen = json.loads(QWEN_ACQUISITION.read_text(encoding="utf-8"))
    coretail = json.loads(CORETAIL.read_text(encoding="utf-8"))
    atomic = json.loads(ATOMIC_FULL.read_text(encoding="utf-8"))
    route_bank = {domain: np.empty((32768, 48, 8), dtype=np.int16) for domain in DOMAINS}
    route_hashes_ok = True
    for layer in range(48):
        record = hera["artifacts"][str(layer)]
        path = ROOT / record["artifact"]
        route_hashes_ok &= sha256(path) == record["artifact_sha256"]
        tensors = load_file(path)
        for domain in DOMAINS:
            route_bank[domain][:, layer, :] = tensors[f"{domain}_router_ids"]
    curves = {domain: union_curve(routes) for domain, routes in route_bank.items()}

    models_dir = ROOT / "models"
    local_model_directories = sorted(path.name for path in models_dir.iterdir() if path.is_dir())
    k3_candidates = [name for name in local_model_directories if "k3" in name.lower() or "kimi" in name.lower()]
    qwen_path = Path(qwen["snapshot_path"])
    qwen_shards = list(qwen_path.glob("model-*-of-*.safetensors")) if qwen_path.is_dir() else []
    k3_expected = 896 * (1 - (1 - 16 / 896) ** 8)
    supplied_k3_value = 118.6
    hot_actual = hera["hot_experts_by_domain"]
    hot_claimed = {"code": 4168, "multilingual": 4320, "general": 4453, "math": 4823, "instruction": 4957}
    hot_differences = {domain: hot_actual[domain] - hot_claimed[domain] for domain in hot_claimed}
    payload = {
        "kind": "offload_roofline_remaining_proposal_audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "p_d_preregistration_sha256": sha256(PREREG), "hera_sha256": sha256(HERA),
            "qwen_acquisition_sha256": sha256(QWEN_ACQUISITION), "coretail_sha256": sha256(CORETAIL),
            "atomic_full_sha256": sha256(ATOMIC_FULL), "all_48_route_hashes_ok": bool(route_hashes_ok),
        },
        "P_A_QWEN_WALLCLOCK": {
            "status": "blocked_missing_full_bank_gptq_and_runtime",
            "qwen_bf16_checkpoint_verified": qwen["status"] == "complete_verified" and qwen["local_sha256_verified"],
            "qwen_bf16_shards_present": len(qwen_shards),
            "canonical_actual_gptq_experts_present": coretail["inputs"]["canonical_experts"],
            "canonical_actual_gptq_experts_required": 6144,
            "full_bank_gptq_present": coretail["inputs"]["canonical_experts"] == 6144,
            "packed_lfu_async_runtime_present": False,
            "tokens_per_second_measured": False,
            "quality_vram_rss_rollout_gates_measured": False,
        },
        "P_C_FULL_K3": {
            "status": "blocked_missing_k3_checkpoint_and_runtime",
            "local_model_directories": local_model_directories,
            "k3_candidate_directories": k3_candidates,
            "actual_k3_trunk_bytes_measured": False,
            "actual_64_token_k3_decode_measured": False,
        },
        "P_D_SPECULATIVE": {
            "status": "blocked_acceptance_unmeasured",
            "k3_target_present": bool(k3_candidates),
            "h3_is_standalone_autoregressive_drafter": False,
            "h3_candidate_validation_eligible": atomic["candidate_validation_eligible"],
            "small_external_draft_checkpoint_present": False,
            "accepted_tokens_per_pass_measured": False,
            "acceptance_gate_ge_4_tested": False,
            "supplied_k3_uniform_union": supplied_k3_value,
            "recomputed_k3_uniform_union": k3_expected,
            "supplied_absolute_error": abs(k3_expected - supplied_k3_value),
            "recomputed_naive_over_unique_factor": 128 / k3_expected,
            "qwen_actual_top8_union_curves": curves,
            "claim_boundary": "Qwen route union is measured; K3 top-16 route union and speculative acceptance are not.",
        },
        "SOURCE_ANALYSIS_CORRECTIONS": {
            "hera_hot_counts_claimed": hot_claimed,
            "hera_hot_counts_actual": hot_actual,
            "actual_minus_claimed": hot_differences,
            "static_hera_verdict_remains_valid": hera["verdict"] == "static_tier_negative",
            "p_b_cache_claim_now_tested_negative": True,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Resterende offloadvoorstellen — uitvoerbaarheidsaudit", "",
        "- **P-A geblokkeerd:** Qwen BF16 is compleet, maar slechts 16/6.144 echte GPTQ-experts en geen packed LFU/async-runtime zijn aanwezig.",
        "- **Volledige P-C geblokkeerd:** geen lokaal K3-checkpoint, geen gemeten actieve trunkbytes en geen 64-token K3-decode.",
        "- **P-D geblokkeerd:** geen K3-target of werkende autoregressieve drafter; acceptatie is niet meetbaar.", "",
        f"De K3-unieformule geeft {k3_expected:.6f}, niet 118,6; naive/uniek is {128/k3_expected:.4f}×, niet circa 1,08×.", "",
        "Wel gemeten op de echte Qwen top-8-routes (gemiddelde unieke experts per laag bij diepte 8):",
    ]
    for domain in DOMAINS:
        row = curves[domain]["8"]
        lines.append(f"- {domain}: {row['mean']:.3f}/64 ({row['mean_fraction_of_naive_k_times_s']:.3f}); naive/uniek {row['mean_naive_over_unique_factor']:.3f}×.")
    lines.extend(["", "Deze uniemeting zegt niets over speculative acceptatie; P-D blijft daarom niet geslaagd.", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"P_A": payload["P_A_QWEN_WALLCLOCK"]["status"], "P_C": payload["P_C_FULL_K3"]["status"], "P_D": payload["P_D_SPECULATIVE"]["status"], "k3_union_recomputed": k3_expected, "qwen_depth8": {d: curves[d]["8"]["mean"] for d in DOMAINS}}, indent=2))
