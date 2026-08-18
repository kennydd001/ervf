from __future__ import annotations

import argparse
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RANKS = (128, 256, 384, 512, 768, 1024)
FALLBACK_RATES = (0.0, 0.10, 0.25, 0.50, 1.0)
TOKENS_PER_PROMPT = 64


def prompts(repo: Path) -> tuple[list[dict], list[dict]]:
    from transformers import AutoTokenizer

    rows = json.loads((repo / "pro_research" / "S100_PHASE3_PROMPTS.json").read_text())["prompts"]
    model = Path(os.environ.get("LS_MODEL_DIR", "models/nemotron_3_5_lightning"))
    if not model.is_absolute():
        model = repo / model
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=True, use_fast=True)
    cal, val = [], []
    for row in rows:
        ids = tok.encode(row["prompt"], add_special_tokens=False)
        if not ids:
            continue
        item = {"id": row["id"], "domain": row["domain"], "prompt_ids": [int(x) for x in ids]}
        if row["id"].endswith("_01"):
            cal.append(item)
        elif row["id"].endswith("_02"):
            val.append(item)
    if len(cal) != 10 or len(val) != 10:
        raise RuntimeError(f"expected 10 _01 and 10 _02 prompts, got {len(cal)} / {len(val)}")
    return cal, val


def capture_split(rt, rows: list[dict], tokens: int) -> dict[str, dict[int, list[np.ndarray]]]:
    import cupy as cp

    buckets: dict[str, dict[int, list[np.ndarray]]] = {
        "mamba_in": {}, "mamba_out": {}, "attention_input": {}, "moe_input": {}, "final_norm": {0: []}
    }
    active = {"value": False}
    original_mamba = rt._mamba
    original_attention = rt._attention
    original_moe = rt._moe

    def save(family: str, layer: int, value) -> None:
        if not active["value"]:
            return
        buckets.setdefault(family, {}).setdefault(int(layer), []).append(
            cp.asnumpy(value).astype(np.float16, copy=True)
        )

    def mamba(self, i, out):
        save("mamba_in", i, self.normed)
        original_mamba(i, out)
        save("mamba_out", i, self.gn)

    def attention(self, i, out):
        save("attention_input", i, self.normed)
        original_attention(i, out)

    def moe(self, i, out):
        save("moe_input", i, self.normed)
        return original_moe(i, out)

    rt._mamba = types.MethodType(mamba, rt)
    rt._attention = types.MethodType(attention, rt)
    rt._moe = types.MethodType(moe, rt)
    try:
        for row in rows:
            rt.reset()
            nxt = None
            for token in row["prompt_ids"]:
                nxt = rt.step(int(token))
            active["value"] = True
            for _ in range(tokens):
                nxt = rt.step(int(nxt))
                save("final_norm", 0, rt.normed)
            active["value"] = False
    finally:
        rt._mamba = original_mamba
        rt._attention = original_attention
        rt._moe = original_moe
    return buckets


def pool(bucket: dict[int, list[np.ndarray]], max_rows: int = 1024) -> np.ndarray:
    parts = [np.stack(rows) for _, rows in sorted(bucket.items()) if rows]
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    x = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if x.shape[0] > max_rows:
        indices = np.linspace(0, x.shape[0] - 1, max_rows, dtype=np.int64)
        x = x[indices]
    return x


def fit_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] < 2:
        return np.zeros((x.shape[1], 0), dtype=np.float32), np.zeros(0, dtype=np.float64)
    # The origin-subspace is intentional: SR-ERVF uses U U^T x, not a centered PCA.
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt.astype(np.float32, copy=False).T
    energy = np.square(singular.astype(np.float64))
    return basis, energy


def residual_energy(x: np.ndarray, basis: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0 or basis.shape[1] < rank:
        return np.ones(x.shape[0], dtype=np.float64)
    u = basis[:, :rank]
    coeff = x @ u
    residual = x - coeff @ u.T
    num = np.square(residual.astype(np.float64)).sum(axis=1)
    den = np.square(x.astype(np.float64)).sum(axis=1)
    return num / np.maximum(den, 1e-30)


def dense_shapes(rt) -> dict[str, tuple[int, int, int, int]]:
    hidden, d_inner, shared = int(rt.hidden), int(rt.d_inner), int(rt.shared_inter)
    hq, kv = int(rt.n_heads * rt.head_dim), int(rt.kv_dim)
    # (input dimension, original aggregate bytes, original weight count, basis count)
    return {
        "mamba_in": (hidden, sum(int(rt.proj.size) * hidden * (1 if rt.layer[i]["in_k"] == "fp8_tensor" else 2) for i in rt.mamba_layers), sum(int(rt.proj.size) * hidden for _ in rt.mamba_layers), len(rt.mamba_layers)),
        "mamba_out": (d_inner, sum(hidden * d_inner * (1 if rt.layer[i]["out_k"] == "fp8_tensor" else 2) for i in rt.mamba_layers), sum(hidden * d_inner for _ in rt.mamba_layers), len(rt.mamba_layers)),
        "attention_input": (hidden, len(rt.attn_layers) * (hq + 2 * kv + hq) * hidden * 2, len(rt.attn_layers) * (hq + 2 * kv + hq) * hidden, len(rt.attn_layers)),
        "moe_input": (hidden, len(rt.moe_layers) * (shared * hidden + hidden * shared) * 1, len(rt.moe_layers) * (shared * hidden + hidden * shared), len(rt.moe_layers)),
    }


def family_metrics(name: str, cal: np.ndarray, val: np.ndarray, shapes: dict) -> dict:
    basis, singular_energy = fit_basis(cal)
    rows = []
    input_dim, original_bytes, _, basis_count = shapes.get(name, (cal.shape[1], 0, 0, 1))
    for rank in RANKS:
        if rank > min(cal.shape):
            rows.append({"rank": rank, "status": "unsupported_sample_count"})
            continue
        cal_res = residual_energy(cal, basis, rank)
        val_res = residual_energy(val, basis, rank)
        # WU uses the original matrix element width; U is stored as FP16 for the estimate.
        candidate_bytes = original_bytes * 0.0
        if original_bytes:
            # Aggregate rows are recovered from bytes/weights; this preserves the
            # family-specific FP8/BF16 mix without pretending to be a kernel cost.
            candidate_bytes = (original_bytes * rank / input_dim) + (input_dim * rank * 2 * basis_count)
        raw_reduction = 1.0 - candidate_bytes / original_bytes if original_bytes else float("nan")
        gates = {}
        for fallback in FALLBACK_RATES:
            threshold = float(np.quantile(cal_res, 1.0 - fallback))
            fast_fraction = float(np.mean(val_res <= threshold))
            gates[str(fallback)] = {
                "calibration_threshold_residual_energy": threshold,
                "validation_fast_fraction": fast_fraction,
                "projected_expected_byte_reduction": fast_fraction * raw_reduction,
            }
        rows.append({
            "rank": rank,
            "status": "measured",
            "calibration_rows": int(cal.shape[0]),
            "validation_rows": int(val.shape[0]),
            "calibration_residual_energy_mean": float(cal_res.mean()),
            "validation_residual_energy_mean": float(val_res.mean()),
            "validation_residual_energy_p95": float(np.percentile(val_res, 95)),
            "candidate_bytes_estimate": float(candidate_bytes),
            "original_bytes_estimate": int(original_bytes),
            "projected_dense_byte_reduction": float(raw_reduction),
            "gates": gates,
        })
    return {
        "family": name,
        "input_dim": int(input_dim),
        "basis_count": int(basis_count),
        "calibration_rows": int(cal.shape[0]),
        "validation_rows": int(val.shape[0]),
        "basis_rank_limit": int(min(cal.shape)),
        "singular_energy_top_fraction": float(singular_energy[: min(1024, singular_energy.size)].sum() / max(singular_energy.sum(), 1e-30)),
        "ranks": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/nemotron_3_5_lightning")
    ap.add_argument("--tokens-per-prompt", type=int, default=TOKENS_PER_PROMPT)
    ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13b/S100_PHASE13B_ACTIVATION.json"))
    args = ap.parse_args()
    repo = REPO
    sys.path.insert(0, str(repo / "src"))
    os.environ["LS_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    cal_prompts, val_prompts = prompts(repo)
    rt = LightningRuntime(Path(args.model_dir).resolve(), contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    print(f"collect calibration {len(cal_prompts)} prompts x {args.tokens_per_prompt} tokens", flush=True)
    cal = capture_split(rt, cal_prompts, args.tokens_per_prompt)
    print(f"collect validation {len(val_prompts)} prompts x {args.tokens_per_prompt} tokens", flush=True)
    val = capture_split(rt, val_prompts, args.tokens_per_prompt)
    shapes = dense_shapes(rt)
    families = ["mamba_in", "mamba_out", "attention_input", "moe_input", "final_norm"]
    metrics = {}
    for family in families:
        cal_x = pool(cal.get(family, {}))
        val_x = pool(val.get(family, {}))
        if family == "final_norm":
            metrics[family] = {"family": family, "status": "activation_only", "calibration_rows": int(cal_x.shape[0]), "validation_rows": int(val_x.shape[0]), "ranks": []}
        else:
            metrics[family] = family_metrics(family, cal_x, val_x, shapes)
        print(f"analyzed {family}: {cal_x.shape} -> {val_x.shape}", flush=True)
    result = {
        "kind": "s100_phase13b_activation_census",
        "status": "measured",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(Path(args.model_dir).resolve()),
        "claim_boundary": "activation-subspace screen; no substituted model output, token fidelity, or speed claim",
        "splits": {"calibration": [x["id"] for x in cal_prompts], "validation": [x["id"] for x in val_prompts]},
        "tokens_per_prompt": int(args.tokens_per_prompt),
        "method": {"basis": "uncentered origin SVD U; projection U U^T x", "pooling": "deterministic max-1024 rows per family after layer pooling", "missing": ["W U output NRMSE", "top1/top5 token effects", "official validation fidelity", "GPU bytes and latency"]},
        "families": metrics,
        "screen_gate": {"promotion_open": False, "reason": "official output and end-to-end quality gates were not measured in this screen"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "promotion_open": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
