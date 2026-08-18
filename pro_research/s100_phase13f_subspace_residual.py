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
RANKS = (128, 256, 512)
FALLBACK_RATES = (0.0, 0.10, 0.25, 0.50)
TOKENS = 64


def capture(rt, rows, tokens, active_split):
    import cupy as cp

    buckets = {"mamba_in": {}, "mamba_out": {}}
    active = {"value": False}
    original_mamba = rt._mamba

    def save(family, layer, value):
        if active["value"] and rt.layer[int(layer)].get(f"{('in' if family == 'mamba_in' else 'out')}_k") == "bf16":
            buckets[family].setdefault(int(layer), []).append(cp.asnumpy(value).astype(np.float32, copy=True))

    def mamba(self, i, out):
        save("mamba_in", i, self.normed)
        original_mamba(i, out)
        save("mamba_out", i, self.gn)

    rt._mamba = types.MethodType(mamba, rt)
    try:
        for row in rows:
            rt.reset()
            nxt = None
            for token in row["prompt_ids"]:
                nxt = rt.step(int(token))
            active["value"] = True
            for _ in range(tokens):
                nxt = rt.step(int(nxt))
            active["value"] = False
    finally:
        rt._mamba = original_mamba
    return buckets


def pool(bucket):
    return np.stack(bucket).astype(np.float32, copy=False) if bucket else np.zeros((0, 0), dtype=np.float32)


def case_metrics(rt, family, layer, cal, val, rank, w_t):
    if cal.shape[0] < rank or val.shape[0] == 0:
        return {"status": "unsupported", "rank": rank}
    _, _, vt = np.linalg.svd(cal, full_matrices=False)
    u = vt[:rank].T.astype(np.float32, copy=False)
    x_cal = np.asarray(cal, dtype=np.float32)
    x_val = np.asarray(val, dtype=np.float32)
    with __import__("torch").no_grad():
        import torch
        U = torch.from_numpy(u).to(device="cuda", dtype=torch.float32)
        W = w_t.float()
        WU = torch.mm(W, U)
        X = torch.from_numpy(x_val).to(device="cuda", dtype=torch.float32)
        ref = torch.mm(X, W.t())
        approx = torch.mm(torch.mm(X, U), WU.t())
        diff = approx - ref
        val_res = np.sum(np.square(x_val - (x_val @ u) @ u.T), axis=1) / np.maximum(np.sum(np.square(x_val), axis=1), 1e-30)
        cal_res = np.sum(np.square(x_cal - (x_cal @ u) @ u.T), axis=1) / np.maximum(np.sum(np.square(x_cal), axis=1), 1e-30)
        output_nrmse = torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(ref, dim=1).clamp_min(1e-12)
        per_token = output_nrmse.detach().cpu().numpy()
        rows = []
        for fallback in FALLBACK_RATES:
            threshold = float(np.quantile(cal_res, 1.0 - fallback)) if fallback else float(np.max(cal_res) + 1e-12)
            fast = val_res <= threshold
            mixed = per_token.copy()
            mixed[~fast] = 0.0  # exact fallback has no approximation error
            rows.append({
                "fallback_rate_target": fallback,
                "calibration_residual_threshold": threshold,
                "validation_fast_fraction": float(fast.mean()),
                "fast_output_nrmse_mean": float(per_token[fast].mean()) if fast.any() else 0.0,
                "fast_output_nrmse_p95": float(np.percentile(per_token[fast], 95)) if fast.any() else 0.0,
                "mixed_output_nrmse_mean_with_exact_fallback": float(mixed.mean()),
            })
        return {
            "status": "measured",
            "family": family,
            "layer": int(layer),
            "rank": int(rank),
            "calibration_rows": int(cal.shape[0]),
            "validation_rows": int(val.shape[0]),
            "validation_residual_energy_mean": float(val_res.mean()),
            "validation_residual_energy_p95": float(np.percentile(val_res, 95)),
            "output_nrmse_mean_no_fallback": float(per_token.mean()),
            "output_nrmse_p95_no_fallback": float(np.percentile(per_token, 95)),
            "gates": rows,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/nemotron_3_5_lightning")
    ap.add_argument("--tokens-per-prompt", type=int, default=TOKENS)
    ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13f/S100_PHASE13F_SUBSPACE_RESIDUAL.json"))
    args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "pro_research"))
    os.environ["LS_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    import torch
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    from s100_phase13b_activation_census import prompts

    cal_prompts, val_prompts = prompts(REPO)
    rt = LightningRuntime(Path(args.model_dir).resolve(), contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    cal = capture(rt, cal_prompts, args.tokens_per_prompt, "calibration")
    val = capture(rt, val_prompts, args.tokens_per_prompt, "validation")
    records = []
    for family, side in (("mamba_in", "in"), ("mamba_out", "out")):
        for layer in sorted(set(cal[family]) & set(val[family])):
            d = rt.layer[int(layer)]
            rows, cols = ((int(rt.proj.size), int(rt.hidden)) if side == "in" else (int(rt.hidden), int(rt.d_inner)))
            w_t = torch.utils.dlpack.from_dlpack(d[f"{side}_w"]).view(torch.bfloat16).reshape(rows, cols).clone()
            for rank in RANKS:
                record = case_metrics(rt, family, layer, pool(cal[family][layer]), pool(val[family][layer]), rank, w_t)
                record["weight_shape"] = [rows, cols]
                record["projected_weight_read_fraction"] = float(rank / cols)
                records.append(record)
            del w_t
            torch.cuda.empty_cache()
            print(f"measured {family} layer={layer}", flush=True)
    result = {
        "kind": "s100_phase13f_subspace_residual_ervf",
        "status": "measured",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(Path(args.model_dir).resolve()),
        "claim_boundary": "causal activation/output component screen; no modified generation path or official quality claim",
        "method": {
            "basis": "origin SVD U on calibration activations",
            "candidate": "(W U)(U^T x)",
            "fallback": "exact W x for residual-gated validation tokens",
            "tested_families": ["BF16 Mamba in/out"],
            "missing": ["persistent GPU gate", "Mamba state refresh", "heldout generation quality", "FP8/NVFP4 layers"],
        },
        "splits": {"calibration": [x["id"] for x in cal_prompts], "validation": [x["id"] for x in val_prompts]},
        "records": records,
        "gates": {"output_quality_green": False, "end_to_end_quality_green": False, "promotion_open": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "records": len(records), "promotion_open": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
