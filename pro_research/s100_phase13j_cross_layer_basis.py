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
MAX_ROWS = 512


def capture(rt, prompts, tokens):
    import cupy as cp
    buckets = {"attention": {}, "moe": {}}
    active = {"value": False}
    old_attn, old_moe = rt._attention, rt._moe

    def save(family, layer, value):
        if active["value"]: buckets[family].setdefault(int(layer), []).append(cp.asnumpy(value).astype(np.float32, copy=True))
    def attn(self, i, out): save("attention", i, self.normed); return old_attn(i, out)
    def moe(self, i, out): save("moe", i, self.normed); return old_moe(i, out)
    rt._attention = types.MethodType(attn, rt); rt._moe = types.MethodType(moe, rt)
    try:
        for row in prompts:
            rt.reset(); nxt = None
            for token in row["prompt_ids"]: nxt = rt.step(int(token))
            active["value"] = True
            for _ in range(tokens): nxt = rt.step(int(nxt))
            active["value"] = False
    finally:
        rt._attention, rt._moe = old_attn, old_moe
    return buckets


def pooled(parts):
    x = np.concatenate([np.stack(v) for v in parts.values() if v], axis=0).astype(np.float32, copy=False)
    if x.shape[0] > MAX_ROWS: x = x[np.linspace(0, x.shape[0] - 1, MAX_ROWS, dtype=np.int64)]
    return x


def basis(x, rank):
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:rank].T.astype(np.float32, copy=False)


def residual(x, u):
    z = x @ u; r = x - z @ u.T
    return np.sum(r * r, axis=1) / np.maximum(np.sum(x * x, axis=1), 1e-30)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", default="models/nemotron_3_5_lightning"); ap.add_argument("--tokens-per-prompt", type=int, default=16); ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13j/S100_PHASE13J_CROSS_LAYER_BASIS.json")); args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "pro_research")); os.environ["LS_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    from s100_phase13b_activation_census import prompts
    cal_prompts, val_prompts = prompts(REPO)
    rt = LightningRuntime(Path(args.model_dir).resolve(), contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False); rt.enable_cache(72); rt.load_routed_bank(); rt.deterministic_accum = True
    cal = capture(rt, cal_prompts, args.tokens_per_prompt); val = capture(rt, val_prompts, args.tokens_per_prompt)
    cal_family = {f: pooled(cal[f]) for f in ("attention", "moe")}; val_family = {f: pooled(val[f]) for f in ("attention", "moe")}
    cal_all = pooled({(family, layer): rows for family, family_rows in cal.items() for layer, rows in family_rows.items()}); val_all = pooled({(family, layer): rows for family, family_rows in val.items() for layer, rows in family_rows.items()})
    records = []
    source_count = len(set(cal["attention"]) | set(cal["moe"]))
    for rank in RANKS:
        separate = {f: basis(cal_family[f], rank) for f in cal_family}
        shared = basis(cal_all, rank)
        for family in ("attention", "moe"):
            records.append({"rank": rank, "family": family, "separate_basis_residual_mean": float(residual(val_family[family], separate[family]).mean()), "shared_cross_layer_basis_residual_mean": float(residual(val_family[family], shared).mean()), "separate_basis_rows": int(cal_family[family].shape[0]), "shared_basis_rows": int(cal_all.shape[0])})
    result = {"kind": "s100_phase13j_cross_layer_activation_basis", "status": "measured", "created_utc": datetime.now(timezone.utc).isoformat(), "model_dir": str(Path(args.model_dir).resolve()), "claim_boundary": "shared-input-basis residual screen; no WU output quality, runtime fusion, or end-to-end claim", "method": {"families": ["attention_input", "moe_input"], "basis": "origin SVD", "max_rows_per_pool": MAX_ROWS, "source_layer_count": source_count, "missing": ["shared WU output", "kernel amortization", "heldout quality"]}, "records": records, "gates": {"shared_basis_input_residual_green": False, "promotion_open": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"status": result["status"], "records": len(records), "promotion_open": False}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
