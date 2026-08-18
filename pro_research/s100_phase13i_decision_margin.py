from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
NOISE_SCALES = (0.01, 0.05, 0.10)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", default="models/nemotron_3_5_lightning"); ap.add_argument("--tokens-per-prompt", type=int, default=32); ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13i/S100_PHASE13I_DECISION_MARGIN.json")); args = ap.parse_args()
    sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "pro_research")); os.environ["LS_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    from s100_phase13b_activation_census import prompts
    cal, val = prompts(REPO)
    rt = LightningRuntime(Path(args.model_dir).resolve(), contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False)
    rt.enable_cache(72); rt.load_routed_bank(); rt.deterministic_accum = True
    rows = []; rng = np.random.default_rng(1309)
    for prompt in val:
        rt.reset(); nxt = None
        for token in prompt["prompt_ids"]: nxt = rt.step(int(token))
        for step in range(args.tokens_per_prompt):
            nxt = rt.step(int(nxt)); logits = rt.cp.asnumpy(rt.logits).astype(np.float32, copy=True)
            top = np.argpartition(logits, -2)[-2:]; top = top[np.argsort(logits[top])[::-1]]
            margin = float(logits[top[0]] - logits[top[1]]); scale = float(np.std(logits))
            row = {"prompt": prompt["id"], "step": step, "margin": margin, "normalized_margin": margin / max(scale, 1e-12), "exact_token": int(top[0])}
            for noise_scale in NOISE_SCALES:
                perturbed = logits + rng.normal(0.0, noise_scale * max(scale, 1e-12), size=logits.shape).astype(np.float32)
                row[f"stable_noise_{noise_scale}"] = bool(int(np.argmax(perturbed)) == int(top[0]))
            rows.append(row)
        print(f"measured margin {prompt['id']}", flush=True)
    margins = np.asarray([r["normalized_margin"] for r in rows], dtype=np.float64)
    aggregates = []
    for threshold_quantile in (0.50, 0.75, 0.90):
        threshold = float(np.quantile(margins, threshold_quantile))
        gate = margins >= threshold
        item = {"threshold_quantile": threshold_quantile, "normalized_margin_threshold": threshold, "fast_fraction": float(gate.mean()), "noise": {}}
        for noise_scale in NOISE_SCALES:
            stable = np.asarray([r[f"stable_noise_{noise_scale}"] for r in rows], dtype=bool)
            item["noise"][str(noise_scale)] = {"high_margin_stability": float(stable[gate].mean()), "low_margin_stability": float(stable[~gate].mean()), "high_margin_error_rate": float((~stable[gate]).mean())}
        aggregates.append(item)
    result = {"kind": "s100_phase13i_decision_directed_margin", "status": "measured", "created_utc": datetime.now(timezone.utc).isoformat(), "model_dir": str(Path(args.model_dir).resolve()), "claim_boundary": "logit-margin robustness screen with synthetic controlled perturbations; no approximate layer path or causal gate", "splits": {"validation": [x["id"] for x in val], "calibration_reference": [x["id"] for x in cal]}, "tokens": len(rows), "noise_scales_relative_to_logit_std": list(NOISE_SCALES), "aggregates": aggregates, "gates": {"margin_separates_stability": all(a["noise"]["0.1"]["high_margin_stability"] > a["noise"]["0.1"]["low_margin_stability"] for a in aggregates), "approximate_path_green": False, "promotion_open": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"status": result["status"], "tokens": len(rows), "promotion_open": False}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
