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
TOP_K = (32, 64, 128, 256, 512)
TOKENS_PER_PROMPT = 64


def load_prompt_splits(repo: Path):
    from transformers import AutoTokenizer

    rows = json.loads((repo / "pro_research" / "S100_PHASE3_PROMPTS.json").read_text())["prompts"]
    model = Path(os.environ.get("LS_MODEL_DIR", "models/nemotron_3_5_lightning"))
    if not model.is_absolute():
        model = repo / model
    tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=True, use_fast=True)
    splits = {"calibration": [], "validation": []}
    for row in rows:
        ids = tok.encode(row["prompt"], add_special_tokens=False)
        if not ids:
            continue
        split = "calibration" if row["id"].endswith("_01") else "validation" if row["id"].endswith("_02") else None
        if split:
            splits[split].append({"id": row["id"], "prompt_ids": [int(x) for x in ids]})
    if len(splits["calibration"]) != 10 or len(splits["validation"]) != 10:
        raise RuntimeError("expected ten prompts per split")
    return splits


def empty_acc():
    return {"samples": 0, "cosine": [], "norm_ratio": [], "delta_norm": [], "topk_energy": {str(k): [] for k in TOP_K}, "int8_nrmse": [], "int4_nrmse": []}


def add_delta(acc, current: np.ndarray, previous: np.ndarray) -> None:
    current = current.astype(np.float32, copy=False)
    previous = previous.astype(np.float32, copy=False)
    delta = current - previous
    cn = float(np.linalg.norm(current))
    pn = float(np.linalg.norm(previous))
    dn = float(np.linalg.norm(delta))
    acc["samples"] += 1
    acc["cosine"].append(float(np.dot(current, previous) / max(cn * pn, 1e-12)))
    acc["norm_ratio"].append(dn / max(pn, 1e-12))
    acc["delta_norm"].append(dn)
    energy = np.sort(np.square(delta.astype(np.float64)))[::-1]
    total = max(float(energy.sum()), 1e-30)
    for k in TOP_K:
        acc["topk_energy"][str(k)].append(float(energy[: min(k, energy.size)].sum() / total))
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    for bits, key in ((8, "int8_nrmse"), (4, "int4_nrmse")):
        qmax = (1 << (bits - 1)) - 1
        scale = max(max_abs / qmax, 1e-12)
        recon = np.clip(np.rint(delta / scale), -qmax - 1, qmax).astype(np.float32) * scale
        acc[key].append(float(np.linalg.norm(recon - delta) / max(dn, 1e-12)))


def capture(rt, rows, tokens):
    import cupy as cp

    family_names = ("mamba_in", "mamba_out", "attention_input", "moe_input", "final_norm")
    acc = {family: empty_acc() for family in family_names}
    state = {"active": False, "previous": {}}
    originals = (rt._mamba, rt._attention, rt._moe)

    def save(family: str, layer: int, value) -> None:
        if not state["active"]:
            return
        key = (family, int(layer))
        current = cp.asnumpy(value).astype(np.float32, copy=True)
        previous = state["previous"].get(key)
        if previous is not None:
            add_delta(acc[family], current, previous)
        state["previous"][key] = current

    def mamba(self, i, out):
        save("mamba_in", i, self.normed)
        originals[0](i, out)
        save("mamba_out", i, self.gn)

    def attention(self, i, out):
        save("attention_input", i, self.normed)
        originals[1](i, out)

    def moe(self, i, out):
        save("moe_input", i, self.normed)
        return originals[2](i, out)

    rt._mamba = types.MethodType(mamba, rt)
    rt._attention = types.MethodType(attention, rt)
    rt._moe = types.MethodType(moe, rt)
    try:
        for row in rows:
            rt.reset()
            state["previous"].clear()
            nxt = None
            for token in row["prompt_ids"]:
                nxt = rt.step(int(token))
            state["active"] = True
            for _ in range(tokens):
                nxt = rt.step(int(nxt))
                save("final_norm", 0, rt.normed)
            state["active"] = False
    finally:
        rt._mamba, rt._attention, rt._moe = originals
    return acc


def summarize(acc: dict) -> dict:
    def stats(values):
        if not values:
            return {"count": 0}
        x = np.asarray(values, dtype=np.float64)
        return {"count": int(x.size), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95))}
    return {
        "samples": int(acc["samples"]),
        "cosine": stats(acc["cosine"]),
        "norm_ratio": stats(acc["norm_ratio"]),
        "delta_norm": stats(acc["delta_norm"]),
        "topk_energy": {k: stats(v) for k, v in acc["topk_energy"].items()},
        "int8_nrmse": stats(acc["int8_nrmse"]),
        "int4_nrmse": stats(acc["int4_nrmse"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/nemotron_3_5_lightning")
    ap.add_argument("--tokens-per-prompt", type=int, default=TOKENS_PER_PROMPT)
    ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13c/S100_PHASE13C_TEMPORAL.json"))
    args = ap.parse_args()
    os.environ["LS_MODEL_DIR"] = str(Path(args.model_dir).resolve())
    sys.path.insert(0, str(REPO / "src"))
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    splits = load_prompt_splits(REPO)
    rt = LightningRuntime(Path(args.model_dir).resolve(), contexts_max=4096, embed_on_host=True, fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    report = {}
    for split, rows in splits.items():
        print(f"collect {split} {len(rows)} prompts x {args.tokens_per_prompt} tokens", flush=True)
        report[split] = {family: summarize(values) for family, values in capture(rt, rows, args.tokens_per_prompt).items()}
    result = {
        "kind": "s100_phase13c_temporal_delta_census",
        "status": "measured",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(Path(args.model_dir).resolve()),
        "tokens_per_prompt": int(args.tokens_per_prompt),
        "claim_boundary": "activation-delta census; no W delta output energy, model substitution, speed, or quality claim",
        "method": {"delta": "x_t - x_(t-1) on generated tokens, first generated token excluded", "topk": list(TOP_K), "quantization": "per-vector symmetric int8/int4 reconstruction", "missing": ["output energy under W delta", "sparse-column kernel", "official validation fidelity"]},
        "splits": report,
        "gate": {"output_energy_gate_measured": False, "promotion_open": False, "reason": "W delta output energy is required by the preregistration and was not inferred from coordinate energy"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "promotion_open": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
