from __future__ import annotations

import gc
import json
import statistics
import traceback

import numpy as np

from common import REPO, require_model_dir, utc_now, write_json_atomic
from s100_phase17_kernels import Phase17Kernels
from s100_phase17_mamba_block import capture_sequences, nrmse, timed
from s100_phase19_residual_projection import (
    FP8ProjectionBlock,
    torch_bf16_terms,
)

OUT = REPO / "pro_research" / "results" / "s100_phase19_full_layer.json"
H = 4
TERMS = 2
CORR_FULL = 1e-4
CORR_STATE = 5e-5
CORR_CONV = 5e-6


def residual_fp8_project(cp, block, d, side, x, out):
    W = d["in_w8"] if side == "in" else d["out_w8"]
    scale = d["in_s"] if side == "in" else d["out_s"]
    block.apply_residual2(W, x, out, scale)


def run_layer(rt, kernels, fp8_block, torch, cp, layer, cap):
    d = rt.layer[layer]
    if d["in_k"] != "fp8_tensor" or d["out_k"] != "fp8_tensor":
        raise RuntimeError(f"layer {layer} is not FP8/FP8: {d['in_k']}/{d['out_k']}")

    Hh, P, N = int(rt.m_heads), int(rt.m_hdim), int(rt.n_state)
    hpg, G = int(rt.hpg), int(rt.m_heads) // int(rt.hpg)
    conv_dim, d_inner = int(rt.conv_dim), int(rt.d_inner)
    proj_size, group_size = int(rt.proj.size), d_inner // int(rt.n_groups)
    normed = cp.asarray(cap["normed"][:H])
    exact_out = np.asarray(cap["out"][:H])
    conv0 = cp.asarray(cap["conv0"])
    ssm0 = cp.asarray(cap["ssm0"])

    xbc_offset = d_inner
    dtr_offset = d_inner + conv_dim
    B_offset = d_inner
    C_offset = d_inner + int(rt.n_groups) * N

    base_proj = cp.empty((H, proj_size), cp.float32)
    base_out = cp.empty((H, int(rt.hidden)), cp.float32)
    base_conv = cp.empty_like(conv0)
    base_ssm = cp.empty_like(ssm0)
    base_convo = cp.empty((H, conv_dim), cp.float32)
    base_dt = cp.empty((H, Hh), cp.float32)
    base_y = cp.empty((H, d_inner), cp.float32)
    base_gn = cp.empty((H, d_inner), cp.float32)

    cand_proj = cp.empty_like(base_proj)
    cand_out = cp.empty_like(base_out)
    cand_convo = cp.empty_like(base_convo)
    cand_dt = cp.empty_like(base_dt)
    cand_y = cp.empty_like(base_y)
    cand_gn = cp.empty_like(base_gn)
    cand_conv_final = cp.empty_like(conv0)
    cand_states = cp.empty((H, int(ssm0.size)), cp.float32)
    cand_dx = cp.empty((H, d_inner), cp.float32)
    cand_decay = cp.empty((H, Hh), cp.float32)

    def reset_base():
        cp.copyto(base_conv, conv0)
        cp.copyto(base_ssm, ssm0)

    def baseline():
        for t in range(H):
            import s100_phase17_mamba_block as p17
            p17.in_proj(rt, d, base_proj[t], normed[t])
            rt.k.conv_step(
                base_convo[t], base_conv,
                base_proj[t, xbc_offset:xbc_offset + conv_dim],
                d["conv_w"], d["conv_b"], conv_dim, int(rt.conv_k),
            )
            rt.k.dt_activate(
                base_dt[t], base_proj[t, dtr_offset:dtr_offset + Hh],
                d["dt_bias"], Hh, 0.0, 3.4e38,
            )
            row = base_convo[t]
            rt.k.ssm_step(
                base_y[t], base_ssm, row[:d_inner],
                row[B_offset:B_offset + G * N],
                row[C_offset:C_offset + G * N], base_dt[t],
                d["A_log"], d["D"], Hh, P, N, hpg,
            )
            rt.k.gated_norm(
                base_gn[t], base_y[t], base_proj[t, :d_inner],
                d["m_norm"], d_inner, group_size, float(rt.eps),
            )
            p17.out_proj(rt, d, base_out[t], base_gn[t])

    def candidate():
        residual_fp8_project(cp, fp8_block, d, "in", normed, cand_proj)
        kernels.block_conv(
            H, conv0, cand_proj, d["conv_w"], d["conv_b"],
            cand_convo, cand_conv_final, conv_dim, int(rt.conv_k),
            x_stride=proj_size, x_offset=xbc_offset,
        )
        kernels.block_dt(
            H, cand_proj, d["dt_bias"], cand_dt, Hh,
            dtr_stride=proj_size, dtr_offset=dtr_offset,
        )
        kernels.prepare(
            H, cand_convo, cand_dt, d["A_log"], cand_dx, cand_decay,
            Hh, P, x_stride=conv_dim, x_offset=0,
        )
        kernels.scan(
            "serial", H, ssm0, cand_dx, cand_convo, cand_decay,
            cand_states, Hh, P, N, hpg,
            b_stride=conv_dim, b_offset=B_offset,
        )
        kernels.y(
            H, cand_states, cand_convo, cand_convo, d["D"], cand_y,
            Hh, P, N, hpg, c_stride=conv_dim, c_offset=C_offset,
            x_stride=conv_dim, x_offset=0,
        )
        kernels.gated(
            H, cand_y, cand_proj, d["m_norm"], cand_gn,
            d_inner, group_size, float(rt.eps),
            z_stride=proj_size, z_offset=0,
        )
        residual_fp8_project(cp, fp8_block, d, "out", cand_gn, cand_out)

    reset_base()
    baseline()
    cp.cuda.get_current_stream().synchronize()
    base_np = cp.asnumpy(base_out)
    baseline_capture = {
        "output_nrmse": nrmse(base_np, exact_out),
        "conv_final_nrmse": nrmse(cp.asnumpy(base_conv), cap["conv_post"][H - 1]),
        "ssm_final_nrmse": nrmse(cp.asnumpy(base_ssm), cap["ssm_post"][H - 1]),
    }
    candidate()
    cp.cuda.get_current_stream().synchronize()
    full_corr = {
        "output_nrmse": nrmse(cp.asnumpy(cand_out), base_np),
        "conv_final_nrmse": nrmse(cp.asnumpy(cand_conv_final), cp.asnumpy(base_conv)),
        "ssm_final_nrmse": nrmse(cp.asnumpy(cand_states[H - 1]), cp.asnumpy(base_ssm)),
    }
    full_corr["pass"] = (
        full_corr["output_nrmse"] <= CORR_FULL
        and full_corr["conv_final_nrmse"] <= CORR_CONV
        and full_corr["ssm_final_nrmse"] <= CORR_STATE
    )
    base_t = timed(cp, baseline, reset_base)
    cand_t = timed(cp, candidate)
    result = {
        "layer": layer,
        "H": H,
        "residual_terms": TERMS,
        "in_format": d["in_k"], "out_format": d["out_k"],
        "baseline_reproduction": baseline_capture,
        "correctness": full_corr,
        "baseline": base_t, "candidate": cand_t,
        "speedup": base_t["median_ms"] / cand_t["median_ms"],
    }
    print(
        f"P19 FULL layer={layer} H={H}: speed={result['speedup']:.3f}x "
        f"out_nrmse={full_corr['output_nrmse']:.2e} pass={full_corr['pass']}",
        flush=True,
    )
    return result


def main():
    payload = {
        "kind": "s100_phase19_full_layer",
        "status": "started",
        "model_id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "H": H, "residual_terms": TERMS,
        "claim_boundary": "full H=4 Mamba-layer retest; not end-to-end decode",
        "started_utc": utc_now(),
    }
    try:
        import torch
        import cupy as cp
        from transformers import AutoTokenizer
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        model_dir = require_model_dir()
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        rt = LightningRuntime(model_dir, contexts_max=512,
                              embed_on_host=True, fp8_kv=True, verbose=False)
        rt.load_routed_bank()
        rt.deterministic_accum = True
        layers = [int(x) for x in rt.mamba_layers]
        chosen = sorted({layers[0], layers[len(layers) // 2], layers[-1]})
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                             trust_remote_code=True, use_fast=True)
        prompt_ids = tok.encode(
            "The history of computing and artificial intelligence",
            add_special_tokens=False,
        )
        captures = capture_sequences(rt, chosen, prompt_ids)
        rt.bank = {}
        gc.collect()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        kernels = Phase17Kernels()
        fp8_block = FP8ProjectionBlock(cp)
        results = [run_layer(rt, kernels, fp8_block, torch, cp, layer, captures[layer])
                   for layer in chosen]
        h4_green = all(
            r["correctness"]["pass"] and r["speedup"] >= 1.10
            for r in results
        )
        payload.update({
            "status": "measured",
            "model_dir": str(model_dir),
            "config_identity": {
                "architectures": cfg.get("architectures"),
                "model_type": cfg.get("model_type"),
                "num_hidden_layers": cfg.get("num_hidden_layers"),
                "hidden_size": cfg.get("hidden_size"),
            },
            "sampled_layers": chosen,
            "results": results,
            "PHASE20_FULL_BLOCK_VERIFIER_OPEN": bool(h4_green),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "sampled_layers": payload.get("sampled_layers"),
        "PHASE20_FULL_BLOCK_VERIFIER_OPEN": payload.get(
            "PHASE20_FULL_BLOCK_VERIFIER_OPEN"
        ),
        "layer_H4": [
            {"layer": r["layer"], "speedup": r["speedup"],
             "output_nrmse": r["correctness"]["output_nrmse"],
             "pass": r["correctness"]["pass"]}
            for r in payload.get("results", [])
        ],
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
