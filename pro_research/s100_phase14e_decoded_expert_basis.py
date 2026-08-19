from __future__ import annotations

import json
import traceback

import numpy as np

from common import REPO, write_json_atomic, utc_now
from diag_s100_d4_weight_only_dense import E4_FULL, quantize_matrix
from moe_dev_batched import UP_CODE, UP_SCALE
from scale_resident_kernels import (
    DOWN_PANEL_BYTES, PANEL_STRIDE, ROWHALF, HIDDEN, INTER, NPANEL,
)
from s100_phase10a_runtime import build
from s100_phase14_common import (
    RESULTS, E2, bf16_round, metrics, ensure_results
)

OUT = RESULTS / "S100_PHASE14E_DECODED_BASIS.json"
RANKS = (4, 8, 16, 32)
RESIDUAL_BLOCK_FRACTIONS = (0.0625, 0.125, 0.25)
SAMPLE_ROWS = 256
SAMPLE_COLS = 256
BLOCK = 16


def decode_q(q):
    rows, cols = q["shape"]
    codes = np.asarray(q["codes"], np.uint8).reshape(rows, cols // 2)
    scales = np.asarray(q["scales"], np.uint8).reshape(rows, cols // 16)
    nib = np.empty((rows, cols), np.uint8)
    nib[:, 0::2] = codes & 15
    nib[:, 1::2] = codes >> 4
    sf = E4_FULL[scales.astype(np.int32)].astype(np.float32)
    sf *= np.float32(q["global"])
    return E2[nib.astype(np.int32)] * np.repeat(sf, 16, axis=1)


def quantize_bases(bases):
    decoded = []
    bytes_total = 0
    for basis in bases:
        q = quantize_matrix(np.ascontiguousarray(basis, np.float32), "CEIL")
        q["shape"] = basis.shape
        decoded.append(decode_q(q))
        bytes_total += int(q["bytes"])
    return np.stack(decoded), bytes_total


def decode_up_sample(bank, expert_count, inter, hidden, row_idx, col_idx):
    codes = np.asarray(bank["up_codes"], np.uint8).reshape(
        expert_count, inter, hidden // 2
    )
    scales = np.asarray(bank["up_scales"], np.uint8).reshape(
        expert_count, inter, hidden // 16
    )
    globals_up = np.asarray(bank["globals"], np.float32)[:, 1]
    byte_col = col_idx // 2
    scale_col = col_idx // 16
    high = (col_idx & 1).astype(bool)
    output = np.empty((expert_count, len(row_idx), len(col_idx)), np.float32)
    for expert in range(expert_count):
        packed = codes[expert, row_idx[:, None], byte_col[None, :]]
        nib = np.where(high[None, :], packed >> 4, packed & 15).astype(np.int32)
        scale_code = scales[
            expert, row_idx[:, None], scale_col[None, :]
        ].astype(np.int32)
        output[expert] = (
            E2[nib]
            * E4_FULL[scale_code].astype(np.float32)
            * globals_up[expert]
        )
    return output


def decode_down_sample(bank, expert_count, hidden, inter, row_idx, col_idx):
    records = np.asarray(bank["down_pm"], np.uint8).reshape(
        expert_count, DOWN_PANEL_BYTES
    )
    globals_down = np.asarray(bank["globals"], np.float32)[:, 0]
    panel = col_idx // 16
    in_panel = col_idx & 15
    half_row = row_idx // 2
    high = (row_idx & 1).astype(bool)
    scale_offsets = panel[None, :] * PANEL_STRIDE + row_idx[:, None]
    code_offsets = (
        panel[None, :] * PANEL_STRIDE + hidden
        + in_panel[None, :] * ROWHALF + half_row[:, None]
    )
    output = np.empty((expert_count, len(row_idx), len(col_idx)), np.float32)
    for expert in range(expert_count):
        rec = records[expert]
        packed = rec[code_offsets]
        nib = np.where(high[:, None], packed >> 4, packed & 15).astype(np.int32)
        scale_code = rec[scale_offsets].astype(np.int32)
        output[expert] = (
            E2[nib]
            * E4_FULL[scale_code].astype(np.float32)
            * globals_down[expert]
        )
    return output


def residual_block_mask(residual, energy, fraction):
    experts, rows, cols = residual.shape
    blocks = cols // BLOCK
    weighted = residual * np.sqrt(np.maximum(energy, 1e-30))[None, None, :]
    scores = np.sum(
        weighted.reshape(experts, rows, blocks, BLOCK) ** 2,
        axis=3,
    )
    keep = max(1, int(round(scores.size * fraction)))
    selected = np.argpartition(scores.reshape(-1), -keep)[-keep:]
    mask_blocks = np.zeros(scores.size, bool)
    mask_blocks[selected] = True
    mask_blocks = mask_blocks.reshape(scores.shape)
    return np.repeat(mask_blocks, BLOCK, axis=2)


def screen(W, cal_energy, val_vectors, val_ids=None):
    """Fit fixed expert-axis basis and evaluate actual sampled GEMV output."""
    weighted = W * np.sqrt(np.maximum(cal_energy, 1e-30))[None, None, :]
    flat = weighted.reshape(W.shape[0], -1)
    U, singular, Vt = np.linalg.svd(flat, full_matrices=False)

    if val_ids is None:
        exact_output = np.einsum("erc,sc->esr", W, val_vectors, optimize=True)
    else:
        # val_vectors [samples, top_k, cols], ids [samples, top_k]
        n, k, _ = val_vectors.shape
        exact_output = np.empty((n, k, W.shape[1]), np.float32)
        for sample in range(n):
            for slot in range(k):
                exact_output[sample, slot] = (
                    W[int(val_ids[sample, slot])] @ val_vectors[sample, slot]
                )

    candidates = []
    for rank in RANKS:
        coefficients = bf16_round(U[:, :rank] * singular[:rank])
        basis_weighted = Vt[:rank].reshape(rank, W.shape[1], W.shape[2])
        basis = basis_weighted / np.sqrt(
            np.maximum(cal_energy, 1e-30)
        )[None, None, :]
        basis_q, basis_bytes_sample = quantize_bases(basis)
        main = np.einsum("er,rij->eij", coefficients, basis_q, optimize=True)
        residual = W - main

        for fraction in RESIDUAL_BLOCK_FRACTIONS:
            mask = residual_block_mask(residual, cal_energy, fraction)
            candidate_w = main + residual * mask
            if val_ids is None:
                candidate_output = np.einsum(
                    "erc,sc->esr", candidate_w, val_vectors, optimize=True
                )
                ref_rows = exact_output.reshape(-1, W.shape[1])
                cand_rows = candidate_output.reshape(-1, W.shape[1])
            else:
                n, k, _ = val_vectors.shape
                candidate_output = np.empty_like(exact_output)
                for sample in range(n):
                    for slot in range(k):
                        candidate_output[sample, slot] = (
                            candidate_w[int(val_ids[sample, slot])]
                            @ val_vectors[sample, slot]
                        )
                ref_rows = exact_output.reshape(-1, W.shape[1])
                cand_rows = candidate_output.reshape(-1, W.shape[1])

            score = metrics(ref_rows, cand_rows)
            basis_ratio = rank / W.shape[0]
            coefficient_ratio = (
                W.shape[0] * rank * 2
                / max(W.shape[0] * W.shape[1] * W.shape[2] * 0.5625, 1)
            )
            bitmap_ratio = 1.0 / (BLOCK * 8.0)
            byte_ratio = basis_ratio + coefficient_ratio + fraction + bitmap_ratio
            gates = {
                "byte_ratio_le_0_70": byte_ratio <= 0.70,
                "output_nrmse_le_0_05": score["nrmse"] <= 0.05,
                "cosine_ge_0_999": score["mean_cosine"] >= 0.999,
                "finite": score["finite"],
            }
            candidates.append({
                "rank": rank,
                "residual_block_fraction": fraction,
                "estimated_full_byte_ratio": float(byte_ratio),
                "sample_basis_nvfp4_bytes": int(basis_bytes_sample),
                "validation_sampled_gemv": score,
                "gates": gates,
                "pass": all(gates.values()),
            })
    passed = [x for x in candidates if x["pass"]]
    selected = min(
        passed,
        key=lambda x: (
            x["estimated_full_byte_ratio"],
            x["validation_sampled_gemv"]["nrmse"],
        ),
    ) if passed else None
    return candidates, selected


def main():
    ensure_results()
    payload = {
        "kind": "s100_phase14e_decoded_expert_basis",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        capture_path = RESULTS / "S100_PHASE14_MOE_INPUTS.npz"
        if not capture_path.exists():
            raise FileNotFoundError("14B2 MoE capture is required before 14E2")

        with np.load(capture_path) as cap:
            captures = {key: cap[key] for key in cap.files}
            available = sorted({
                int(key.split("_")[1])
                for key in cap.files
                if key.startswith("moe_") and key.endswith("_cal")
            })
        if len(available) < 3:
            raise RuntimeError(f"expected 3 MoE input captures, got {available}")

        bundle = build()
        rt = bundle.rt
        chosen = [available[0], available[len(available)//2], available[-1]]
        records = []

        for layer in chosen:
            cal_x = captures[f"moe_{layer}_cal"].astype(np.float32)
            val_x = captures[f"moe_{layer}_val"].astype(np.float32)
            cal_hidden_energy = np.mean(cal_x ** 2, axis=0)
            up_cols = np.sort(np.argsort(-cal_hidden_energy)[:SAMPLE_COLS].astype(np.int32))
            up_rows = np.linspace(0, int(rt.moe_inter)-1, SAMPLE_ROWS, dtype=np.int32)
            Wup = decode_up_sample(
                rt.bank[layer], int(rt.n_experts), int(rt.moe_inter),
                int(rt.hidden), up_rows, up_cols,
            )
            up_candidates, up_selected = screen(
                Wup, cal_hidden_energy[up_cols], val_x[:, up_cols]
            )

            act_cal_key = f"moe_{layer}_act_cal"
            act_val_key = f"moe_{layer}_act_val"
            id_val_key = f"moe_{layer}_id_val"
            if not all(key in captures for key in (act_cal_key, act_val_key, id_val_key)):
                down = {
                    "status": "incomplete",
                    "reason": "exposed routed activation/id capture unavailable",
                    "candidates": [],
                    "selected": None,
                }
            else:
                cal_act = captures[act_cal_key].astype(np.float32)
                val_act = captures[act_val_key].astype(np.float32)
                val_ids = captures[id_val_key].astype(np.int32)
                cal_down_energy = np.mean(cal_act ** 2, axis=(0, 1))
                down_cols = np.sort(
                    np.argsort(-cal_down_energy)[:SAMPLE_COLS].astype(np.int32)
                )
                down_rows = np.linspace(
                    0, int(rt.hidden)-1, SAMPLE_ROWS, dtype=np.int32
                )
                Wdown = decode_down_sample(
                    rt.bank[layer], int(rt.n_experts), int(rt.hidden),
                    int(rt.moe_inter), down_rows, down_cols,
                )
                down_candidates, down_selected = screen(
                    Wdown,
                    cal_down_energy[down_cols],
                    val_act[:, :, down_cols],
                    val_ids=val_ids,
                )
                down = {
                    "status": "measured",
                    "sample_rows": SAMPLE_ROWS,
                    "sample_cols": SAMPLE_COLS,
                    "candidates": down_candidates,
                    "selected": down_selected,
                }

            record = {
                "layer": layer,
                "calibration_rows": int(cal_x.shape[0]),
                "validation_rows": int(val_x.shape[0]),
                "up": {
                    "status": "measured",
                    "sample_rows": SAMPLE_ROWS,
                    "sample_cols": SAMPLE_COLS,
                    "candidates": up_candidates,
                    "selected": up_selected,
                },
                "down": down,
                "layer_pass": bool(
                    up_selected is not None
                    and down.get("selected") is not None
                ),
            }
            records.append(record)
            print(
                f"14E2 layer {layer}: up={up_selected is not None} "
                f"down={down.get('selected') is not None}",
                flush=True,
            )

        complete = all(
            row["up"]["status"] == "measured"
            and row["down"]["status"] == "measured"
            for row in records
        )
        open_all = bool(complete and all(row["layer_pass"] for row in records))
        payload.update({
            "status": "measured" if complete else "incomplete",
            "layers": chosen,
            "method": (
                "actual decoded NVFP4 up/down expert samples; calibration "
                "activation-weighted expert-axis SVD; BF16 coefficients; "
                "NVFP4-quantized shared bases; frozen exact NVFP4 residual "
                "blocks; validation sampled GEMV output error"
            ),
            "records": records,
            "EXPERT_BASIS_RUNTIME_BUILD_OPEN": open_all if complete else None,
            "completed_utc": utc_now(),
        })
        bundle.restore_combined()
        bundle.restore_sel()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "layers": payload.get("layers"),
        "EXPERT_BASIS_RUNTIME_BUILD_OPEN": payload.get(
            "EXPERT_BASIS_RUNTIME_BUILD_OPEN"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") in {"measured", "incomplete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
