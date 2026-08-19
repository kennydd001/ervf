from __future__ import annotations

import json
import traceback

import numpy as np

from common import REPO, write_json_atomic, utc_now
from diag_s100_d4_weight_only_dense import E4_FULL, quantize_matrix
from scale_resident_kernels import (
    DOWN_PANEL_BYTES, PANEL_STRIDE, ROWHALF,
)
from s100_phase10a_runtime import build
from s100_phase14r_common import (
    RESULTS, E2, bf16_round, metrics, ensure_results
)

OUT = RESULTS / "S100_PHASE14R_EXPERT_BASIS.json"
CAPTURE = RESULTS / "S100_PHASE14R_MOE_INPUTS.npz"
RANKS = (4, 8, 16, 32)
RESIDUAL_BLOCK_FRACTIONS = (0.0625, 0.125, 0.25)
SAMPLE_ROWS = 256
SAMPLE_COLS = 256
BLOCK = 16

def decode_quantized(record):
    rows, cols = record["shape"]
    codes = np.asarray(
        record["codes"], np.uint8
    ).reshape(rows, cols // 2)
    scales = np.asarray(
        record["scales"], np.uint8
    ).reshape(rows, cols // 16)
    nibble = np.empty((rows, cols), np.uint8)
    nibble[:, 0::2] = codes & 15
    nibble[:, 1::2] = codes >> 4
    scale = E4_FULL[scales.astype(np.int32)].astype(np.float32)
    scale *= np.float32(record["global"])
    return E2[nibble.astype(np.int32)] * np.repeat(scale, 16, axis=1)

def quantize_bases(bases):
    decoded = []
    byte_count = 0
    for basis in bases:
        record = quantize_matrix(
            np.ascontiguousarray(basis, np.float32), "CEIL"
        )
        record["shape"] = basis.shape
        decoded.append(decode_quantized(record))
        byte_count += int(record["bytes"])
    return np.stack(decoded), byte_count

def decode_up_sample(bank, experts, inter, hidden, rows, cols):
    code = np.asarray(bank["up_codes"], np.uint8).reshape(
        experts, inter, hidden // 2
    )
    scale = np.asarray(bank["up_scales"], np.uint8).reshape(
        experts, inter, hidden // 16
    )
    global_scale = np.asarray(bank["globals"], np.float32)[:, 1]
    byte_col = cols // 2
    scale_col = cols // 16
    high = (cols & 1).astype(bool)
    output = np.empty(
        (experts, len(rows), len(cols)), np.float32
    )
    for expert in range(experts):
        packed = code[expert, rows[:, None], byte_col[None, :]]
        nibble = np.where(
            high[None, :], packed >> 4, packed & 15
        ).astype(np.int32)
        scale_code = scale[
            expert, rows[:, None], scale_col[None, :]
        ].astype(np.int32)
        output[expert] = (
            E2[nibble]
            * E4_FULL[scale_code].astype(np.float32)
            * global_scale[expert]
        )
    return output

def decode_down_sample(bank, experts, hidden, rows, cols):
    records = np.asarray(bank["down_pm"], np.uint8).reshape(
        experts, DOWN_PANEL_BYTES
    )
    global_scale = np.asarray(bank["globals"], np.float32)[:, 0]
    panel = cols // 16
    in_panel = cols & 15
    half_row = rows // 2
    high = (rows & 1).astype(bool)
    scale_offsets = panel[None, :] * PANEL_STRIDE + rows[:, None]
    code_offsets = (
        panel[None, :] * PANEL_STRIDE + hidden
        + in_panel[None, :] * ROWHALF + half_row[:, None]
    )
    output = np.empty(
        (experts, len(rows), len(cols)), np.float32
    )
    for expert in range(experts):
        record = records[expert]
        packed = record[code_offsets]
        nibble = np.where(
            high[:, None], packed >> 4, packed & 15
        ).astype(np.int32)
        scale_code = record[scale_offsets].astype(np.int32)
        output[expert] = (
            E2[nibble]
            * E4_FULL[scale_code].astype(np.float32)
            * global_scale[expert]
        )
    return output

def residual_mask(residual, energy, fraction):
    experts, rows, cols = residual.shape
    block_count = cols // BLOCK
    weighted = residual * np.sqrt(
        np.maximum(energy, 1e-30)
    )[None, None, :]
    scores = np.sum(
        weighted.reshape(experts, rows, block_count, BLOCK) ** 2,
        axis=3,
    )
    keep = max(1, int(round(scores.size * fraction)))
    selected = np.argpartition(scores.reshape(-1), -keep)[-keep:]
    block_mask = np.zeros(scores.size, bool)
    block_mask[selected] = True
    return np.repeat(
        block_mask.reshape(scores.shape), BLOCK, axis=2
    )

def screen(weights, calibration_energy, validation_vectors, ids=None):
    weighted = weights * np.sqrt(
        np.maximum(calibration_energy, 1e-30)
    )[None, None, :]
    flat = weighted.reshape(weights.shape[0], -1)
    U, singular, Vt = np.linalg.svd(flat, full_matrices=False)

    if ids is None:
        exact = np.einsum(
            "erc,sc->esr", weights, validation_vectors,
            optimize=True,
        )
    else:
        sample_count, top_k, _ = validation_vectors.shape
        exact = np.empty(
            (sample_count, top_k, weights.shape[1]), np.float32
        )
        for sample in range(sample_count):
            for slot in range(top_k):
                exact[sample, slot] = (
                    weights[int(ids[sample, slot])]
                    @ validation_vectors[sample, slot]
                )

    candidates = []
    for rank in RANKS:
        coefficients = bf16_round(U[:, :rank] * singular[:rank])
        basis_weighted = Vt[:rank].reshape(
            rank, weights.shape[1], weights.shape[2]
        )
        basis = basis_weighted / np.sqrt(
            np.maximum(calibration_energy, 1e-30)
        )[None, None, :]
        basis_quantized, sample_basis_bytes = quantize_bases(basis)
        main = np.einsum(
            "er,rij->eij", coefficients, basis_quantized,
            optimize=True,
        )
        residual = weights - main

        for fraction in RESIDUAL_BLOCK_FRACTIONS:
            mask = residual_mask(
                residual, calibration_energy, fraction
            )
            candidate_weights = main + residual * mask

            if ids is None:
                candidate = np.einsum(
                    "erc,sc->esr",
                    candidate_weights, validation_vectors,
                    optimize=True,
                )
            else:
                sample_count, top_k, _ = validation_vectors.shape
                candidate = np.empty_like(exact)
                for sample in range(sample_count):
                    for slot in range(top_k):
                        candidate[sample, slot] = (
                            candidate_weights[int(ids[sample, slot])]
                            @ validation_vectors[sample, slot]
                        )

            score = metrics(
                exact.reshape(-1, weights.shape[1]),
                candidate.reshape(-1, weights.shape[1]),
            )
            basis_ratio = rank / weights.shape[0]
            coefficient_ratio = (
                weights.shape[0] * rank * 2
                / max(
                    weights.shape[0] * weights.shape[1]
                    * weights.shape[2] * 0.5625,
                    1,
                )
            )
            bitmap_ratio = 1.0 / (BLOCK * 8.0)
            byte_ratio = (
                basis_ratio + coefficient_ratio
                + fraction + bitmap_ratio
            )
            gates = {
                "byte_ratio_le_0_70": byte_ratio <= 0.70,
                "nrmse_le_0_05": score["nrmse"] <= 0.05,
                "cosine_ge_0_999": score["mean_cosine"] >= 0.999,
                "finite": score["finite"],
            }
            candidates.append({
                "rank": rank,
                "residual_block_fraction": fraction,
                "estimated_full_byte_ratio": float(byte_ratio),
                "sample_basis_nvfp4_bytes": int(sample_basis_bytes),
                "validation_sampled_gemv": score,
                "gates": gates,
                "pass": all(gates.values()),
            })

    passed = [row for row in candidates if row["pass"]]
    selected = min(
        passed,
        key=lambda row: (
            row["estimated_full_byte_ratio"],
            row["validation_sampled_gemv"]["nrmse"],
        ),
    ) if passed else None
    return candidates, selected

def main():
    ensure_results()
    payload = {
        "kind": "s100_phase14r_expert_basis",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        if not CAPTURE.exists():
            raise FileNotFoundError(
                "repaired 14B2 MoE capture is required"
            )

        with np.load(CAPTURE) as capture:
            values = {key: capture[key] for key in capture.files}
            available = sorted({
                int(key.split("_")[1])
                for key in capture.files
                if key.startswith("moe_") and key.endswith("_cal")
            })
        if len(available) < 3:
            raise RuntimeError(
                f"expected three MoE captures, got {available}"
            )

        bundle = build()
        rt = bundle.rt
        chosen = [
            available[0],
            available[len(available)//2],
            available[-1],
        ]
        records = []

        for layer in chosen:
            cal_x = values[f"moe_{layer}_cal"].astype(np.float32)
            val_x = values[f"moe_{layer}_val"].astype(np.float32)
            hidden_energy = np.mean(cal_x ** 2, axis=0)
            up_cols = np.sort(
                np.argsort(-hidden_energy)[:SAMPLE_COLS]
                .astype(np.int32)
            )
            up_rows = np.linspace(
                0, int(rt.moe_inter) - 1,
                SAMPLE_ROWS, dtype=np.int32,
            )
            up_weights = decode_up_sample(
                rt.bank[layer], int(rt.n_experts),
                int(rt.moe_inter), int(rt.hidden),
                up_rows, up_cols,
            )
            up_candidates, up_selected = screen(
                up_weights,
                hidden_energy[up_cols],
                val_x[:, up_cols],
            )

            required = (
                f"moe_{layer}_act_cal",
                f"moe_{layer}_act_val",
                f"moe_{layer}_id_val",
            )
            if not all(key in values for key in required):
                down = {
                    "status": "incomplete",
                    "reason": "routed activation/id capture unavailable",
                    "selected": None,
                    "candidates": [],
                }
            else:
                cal_act = values[
                    f"moe_{layer}_act_cal"
                ].astype(np.float32)
                val_act = values[
                    f"moe_{layer}_act_val"
                ].astype(np.float32)
                val_ids = values[
                    f"moe_{layer}_id_val"
                ].astype(np.int32)
                down_energy = np.mean(cal_act ** 2, axis=(0, 1))
                down_cols = np.sort(
                    np.argsort(-down_energy)[:SAMPLE_COLS]
                    .astype(np.int32)
                )
                down_rows = np.linspace(
                    0, int(rt.hidden) - 1,
                    SAMPLE_ROWS, dtype=np.int32,
                )
                down_weights = decode_down_sample(
                    rt.bank[layer], int(rt.n_experts),
                    int(rt.hidden), down_rows, down_cols,
                )
                down_candidates, down_selected = screen(
                    down_weights,
                    down_energy[down_cols],
                    val_act[:, :, down_cols],
                    ids=val_ids,
                )
                down = {
                    "status": "measured",
                    "sample_rows": SAMPLE_ROWS,
                    "sample_cols": SAMPLE_COLS,
                    "candidates": down_candidates,
                    "selected": down_selected,
                }

            records.append({
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
            })
            print(
                f"14E2 layer={layer} "
                f"up={up_selected is not None} "
                f"down={down.get('selected') is not None}",
                flush=True,
            )

        complete = all(
            record["up"]["status"] == "measured"
            and record["down"]["status"] == "measured"
            for record in records
        )
        open_all = bool(
            complete and all(record["layer_pass"] for record in records)
        )
        payload.update({
            "status": "measured" if complete else "incomplete",
            "layers": chosen,
            "records": records,
            "EXPERT_BASIS_RUNTIME_BUILD_OPEN": (
                open_all if complete else None
            ),
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
