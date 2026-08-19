"""Fresh Lightning transfer screen for DFlash2 suffix and lattice ideas."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import traceback

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    RESULTS, assert_lightning, ensure_results,
    normalize_eager_moe, prompt_manifest,
)

OUT = RESULTS / "S100_LIGHTNING16_DFLASH2_PROXY.json"
CAPTURE = RESULTS / "S100_LIGHTNING16_DFLASH2_CAPTURE.npz"

TOKENS_PER_PROMPT = 72
DRAFT_SLOTS = 7
WINDOWS_PER_PROMPT = 32
TOP_K = 16
RANKS = (64, 128, 192)
GROUP_SIZES = (64, 128)
RIDGE_REL = 1e-3
ALPHA_CLIP = 2.0
SELECTOR_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
SEED = 20260819

@dataclass
class Windows:
    x: np.ndarray
    future: np.ndarray
    labels: np.ndarray
    anchors: np.ndarray
    prompt_index: np.ndarray
    anchor_index: np.ndarray

@dataclass
class BaseModel:
    mean_x: np.ndarray
    basis: np.ndarray
    mean_y: np.ndarray
    coefficients: np.ndarray
    ridge: float

@dataclass
class Correction:
    group_size: int
    mean: np.ndarray
    coefficients: np.ndarray
    ridge: float

def ridge_value(gram):
    scale = float(np.trace(gram) / max(len(gram), 1))
    return max(scale * RIDGE_REL, 1e-6)

def fit_base(x, y, rank):
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    mean_x = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_y = y.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = x - mean_x
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    actual = min(rank, vt.shape[0])
    basis = vt[:actual].T.astype(np.float32)
    z = centered @ basis
    gram = z.T @ z
    ridge = ridge_value(gram)
    lhs = gram + np.eye(actual, dtype=np.float32) * ridge
    coefficients = []
    for slot in range(y.shape[1]):
        rhs = z.T @ (y[:, slot] - mean_y[slot])
        coefficients.append(
            np.linalg.solve(lhs, rhs).astype(np.float32)
        )
    return BaseModel(
        mean_x, basis, mean_y,
        np.stack(coefficients), ridge,
    )

def predict_base(model, x):
    z = (np.asarray(x, np.float32) - model.mean_x) @ model.basis
    prediction = np.einsum(
        "nr,srh->nsh", z, model.coefficients, optimize=True
    )
    prediction += model.mean_y[None]
    return prediction.astype(np.float32), z.astype(np.float32)

def fit_correction(x, y, base, z, group_size):
    n, slots, hidden = y.shape
    if hidden % group_size:
        raise ValueError((hidden, group_size))
    groups = hidden // group_size
    gram = z.T @ z
    ridge = ridge_value(gram)
    lhs = gram + np.eye(len(gram), dtype=np.float32) * ridge
    mean = np.empty((slots, groups), np.float32)
    coefficients = np.empty(
        (slots, z.shape[1], groups), np.float32
    )

    for slot in range(slots):
        previous = x if slot == 0 else base[:, slot - 1]
        residual = y[:, slot] - base[:, slot]
        p = previous.reshape(n, groups, group_size)
        r = residual.reshape(n, groups, group_size)
        target = np.clip(
            np.sum(p * r, axis=2)
            / (np.sum(p * p, axis=2) + 1e-8),
            -ALPHA_CLIP, ALPHA_CLIP,
        )
        mean[slot] = target.mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
        coefficients[slot] = np.linalg.solve(
            lhs, z.T @ (target - mean[slot])
        ).astype(np.float32)
    return Correction(group_size, mean, coefficients, ridge)

def apply_correction(model, x, base, z):
    n, slots, hidden = base.shape
    groups = hidden // model.group_size
    corrected = base.copy()
    alpha_all = np.empty((n, slots, groups), np.float32)
    for slot in range(slots):
        alpha = np.clip(
            model.mean[slot] + z @ model.coefficients[slot],
            -ALPHA_CLIP, ALPHA_CLIP,
        ).astype(np.float32)
        previous = x if slot == 0 else base[:, slot - 1]
        corrected[:, slot] += (
            previous.reshape(n, groups, model.group_size)
            * alpha[:, :, None]
        ).reshape(n, hidden)
        alpha_all[:, slot] = alpha
    return corrected, alpha_all

def hidden_metrics(reference, candidate):
    reference = np.asarray(reference, np.float64)
    candidate = np.asarray(candidate, np.float64)
    rows = []
    for slot in range(reference.shape[1]):
        r = reference[:, slot]
        c = candidate[:, slot]
        d = c - r
        rn = np.linalg.norm(r, axis=1)
        cn = np.linalg.norm(c, axis=1)
        cosine = np.sum(r * c, axis=1) / np.maximum(
            rn * cn, 1e-30
        )
        rows.append({
            "slot": slot + 1,
            "nrmse": float(
                np.linalg.norm(d) / max(np.linalg.norm(r), 1e-30)
            ),
            "mean_cosine": float(np.mean(cosine)),
            "p05_cosine": float(np.percentile(cosine, 5)),
        })
    nrmse = np.asarray([row["nrmse"] for row in rows])
    return {
        "per_slot": rows,
        "mean_nrmse": float(nrmse.mean()),
        "last3_mean_nrmse": float(nrmse[-3:].mean()),
        "nrmse_suffix_slope": float(np.polyfit(
            np.arange(1, len(rows) + 1), nrmse, 1
        )[0]),
    }

def split_prompts():
    from transformers import AutoTokenizer
    manifest = prompt_manifest()
    tokenizer = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    calibration, validation = [], []
    for row in manifest.values():
        item = {
            "id": row["id"],
            "domain": row["domain"],
            "prompt_ids": [
                int(x) for x in tokenizer.encode(
                    row["prompt"], add_special_tokens=False
                )
            ],
        }
        if row["id"].endswith("_01"):
            calibration.append(item)
        elif row["id"].endswith("_02"):
            validation.append(item)
    calibration.sort(key=lambda row: row["id"])
    validation.sort(key=lambda row: row["id"])
    if len(calibration) != 10 or len(validation) != 10:
        raise RuntimeError(
            f"prompt split mismatch {len(calibration)}/{len(validation)}"
        )
    return calibration, validation

def capture(rt, prompts, label):
    import cupy as cp
    sequences = []
    steps = TOKENS_PER_PROMPT + DRAFT_SLOTS
    for pi, row in enumerate(prompts):
        rt.reset()
        next_token = None
        for token in row["prompt_ids"]:
            next_token = rt.step(int(token))
        states = np.empty((steps, rt.hidden), np.float16)
        tokens = np.empty(steps, np.int32)
        for step in range(steps):
            states[step] = cp.asnumpy(rt.normed).astype(np.float16)
            tokens[step] = int(next_token)
            next_token = rt.step(int(next_token))
        sequences.append({
            "id": row["id"],
            "states": states,
            "tokens": tokens,
        })
        print(
            f"16E capture {label} {pi+1:02d}/{len(prompts)} "
            f"{row['id']}",
            flush=True,
        )
    return sequences

def make_windows(sequences):
    x, future, labels, anchors = [], [], [], []
    prompt_index, anchor_index = [], []
    for pi, sequence in enumerate(sequences):
        states = sequence["states"]
        tokens = sequence["tokens"]
        available = len(states) - DRAFT_SLOTS
        indices = np.unique(np.linspace(
            0, available - 1,
            min(WINDOWS_PER_PROMPT, available),
            dtype=np.int32,
        ))
        for start in indices:
            x.append(states[start])
            future.append(
                states[start + 1:start + 1 + DRAFT_SLOTS]
            )
            labels.append(
                tokens[start + 1:start + 1 + DRAFT_SLOTS]
            )
            anchors.append(tokens[start])
            prompt_index.append(pi)
            anchor_index.append(start)
    return Windows(
        np.stack(x).astype(np.float32),
        np.stack(future).astype(np.float32),
        np.stack(labels).astype(np.int32),
        np.asarray(anchors, np.int32),
        np.asarray(prompt_index, np.int16),
        np.asarray(anchor_index, np.int16),
    )

def subset(windows, mask):
    return Windows(
        windows.x[mask],
        windows.future[mask],
        windows.labels[mask],
        windows.anchors[mask],
        windows.prompt_index[mask],
        windows.anchor_index[mask],
    )

def lm_topk(rt, hidden, label):
    import cupy as cp
    n, slots, width = hidden.shape
    ids = np.empty((n, slots, TOP_K), np.int32)
    scores = np.empty((n, slots, TOP_K), np.float32)
    x = cp.empty(width, cp.float32)
    logits = cp.empty(rt.vocab, cp.float32)
    for row in range(n):
        for slot in range(slots):
            x.set(np.ascontiguousarray(
                hidden[row, slot], np.float32
            ))
            if rt.lm_head_kind == "nvfp4":
                rt.fused.gemv_into(
                    logits,
                    rt.lm_head_codes,
                    rt.lm_head_scales,
                    x,
                    rt.lm_head_g,
                    rt.vocab,
                    rt.hidden,
                )
            else:
                rt.k.mv_bf16(
                    logits, rt.lm_head, x,
                    rt.vocab, rt.hidden,
                )
            candidates = cp.argpartition(logits, -TOP_K)[-TOP_K:]
            candidates = candidates[
                cp.argsort(-logits[candidates])
            ]
            ids[row, slot] = cp.asnumpy(candidates)
            scores[row, slot] = cp.asnumpy(logits[candidates])
        if (row + 1) % 32 == 0 or row + 1 == n:
            print(
                f"16E LM {label}: {row+1}/{n}",
                flush=True,
            )
    return ids, scores

def prefix_lengths(path, labels):
    return np.cumprod(path == labels, axis=1).sum(axis=1)

def candidate_metrics(ids, labels):
    top1 = ids[:, :, 0]
    covered = np.any(ids == labels[:, :, None], axis=2)
    independent = prefix_lengths(top1, labels)
    oracle = np.cumprod(covered, axis=1).sum(axis=1)
    return {
        "per_slot": [
            {
                "slot": slot + 1,
                "top1_accuracy": float(
                    np.mean(top1[:, slot] == labels[:, slot])
                ),
                "topk_recall": float(
                    np.mean(covered[:, slot])
                ),
            }
            for slot in range(labels.shape[1])
        ],
        "mean_top1_accuracy": float(np.mean(top1 == labels)),
        "mean_topk_recall": float(np.mean(covered)),
        "mean_acceptance_independent_including_anchor": float(
            1.0 + np.mean(independent)
        ),
        "mean_acceptance_oracle_lattice_including_anchor": float(
            1.0 + np.mean(oracle)
        ),
        "oracle_selector_headroom_tokens": float(
            np.mean(oracle - independent)
        ),
        "full_block_candidate_coverage_fraction": float(
            np.mean(oracle == labels.shape[1])
        ),
    }

def bf16_to_float(raw):
    value = np.asarray(raw, np.uint16).astype(np.uint32)
    return (value << np.uint32(16)).view(np.float32)

def embedding_rows(rt, token_ids):
    unique = np.unique(token_ids.astype(np.int64))
    if getattr(rt, "embed_host", None) is not None:
        table = rt.embed_host.reshape(rt.vocab, rt.hidden)
        rows = bf16_to_float(table[unique])
    else:
        import cupy as cp
        table = rt.embed.reshape(rt.vocab, rt.hidden)
        rows = bf16_to_float(cp.asnumpy(table[unique]))
    rows /= np.maximum(
        np.linalg.norm(rows, axis=1, keepdims=True), 1e-8
    )
    return {
        int(token): rows[index].astype(np.float32)
        for index, token in enumerate(unique)
    }

def normalized_unary(scores):
    return (
        scores - scores.mean(axis=-1, keepdims=True)
    ) / np.maximum(scores.std(axis=-1, keepdims=True), 1e-5)

def selector_path(ids, scores, anchors, embedding, weight):
    unary = normalized_unary(scores)
    n, slots, top_k = ids.shape
    paths = np.empty((n, slots), np.int32)
    for row in range(n):
        back = np.empty((slots, top_k), np.int16)
        first = ids[row, 0]
        previous = embedding[int(anchors[row])]
        current = np.stack([
            embedding[int(token)] for token in first
        ])
        dp = unary[row, 0] + weight * (current @ previous)
        back[0].fill(-1)
        for slot in range(1, slots):
            prev_ids = ids[row, slot - 1]
            cur_ids = ids[row, slot]
            prev = np.stack([
                embedding[int(token)] for token in prev_ids
            ])
            cur = np.stack([
                embedding[int(token)] for token in cur_ids
            ])
            transition = prev @ cur.T
            total = (
                dp[:, None] + unary[row, slot][None, :]
                + weight * transition
            )
            back[slot] = np.argmax(total, axis=0)
            dp = np.max(total, axis=0)
        index = int(np.argmax(dp))
        for slot in range(slots - 1, -1, -1):
            paths[row, slot] = int(ids[row, slot, index])
            if slot:
                index = int(back[slot, index])
    return paths

def selector_eval(rt, cal_ids, cal_scores, cal, val_ids, val_scores, val):
    all_tokens = np.concatenate([
        cal.anchors, val.anchors,
        cal_ids.reshape(-1), val_ids.reshape(-1),
    ])
    embedding = embedding_rows(rt, all_tokens)
    trials = []
    for weight in SELECTOR_LAMBDAS:
        path = selector_path(
            cal_ids, cal_scores, cal.anchors,
            embedding, weight,
        )
        trials.append({
            "weight": weight,
            "calibration_acceptance": float(
                1.0 + np.mean(prefix_lengths(path, cal.labels))
            ),
        })
    selected = max(
        trials,
        key=lambda row: (row["calibration_acceptance"], -row["weight"]),
    )
    path = selector_path(
        val_ids, val_scores, val.anchors,
        embedding, selected["weight"],
    )
    selected_prefix = prefix_lengths(path, val.labels)
    independent = prefix_lengths(
        val_ids[:, :, 0], val.labels
    )
    covered = np.any(
        val_ids == val.labels[:, :, None], axis=2
    )
    oracle = np.cumprod(covered, axis=1).sum(axis=1)
    headroom = float(np.mean(oracle - independent))
    gain = float(np.mean(selected_prefix - independent))
    return {
        "kind": "embedding_transition_dp_proxy",
        "trials": trials,
        "selected_weight": selected["weight"],
        "validation_acceptance_including_anchor": float(
            1.0 + np.mean(selected_prefix)
        ),
        "validation_gain_vs_independent_tokens": gain,
        "validation_oracle_headroom_tokens": headroom,
        "fraction_of_oracle_headroom_recovered": float(
            gain / max(headroom, 1e-9)
        ),
    }

def main():
    ensure_results()
    payload = {
        "kind": "s100_lightning16_dflash2_proxy",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": (
            "fresh Lightning target-state transfer proxy; "
            "not a trained DFlash2 drafter"
        ),
    }
    bundle = None
    try:
        ident = assert_lightning()
        calibration_prompts, validation_prompts = split_prompts()
        bundle = build()
        rt = bundle.rt
        rt._graph = None
        rt.graph_mode = False
        normalize_eager_moe(rt)

        calibration_sequences = capture(
            rt, calibration_prompts, "calibration"
        )
        validation_sequences = capture(
            rt, validation_prompts, "validation"
        )
        calibration = make_windows(calibration_sequences)
        validation = make_windows(validation_sequences)
        np.savez_compressed(
            CAPTURE,
            calibration_x=calibration.x.astype(np.float16),
            calibration_future=calibration.future.astype(np.float16),
            calibration_labels=calibration.labels,
            calibration_anchors=calibration.anchors,
            validation_x=validation.x.astype(np.float16),
            validation_future=validation.future.astype(np.float16),
            validation_labels=validation.labels,
            validation_anchors=validation.anchors,
        )

        select_mask = calibration.anchor_index % 4 == 3
        train = subset(calibration, ~select_mask)
        select = subset(calibration, select_mask)
        trials = []

        for rank in RANKS:
            for group_size in GROUP_SIZES:
                base_model = fit_base(
                    train.x, train.future, rank
                )
                train_base, train_z = predict_base(
                    base_model, train.x
                )
                correction = fit_correction(
                    train.x, train.future,
                    train_base, train_z, group_size,
                )
                select_base, select_z = predict_base(
                    base_model, select.x
                )
                select_corrected, _ = apply_correction(
                    correction, select.x,
                    select_base, select_z,
                )
                base_score = hidden_metrics(
                    select.future, select_base
                )
                corrected_score = hidden_metrics(
                    select.future, select_corrected
                )
                trials.append({
                    "rank": rank,
                    "group_size": group_size,
                    "base": base_score,
                    "corrected": corrected_score,
                })

        selected = min(
            trials,
            key=lambda row: (
                row["corrected"]["mean_nrmse"],
                row["rank"],
                row["group_size"],
            ),
        )
        base_model = fit_base(
            calibration.x, calibration.future,
            selected["rank"],
        )
        cal_base, cal_z = predict_base(
            base_model, calibration.x
        )
        correction = fit_correction(
            calibration.x, calibration.future,
            cal_base, cal_z, selected["group_size"],
        )
        cal_corrected, _ = apply_correction(
            correction, calibration.x,
            cal_base, cal_z,
        )
        val_base, val_z = predict_base(
            base_model, validation.x
        )
        val_corrected, alpha = apply_correction(
            correction, validation.x,
            val_base, val_z,
        )

        base_hidden = hidden_metrics(
            validation.future, val_base
        )
        corrected_hidden = hidden_metrics(
            validation.future, val_corrected
        )
        val_base_ids, _ = lm_topk(
            rt, val_base, "validation/base"
        )
        cal_ids, cal_scores = lm_topk(
            rt, cal_corrected, "calibration/corrected"
        )
        val_ids, val_scores = lm_topk(
            rt, val_corrected, "validation/corrected"
        )
        base_candidates = candidate_metrics(
            val_base_ids, validation.labels
        )
        corrected_candidates = candidate_metrics(
            val_ids, validation.labels
        )
        selector = selector_eval(
            rt,
            cal_ids, cal_scores, calibration,
            val_ids, val_scores, validation,
        )

        base_last3 = base_hidden["last3_mean_nrmse"]
        corrected_last3 = corrected_hidden[
            "last3_mean_nrmse"
        ]
        base_recall = float(np.mean([
            row["topk_recall"]
            for row in base_candidates["per_slot"][-3:]
        ]))
        corrected_recall = float(np.mean([
            row["topk_recall"]
            for row in corrected_candidates["per_slot"][-3:]
        ]))
        suffix_signal = bool(
            corrected_last3 <= 0.90 * base_last3
            or corrected_recall >= base_recall + 0.02
        )
        lattice_signal = bool(
            corrected_candidates[
                "mean_acceptance_oracle_lattice_including_anchor"
            ] >= 3.0
            and corrected_candidates[
                "oracle_selector_headroom_tokens"
            ] >= 0.75
        )
        selector_signal = bool(
            selector[
                "validation_gain_vs_independent_tokens"
            ] >= 0.10
            and selector[
                "fraction_of_oracle_headroom_recovered"
            ] >= 0.10
        )
        transfer = bool(
            lattice_signal and (suffix_signal or selector_signal)
        )

        payload.update({
            "status": "measured",
            "identity": ident,
            "dataset": {
                "calibration_windows": len(calibration.x),
                "validation_windows": len(validation.x),
                "tokens_per_prompt": TOKENS_PER_PROMPT,
                "draft_slots": DRAFT_SLOTS,
                "capture": str(CAPTURE),
            },
            "configuration_trials": trials,
            "selected_configuration": {
                "rank": selected["rank"],
                "group_size": selected["group_size"],
                "alpha_abs_p95": float(
                    np.percentile(np.abs(alpha), 95)
                ),
            },
            "validation_hidden": {
                "base": base_hidden,
                "corrected": corrected_hidden,
                "last3_relative_nrmse": float(
                    corrected_last3 / max(base_last3, 1e-12)
                ),
            },
            "validation_candidates": {
                "base": base_candidates,
                "corrected": corrected_candidates,
                "last3_topk_recall_delta": (
                    corrected_recall - base_recall
                ),
            },
            "selector_proxy": selector,
            "gates": {
                "SUFFIX_DECAY_CORRECTION_SIGNAL_OPEN": suffix_signal,
                "CANDIDATE_LATTICE_HEADROOM_OPEN": lattice_signal,
                "SELECTOR_PROXY_SIGNAL_OPEN": selector_signal,
                "DFLASH2_LIGHTNING_SIGNAL_OPEN": transfer,
            },
            "completed_utc": utc_now(),
        })
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
    finally:
        if bundle is not None:
            try:
                bundle.restore_combined()
                bundle.restore_sel()
            except Exception:
                pass

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "selected_configuration": payload.get(
            "selected_configuration"
        ),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
