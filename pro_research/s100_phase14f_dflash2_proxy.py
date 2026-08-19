from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import REPO, write_json_atomic, utc_now
from s100_phase13b_activation_census import prompts
from s100_phase14_common import RESULTS, ensure_results

OUT = RESULTS / "S100_PHASE14F_DFLASH2_PROXY.json"
CAPTURE = RESULTS / "S100_PHASE14F_DFLASH2_CAPTURE.npz"

TOKENS_PER_PROMPT = 72
DRAFT_SLOTS = 7                 # block size 8: one anchor + seven draft tokens
MAX_WINDOWS_PER_PROMPT = 32
TOP_K = 16
RANKS = (64, 128, 192)
GROUP_SIZES = (64, 128)
RIDGE_REL = 1e-3
ALPHA_CLIP = 2.0
SELECTOR_PROXY_RANK = 32
SELECTOR_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SEED = 20260819


@dataclass
class WindowSet:
    x: np.ndarray                  # [N, H]
    future_hidden: np.ndarray      # [N, S, H]
    labels: np.ndarray             # [N, S]
    anchor_ids: np.ndarray         # [N]
    prompt_index: np.ndarray       # [N]
    anchor_index: np.ndarray       # [N]


@dataclass
class BaseModel:
    mean_x: np.ndarray
    basis: np.ndarray
    mean_y: np.ndarray
    coefficients: np.ndarray
    ridge: float


@dataclass
class CorrectionModel:
    group_size: int
    alpha_mean: np.ndarray
    alpha_coefficients: np.ndarray
    ridge: float


def relative_ridge(gram: np.ndarray) -> float:
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    return max(scale * RIDGE_REL, 1e-6)


def fit_base(x: np.ndarray, y: np.ndarray, rank: int) -> BaseModel:
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    mean_x = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_y = y.mean(axis=0, dtype=np.float64).astype(np.float32)
    xc = x - mean_x
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    actual = min(int(rank), vt.shape[0])
    if actual < 2:
        raise RuntimeError(f"insufficient rank: requested={rank}, actual={actual}")
    basis = vt[:actual].T.astype(np.float32, copy=False)
    z = xc @ basis
    gram = z.T @ z
    ridge = relative_ridge(gram)
    lhs = gram + np.eye(actual, dtype=np.float32) * ridge
    coefficients = []
    for slot in range(y.shape[1]):
        rhs = z.T @ (y[:, slot] - mean_y[slot])
        coefficients.append(np.linalg.solve(lhs, rhs).astype(np.float32))
    return BaseModel(
        mean_x=mean_x,
        basis=basis,
        mean_y=mean_y,
        coefficients=np.stack(coefficients),
        ridge=ridge,
    )


def predict_base(model: BaseModel, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, np.float32)
    z = (x - model.mean_x) @ model.basis
    # coefficients: [S, R, H]
    prediction = np.einsum("nr,srh->nsh", z, model.coefficients, optimize=True)
    prediction += model.mean_y[None]
    return prediction.astype(np.float32), z.astype(np.float32)


def fit_correction(
    x: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    z: np.ndarray,
    group_size: int,
) -> CorrectionModel:
    n, slots, hidden = y.shape
    if hidden % group_size:
        raise ValueError(f"hidden {hidden} is not divisible by group size {group_size}")
    groups = hidden // group_size
    gram = z.T @ z
    ridge = relative_ridge(gram)
    lhs = gram + np.eye(gram.shape[0], dtype=np.float32) * ridge
    alpha_mean = np.empty((slots, groups), np.float32)
    alpha_coefficients = np.empty((slots, z.shape[1], groups), np.float32)

    for slot in range(slots):
        previous = x if slot == 0 else base[:, slot - 1]
        residual = y[:, slot] - base[:, slot]
        p = previous.reshape(n, groups, group_size)
        r = residual.reshape(n, groups, group_size)
        numerator = np.sum(p * r, axis=2)
        denominator = np.sum(p * p, axis=2) + 1e-8
        target = np.clip(numerator / denominator, -ALPHA_CLIP, ALPHA_CLIP)
        mean = target.mean(axis=0, dtype=np.float64).astype(np.float32)
        alpha_mean[slot] = mean
        rhs = z.T @ (target - mean)
        alpha_coefficients[slot] = np.linalg.solve(lhs, rhs).astype(np.float32)

    return CorrectionModel(
        group_size=group_size,
        alpha_mean=alpha_mean,
        alpha_coefficients=alpha_coefficients,
        ridge=ridge,
    )


def apply_correction(
    model: CorrectionModel,
    x: np.ndarray,
    base: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n, slots, hidden = base.shape
    groups = hidden // model.group_size
    corrected = base.copy()
    alpha_out = np.empty((n, slots, groups), np.float32)
    for slot in range(slots):
        alpha = model.alpha_mean[slot] + z @ model.alpha_coefficients[slot]
        alpha = np.clip(alpha, -ALPHA_CLIP, ALPHA_CLIP).astype(np.float32)
        previous = x if slot == 0 else base[:, slot - 1]
        correction = (
            previous.reshape(n, groups, model.group_size)
            * alpha[:, :, None]
        ).reshape(n, hidden)
        corrected[:, slot] += correction
        alpha_out[:, slot] = alpha
    return corrected, alpha_out


def hidden_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = np.asarray(reference, np.float64)
    cand = np.asarray(candidate, np.float64)
    diff = cand - ref
    rows = []
    for slot in range(ref.shape[1]):
        r = ref[:, slot]
        c = cand[:, slot]
        d = diff[:, slot]
        rn = np.linalg.norm(r, axis=1)
        cn = np.linalg.norm(c, axis=1)
        cosine = np.sum(r * c, axis=1) / np.maximum(rn * cn, 1e-30)
        rows.append({
            "slot": slot + 1,
            "nrmse": float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-30)),
            "mean_cosine": float(np.mean(cosine)),
            "p05_cosine": float(np.percentile(cosine, 5)),
        })
    nrmse = np.asarray([row["nrmse"] for row in rows], np.float64)
    slope = float(np.polyfit(np.arange(1, len(rows) + 1), nrmse, 1)[0])
    return {
        "per_slot": rows,
        "mean_nrmse": float(np.mean(nrmse)),
        "last3_mean_nrmse": float(np.mean(nrmse[-3:])),
        "nrmse_suffix_slope": slope,
    }


def capture_split(rt, rows: list[dict], split_name: str) -> list[dict]:
    import cupy as cp

    sequences = []
    steps = TOKENS_PER_PROMPT + DRAFT_SLOTS
    for pi, row in enumerate(rows):
        rt.reset()
        nxt = None
        for token in row["prompt_ids"]:
            nxt = rt.step(int(token))
        if nxt is None:
            raise RuntimeError(f"empty prompt {row['id']}")

        states = np.empty((steps, int(rt.hidden)), np.float16)
        tokens = np.empty(steps, np.int32)
        for step in range(steps):
            states[step] = cp.asnumpy(rt.normed).astype(np.float16, copy=False)
            tokens[step] = int(nxt)
            nxt = rt.step(int(nxt))
        sequences.append({
            "id": row["id"],
            "states": states,
            "tokens": tokens,
        })
        print(f"14F capture {split_name} {pi + 1:02d}/{len(rows)} {row['id']}", flush=True)
    return sequences


def make_windows(sequences: list[dict]) -> WindowSet:
    xs, ys, labels, anchors, prompts_i, anchor_i = [], [], [], [], [], []
    for pi, seq in enumerate(sequences):
        states = seq["states"]
        tokens = seq["tokens"]
        available = len(states) - DRAFT_SLOTS
        count = min(MAX_WINDOWS_PER_PROMPT, available)
        indices = np.unique(np.linspace(0, available - 1, count, dtype=np.int32))
        for start in indices:
            xs.append(states[start])
            ys.append(states[start + 1:start + 1 + DRAFT_SLOTS])
            labels.append(tokens[start + 1:start + 1 + DRAFT_SLOTS])
            anchors.append(tokens[start])
            prompts_i.append(pi)
            anchor_i.append(int(start))
    return WindowSet(
        x=np.stack(xs).astype(np.float32),
        future_hidden=np.stack(ys).astype(np.float32),
        labels=np.stack(labels).astype(np.int32),
        anchor_ids=np.asarray(anchors, np.int32),
        prompt_index=np.asarray(prompts_i, np.int16),
        anchor_index=np.asarray(anchor_i, np.int16),
    )


def lm_topk(rt, hidden: np.ndarray, top_k: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp

    n, slots, width = hidden.shape
    if width != int(rt.hidden):
        raise ValueError(f"hidden width mismatch {width} != {rt.hidden}")
    ids = np.empty((n, slots, top_k), np.int32)
    scores = np.empty((n, slots, top_k), np.float32)
    x = cp.empty(width, cp.float32)
    logits = cp.empty(int(rt.vocab), cp.float32)
    total = n * slots
    done = 0
    for row in range(n):
        for slot in range(slots):
            x.set(np.ascontiguousarray(hidden[row, slot], dtype=np.float32))
            if rt.lm_head_kind == "nvfp4":
                rt.fused.gemv_into(
                    logits,
                    rt.lm_head_codes,
                    rt.lm_head_scales,
                    x,
                    rt.lm_head_g,
                    int(rt.vocab),
                    int(rt.hidden),
                )
            else:
                rt.k.mv_bf16(
                    logits, rt.lm_head, x, int(rt.vocab), int(rt.hidden)
                )
            candidate = cp.argpartition(logits, -top_k)[-top_k:]
            order = cp.argsort(logits[candidate])[::-1]
            candidate = candidate[order]
            ids[row, slot] = cp.asnumpy(candidate).astype(np.int32, copy=False)
            scores[row, slot] = cp.asnumpy(logits[candidate]).astype(np.float32, copy=False)
            done += 1
        if (row + 1) % 32 == 0 or row + 1 == n:
            print(f"14F LM-head {label}: {done}/{total}", flush=True)
    return ids, scores


def prefix_lengths(path: np.ndarray, labels: np.ndarray) -> np.ndarray:
    equal = np.asarray(path == labels, bool)
    return np.cumprod(equal, axis=1).sum(axis=1).astype(np.int32)


def candidate_metrics(ids: np.ndarray, labels: np.ndarray) -> dict:
    top1 = ids[:, :, 0]
    top1_ok = top1 == labels
    covered = np.any(ids == labels[:, :, None], axis=2)
    independent_prefix = prefix_lengths(top1, labels)
    oracle_prefix = np.cumprod(covered, axis=1).sum(axis=1).astype(np.int32)
    per_slot = []
    for slot in range(labels.shape[1]):
        per_slot.append({
            "slot": slot + 1,
            "top1_accuracy": float(np.mean(top1_ok[:, slot])),
            "topk_recall": float(np.mean(covered[:, slot])),
        })
    return {
        "per_slot": per_slot,
        "mean_top1_accuracy": float(np.mean(top1_ok)),
        "mean_topk_recall": float(np.mean(covered)),
        "mean_acceptance_independent_including_anchor": float(
            1.0 + np.mean(independent_prefix)
        ),
        "mean_acceptance_oracle_lattice_including_anchor": float(
            1.0 + np.mean(oracle_prefix)
        ),
        "oracle_selector_headroom_tokens": float(
            np.mean(oracle_prefix - independent_prefix)
        ),
        "full_block_candidate_coverage_fraction": float(
            np.mean(oracle_prefix == labels.shape[1])
        ),
    }


def bf16_to_float(raw_u16: np.ndarray) -> np.ndarray:
    u32 = np.asarray(raw_u16, np.uint16).astype(np.uint32)
    return (u32 << np.uint32(16)).view(np.float32)


def token_projection_maps(rt, token_ids: np.ndarray, projection: np.ndarray) -> dict[int, np.ndarray]:
    unique = np.unique(token_ids.astype(np.int64, copy=False))
    if getattr(rt, "embed_host", None) is not None:
        table = rt.embed_host.reshape(int(rt.vocab), int(rt.hidden))
        rows = bf16_to_float(table[unique])
    else:
        import cupy as cp
        table = rt.embed.reshape(int(rt.vocab), int(rt.hidden))
        rows = bf16_to_float(cp.asnumpy(table[unique]))
    projected = rows @ projection
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / np.maximum(norms, 1e-8)
    return {int(token): projected[i].astype(np.float32) for i, token in enumerate(unique)}


def normalized_unary(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=-1, keepdims=True)
    std = scores.std(axis=-1, keepdims=True)
    return (scores - mean) / np.maximum(std, 1e-5)


def selector_paths(
    *,
    candidate_ids: np.ndarray,
    unary_scores: np.ndarray,
    anchor_ids: np.ndarray,
    hidden: np.ndarray,
    embedding_map_a: dict[int, np.ndarray],
    embedding_map_b: dict[int, np.ndarray],
    hidden_projection: np.ndarray,
    transition_weight: float,
) -> np.ndarray:
    n, slots, top_k = candidate_ids.shape
    rank = hidden_projection.shape[1]
    unary = normalized_unary(unary_scores)
    context = hidden @ hidden_projection
    context = np.tanh(context).astype(np.float32)
    paths = np.empty((n, slots), np.int32)

    for row in range(n):
        back = np.empty((slots, top_k), np.int16)
        current_ids = candidate_ids[row, 0]
        previous = embedding_map_a[int(anchor_ids[row])]
        successor = np.stack([embedding_map_b[int(x)] for x in current_ids])
        transition = (previous[None] * context[row, 0][None] * successor).sum(axis=1)
        dp = unary[row, 0] + transition_weight * transition
        back[0].fill(-1)

        for slot in range(1, slots):
            previous_ids = candidate_ids[row, slot - 1]
            current_ids = candidate_ids[row, slot]
            predecessor = np.stack([embedding_map_a[int(x)] for x in previous_ids])
            successor = np.stack([embedding_map_b[int(x)] for x in current_ids])
            transition = (
                (predecessor * context[row, slot][None]) @ successor.T
            )
            total = dp[:, None] + unary[row, slot][None, :] + transition_weight * transition
            back[slot] = np.argmax(total, axis=0).astype(np.int16)
            dp = np.max(total, axis=0)

        index = int(np.argmax(dp))
        for slot in range(slots - 1, -1, -1):
            paths[row, slot] = int(candidate_ids[row, slot, index])
            if slot:
                index = int(back[slot, index])
    return paths


def selector_eval(
    rt,
    *,
    cal_ids: np.ndarray,
    cal_scores: np.ndarray,
    cal_anchor: np.ndarray,
    cal_hidden: np.ndarray,
    cal_labels: np.ndarray,
    val_ids: np.ndarray,
    val_scores: np.ndarray,
    val_anchor: np.ndarray,
    val_hidden: np.ndarray,
    val_labels: np.ndarray,
) -> dict:
    rng = np.random.default_rng(SEED)
    h = int(rt.hidden)
    pa = rng.standard_normal((h, SELECTOR_PROXY_RANK), dtype=np.float32) / math.sqrt(h)
    pb = rng.standard_normal((h, SELECTOR_PROXY_RANK), dtype=np.float32) / math.sqrt(h)
    ph = rng.standard_normal((h, SELECTOR_PROXY_RANK), dtype=np.float32) / math.sqrt(h)

    all_a = np.concatenate([cal_anchor.reshape(-1), val_anchor.reshape(-1), cal_ids.reshape(-1), val_ids.reshape(-1)])
    all_b = np.concatenate([cal_ids.reshape(-1), val_ids.reshape(-1)])
    map_a = token_projection_maps(rt, all_a, pa)
    map_b = token_projection_maps(rt, all_b, pb)

    trials = []
    for weight in SELECTOR_LAMBDAS:
        path = selector_paths(
            candidate_ids=cal_ids,
            unary_scores=cal_scores,
            anchor_ids=cal_anchor,
            hidden=cal_hidden,
            embedding_map_a=map_a,
            embedding_map_b=map_b,
            hidden_projection=ph,
            transition_weight=weight,
        )
        prefix = prefix_lengths(path, cal_labels)
        trials.append({
            "transition_weight": weight,
            "calibration_mean_acceptance_including_anchor": float(1.0 + np.mean(prefix)),
        })
    selected = max(
        trials,
        key=lambda row: (
            row["calibration_mean_acceptance_including_anchor"],
            -row["transition_weight"],
        ),
    )
    weight = float(selected["transition_weight"])
    val_path = selector_paths(
        candidate_ids=val_ids,
        unary_scores=val_scores,
        anchor_ids=val_anchor,
        hidden=val_hidden,
        embedding_map_a=map_a,
        embedding_map_b=map_b,
        hidden_projection=ph,
        transition_weight=weight,
    )
    val_prefix = prefix_lengths(val_path, val_labels)
    independent_prefix = prefix_lengths(val_ids[:, :, 0], val_labels)
    covered = np.any(val_ids == val_labels[:, :, None], axis=2)
    oracle_prefix = np.cumprod(covered, axis=1).sum(axis=1)
    headroom = float(np.mean(oracle_prefix - independent_prefix))
    gain = float(np.mean(val_prefix - independent_prefix))
    return {
        "kind": "frozen_embedding_factorized_selector_proxy",
        "claim_boundary": (
            "constrained lower-capacity proxy for DFlash2 path selection; "
            "not the learned DFlash2 codebooks"
        ),
        "rank": SELECTOR_PROXY_RANK,
        "trials": trials,
        "selected_transition_weight": weight,
        "validation_mean_acceptance_including_anchor": float(1.0 + np.mean(val_prefix)),
        "validation_gain_vs_independent_tokens": gain,
        "validation_oracle_headroom_tokens": headroom,
        "fraction_of_oracle_headroom_recovered": float(gain / max(headroom, 1e-9)),
    }


def fit_and_score_configuration(
    train: WindowSet,
    select: WindowSet,
    rank: int,
    group_size: int,
) -> tuple[dict, BaseModel, CorrectionModel]:
    base_model = fit_base(train.x, train.future_hidden, rank)
    train_base, train_z = predict_base(base_model, train.x)
    correction = fit_correction(
        train.x, train.future_hidden, train_base, train_z, group_size
    )
    select_base, select_z = predict_base(base_model, select.x)
    select_corrected, _ = apply_correction(correction, select.x, select_base, select_z)
    base_score = hidden_metrics(select.future_hidden, select_base)
    corrected_score = hidden_metrics(select.future_hidden, select_corrected)
    return ({
        "rank": rank,
        "group_size": group_size,
        "selection_base": base_score,
        "selection_corrected": corrected_score,
        "selection_last3_relative_nrmse": float(
            corrected_score["last3_mean_nrmse"]
            / max(base_score["last3_mean_nrmse"], 1e-12)
        ),
    }, base_model, correction)


def subset(w: WindowSet, mask: np.ndarray) -> WindowSet:
    return WindowSet(
        x=w.x[mask],
        future_hidden=w.future_hidden[mask],
        labels=w.labels[mask],
        anchor_ids=w.anchor_ids[mask],
        prompt_index=w.prompt_index[mask],
        anchor_index=w.anchor_index[mask],
    )


def main() -> int:
    ensure_results()
    payload: dict = {
        "kind": "s100_phase14f_dflash2_proxy",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": (
            "target-state transfer screen for DFlash2 principles; no trained "
            "DFlash2 drafter, no end-to-end speculative throughput claim"
        ),
    }
    bundle = None
    try:
        from s100_phase10a_runtime import build

        cal_prompts, val_prompts = prompts(REPO)
        bundle = build()
        rt = bundle.rt
        # The quality parent is graph-captured, but capture and proxy inference
        # need explicit host-visible hidden states. Keep the same weights/cache/
        # ERVF settings and only disable replay.
        rt._graph = None
        rt.graph_mode = False

        cal_sequences = capture_split(rt, cal_prompts, "calibration")
        val_sequences = capture_split(rt, val_prompts, "validation")
        cal = make_windows(cal_sequences)
        val = make_windows(val_sequences)
        np.savez_compressed(
            CAPTURE,
            calibration_x=cal.x.astype(np.float16),
            calibration_future_hidden=cal.future_hidden.astype(np.float16),
            calibration_labels=cal.labels,
            calibration_anchor_ids=cal.anchor_ids,
            calibration_prompt_index=cal.prompt_index,
            calibration_anchor_index=cal.anchor_index,
            validation_x=val.x.astype(np.float16),
            validation_future_hidden=val.future_hidden.astype(np.float16),
            validation_labels=val.labels,
            validation_anchor_ids=val.anchor_ids,
            validation_prompt_index=val.prompt_index,
            validation_anchor_index=val.anchor_index,
        )

        select_mask = (cal.anchor_index % 4) == 3
        if select_mask.sum() < 16 or (~select_mask).sum() < 32:
            raise RuntimeError("calibration internal split is too small")
        train = subset(cal, ~select_mask)
        select = subset(cal, select_mask)

        configurations = []
        for rank in RANKS:
            for group_size in GROUP_SIZES:
                score, _, _ = fit_and_score_configuration(
                    train, select, rank, group_size
                )
                configurations.append(score)
                print(
                    f"14F proxy rank={rank} group={group_size} "
                    f"last3_ratio={score['selection_last3_relative_nrmse']:.4f}",
                    flush=True,
                )
        selected = min(
            configurations,
            key=lambda row: (
                row["selection_corrected"]["mean_nrmse"],
                row["rank"],
                row["group_size"],
            ),
        )

        base_model = fit_base(cal.x, cal.future_hidden, int(selected["rank"]))
        cal_base, cal_z = predict_base(base_model, cal.x)
        correction = fit_correction(
            cal.x,
            cal.future_hidden,
            cal_base,
            cal_z,
            int(selected["group_size"]),
        )
        cal_corrected, _ = apply_correction(
            correction, cal.x, cal_base, cal_z
        )
        val_base, val_z = predict_base(base_model, val.x)
        val_corrected, val_alpha = apply_correction(
            correction, val.x, val_base, val_z
        )

        base_hidden = hidden_metrics(val.future_hidden, val_base)
        corrected_hidden = hidden_metrics(val.future_hidden, val_corrected)

        val_base_ids, _ = lm_topk(rt, val_base, TOP_K, "validation/base")
        cal_corrected_ids, cal_corrected_scores = lm_topk(
            rt, cal_corrected, TOP_K, "calibration/corrected"
        )
        val_corrected_ids, val_corrected_scores = lm_topk(
            rt, val_corrected, TOP_K, "validation/corrected"
        )
        base_candidates = candidate_metrics(val_base_ids, val.labels)
        corrected_candidates = candidate_metrics(val_corrected_ids, val.labels)

        selector = selector_eval(
            rt,
            cal_ids=cal_corrected_ids,
            cal_scores=cal_corrected_scores,
            cal_anchor=cal.anchor_ids,
            cal_hidden=cal_corrected,
            cal_labels=cal.labels,
            val_ids=val_corrected_ids,
            val_scores=val_corrected_scores,
            val_anchor=val.anchor_ids,
            val_hidden=val_corrected,
            val_labels=val.labels,
        )

        base_last3 = base_hidden["last3_mean_nrmse"]
        corrected_last3 = corrected_hidden["last3_mean_nrmse"]
        base_topk_last3 = float(np.mean([
            row["topk_recall"] for row in base_candidates["per_slot"][-3:]
        ]))
        corrected_topk_last3 = float(np.mean([
            row["topk_recall"] for row in corrected_candidates["per_slot"][-3:]
        ]))
        early_no_regression = (
            corrected_candidates["per_slot"][0]["topk_recall"]
            >= base_candidates["per_slot"][0]["topk_recall"] - 0.01
        )
        suffix_signal = bool(
            early_no_regression
            and (
                corrected_last3 <= 0.90 * base_last3
                or corrected_topk_last3 >= base_topk_last3 + 0.02
            )
        )
        oracle_mean = corrected_candidates[
            "mean_acceptance_oracle_lattice_including_anchor"
        ]
        oracle_headroom = corrected_candidates["oracle_selector_headroom_tokens"]
        path_headroom_signal = bool(oracle_mean >= 3.0 and oracle_headroom >= 0.75)
        selector_proxy_signal = bool(
            selector["validation_gain_vs_independent_tokens"] >= 0.10
            and selector["fraction_of_oracle_headroom_recovered"] >= 0.10
        )
        transfer_open = bool(suffix_signal and path_headroom_signal)

        payload.update({
            "status": "measured",
            "completed_utc": utc_now(),
            "model_shape": {
                "hidden": int(rt.hidden),
                "vocab": int(rt.vocab),
                "draft_slots": DRAFT_SLOTS,
                "block_size_including_anchor": DRAFT_SLOTS + 1,
            },
            "dataset": {
                "calibration_prompts": [row["id"] for row in cal_prompts],
                "validation_prompts": [row["id"] for row in val_prompts],
                "tokens_per_prompt": TOKENS_PER_PROMPT,
                "calibration_windows": int(len(cal.x)),
                "validation_windows": int(len(val.x)),
                "capture_path": str(CAPTURE.relative_to(REPO)),
            },
            "method": {
                "base_proxy": (
                    "calibration-only low-rank map from anchor final hidden to "
                    "seven future final-hidden states"
                ),
                "suffix_correction_proxy": (
                    "anchor-conditioned grouped dynamic two-tap correction; "
                    "all base slots computed before correction"
                ),
                "candidate_selector_proxy": (
                    "top-16 candidate lattice plus frozen embedding-factorized "
                    "transition scorer and dynamic programming"
                ),
                "hyperparameter_selection": (
                    "rank/group selected on a deterministic calibration-only "
                    "internal split; validation untouched"
                ),
            },
            "configuration_trials": configurations,
            "selected_configuration": {
                "rank": int(selected["rank"]),
                "group_size": int(selected["group_size"]),
                "base_ridge": float(base_model.ridge),
                "correction_ridge": float(correction.ridge),
                "top_k": TOP_K,
                "alpha_clip": ALPHA_CLIP,
                "validation_alpha_abs_p95": float(
                    np.percentile(np.abs(val_alpha), 95)
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
                "last3_topk_recall_delta": float(
                    corrected_topk_last3 - base_topk_last3
                ),
            },
            "selector_proxy": selector,
            "gates": {
                "SUFFIX_DECAY_CORRECTION_SIGNAL_OPEN": suffix_signal,
                "CANDIDATE_LATTICE_HEADROOM_OPEN": path_headroom_signal,
                "FROZEN_SELECTOR_PROXY_SIGNAL_OPEN": selector_proxy_signal,
                "DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN": transfer_open,
            },
            "decision": (
                "TRANSFER_SIGNAL_SUPPORTS_A_TRAINED_DRAFTER_PILOT"
                if transfer_open
                else "DO_NOT_TRAIN_YET_FROM_THIS_PROXY_EVIDENCE"
            ),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
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
        "selected_configuration": payload.get("selected_configuration"),
        "gates": payload.get("gates"),
        "decision": payload.get("decision"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
