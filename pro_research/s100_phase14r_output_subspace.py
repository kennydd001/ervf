from __future__ import annotations

import json
import os
import traceback
import types

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_phase14r_common import (
    RESULTS, PROMPTS, bf16_round, metrics, ensure_results,
    normalize_eager_moe,
)

OUT = RESULTS / "S100_PHASE14R_OUTPUT_SUBSPACE.json"
CAPTURE = RESULTS / "S100_PHASE14R_MOE_INPUTS.npz"
RANKS = (32, 64, 128, 192, 256, 384)
TOKENS_PER_PROMPT = 48
MAX_ROWS = 320

def prompt_splits():
    from transformers import AutoTokenizer
    rows = json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(require_model_dir()),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    calibration = []
    validation = []
    for row in rows:
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        item = {
            "id": row["id"],
            "domain": row["domain"],
            "prompt_ids": [int(value) for value in ids],
        }
        if row["id"].endswith("_01"):
            calibration.append(item)
        elif row["id"].endswith("_02"):
            validation.append(item)
    if len(calibration) != 10 or len(validation) != 10:
        raise RuntimeError(
            f"expected 10 calibration and validation prompts, got "
            f"{len(calibration)} / {len(validation)}"
        )
    return calibration, validation

def selected_layers(values):
    values = [int(value) for value in values]
    if len(values) <= 3:
        return values
    return sorted({values[0], values[len(values)//2], values[-1]})

def append_pair(store, key, x, y, active):
    if not active["value"]:
        return
    row = store.setdefault(key, {"x": [], "y": []})
    if len(row["x"]) >= MAX_ROWS:
        return
    import cupy as cp
    row["x"].append(cp.asnumpy(x).astype(np.float16, copy=True))
    row["y"].append(cp.asnumpy(y).astype(np.float16, copy=True))

def append_input(store, key, x, active):
    if not active["value"]:
        return
    rows = store.setdefault(key, [])
    if len(rows) >= MAX_ROWS:
        return
    import cupy as cp
    rows.append(cp.asnumpy(x).astype(np.float16, copy=True))

def capture(bundle, prompt_rows):
    rt = bundle.rt
    store = {}
    moe_inputs = {}
    moe_acts = {}
    moe_ids = {}
    active = {"value": False}
    mamba_set = set(selected_layers(rt.mamba_layers))
    attention_set = {int(value) for value in rt.attn_layers}
    moe_set = set(selected_layers(rt.moe_layers))

    original_mamba = rt._mamba
    original_attention = rt._attention
    original_moe = rt._moe

    def mamba(self, layer, out):
        x_in = self.normed
        original_mamba(layer, out)
        if int(layer) in mamba_set:
            append_pair(
                store, f"mamba_{layer}_in", x_in, self.proj, active
            )
            append_pair(
                store, f"mamba_{layer}_out", self.gn, out, active
            )

    def attention(self, layer, out):
        x_in = self.normed
        original_attention(layer, out)
        if int(layer) in attention_set:
            append_pair(
                store, f"attention_{layer}_q", x_in, self.qv, active
            )
            append_pair(
                store, f"attention_{layer}_k", x_in, self.kv_, active
            )
            append_pair(
                store, f"attention_{layer}_v", x_in, self.vv, active
            )
            append_pair(
                store, f"attention_{layer}_o", self.ctx, out, active
            )

    def moe(self, layer, out):
        selected = int(layer) in moe_set
        if selected:
            append_input(
                moe_inputs, f"moe_{layer}", self.normed, active
            )
        result = original_moe(layer, out)
        if selected and active["value"]:
            state = getattr(bundle, "state", {}).get(int(layer))
            dev = getattr(self, "_dev_cache", {}).get(int(layer))
            if state is not None and dev is not None:
                import cupy as cp
                key = f"moe_{layer}"
                if len(moe_acts.setdefault(key, [])) < MAX_ROWS:
                    moe_acts[key].append(
                        cp.asnumpy(state["act"]).reshape(
                            int(self.top_k), int(self.moe_inter)
                        ).astype(np.float32, copy=True)
                    )
                    moe_ids.setdefault(key, []).append(
                        cp.asnumpy(dev["ids"][:self.top_k])
                        .astype(np.int32, copy=True)
                    )
        return (None, None) if result is None else result

    rt._mamba = types.MethodType(mamba, rt)
    rt._attention = types.MethodType(attention, rt)
    rt._moe = types.MethodType(moe, rt)
    try:
        for prompt in prompt_rows:
            rt.reset()
            next_token = None
            for token in prompt["prompt_ids"]:
                next_token = rt.step(int(token))
            active["value"] = True
            for _ in range(TOKENS_PER_PROMPT):
                next_token = rt.step(int(next_token))
            active["value"] = False
    finally:
        rt._mamba = original_mamba
        rt._attention = original_attention
        rt._moe = original_moe
    return store, moe_inputs, moe_acts, moe_ids

def source_bytes(rt, key):
    parts = key.split("_")
    if parts[0] == "mamba":
        layer = int(parts[1])
        side = parts[2]
        data = rt.layer[layer]
        kind = data[f"{side}_k"]
        if kind == "fp8_tensor":
            return int(data[f"{side}_w8"].nbytes) + 4
        if kind == "bf16":
            return int(data[f"{side}_w"].nbytes)
        if kind == "nvfp4":
            return (
                int(data[f"{side}_codes"].nbytes)
                + int(data[f"{side}_scales"].nbytes) + 4
            )
    if parts[0] == "attention":
        layer = int(parts[1])
        side = parts[2]
        data = rt.layer[layer]
        if side == "q" and data.get("q_kind") == "nvfp4":
            return (
                int(data["q_codes"].nbytes)
                + int(data["q_scales"].nbytes) + 4
            )
        return int(data[f"{side}_proj"].nbytes)
    raise KeyError(key)

def family(key):
    parts = key.split("_")
    return (
        f"mamba_{parts[2]}"
        if parts[0] == "mamba"
        else f"attention_{parts[2]}"
    )

def reduced_rank(cal_x, cal_y, val_x, val_y, original_bytes):
    X = np.asarray(cal_x, np.float32)
    Y = np.asarray(cal_y, np.float32)
    VX = np.asarray(val_x, np.float32)
    VY = np.asarray(val_y, np.float32)

    U, singular, Vt = np.linalg.svd(X, full_matrices=False)
    keep = singular > max(float(singular[0]) * 1e-5, 1e-7)
    U = U[:, keep]
    singular = singular[keep]
    Vt = Vt[keep]
    q = len(singular)
    if q < 2:
        return []

    projected_y = U.T @ Y
    gram = projected_y @ projected_y.T
    eigenvalue, eigenvector = np.linalg.eigh(gram)
    order = np.argsort(-eigenvalue)
    basis = eigenvector[:, order]
    input_basis = Vt.T
    max_rank = min(max(RANKS), q)
    reduced = basis[:, :max_rank]
    left_full = input_basis @ (reduced / singular[:, None])
    right_full = reduced.T @ projected_y

    records = []
    for rank in RANKS:
        if rank > max_rank:
            records.append({"rank": rank, "status": "unsupported"})
            continue
        left = bf16_round(left_full[:, :rank])
        right = bf16_round(right_full[:rank])
        prediction = (VX @ left) @ right
        score = metrics(VY, prediction)
        candidate_bytes = int(
            2 * (VX.shape[1] * rank + rank * VY.shape[1])
        )
        reduction = 1.0 - candidate_bytes / max(original_bytes, 1)
        gates = {
            "bytes_saved_ge_35pct": reduction >= 0.35,
            "nrmse_le_0_03": score["nrmse"] <= 0.03,
            "cosine_ge_0_9995": score["mean_cosine"] >= 0.9995,
            "p95_relative_le_0_08": (
                score["p95_relative_row_error"] <= 0.08
            ),
            "finite": score["finite"],
        }
        records.append({
            "rank": rank,
            "status": "measured",
            "factor_bytes_bf16": candidate_bytes,
            "original_weight_bytes": int(original_bytes),
            "physical_byte_reduction": float(reduction),
            "validation": score,
            "gates": gates,
            "pass": all(gates.values()),
        })
    return records

def main():
    ensure_results()
    payload = {
        "kind": "s100_phase14r_output_subspace",
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        os.environ.setdefault("LS_MODEL_DIR", str(require_model_dir()))
        try:
            bundle = build(expose=True)
        except TypeError:
            bundle = build()
        rt = bundle.rt
        rt._graph = None
        rt.graph_mode = False
        normalize_eager_moe(rt)

        calibration_prompts, validation_prompts = prompt_splits()
        print("14B2 capture calibration", flush=True)
        cal, cal_moe, cal_acts, cal_ids = capture(
            bundle, calibration_prompts
        )
        print("14B2 capture validation", flush=True)
        val, val_moe, val_acts, val_ids = capture(
            bundle, validation_prompts
        )

        capture_payload = {}
        for key, rows in cal_moe.items():
            capture_payload[f"{key}_cal"] = np.stack(rows)
        for key, rows in val_moe.items():
            capture_payload[f"{key}_val"] = np.stack(rows)
        for key, rows in cal_acts.items():
            capture_payload[f"{key}_act_cal"] = np.stack(rows)
        for key, rows in val_acts.items():
            capture_payload[f"{key}_act_val"] = np.stack(rows)
        for key, rows in cal_ids.items():
            capture_payload[f"{key}_id_cal"] = np.stack(rows)
        for key, rows in val_ids.items():
            capture_payload[f"{key}_id_val"] = np.stack(rows)
        np.savez_compressed(CAPTURE, **capture_payload)

        cases = {}
        for key in sorted(set(cal) & set(val)):
            cal_x = np.stack(cal[key]["x"]).astype(np.float32)
            cal_y = np.stack(cal[key]["y"]).astype(np.float32)
            val_x = np.stack(val[key]["x"]).astype(np.float32)
            val_y = np.stack(val[key]["y"]).astype(np.float32)
            cal_count = min(len(cal_x), len(cal_y), MAX_ROWS)
            val_count = min(len(val_x), len(val_y), MAX_ROWS)
            records = reduced_rank(
                cal_x[:cal_count], cal_y[:cal_count],
                val_x[:val_count], val_y[:val_count],
                source_bytes(rt, key),
            )
            passed = [row for row in records if row.get("pass")]
            cases[key] = {
                "family": family(key),
                "calibration_rows": cal_count,
                "validation_rows": val_count,
                "input_dim": int(cal_x.shape[1]),
                "output_dim": int(cal_y.shape[1]),
                "ranks": records,
                "selected": min(
                    passed, key=lambda row: row["rank"]
                ) if passed else None,
            }
            print(
                f"14B2 {key}: "
                f"{(cases[key]['selected'] or {}).get('rank')}",
                flush=True,
            )

        families = {}
        for key, record in cases.items():
            family_name = record["family"]
            row = families.setdefault(
                family_name,
                {"cases": 0, "passed": 0, "case_names": []},
            )
            row["cases"] += 1
            row["passed"] += int(record["selected"] is not None)
            row["case_names"].append(key)
        for row in families.values():
            row["pass_fraction"] = row["passed"] / max(row["cases"], 1)
            row["runtime_build_open"] = row["pass_fraction"] >= 0.80

        open_families = [
            key for key, row in families.items()
            if row["runtime_build_open"]
        ]
        payload.update({
            "status": "measured",
            "tokens_per_prompt": TOKENS_PER_PROMPT,
            "max_rows": MAX_ROWS,
            "cases": cases,
            "families": families,
            "open_families": open_families,
            "SUBSPACE_RUNTIME_BUILD_OPEN": bool(open_families),
            "moe_capture_path": str(CAPTURE.relative_to(REPO)),
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
        "open_families": payload.get("open_families"),
        "SUBSPACE_RUNTIME_BUILD_OPEN": payload.get(
            "SUBSPACE_RUNTIME_BUILD_OPEN"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
