"""Phase 20A: identity, schema, consumption and all-Mamba screening.

This runner is deliberately fail-closed.  It treats the local checkpoint as
the source of truth, records every safetensors entry, and does not open the
full-model block verifier unless an independent reference is available.
"""

from __future__ import annotations

import json
import re
import struct
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, require_model_dir, utc_now, write_json_atomic, write_text_atomic

OUT = REPO / "pro_research" / "results" / "s100_phase20a_identity.json"
SCHEMA_OUT = REPO / "pro_research" / "results" / "s100_phase20_model_schema.json"
REPORT_OUT = REPO / "reports" / "S100_PHASE20_RUN_REPORT.md"
HARD_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
HARD_SNAPSHOT = "e8f3c7c4de75ad84fe1bcef95d38eca76214480b"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _entries(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Read all shard headers; no tensor payloads are loaded."""
    index = _load_json(model_dir / "model.safetensors.index.json")
    result: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        path = model_dir / shard
        with path.open("rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(header_len).decode("utf-8"))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = [int(x) for x in meta["data_offsets"]]
            result[name] = {
                "name": name,
                "shard": shard,
                "dtype": meta["dtype"],
                "shape": [int(x) for x in meta["shape"]],
                "data_offsets": [start, end],
                "byte_count": end - start,
            }
    return result


def _base_name(name: str) -> str:
    for suffix in (".weight_scale_2", ".input_scale", ".weight_scale", ".weight"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _quant_groups(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    q = cfg.get("quantization_config") or {}
    groups: dict[str, dict[str, Any]] = {}
    for group_name, group in (q.get("config_groups") or {}).items():
        weights = group.get("weights") or {}
        info = {
            "group": group_name,
            "quant_algo": q.get("quant_algo"),
            "num_bits": weights.get("num_bits"),
            "type": weights.get("type"),
            "group_size": weights.get("group_size"),
        }
        for target in group.get("targets") or []:
            groups[target] = info
    return groups


def _tensor_quant(name: str, entry: dict[str, Any], groups: dict[str, dict[str, Any]], base_map: dict[str, list[str]]) -> dict[str, Any]:
    base = _base_name(name)
    group = groups.get(base)
    companions = [k for k in base_map.get(base, []) if k != name]
    if group and group.get("num_bits") == 4:
        kind = "nvfp4"
    elif group and group.get("num_bits") == 8:
        kind = "fp8_tensor"
    else:
        kind = "plain"
    return {
        "kind": kind,
        "base": base,
        "quant_group": group,
        "companions": companions,
        "storage_dtype": entry["dtype"],
    }


def _layer_inventory(cfg: dict[str, Any], entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _quant_groups(cfg)
    base_map: dict[str, list[str]] = {}
    for name in entries:
        base_map.setdefault(_base_name(name), []).append(name)
    schedule = list(cfg.get("layers_block_type") or [])
    layers: list[dict[str, Any]] = []
    for i, kind in enumerate(schedule):
        prefix = f"backbone.layers.{i}."
        names = sorted(k for k in entries if k.startswith(prefix))
        tensors = []
        for name in names:
            item = dict(entries[name])
            item["quantization"] = _tensor_quant(name, item, groups, base_map)
            tensors.append(item)
        layers.append({
            "layer": i,
            "type": kind,
            "tensor_count": len(tensors),
            "tensors": tensors,
            "formats": {
                "mamba_in_proj": next((t["quantization"]["kind"] for t in tensors if t["name"].endswith("mixer.in_proj.weight")), None),
                "mamba_out_proj": next((t["quantization"]["kind"] for t in tensors if t["name"].endswith("mixer.out_proj.weight")), None),
                "attention_qkv_o": sorted({t["quantization"]["kind"] for t in tensors if re.search(r"mixer\.[qkvo]_proj\.weight$", t["name"])}),
                "moe_experts": sorted({t["quantization"]["kind"] for t in tensors if ".mixer.experts." in t["name"]}),
                "shared_experts": sorted({t["quantization"]["kind"] for t in tensors if ".mixer.shared_experts." in t["name"]}),
            },
        })
    return layers


def _expected_runtime_keys(cfg: dict[str, Any], entries: dict[str, dict[str, Any]]) -> tuple[set[str], set[str], dict[str, str]]:
    """Declare the exact target-path keys consumed by LightningRuntime.

    Quantizer-only ``input_scale`` records and the optional MTP subtree are
    recorded separately, rather than silently treated as unknown omissions.
    """
    schedule = list(cfg.get("layers_block_type") or [])
    consumed: set[str] = set()
    intentional: set[str] = set()
    reasons: dict[str, str] = {}

    def take(name: str) -> None:
        if name in entries:
            consumed.add(name)

    take("backbone.embeddings.weight")
    take("backbone.norm_f.weight")
    for i, kind in enumerate(schedule):
        p = f"backbone.layers.{i}"
        take(f"{p}.norm.weight")
        m = f"{p}.mixer"
        if kind == "mamba":
            for field in ("in_proj", "out_proj"):
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    take(f"{m}.{field}.{suffix}")
                # The runtime's FP8 weight-only path does not consume the
                # modelopt activation input_scale; classify it explicitly.
                n = f"{m}.{field}.input_scale"
                if n in entries:
                    intentional.add(n)
                    reasons[n] = "modelopt activation metadata; runtime uses FP8 weight-only dequantization"
            for field in ("conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias", "norm.weight"):
                take(f"{m}.{field}")
        elif kind == "attention":
            for field in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
                take(f"{m}.{field}")
        elif kind == "moe":
            for field in ("gate.weight", "gate.e_score_correction_bias"):
                take(f"{m}.{field}")
            for field in ("shared_experts.up_proj", "shared_experts.down_proj"):
                for suffix in ("weight", "weight_scale", "weight_scale_2"):
                    take(f"{m}.{field}.{suffix}")
            n_experts = int(cfg["n_routed_experts"])
            for e in range(n_experts):
                for field in ("up_proj", "down_proj"):
                    for suffix in ("weight", "weight_scale", "weight_scale_2"):
                        take(f"{m}.experts.{e}.{field}.{suffix}")

    for suffix in ("weight", "weight_scale", "weight_scale_2"):
        take(f"lm_head.{suffix}")

    for name in entries:
        if name.startswith("mtp."):
            intentional.add(name)
            reasons[name] = "optional MTP/draft subtree; excluded from the target-only LightningRuntime"
        elif name.endswith(".input_scale") and name not in consumed:
            intentional.add(name)
            reasons[name] = "quantizer activation metadata not consumed by target runtime"
    return consumed, intentional, reasons


def _schema_payload(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _load_json(model_dir / "config.json")
    entries = _entries(model_dir)
    layers = _layer_inventory(cfg, entries)
    groups = _quant_groups(cfg)
    base_map: dict[str, list[str]] = {}
    for name in entries:
        base_map.setdefault(_base_name(name), []).append(name)
    consumed, intentional, reasons = _expected_runtime_keys(cfg, entries)
    unknown = sorted(set(entries) - consumed - intentional)
    missing = sorted(consumed - set(entries))
    schema = {
        "kind": "s100_phase20_model_schema",
        "model_id": HARD_MODEL_ID,
        "snapshot": model_dir.name,
        "model_dir": str(model_dir),
        "config": cfg,
        "shard_count": len(set(v["shard"] for v in entries.values())),
        "tensor_count": len(entries),
        "all_tensors": [entries[k] | {"quantization": _tensor_quant(k, entries[k], groups, base_map)} for k in sorted(entries)],
        "layers": layers,
    }
    audit = {
        "consumed_target_weights": sorted(consumed),
        "intentional_non_target_weights": [
            {"name": n, "reason": reasons[n]} for n in sorted(intentional)
        ],
        "UNKNOWN_UNUSED_WEIGHTS": unknown,
        "EXPECTED_BUT_MISSING_WEIGHTS": missing,
        "unknown_unused_target_weights_count": len(unknown),
        "expected_but_missing_count": len(missing),
        "target_consumption_gate": len(unknown) == 0 and len(missing) == 0,
    }
    return schema, audit


def _write_report(payload: dict[str, Any]) -> None:
    identity = payload.get("identity", {})
    audit = payload.get("consumption_audit", {})
    mamba = payload.get("all_23_mamba_screen", {})
    parity = payload.get("independent_reference_parity", {})
    lines = [
        "# S100 Phase 20 — Real Nemotron 3.5 Lightning",
        "",
        f"Status: **{payload.get('status')}**",
        "",
        f"Model: `{identity.get('model_id')}`",
        f"Snapshot: `{identity.get('snapshot')}`",
        f"Architecture: `{identity.get('architectures')}` / `{identity.get('model_type')}`",
        "",
        "## 20A identity and schema",
        "",
        f"- Layers: {identity.get('num_hidden_layers')} ({identity.get('layer_counts')})",
        f"- Tensor entries: {identity.get('tensor_count')} across {identity.get('shard_count')} shards",
        f"- Consumption gate: **{audit.get('target_consumption_gate')}**",
        f"- Unknown unused target weights: **{audit.get('unknown_unused_target_weights_count')}**",
        f"- Expected-but-missing weights: **{audit.get('expected_but_missing_count')}**",
        "",
        "## All 23 Mamba layers",
        "",
        f"- Status: `{mamba.get('status')}`",
        f"- Tested layers: {mamba.get('tested_layers')}",
        f"- Median/min/max speedup: {mamba.get('median_speedup')} / {mamba.get('min_speedup')} / {mamba.get('max_speedup')}",
        f"- Maximum output NRMSE: {mamba.get('max_output_nrmse')}",
        f"- Maximum state NRMSE: {mamba.get('max_state_nrmse')}",
        f"- Sabotage control: observable={((mamba.get('sabotage_control') or {}).get('observable_change'))}, max logit delta={((mamba.get('sabotage_control') or {}).get('max_abs_logit_delta'))}",
        "",
        "## Independent reference parity",
        "",
        f"- Status: `{parity.get('status')}`",
        f"- `PHASE20A_OFFICIAL_PARITY_GREEN`: **{parity.get('PHASE20A_OFFICIAL_PARITY_GREEN')}**",
        f"- Reason: {parity.get('reason')}",
        "",
        "20B remains closed unless the independent parity gate is green. No S100 claim is made from this Phase 20A result.",
    ]
    write_text_atomic(REPORT_OUT, "\n".join(lines), archive=True)


def _run_all23_mamba(model_dir: Path) -> dict[str, Any]:
    """Reuse the already validated Phase19 H=4 block harness for every Mamba layer."""
    import gc
    import numpy as np
    import cupy as cp
    from transformers import AutoTokenizer
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    from s100_phase17_kernels import Phase17Kernels
    from s100_phase17_mamba_block import capture_sequences
    from s100_phase19_full_layer import run_layer
    from s100_phase19_residual_projection import FP8ProjectionBlock

    rt = LightningRuntime(model_dir, contexts_max=512, embed_on_host=True, fp8_kv=True, verbose=False)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    layers = [int(x) for x in rt.mamba_layers]
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True, use_fast=True)
    prompt_ids = tok.encode("The history of computing and artificial intelligence", add_special_tokens=False)
    captures = capture_sequences(rt, layers, prompt_ids)
    kernels = Phase17Kernels()
    fp8_block = FP8ProjectionBlock(cp)
    results = [run_layer(rt, kernels, fp8_block, None, cp, layer, captures[layer]) for layer in layers]
    for r in results:
        r.pop("baseline", None)
        r.pop("candidate", None)
    speeds = [float(r["speedup"]) for r in results]
    outs = [float(r["correctness"]["output_nrmse"]) for r in results]
    states = [float(r["correctness"]["ssm_final_nrmse"]) for r in results]
    # Candidate-path sabotage: zero one known target tensor in layer 0 and
    # prove that the custom runtime's observable logits move. This is not an
    # independent parity proof; it only guards against a disconnected audit.
    def run_prompt() -> tuple[int, Any]:
        rt.reset()
        nxt = None
        for token in prompt_ids:
            nxt = int(rt.step(int(token)))
        cp.cuda.get_current_stream().synchronize()
        return int(nxt), cp.asnumpy(rt.logits).astype(np.float32, copy=True)

    base_token, base_logits = run_prompt()
    target = rt.layer[0]["in_w8"]
    original = cp.asnumpy(target).copy()
    target.fill(0)
    sabotaged_token, sabotaged_logits = run_prompt()
    target.set(original)
    cp.cuda.get_current_stream().synchronize()
    delta = float(np.max(np.abs(base_logits - sabotaged_logits)))
    sabotage = {
        "tensor": "backbone.layers.0.mixer.in_proj.weight",
        "mutation": "zero_all_storage_bytes",
        "base_token": base_token,
        "sabotaged_token": sabotaged_token,
        "max_abs_logit_delta": delta,
        "observable_change": bool(delta > 0.0 or base_token != sabotaged_token),
    }
    rt.bank = {}
    gc.collect()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return {
        "status": "green" if all(r["correctness"]["pass"] and r["speedup"] >= 1.0 for r in results) else "failed_gate",
        "tested_layers": layers,
        "results": results,
        "median_speedup": float(np.median(speeds)),
        "min_speedup": float(np.min(speeds)),
        "max_speedup": float(np.max(speeds)),
        "max_output_nrmse": float(np.max(outs)),
        "max_state_nrmse": float(np.max(states)),
        "sabotage_control": sabotage,
    }


def _reference_probe(model_dir: Path) -> dict[str, Any]:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
        return {"status": "config_available_only", "config_type": type(cfg).__name__, "PHASE20A_OFFICIAL_PARITY_GREEN": False, "reason": "Transformers config is available, but no independent full reference was loaded or compared"}
    except Exception as exc:
        return {"status": "blocked", "PHASE20A_OFFICIAL_PARITY_GREEN": False, "reason": f"{type(exc).__name__}: {exc}", "sabotage_control": "not_run"}


def main() -> int:
    payload: dict[str, Any] = {"kind": "s100_phase20a_identity", "status": "started", "started_utc": utc_now()}
    try:
        model_dir = require_model_dir()
        cfg = _load_json(model_dir / "config.json")
        if cfg.get("architectures") != ["NemotronHForCausalLM"] or cfg.get("model_type") != "nemotron_h":
            raise RuntimeError(f"unexpected Lightning architecture: {cfg.get('architectures')} / {cfg.get('model_type')}")
        if len(cfg.get("layers_block_type", [])) != 52:
            raise RuntimeError("snapshot does not contain exactly 52 target layers")
        entries = _entries(model_dir)
        schema, audit = _schema_payload(model_dir)
        write_json_atomic(SCHEMA_OUT, schema, archive=True)
        counts = {k: cfg["layers_block_type"].count(k) for k in sorted(set(cfg["layers_block_type"]))}
        payload["identity"] = {
            "model_id": HARD_MODEL_ID,
            "snapshot": model_dir.name,
            "expected_snapshot": HARD_SNAPSHOT,
            "snapshot_match": model_dir.name == HARD_SNAPSHOT,
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "hidden_size": cfg.get("hidden_size"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "layer_counts": counts,
            "tensor_count": len(entries),
            "shard_count": len(set(v["shard"] for v in entries.values())),
            "config_fields": {k: cfg.get(k) for k in sorted(cfg)},
        }
        payload["consumption_audit"] = audit
        payload["independent_reference_parity"] = _reference_probe(model_dir)
        # The all-23 Mamba screen is still useful diagnostic evidence when the
        # target-consumption or independent-reference gate is closed, but its
        # result can never promote 20B by itself.
        if not audit["expected_but_missing_count"]:
            payload["all_23_mamba_screen"] = _run_all23_mamba(model_dir)
        if not audit["target_consumption_gate"]:
            payload["status"] = "phase20a_blocked_target_consumption"
        else:
            payload["status"] = "phase20b_closed" if not payload["independent_reference_parity"].get("PHASE20A_OFFICIAL_PARITY_GREEN") else "phase20a_green"
        payload["completed_utc"] = utc_now()
    except Exception as exc:
        payload.update({"status": "technical_failure", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "completed_utc": utc_now()})
    write_json_atomic(OUT, payload, archive=True)
    _write_report(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "identity": {k: payload.get("identity", {}).get(k) for k in ("model_id", "snapshot", "snapshot_match", "architectures", "model_type", "hidden_size", "num_hidden_layers", "layer_counts", "tensor_count", "shard_count")},
        "consumption_audit": {k: payload.get("consumption_audit", {}).get(k) for k in ("unknown_unused_target_weights_count", "expected_but_missing_count", "target_consumption_gate")},
        "mamba": {k: payload.get("all_23_mamba_screen", {}).get(k) for k in ("status", "tested_layers", "median_speedup", "min_speedup", "max_speedup", "max_output_nrmse", "max_state_nrmse")},
        "parity": payload.get("independent_reference_parity"),
        "schema": str(SCHEMA_OUT), "report": str(REPORT_OUT),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") in {"phase20a_green", "phase20b_closed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
