"""Integrated target-only Ornith H4 correctness gate for Phase84."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, utc_now, write_json_atomic
from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm
from s100_phase35_c3c_quantizer import FusedStaticNVFP4Quantizer
from s100_phase48_ornith_swiglu_h8 import _decode, _load_projection
from s100_phase58_ornith_fp8_h4_kernels import OrnithFP8H4Kernels
from s100_phase59_ornith_bulk_expert_kernels import OrnithNVFP4BulkM1
from s100_phase60_ornith_route_adaptive_kernels import OrnithNVFP4RouteAdaptive
from s100_phase64_ornith_shortlist_kernel import ExactERVFShortlist
from s100_phase67_ornith_linear_h4 import _load_auxiliary, _reference as _linear_reference
from s100_phase67_ornith_linear_h4_kernels import OrnithLinearH4Kernels
from s100_phase68_ornith_full_attn_h4 import (
    _load_norms as _load_attention_norms,
    _reference as _full_attention_reference,
    _rope,
)
from s100_phase68_ornith_full_attn_h4_kernels import OrnithFullAttentionH4Kernels
from s100_phase69_ornith_support_h4_kernels import OrnithSupportH4Kernels


SRC = REPO / "src"
RESULTS = REPO / "pro_research" / "results" / "s100_phase84_target_verifier"
PREREG = REPO / "pro_research" / "S100_PHASE84_ORNITH_TARGET_VERIFIER_H4_PREREGISTRATION.md"
SCRIPT = REPO / "pro_research" / "s100_phase84_ornith_target_verifier_h4.py"
TRACE_DEFAULT = (
    REPO / "pro_research" / "results" / "s100_phase76" /
    "ornith_64_hidden_trace.json.gz"
)
HIDDEN = 2048
INTERMEDIATE = 512
EXPERTS = 256
TOP_K = 8
LAYERS = 40
HEAD_ROWS = 248_320
SHORTLIST = 64


class DiagnosticComplete(Exception):
    """Internal control flow for a preregistered short localization run."""


def _load_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text("utf-8"))


def _weight_map(snapshot: Path) -> dict[str, str]:
    return json.loads(
        (snapshot / "model.safetensors.index.json").read_text("utf-8")
    )["weight_map"]


def _load_tensor(snapshot: Path, weight_map: dict[str, str], name: str):
    from safetensors import safe_open

    with safe_open(
        str(snapshot / weight_map[name]), framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(name).contiguous()


def _load_bf16(snapshot: Path, weight_map: dict[str, str], name: str):
    import torch

    tensor = _load_tensor(snapshot, weight_map, name)
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name}: expected BF16, got {tensor.dtype}")
    return (
        tensor.view(torch.uint16).numpy().copy().reshape(tuple(tensor.shape)),
        tensor.float().numpy().copy().reshape(tuple(tensor.shape)),
    )


def _load_fp8(snapshot: Path, weight_map: dict[str, str], prefix: str):
    import torch

    weight = _load_tensor(snapshot, weight_map, prefix + ".weight")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{prefix}.weight: expected E4M3, got {weight.dtype}")
    weight_scale = _load_tensor(snapshot, weight_map, prefix + ".weight_scale")
    input_scale = _load_tensor(snapshot, weight_map, prefix + ".input_scale")
    return (
        weight.view(torch.uint8).numpy().copy().reshape(tuple(weight.shape)),
        float(weight_scale.float().reshape(-1)[0]),
        float(input_scale.float().reshape(-1)[0]),
    )


def _rmsnorm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    inverse = np.reciprocal(
        np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + np.float32(1e-6))
    )
    return np.asarray(value * inverse * (np.float32(1.0) + weight), dtype=np.float32)


def _fp8_reference(torch, x: np.ndarray, projection) -> np.ndarray:
    raw, weight_scale, input_scale = projection
    xq = torch.from_numpy(
        np.asarray(x / np.float32(input_scale), dtype=np.float32)
    ).to(torch.float8_e4m3fn).float().numpy()
    weight = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).float().numpy()
    return np.asarray(
        (xq * np.float32(input_scale))
        @ (weight * np.float32(weight_scale)).T,
        dtype=np.float32,
    )


def _nrmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = max(
        float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))), 1e-12
    )
    return float(np.sqrt(np.mean(delta ** 2)) / denominator)


def _top8(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.empty((4, TOP_K), dtype=np.int32)
    weights = np.empty((4, TOP_K), dtype=np.float32)
    axis = np.arange(EXPERTS, dtype=np.int32)
    for token in range(4):
        order = np.lexsort((axis, -logits[token]))[:TOP_K]
        ids[token] = order
        selected = logits[token, order]
        exponential = np.exp(selected - selected[0]).astype(np.float32)
        weights[token] = exponential / np.sum(exponential, dtype=np.float32)
    return ids, weights


def _bucket_routes(ids: np.ndarray):
    occurrences: dict[int, list[tuple[int, int]]] = {}
    for token in range(4):
        for route in range(TOP_K):
            occurrences.setdefault(int(ids[token, route]), []).append(
                (token, token * TOP_K + route)
            )
    buckets: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
    for expert, rows in occurrences.items():
        buckets.setdefault(len(rows), []).append((expert, rows))
    return occurrences, buckets


def _host_moe_reference(
    snapshot: Path,
    weight_map: dict[str, str],
    layer: int,
    hidden: np.ndarray,
    ids: np.ndarray,
    route_weights: np.ndarray,
    shared_logits: np.ndarray,
) -> np.ndarray:
    outputs = np.empty((32, HIDDEN), dtype=np.float32)
    occurrences, _ = _bucket_routes(ids)
    for expert, assignments in occurrences.items():
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        gate = _decode(
            _load_projection(snapshot, weight_map, base + ".gate_proj"),
            INTERMEDIATE, HIDDEN,
        )
        up = _decode(
            _load_projection(snapshot, weight_map, base + ".up_proj"),
            INTERMEDIATE, HIDDEN,
        )
        down = _decode(
            _load_projection(snapshot, weight_map, base + ".down_proj"),
            HIDDEN, INTERMEDIATE,
        )
        token_ids = [token for token, _route in assignments]
        source = hidden[token_ids]
        gate_value = np.asarray(source @ gate.T, dtype=np.float32)
        up_value = np.asarray(source @ up.T, dtype=np.float32)
        activated = np.asarray(
            gate_value / (np.float32(1.0) + np.exp(-gate_value)) * up_value,
            dtype=np.float32,
        )
        result = np.asarray(activated @ down.T, dtype=np.float32)
        for row, (_token, route) in enumerate(assignments):
            outputs[route] = result[row]
        del gate, up, down

    shared_base = f"model.layers.{layer}.mlp.shared_expert"
    shared_gate = _decode(
        _load_projection(snapshot, weight_map, shared_base + ".gate_proj"),
        INTERMEDIATE, HIDDEN,
    )
    shared_up = _decode(
        _load_projection(snapshot, weight_map, shared_base + ".up_proj"),
        INTERMEDIATE, HIDDEN,
    )
    shared_down = _decode(
        _load_projection(snapshot, weight_map, shared_base + ".down_proj"),
        HIDDEN, INTERMEDIATE,
    )
    gate_value = np.asarray(hidden @ shared_gate.T, dtype=np.float32)
    up_value = np.asarray(hidden @ shared_up.T, dtype=np.float32)
    activated = np.asarray(
        gate_value / (np.float32(1.0) + np.exp(-gate_value)) * up_value,
        dtype=np.float32,
    )
    shared = np.asarray(activated @ shared_down.T, dtype=np.float32)
    routed = np.sum(
        outputs.reshape(4, TOP_K, HIDDEN) * route_weights[:, :, None],
        axis=1,
        dtype=np.float32,
    )
    scale = np.asarray(
        1.0 / (1.0 + np.exp(-shared_logits)), dtype=np.float32
    )
    return np.asarray(routed + shared * scale[:, None], dtype=np.float32)


class IntegratedH4:
    def __init__(self, cp, snapshot: Path, weight_map: dict[str, str]):
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4
        from moe_lab.ornith.static_fp8 import StaticFP8H4Quantizer

        self.cp = cp
        self.snapshot = snapshot
        self.weight_map = weight_map
        self.lookup = FusedNVFP4()
        self.fp8 = OrnithFP8H4Kernels()
        self.quantizer = StaticFP8H4Quantizer()
        self.linear = OrnithLinearH4Kernels()
        self.full = OrnithFullAttentionH4Kernels()
        self.support = OrnithSupportH4Kernels()
        self.bulk = OrnithNVFP4BulkM1()
        self.adaptive = OrnithNVFP4RouteAdaptive()
        self.slot_of = cp.arange(EXPERTS, dtype=cp.int32)
        self.quantized_2048 = cp.empty((4, HIDDEN), dtype=cp.uint8)
        self.quantized_4096 = cp.empty((4, 4096), dtype=cp.uint8)
        self.expert_outputs = cp.empty((32, HIDDEN), dtype=cp.float32)

    def projection(self, projection, x, target):
        raw, weight_scale, input_scale = projection
        q = self.quantized_2048 if x.shape[1] == HIDDEN else self.quantized_4096
        weight = self.cp.asarray(raw)
        self.quantizer.quantize(x, q, input_scale)
        self.fp8.m4(
            weight, q, target, raw.shape[0], raw.shape[1],
            weight_scale * input_scale,
        )
        return weight

    def _device_expert_bank(self, layer: int, expert_ids: list[int]):
        cp = self.cp
        bank: dict[str, Any] = {}
        rows = {"gate": (INTERMEDIATE, HIDDEN), "up": (INTERMEDIATE, HIDDEN),
                "down": (HIDDEN, INTERMEDIATE)}
        for projection, (out_rows, in_cols) in rows.items():
            codes = []
            scales = []
            globals_ = []
            for expert in expert_ids:
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj"
                row = _load_projection(self.snapshot, self.weight_map, prefix)
                codes.append(row["codes"].reshape(out_rows, in_cols // 2))
                scales.append(row["scales"].reshape(out_rows, in_cols // 16))
                globals_.append(row["global_scale"])
            bank[projection + "_codes"] = cp.asarray(np.stack(codes))
            bank[projection + "_scales"] = cp.asarray(np.stack(scales))
            bank[projection + "_global"] = cp.asarray(
                np.asarray(globals_, dtype=np.float32)
            )
        return bank

    def _shared_bank(self, layer: int):
        cp = self.cp
        result = {}
        base = f"model.layers.{layer}.mlp.shared_expert"
        for projection, rows, cols in (
            ("gate", INTERMEDIATE, HIDDEN), ("up", INTERMEDIATE, HIDDEN),
            ("down", HIDDEN, INTERMEDIATE),
        ):
            row = _load_projection(
                self.snapshot, self.weight_map, base + f".{projection}_proj"
            )
            result[projection + "_codes"] = cp.asarray(
                row["codes"].reshape(1, rows, cols // 2)
            )
            result[projection + "_scales"] = cp.asarray(
                row["scales"].reshape(1, rows, cols // 16)
            )
            result[projection + "_global"] = cp.asarray(
                np.asarray([row["global_scale"]], dtype=np.float32)
            )
        return result

    def moe(self, layer: int, hidden, ids, route_weights, shared_logits):
        cp = self.cp
        ids_host = cp.asnumpy(ids)
        occurrences, buckets = _bucket_routes(ids_host)
        expert_ids = list(occurrences)
        expert_to_slot = {expert: slot for slot, expert in enumerate(expert_ids)}
        bank = self._device_expert_bank(layer, expert_ids)
        for multiplicity, groups in sorted(buckets.items()):
            count = len(groups)
            slots = cp.asarray(
                [expert_to_slot[expert] for expert, _rows in groups], dtype=cp.int32
            )
            input_ids = cp.asarray(
                [token for _expert, rows in groups for token, _route in rows],
                dtype=cp.int32,
            )
            route_indices = cp.asarray(
                [route for _expert, rows in groups for _token, route in rows],
                dtype=cp.int32,
            )
            gate = cp.empty((count, multiplicity, INTERMEDIATE), dtype=cp.float32)
            up = cp.empty_like(gate)
            act = cp.empty_like(gate)
            out = cp.empty((count, multiplicity, HIDDEN), dtype=cp.float32)
            for projection, target in (("gate", gate), ("up", up)):
                self.adaptive.nvfp4(
                    multiplicity, bank[projection + "_codes"],
                    bank[projection + "_scales"], self.lookup.e2m1, self.lookup.e4m3,
                    hidden, target, bank[projection + "_global"], slots, input_ids,
                    count, INTERMEDIATE, HIDDEN,
                )
            self.bulk.swiglu(gate, up, act, count * multiplicity)
            self.adaptive.nvfp4(
                multiplicity, bank["down_codes"], bank["down_scales"],
                self.lookup.e2m1, self.lookup.e4m3, act.reshape(-1, INTERMEDIATE),
                out, bank["down_global"], slots,
                cp.arange(count * multiplicity, dtype=cp.int32),
                count, HIDDEN, INTERMEDIATE,
            )
            self.expert_outputs[route_indices] = out.reshape(-1, HIDDEN)

        shared = self._shared_bank(layer)
        shared_slots = cp.zeros(1, dtype=cp.int32)
        rows4 = cp.arange(4, dtype=cp.int32)
        shared_gate = cp.empty((1, 4, INTERMEDIATE), dtype=cp.float32)
        shared_up = cp.empty_like(shared_gate)
        shared_act = cp.empty_like(shared_gate)
        shared_out = cp.empty((1, 4, HIDDEN), dtype=cp.float32)
        for projection, target in (("gate", shared_gate), ("up", shared_up)):
            self.adaptive.nvfp4(
                4, shared[projection + "_codes"], shared[projection + "_scales"],
                self.lookup.e2m1, self.lookup.e4m3, hidden, target,
                shared[projection + "_global"], shared_slots, rows4,
                1, INTERMEDIATE, HIDDEN,
            )
        self.bulk.swiglu(shared_gate, shared_up, shared_act, 4)
        self.adaptive.nvfp4(
            4, shared["down_codes"], shared["down_scales"], self.lookup.e2m1,
            self.lookup.e4m3, shared_act.reshape(4, INTERMEDIATE), shared_out,
            shared["down_global"], shared_slots, rows4, 1, HIDDEN, INTERMEDIATE,
        )
        return shared_out.reshape(4, HIDDEN), bank, shared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=TRACE_DEFAULT)
    parser.add_argument("--diagnostic-stop-layer", type=int, default=None)
    parser.add_argument("--local-parity", action="store_true")
    args = parser.parse_args()
    output_name = (
        "S100_PHASE84_ORNITH_TARGET_VERIFIER_H4_LOCAL_DIAGNOSTIC.json"
        if args.diagnostic_stop_layer is not None
        else "S100_PHASE84_ORNITH_TARGET_VERIFIER_H4.json"
    )
    out = RESULTS / output_name
    payload: dict[str, Any] = {
        "kind": "s100_phase84_ornith_target_verifier_h4",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "one integrated target-only H4 correctness gate; not output tok/s",
    }
    cp = None
    try:
        import cupy as cp_module
        import torch
        import torch.nn.functional as F
        import native_nvfp4_c3a_layout_v2 as layout_v2
        import native_nvfp4_c3a_lib as c3lib
        from diag_native_nvfp4_c3b_realact import native_call

        if not hasattr(F, "ScalingType") or not hasattr(F, "SwizzleType"):
            raise RuntimeError(
                "this integrated ERVF-head run requires the native FP4 Torch "
                "environment with F.ScalingType/F.SwizzleType"
            )
        try:
            import safetensors  # noqa: F401
        except ModuleNotFoundError:
            fallback = (
                Path(sys.executable).resolve().parents[2]
                / ".venv-nemotron" / "Lib" / "site-packages"
            )
            if not fallback.is_dir():
                raise
            sys.path.append(str(fallback))
            import safetensors  # noqa: F401

        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import FusedNVFP4

        cp = cp_module
        payload["gpu_idle_preflight"] = require_gpu_idle_wddm()
        snapshot = args.snapshot.expanduser().resolve()
        trace_path = args.trace.expanduser().resolve()
        weight_map = _weight_map(snapshot)
        trace = _load_trace(trace_path)
        tokens = tuple(int(value) for value in trace["tokens"][:4])
        config = json.loads((snapshot / "config.json").read_text("utf-8"))
        layer_types = list(config["layer_types"])
        if len(layer_types) != LAYERS:
            raise ValueError(f"expected {LAYERS} layer types, got {len(layer_types)}")

        embedding = _load_tensor(snapshot, weight_map, "model.embed_tokens.weight")
        residual_host = embedding[list(tokens)].float().numpy().copy()
        del embedding
        residual_ref = residual_host.copy()
        residual = cp.asarray(residual_host)
        normed = cp.empty_like(residual)
        next_normed = cp.empty_like(residual)
        branch = cp.empty_like(residual)
        router_logits = cp.empty((4, EXPERTS), dtype=cp.float32)
        shared_logits = cp.empty(4, dtype=cp.float32)
        route_ids = cp.empty((4, TOP_K), dtype=cp.int32)
        route_weights = cp.empty((4, TOP_K), dtype=cp.float32)
        route_slots = cp.empty((4, TOP_K), dtype=cp.int32)
        route_need = cp.empty((4, TOP_K), dtype=cp.int32)
        runtime = IntegratedH4(cp, snapshot, weight_map)

        linear_states_gpu: dict[int, tuple[Any, Any]] = {}
        linear_states_ref: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        full_caches_gpu: dict[int, tuple[Any, Any]] = {}
        full_caches_ref: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        layer_records = []
        local_records = []

        first_norm_raw, first_norm_float = _load_bf16(
            snapshot, weight_map, "model.layers.0.input_layernorm.weight"
        )
        runtime.support.norm(residual, cp.asarray(first_norm_raw), normed)
        normed_ref = _rmsnorm(residual_ref, first_norm_float)

        for layer, layer_type in enumerate(layer_types):
            dense_prefix = f"model.layers.{layer}"
            residual_input_candidate = cp.asnumpy(residual)
            normed_input_candidate = cp.asnumpy(normed)
            if layer_type == "linear_attention":
                projections = {
                    "qkv": _load_fp8(snapshot, weight_map, dense_prefix + ".linear_attn.in_proj_qkv"),
                    "z": _load_fp8(snapshot, weight_map, dense_prefix + ".linear_attn.in_proj_z"),
                    "out": _load_fp8(snapshot, weight_map, dense_prefix + ".linear_attn.out_proj"),
                }
                auxiliary = _load_auxiliary(snapshot, dense_prefix + ".linear_attn")
                mixed = cp.empty((4, 8192), dtype=cp.float32)
                z = cp.empty((4, 4096), dtype=cp.float32)
                core = cp.empty((4, 4096), dtype=cp.float32)
                beta = cp.empty((4, 32), dtype=cp.float32)
                g = cp.empty((4, 32), dtype=cp.float32)
                convolved = cp.empty((4, 8192), dtype=cp.float32)
                conv_state = cp.zeros((8192, 4), dtype=cp.float32)
                recurrent_state = cp.zeros((32, 128, 128), dtype=cp.float32)
                linear_states_gpu[layer] = (conv_state, recurrent_state)
                aux_gpu = {name: cp.asarray(row["raw"]) for name, row in auxiliary.items()}
                qkv_weight = runtime.projection(projections["qkv"], normed, mixed)
                z_weight = runtime.projection(projections["z"], normed, z)
                runtime.linear.gates(
                    aux_gpu["in_proj_a.weight"], aux_gpu["in_proj_b.weight"], normed,
                    aux_gpu["A_log"], aux_gpu["dt_bias"], beta, g,
                )
                runtime.linear.convolution(
                    mixed, aux_gpu["conv1d.weight"], conv_state, convolved
                )
                runtime.linear.delta_norm(
                    convolved, z, beta, g, aux_gpu["norm.weight"], recurrent_state, core
                )
                out_weight = runtime.projection(projections["out"], core, branch)

                mixed_ref = _fp8_reference(torch, normed_ref, projections["qkv"])
                z_ref = _fp8_reference(torch, normed_ref, projections["z"])
                initial_conv = np.zeros((8192, 4), dtype=np.float32)
                initial_recurrent = np.zeros((32, 128, 128), dtype=np.float32)
                linear_ref = _linear_reference(
                    auxiliary, normed_ref, mixed_ref, z_ref,
                    initial_conv, initial_recurrent,
                )
                linear_states_ref[layer] = (
                    linear_ref["conv_state"], linear_ref["recurrent_state"]
                )
                branch_ref = _fp8_reference(torch, linear_ref["output"], projections["out"])
                branch_local = None
                if args.local_parity:
                    mixed_local = _fp8_reference(
                        torch, normed_input_candidate, projections["qkv"]
                    )
                    z_local = _fp8_reference(
                        torch, normed_input_candidate, projections["z"]
                    )
                    linear_local = _linear_reference(
                        auxiliary, normed_input_candidate, mixed_local, z_local,
                        np.zeros((8192, 4), dtype=np.float32),
                        np.zeros((32, 128, 128), dtype=np.float32),
                    )
                    branch_local = _fp8_reference(
                        torch, linear_local["output"], projections["out"]
                    )
                del qkv_weight, z_weight, out_weight, aux_gpu
            else:
                projections = {
                    "q": _load_fp8(snapshot, weight_map, dense_prefix + ".self_attn.q_proj"),
                    "k": _load_fp8(snapshot, weight_map, dense_prefix + ".self_attn.k_proj"),
                    "v": _load_fp8(snapshot, weight_map, dense_prefix + ".self_attn.v_proj"),
                    "out": _load_fp8(snapshot, weight_map, dense_prefix + ".self_attn.o_proj"),
                }
                norms = _load_attention_norms(snapshot, dense_prefix + ".self_attn")
                q_gate = cp.empty((4, 8192), dtype=cp.float32)
                key = cp.empty((4, 512), dtype=cp.float32)
                value = cp.empty((4, 512), dtype=cp.float32)
                attended = cp.empty((4, 4096), dtype=cp.float32)
                prepared = cp.empty((4, 4096), dtype=cp.float32)
                key_cache = cp.zeros((2, 4, 256), dtype=cp.float32)
                value_cache = cp.zeros_like(key_cache)
                full_caches_gpu[layer] = (key_cache, value_cache)
                q_weight = runtime.projection(projections["q"], normed, q_gate)
                k_weight = runtime.projection(projections["k"], normed, key)
                v_weight = runtime.projection(projections["v"], normed, value)
                cos4, sin4 = _rope(0)
                cos_gpu, sin_gpu = cp.asarray(cos4), cp.asarray(sin4)
                runtime.full.prepare(
                    q_gate, key, value,
                    cp.asarray(norms["q_norm.weight"]["raw"]),
                    cp.asarray(norms["k_norm.weight"]["raw"]),
                    cos_gpu, sin_gpu, prepared, key_cache, value_cache, 0, 4,
                )
                runtime.full.attention(
                    "g1", prepared, q_gate, key_cache, value_cache, attended, 0, 4
                )
                out_weight = runtime.projection(projections["out"], attended, branch)

                q_ref = _fp8_reference(torch, normed_ref, projections["q"])
                k_ref = _fp8_reference(torch, normed_ref, projections["k"])
                v_ref = _fp8_reference(torch, normed_ref, projections["v"])
                zero_key = np.zeros((2, 4, 256), dtype=np.float32)
                zero_value = np.zeros_like(zero_key)
                full_ref = _full_attention_reference(
                    norms, q_ref, k_ref, v_ref, cos4, sin4,
                    zero_key, zero_value, 0,
                )
                ref_key = zero_key.copy()
                ref_value = zero_value.copy()
                ref_key[:, :4] = full_ref["appended_key"]
                ref_value[:, :4] = full_ref["appended_value"]
                full_caches_ref[layer] = (ref_key, ref_value)
                branch_ref = _fp8_reference(torch, full_ref["output"], projections["out"])
                branch_local = None
                if args.local_parity:
                    q_local = _fp8_reference(
                        torch, normed_input_candidate, projections["q"]
                    )
                    k_local = _fp8_reference(
                        torch, normed_input_candidate, projections["k"]
                    )
                    v_local = _fp8_reference(
                        torch, normed_input_candidate, projections["v"]
                    )
                    full_local = _full_attention_reference(
                        norms, q_local, k_local, v_local, cos4, sin4,
                        np.zeros((2, 4, 256), dtype=np.float32),
                        np.zeros((2, 4, 256), dtype=np.float32), 0,
                    )
                    branch_local = _fp8_reference(
                        torch, full_local["output"], projections["out"]
                    )
                del q_weight, k_weight, v_weight, out_weight

            branch_candidate = cp.asnumpy(branch)

            post_norm_raw, post_norm_float = _load_bf16(
                snapshot, weight_map, dense_prefix + ".post_attention_layernorm.weight"
            )
            runtime.support.add_norm(
                residual, branch, cp.asarray(post_norm_raw), normed
            )
            cp.cuda.get_current_stream().synchronize()
            post_attention_residual_candidate = cp.asnumpy(residual)
            post_normed_candidate = cp.asnumpy(normed)
            residual_ref = np.asarray(residual_ref + branch_ref, dtype=np.float32)
            normed_ref = _rmsnorm(residual_ref, post_norm_float)

            router_raw, router_float = _load_bf16(
                snapshot, weight_map, dense_prefix + ".mlp.gate.weight"
            )
            shared_gate_raw, shared_gate_float = _load_bf16(
                snapshot, weight_map, dense_prefix + ".mlp.shared_expert_gate.weight"
            )
            runtime.support.router_shared(
                cp.asarray(router_raw), cp.asarray(shared_gate_raw), normed,
                router_logits, shared_logits,
            )
            runtime.support.top8_cache(
                router_logits, runtime.slot_of, route_ids, route_weights,
                route_slots, route_need,
            )
            cp.cuda.get_current_stream().synchronize()
            router_logits_ref = np.asarray(normed_ref @ router_float.T, dtype=np.float32)
            shared_logits_ref = np.asarray(
                normed_ref @ shared_gate_float.T, dtype=np.float32
            ).reshape(4)
            ids_ref, weights_ref = _top8(router_logits_ref)
            ids_candidate = cp.asnumpy(route_ids)
            router_logits_candidate = cp.asnumpy(router_logits)
            route_weights_candidate = cp.asnumpy(route_weights)
            local_moe = None
            local_ids = None
            local_weights = None
            local_shared_logits = None
            if args.local_parity:
                local_router_logits = np.asarray(
                    post_normed_candidate @ router_float.T, dtype=np.float32
                )
                local_shared_logits = np.asarray(
                    post_normed_candidate @ shared_gate_float.T, dtype=np.float32
                ).reshape(4)
                local_ids, local_weights = _top8(local_router_logits)
            shared_out, bank, shared_bank = runtime.moe(
                layer, normed, route_ids, route_weights, shared_logits
            )
            moe_ref = _host_moe_reference(
                snapshot, weight_map, layer, normed_ref, ids_ref,
                weights_ref, shared_logits_ref,
            )
            if args.local_parity:
                local_moe = _host_moe_reference(
                    snapshot, weight_map, layer, post_normed_candidate,
                    local_ids, local_weights, local_shared_logits,
                )

            if layer + 1 < LAYERS:
                next_raw, next_float = _load_bf16(
                    snapshot, weight_map,
                    f"model.layers.{layer + 1}.input_layernorm.weight",
                )
            else:
                next_raw, next_float = _load_bf16(
                    snapshot, weight_map, "model.norm.weight"
                )
            runtime.support.combine_norm(
                residual, runtime.expert_outputs, route_weights, shared_out,
                shared_logits, cp.asarray(next_raw), next_normed,
            )
            residual_ref = np.asarray(residual_ref + moe_ref, dtype=np.float32)
            normed_ref = _rmsnorm(residual_ref, next_float)
            cp.cuda.get_current_stream().synchronize()
            residual_candidate = cp.asnumpy(residual)
            normed_candidate = cp.asnumpy(next_normed)
            layer_records.append({
                "layer": layer,
                "type": layer_type,
                "routes_exact": bool(np.array_equal(ids_candidate, ids_ref)),
                "residual_nrmse": _nrmse(residual_candidate, residual_ref),
                "normed_nrmse": _nrmse(normed_candidate, normed_ref),
                "finite": bool(
                    np.isfinite(residual_candidate).all()
                    and np.isfinite(normed_candidate).all()
                ),
            })
            if args.local_parity:
                local_records.append({
                    "layer": layer,
                    "type": layer_type,
                    "attention_branch_nrmse": _nrmse(
                        branch_candidate, branch_local
                    ),
                    "post_attention_residual_nrmse": _nrmse(
                        post_attention_residual_candidate,
                        np.asarray(
                            residual_input_candidate + branch_local,
                            dtype=np.float32,
                        ),
                    ),
                    "router_logits_nrmse": _nrmse(
                        router_logits_candidate, local_router_logits
                    ),
                    "routes_exact": bool(
                        np.array_equal(ids_candidate, local_ids)
                    ),
                    "route_weights_nrmse": _nrmse(
                        route_weights_candidate, local_weights
                    ),
                    "moe_branch_nrmse": _nrmse(
                        residual_candidate - post_attention_residual_candidate,
                        local_moe,
                    ),
                })
            print(json.dumps({
                "layer": layer,
                "routes_exact": layer_records[-1]["routes_exact"],
                "residual_nrmse": layer_records[-1]["residual_nrmse"],
                "normed_nrmse": layer_records[-1]["normed_nrmse"],
            }), flush=True)
            normed, next_normed = next_normed, normed
            del bank, shared_bank
            cp.get_default_memory_pool().free_all_blocks()
            if (
                args.diagnostic_stop_layer is not None
                and layer >= args.diagnostic_stop_layer
            ):
                payload.update({
                    "status": "measured_diagnostic",
                    "completed_utc": utc_now(),
                    "inputs": {
                        "snapshot": str(snapshot),
                        "trace": str(trace_path),
                        "tokens": list(tokens),
                        "diagnostic_stop_layer": layer,
                    },
                    "quality": {
                        "layer_records": layer_records,
                        "local_records": local_records,
                    },
                })
                raise DiagnosticComplete

        final_candidate = cp.asnumpy(normed)

        payload["body_checkpoint"] = {
            "layers_completed": len(layer_records),
            "all_routes_exact": all(row["routes_exact"] for row in layer_records),
            "layer_records": layer_records,
        }

        state_quality = []
        for layer, (conv_gpu, recurrent_gpu) in linear_states_gpu.items():
            conv_ref, recurrent_ref = linear_states_ref[layer]
            state_quality.append({
                "layer": layer,
                "type": "linear_attention",
                "conv_nrmse": _nrmse(cp.asnumpy(conv_gpu), conv_ref),
                "recurrent_nrmse": _nrmse(cp.asnumpy(recurrent_gpu), recurrent_ref),
            })
        for layer, (key_gpu, value_gpu) in full_caches_gpu.items():
            key_ref, value_ref = full_caches_ref[layer]
            state_quality.append({
                "layer": layer,
                "type": "full_attention",
                "key_nrmse": _nrmse(cp.asnumpy(key_gpu), key_ref),
                "value_nrmse": _nrmse(cp.asnumpy(value_gpu), value_ref),
            })

        head = _load_projection(snapshot, weight_map, "lm_head")
        fused = FusedNVFP4()
        codes = cp.asarray(head["codes"])
        scales = cp.asarray(head["scales"])
        logits_candidate = cp.empty((4, HEAD_ROWS), dtype=cp.float32)
        logits_reference = cp.empty_like(logits_candidate)
        reference_gpu = cp.asarray(normed_ref)
        for token in range(4):
            fused.gemv_into(
                logits_candidate[token], codes, scales, normed[token],
                head["global_scale"], HEAD_ROWS, HIDDEN,
            )
            fused.gemv_into(
                logits_reference[token], codes, scales, reference_gpu[token],
                head["global_scale"], HEAD_ROWS, HIDDEN,
            )
        cp.cuda.get_current_stream().synchronize()
        logits_candidate_host = cp.asnumpy(logits_candidate)
        logits_reference_host = cp.asnumpy(logits_reference)
        control_ids = np.argmax(logits_candidate_host, axis=1)
        reference_ids = np.argmax(logits_reference_host, axis=1)

        layout_v2.install(c3lib)
        b = c3lib.make_b(
            torch, head["codes"].tobytes(), head["scales"].tobytes(),
            head["global_scale"], HEAD_ROWS, HIDDEN,
        )
        tensor_scale = float(
            np.max(np.abs(final_candidate)) * 1.10 / (448.0 * 6.0)
        )
        quantizer = FusedStaticNVFP4Quantizer(HIDDEN, 4)
        quantizer.quantize(normed, tensor_scale)
        stream = cp.cuda.get_current_stream()
        stream.synchronize()
        external = torch.cuda.ExternalStream(stream.ptr)
        packed_t = torch.utils.dlpack.from_dlpack(quantizer.packed)
        blocked_t = torch.utils.dlpack.from_dlpack(quantizer.blocked_scales)
        a_global = torch.tensor([tensor_scale], dtype=torch.float32, device="cuda")
        a = {
            "fp4": packed_t.view(torch.float4_e2m1fn_x2),
            "block": blocked_t.view(torch.float8_e4m3fn),
            "global": a_global,
        }
        with torch.cuda.stream(external):
            native = native_call(torch, F, F.ScalingType, F.SwizzleType, a, b)
            _values, ids_t = torch.topk(native, SHORTLIST, dim=1)
        external.synchronize()
        shortlist_ids = cp.from_dlpack(ids_t)
        rerank = cp.empty((4, SHORTLIST), dtype=cp.float32)
        exact = ExactERVFShortlist()
        exact(
            codes, scales, fused.e2m1, fused.e4m3, normed,
            shortlist_ids.data.ptr, rerank.data.ptr,
            head["global_scale"], SHORTLIST, HIDDEN,
        )
        selected = cp.take_along_axis(
            shortlist_ids, cp.argmax(rerank, axis=1).reshape(-1, 1), axis=1
        ).reshape(-1)
        stream.synchronize()
        selected_host = cp.asnumpy(selected)
        shortlist_host = cp.asnumpy(shortlist_ids)

        final_nrmse = _nrmse(final_candidate, normed_ref)
        logits_nrmse = _nrmse(logits_candidate_host, logits_reference_host)
        state_ok = all(
            max(value for key, value in row.items() if key.endswith("_nrmse")) <= 2e-3
            for row in state_quality
        )
        gates = {
            "P84_H4_G1_all40_routes_exact": all(row["routes_exact"] for row in layer_records),
            "P84_H4_G2_full_residual_parity": (
                final_nrmse <= 2e-3 and all(row["finite"] for row in layer_records)
            ),
            "P84_H4_G3_state_parity": state_ok,
            "P84_H4_G4_full_logit_parity": (
                logits_nrmse <= 2e-3 and np.array_equal(control_ids, reference_ids)
            ),
            "P84_H4_G5_ervf_exact_top1": bool(
                np.array_equal(selected_host, control_ids)
                and all(control_ids[row] in shortlist_host[row] for row in range(4))
            ),
        }
        payload.update({
            "status": "measured_pass" if all(gates.values()) else "measured_fail",
            "completed_utc": utc_now(),
            "inputs": {"snapshot": str(snapshot), "trace": str(trace_path), "tokens": list(tokens)},
            "quality": {
                "layer_records": layer_records,
                "state_records": state_quality,
                "final_normed_nrmse": final_nrmse,
                "complete_logits_nrmse": logits_nrmse,
                "control_top1": control_ids.astype(np.int64).tolist(),
                "reference_top1": reference_ids.astype(np.int64).tolist(),
                "ervf_top1": selected_host.astype(np.int64).tolist(),
                "ervf_top64_contains_control": [
                    bool(control_ids[row] in shortlist_host[row]) for row in range(4)
                ],
            },
            "gates": gates,
        })
    except DiagnosticComplete:
        pass
    except Exception as error:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        })
    finally:
        if cp is not None:
            try:
                cp.cuda.get_current_stream().synchronize()
            except Exception:
                pass
        payload["environment"] = environment_snapshot((SCRIPT, PREREG, args.trace))
        write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "quality": payload.get("quality"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2), flush=True)
    return 0 if payload.get("status") in {"measured_pass", "measured_diagnostic"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
