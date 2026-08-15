"""N6-A runner: full-depth 52-layer forward, natural routes, coherence.

Executes ``N6A_FULL_DEPTH_FORWARD_PREREGISTRATION_2026-08-14.md``.

CPU only, float64 accumulation, N3-validated numpy modules.  Weights are
dequantised per layer and released, so no full BF16 model is materialised.
No timing claim is made and the GPU is left free for the protected line.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron import reference as ref  # noqa: E402
from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

PROMPTS = {
    "P1": "The capital of France is",
    "P2": "1, 2, 3, 4,",
    "P3": "def add(a, b):\n    return",
}
PRIMARY = "P1"
PRIMARY_EXPECT = "paris"
GIB = 1024 ** 3


class _MemCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def process_memory() -> dict:
    try:
        c = _MemCounters()
        c.cb = ctypes.sizeof(_MemCounters)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemCounters), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        if not fn(h, ctypes.byref(c), c.cb) or c.WorkingSetSize == 0:
            return {"error": "GetProcessMemoryInfo failed"}
        return {"working_set_bytes": int(c.WorkingSetSize),
                "peak_working_set_bytes": int(c.PeakWorkingSetSize),
                "commit_bytes": int(c.PrivateUsage),
                "peak_commit_bytes": int(c.PeakPagefileUsage)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def forward(index: ShardIndex, token_ids: list[int], cfg: dict) -> dict:
    """Full-depth prefill forward. Returns final logits and natural routes."""
    pattern = cfg["hybrid_override_pattern"]
    eps = cfg["layer_norm_epsilon"]
    hidden_size = cfg["hidden_size"]

    embed = index.get_float32("backbone.embeddings.weight")
    h = embed[np.asarray(token_ids, dtype=np.int64)].astype(np.float64)

    mamba_cfg = {
        "mamba_num_heads": cfg["mamba_num_heads"],
        "mamba_head_dim": cfg["mamba_head_dim"],
        "ssm_state_size": cfg["ssm_state_size"],
        "n_groups": cfg["n_groups"],
        "conv_kernel": cfg["conv_kernel"],
        "layer_norm_epsilon": eps,
        "time_step_limit": (0.0, float("inf")),
    }

    routes: dict[str, dict] = {}
    for layer, kind_char in enumerate(pattern):
        prefix = f"backbone.layers.{layer}"
        norm_w = index.get_float32(f"{prefix}.norm.weight")
        normed = ref.rms_norm(h, norm_w, eps)

        if kind_char == "M":
            # Layers 4, 11, 18, 25, 32 and 41 keep in_proj/out_proj in BF16 --
            # they are in the checkpoint's exclude_modules list. load_linear
            # handles quantised and unquantised uniformly.
            w = {
                "in_proj": index.load_linear(f"{prefix}.mixer.in_proj"),
                "out_proj": index.load_linear(f"{prefix}.mixer.out_proj"),
                "conv1d_weight": index.get_float32(f"{prefix}.mixer.conv1d.weight"),
                "conv1d_bias": index.get_float32(f"{prefix}.mixer.conv1d.bias"),
                "A_log": index.get_float32(f"{prefix}.mixer.A_log"),
                "D": index.get_float32(f"{prefix}.mixer.D"),
                "dt_bias": index.get_float32(f"{prefix}.mixer.dt_bias"),
                "norm_weight": index.get_float32(f"{prefix}.mixer.norm.weight"),
            }
            out = ref.mamba2_forward(normed, w, mamba_cfg)["output"]
            del w

        elif kind_char == "*":
            q = index.get_float32(f"{prefix}.mixer.q_proj.weight")
            k = index.get_float32(f"{prefix}.mixer.k_proj.weight")
            v = index.get_float32(f"{prefix}.mixer.v_proj.weight")
            o = index.get_float32(f"{prefix}.mixer.o_proj.weight")
            out = ref.attention_forward(
                normed, q, k, v, o, cfg["num_attention_heads"],
                cfg["num_key_value_heads"], cfg["head_dim"])["output"]
            del q, k, v, o

        else:  # "E" -- MoE
            gate_w = index.get_float32(f"{prefix}.mixer.gate.weight")
            gate_b = index.get_float32(f"{prefix}.mixer.gate.e_score_correction_bias")
            idx, wts, diag = ref.router(
                normed, gate_w, gate_b, cfg["num_experts_per_tok"],
                cfg["routed_scaling_factor"], cfg["norm_topk_prob"])

            routes[str(layer)] = {
                "indices": idx.tolist(),
                "weights": wts.tolist(),
                "tie_margin": diag["tie_margin"].tolist(),
            }

            needed = sorted({int(e) for row in idx for e in row})
            up_w, down_w = {}, {}
            for e in needed:
                ep = f"{prefix}.mixer.experts.{e}"
                up_w[e] = index.dequantize_linear(f"{ep}.up_proj")
                down_w[e] = index.dequantize_linear(f"{ep}.down_proj")

            sp = f"{prefix}.mixer.shared_experts"
            shared_up = index.dequantize_linear(f"{sp}.up_proj")
            shared_down = index.dequantize_linear(f"{sp}.down_proj")

            moe = ref.moe_forward(
                normed, up_w, down_w, gate_w, gate_b, shared_up, shared_down,
                cfg["num_experts_per_tok"], cfg["routed_scaling_factor"],
                cfg["norm_topk_prob"])
            out = moe["output"]
            del up_w, down_w, shared_up, shared_down, moe, gate_w, gate_b

        h = h + out
        del out, normed, norm_w
        gc.collect()

    final_norm = index.get_float32("backbone.norm_f.weight")
    h = ref.rms_norm(h, final_norm, eps)
    lm_head = index.get_float32("lm_head.weight")
    logits = h[-1] @ lm_head.astype(np.float64).T
    del lm_head

    return {"logits": logits, "routes": routes, "hidden_last": h[-1]}


def main() -> int:
    from transformers import AutoTokenizer

    started = utc_now()
    index = ShardIndex(MODEL_DIR)
    cfg = index.config
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    results = {}
    all_routes = {}
    for pid, text in PROMPTS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(f"{pid}: {text!r} -> {len(ids)} tokens")
        t0 = time.perf_counter()
        out = forward(index, ids, cfg)
        elapsed = time.perf_counter() - t0

        logits = out["logits"]
        finite = bool(np.isfinite(logits).all())
        order = np.argsort(-logits)
        top5 = [{"id": int(i), "logit": float(logits[i]),
                 "text": tokenizer.decode([int(i)])} for i in order[:5]]
        shifted = logits - logits.max()
        probs = np.exp(shifted)
        probs /= probs.sum()

        results[pid] = {
            "prompt": text,
            "token_ids": ids,
            "n_tokens": len(ids),
            "seconds": elapsed,
            "logits_finite": finite,
            "logits_shape": list(logits.shape),
            "top5": top5,
            "top1_id": int(order[0]),
            "top1_text": tokenizer.decode([int(order[0])]),
            "top1_prob": float(probs[order[0]]),
            "entropy_nats": float(-(probs * np.log(probs + 1e-30)).sum()),
            "hidden_last_finite": bool(np.isfinite(out["hidden_last"]).all()),
        }
        all_routes[pid] = out["routes"]
        print(f"   top1 = {results[pid]['top1_text']!r} "
              f"(p={results[pid]['top1_prob']:.4f}, {elapsed:.1f}s)")
        del out
        gc.collect()

    # ------------------------------------------------------------- route audit
    moe_layers = [i for i, c in enumerate(cfg["hybrid_override_pattern"]) if c == "E"]
    route_audit = {"moe_layers_expected": len(moe_layers), "per_prompt": {}}
    routes_ok = True
    for pid, layers in all_routes.items():
        n_layers = len(layers)
        ids_ok = all(
            len(row) == cfg["num_experts_per_tok"] and all(0 <= int(e) < cfg["n_routed_experts"] for e in row)
            for lay in layers.values() for row in lay["indices"])
        w_ok = all(
            np.isfinite(row).all() and (np.asarray(row) > 0).all()
            for lay in layers.values() for row in lay["weights"])
        sums = [float(np.sum(row)) for lay in layers.values() for row in lay["weights"]]
        scale = cfg["routed_scaling_factor"]
        sum_ok = all(abs(s - scale) < 1e-6 for s in sums)
        route_audit["per_prompt"][pid] = {
            "moe_layers_seen": n_layers,
            "layers_match": n_layers == len(moe_layers),
            "indices_valid": ids_ok,
            "weights_valid": w_ok,
            "weight_sums_equal_scaling_factor": sum_ok,
            "example_sum": sums[0] if sums else None,
        }
        routes_ok = routes_ok and n_layers == len(moe_layers) and ids_ok and w_ok and sum_ok

    primary = results[PRIMARY]
    coherent = PRIMARY_EXPECT in primary["top1_text"].strip().lower()
    proc = process_memory()

    gates = {
        "C1_all_prompts_completed_52_layers": len(results) == len(PROMPTS),
        "C2_all_logits_finite": all(r["logits_finite"] and r["hidden_last_finite"]
                                    for r in results.values()),
        "C3_natural_routes_all_moe_layers": routes_ok,
        "C4_route_weights_valid": routes_ok,
        "C5_primary_prompt_coherent": coherent,
        "C6_distribution_not_degenerate": all(r["top1_prob"] < 0.999 for r in results.values()),
        "C7_process_commit_under_32gib": (
            "error" not in proc and 0 < proc.get("peak_commit_bytes", 0) <= 32 * GIB),
    }

    result = {
        "kind": "lightningstream_nemotron_n6a_full_depth_forward",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N6_A_FULL_DEPTH_FORWARD",
        "started_utc": started, "completed_utc": utc_now(),
        "runner_sha256": sha256_path(Path(__file__)),
        "reference_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/reference.py"),
        "device": "cpu",
        "gpu_used": False,
        "timing_claim": False,
        "bf16_model_materialised": False,
        "layers": len(cfg["hybrid_override_pattern"]),
        "hybrid_override_pattern": cfg["hybrid_override_pattern"],
        "prompts": results,
        "route_audit": route_audit,
        "primary_prompt": PRIMARY,
        "primary_expectation": PRIMARY_EXPECT,
        "primary_coherent": coherent,
        "process_memory": proc,
        "gates": gates,
        "gates_all_pass": all(gates.values()),
        "settles": {
            "gated_rmsnorm_deferred_from_n3": coherent,
            "nibble_order_joint_confirmation": coherent,
            "dequant_grouping_joint_confirmation": coherent,
            "note": ("Coherence is a JOINT test of these assumptions. Passing "
                     "supports all of them together; failing would not identify "
                     "which one is wrong."),
        },
        "claim_boundary": (
            "The assembled 52-layer graph runs to completion on real weights and "
            "produces specific next-token predictions for three frozen prompts, "
            "plus the natural routes observed for those prompts. NOT model "
            "quality, benchmark scores, tokens per second, latency, general "
            "runtime correctness, or a representative routing distribution -- "
            "three prompts are three prompts. A component measurement is never "
            "promoted to tok/s."
        ),
    }
    result["terminal_state"] = (
        "n6a_full_depth_coherent" if result["gates_all_pass"]
        else ("n6a_incoherent_assumption_set_wrong" if not coherent
              else "n6a_full_depth_fail"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "n6a_full_depth_forward.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "n6a_natural_routes.json").write_text(
        json.dumps({
            "kind": "lightningstream_nemotron_n6a_natural_routes",
            "provenance": ("Natural routes from a real full-depth forward on three "
                           "frozen prompts. NOT a representative routing "
                           "distribution."),
            "prompts": {k: v["prompt"] for k, v in results.items()},
            "top_k": cfg["num_experts_per_tok"],
            "n_routed_experts": cfg["n_routed_experts"],
            "routes": all_routes,
        }, indent=2) + "\n", encoding="utf-8")

    print()
    for pid, r in results.items():
        print(f"  {pid}: top1={r['top1_text']!r} p={r['top1_prob']:.4f} "
              f"H={r['entropy_nats']:.3f}")
    print()
    for key, value in gates.items():
        print(f"  {'OK  ' if value else 'FAIL'} {key}")
    print(f"terminal state : {result['terminal_state']}")
    return 0 if result["gates_all_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
