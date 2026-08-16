"""Which stage of `_attention` holds its 1.35 ms of headroom?

The in-graph attribution (diag_component_marginals_graph, all gates green, drift
0.378 ms) puts attention at **2.479 ms against a 1.128 ms floor -- 45.5%, the
least efficient path in the model**: Mamba runs at 69.3%, down_masked at 60%,
shared_expert at 90%.

Two things point here and nowhere else. PV2-11 tried to fuse the Q/K/V
projections and came out **2.628 ms slower**, because it replaced the
selective-ERVF Q path with a plain BF16 kernel -- so the inefficiency is not the
projection launch count, and is probably not the projections at all. And
attention is the one component whose marginal is *higher* in the graph
(2.479) than eager (1.917), which nothing yet explains.

`_attention` in the graph path (runtime.py:431) is five stages:

    mv_bf16 x3            Q, K, V projections (Q and O go through selective ERVF)
    kv_write_fp8_dp x2    append K and V to the fp8 cache at the device position
    attention_fp8_gqa4_dp the GQA flash-decode itself, with split partials
    mv_bf16               the O projection

Marginal probes, one stage at a time, in the captured graph -- the same method
that localised down_masked without a profiler. Every stage here is idempotent:
the projections overwrite `qv`/`kv_`/`vv`, the KV writes address the same
device position with the same bytes, the flash-decode overwrites `ctx`, and the
O projection overwrites `out`. That is asserted by the bit-exactness gate, not
assumed -- the naive `_mamba` probe already proved that assumption can be wrong.

At ctx <= 4096 the KV cache is tiny (6 attention layers x 4096 positions x
512 B = 12.6 MB) so the flash-decode should be nearly free and the projections
(281 MB/token) should dominate. If the measurement says otherwise, that is the
finding.

Gates: G1 every arm bit-exact against BASE_A; G2 drift <= 1.0 ms.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import types
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

OUT = REPO / "pro_research" / "diag_attention_stage_marginals.json"
STAGES = ("qkv_proj", "q_proj", "kv_proj", "kv_write", "flash_decode", "o_proj")

# 6 attention layers; bytes per token per stage, from the safetensors headers.
BYTES = {
    "qkv_proj": 6 * (2688 * 4096 + 2 * 2688 * 256) * 2,   # BF16 Q,K,V
    "q_proj":   6 * (2688 * 4096) * 2,                     # BF16 Q alone
    "kv_proj":  6 * (2 * 2688 * 256) * 2,                  # BF16 K+V alone
    "o_proj":   6 * (4096 * 2688) * 2,                     # BF16 O
}


def _recapture(rt) -> None:
    cp = rt.cp
    s = rt._graph_stream
    rt._graph = None
    rt._step_body_graph()
    cp.cuda.Device(0).synchronize()
    s.begin_capture()
    with s:
        rt._step_body_graph()
    rt._graph = s.end_capture()
    s.synchronize()
    rt.reset()


def _reset_exact_state(rt) -> None:
    import cupy as cp

    rt._graph_stream.synchronize()
    rt.reset()
    for dev in getattr(rt, "_dev_cache", {}).values():
        for name in ("ids", "w", "slots", "need", "state2", "stats2"):
            if name in dev:
                dev[name].fill(0)
        for name, val in (("slot_of", -1), ("expert_of", -1), ("last_used", -1)):
            if name in dev:
                dev[name].fill(val)
    rt._ring_i = 0
    rt._ring_np[:] = np.int32(-1)
    cp.cuda.Device(0).synchronize()


def _prefill(rt, prompt_ids: list[int]) -> int:
    start = int(rt._ring_i)
    for tok in prompt_ids:
        rt.step_graph(int(tok))
        rt._graph_stream.synchronize()
    return int(rt.ring_harvest((start + len(prompt_ids) - 1) % int(rt._ring_size), 1)[0])


def _run(rt, prompt_ids, n):
    _reset_exact_state(rt)
    ids, ms = [_prefill(rt, prompt_ids)], []
    for _ in range(n - 1):
        slot = int(rt._ring_i)
        t0 = time.perf_counter_ns()
        rt.step_graph(None)
        tok = int(rt.ring_harvest(slot, 1)[0])
        ms.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(tok)
    return ids, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "diag_attention_stage_marginals",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "why": "attention is 2.479 ms against a 1.128 ms floor in-graph -- 45.5%, the least efficient path (Mamba 69.3%, down_masked 60%, shared_expert 90%). PV2-11's fusion of the Q/K/V projections came out 2.628 ms SLOWER, so the projections are probably not where it sits.",
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _e, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 24) if args.mode == "smoke" else max(n, 192)
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts)}
        payload["environment"] = environment_snapshot()

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, _ = _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)
        rt.setup_graph()

        orig_attn = rt._attention
        scratch_ctx = cp.zeros_like(rt.ctx)
        scratch_out = cp.zeros(rt.hidden, dtype=cp.float32)

        def make_probe(stage: str):
            def attention(self, i, out):
                r = orig_attn(i, out)
                k, d = self.k, self.layer[i]
                qrows = self.n_heads * self.head_dim
                scale = 1.0 / float(np.sqrt(self.head_dim))
                if stage == "qkv_proj":
                    k.mv_bf16(self.qv, d["q_proj"], self.normed, qrows, self.hidden)
                    k.mv_bf16(self.kv_, d["k_proj"], self.normed, self.kv_dim, self.hidden)
                    k.mv_bf16(self.vv, d["v_proj"], self.normed, self.kv_dim, self.hidden)
                elif stage == "q_proj":
                    # Q alone: 132 MB of the 148.6, and the only one that
                    # goes through selective ERVF.
                    k.mv_bf16(self.qv, d["q_proj"], self.normed, qrows, self.hidden)
                elif stage == "kv_proj":
                    # K and V alone: 16.8 MB, but only 256 rows each, i.e.
                    # 16 blocks at ERVF-16 geometry on a 26-SM device.
                    k.mv_bf16(self.kv_, d["k_proj"], self.normed, self.kv_dim, self.hidden)
                    k.mv_bf16(self.vv, d["v_proj"], self.normed, self.kv_dim, self.hidden)
                elif stage == "kv_write":
                    k.kv_write_fp8_dp(self.kc[i], self.kv_, self._pos_dev,
                                      self.n_kv, self.head_dim, self.max_ctx)
                    k.kv_write_fp8_dp(self.vc[i], self.vv, self._pos_dev,
                                      self.n_kv, self.head_dim, self.max_ctx)
                elif stage == "flash_decode":
                    k.attention_fp8_gqa4_dp(scratch_ctx, self.qv, self.kc[i], self.vc[i],
                                            self._pos_dev, self.n_heads, self.head_dim,
                                            self.groups, self.max_ctx, scale,
                                            self.part_acc, self.part_ml)
                elif stage == "o_proj":
                    k.mv_bf16(scratch_out, d["o_proj"], self.ctx, self.hidden, qrows)
                return r
            return attention

        _reset_exact_state(rt)
        _prefill(rt, prompts[0]["prompt_ids"])
        for _ in range(32 if args.mode == "smoke" else 128):
            rt.step_graph(None)
        rt._graph_stream.synchronize()

        def arm(label, install, restore):
            install()
            _recapture(rt)
            ids_by, ms_all = {}, []
            for p in prompts:
                ids, ms = _run(rt, p["prompt_ids"], n)
                ids_by[p["prompt"]] = ids
                ms_all.extend(ms)
            restore()
            return label, ids_by, percentiles(ms_all)

        noop = lambda: None
        results = [arm("BASE_A", noop, noop)]
        for st in STAGES:
            results.append(arm(
                st,
                lambda st=st: setattr(rt, "_attention", types.MethodType(make_probe(st), rt)),
                lambda: setattr(rt, "_attention", orig_attn)))
        results.append(arm("BASE_B", noop, noop))

        by = {lab: (ids, pc) for lab, ids, pc in results}
        base_ids, a = by["BASE_A"]
        _, b = by["BASE_B"]
        drift = abs(float(a["p50"]) - float(b["p50"]))
        midpoint = (float(a["p50"]) + float(b["p50"])) / 2.0

        arms_out, marginals = {}, {}
        for lab, ids, pc in results:
            divs = {p["prompt"]: first_divergence(base_ids[p["prompt"]], ids[p["prompt"]])
                    for p in prompts}
            arms_out[lab] = {"percentiles": pc,
                             "ids_match_base_a": all(v is None for v in divs.values()),
                             "first_divergence": divs}
            if lab in STAGES:
                marg = float(pc["p50"]) - midpoint
                byt = BYTES.get(lab)
                marginals[lab] = {
                    "marginal_ms_per_token": marg,
                    "bytes_per_token": byt,
                    "floor_ms_at_249_GB_s": (byt / (249.0 * 1e9) * 1e3) if byt else None,
                    "achieved_GB_s": (byt / (marg * 1e-3) / 1e9) if (byt and marg > 0) else None,
                }

        total = sum(v["marginal_ms_per_token"] for v in marginals.values())
        gates = {
            "G1_all_arms_ids_match_base_a": all(v["ids_match_base_a"] for v in arms_out.values()),
            "G2_drift_le_1ms": drift <= 1.0,
        }
        payload.update({
            "arms": arms_out,
            "baseline_midpoint_ms": midpoint,
            "drift_ms": drift,
            "marginals": marginals,
            "sum_of_stage_marginals_ms": total,
            "attention_total_marginal_in_graph_ms": 2.479,
            "attention_floor_ms": 1.128,
            "gates": gates,
            "status": ("correctness_failed" if not gates["G1_all_arms_ids_match_base_a"]
                       else "measurement_unstable" if not gates["G2_drift_le_1ms"]
                       else "measured"),
            "completed_utc": utc_now(),
        })
        restore_sel()
    except Exception as exc:
        payload.update({"status": "technical_failure",
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()},
                        "completed_utc": utc_now()})

    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload.get("status"),
                      "baseline_midpoint_ms": payload.get("baseline_midpoint_ms"),
                      "drift_ms": payload.get("drift_ms"),
                      "marginals": payload.get("marginals"),
                      "sum_of_stage_marginals_ms": payload.get("sum_of_stage_marginals_ms"),
                      "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
