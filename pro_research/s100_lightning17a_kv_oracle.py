"""S100 Lightning Phase 17A: exact-replay K/V oracle ceiling inside the CUDA graph.

Question: how many ms/token do the eleven heldout-green K/V projections
(tc1_pair_kv_minus__a19_k) actually cost the production graph parent?

Method: a recorder graph first stores the bit-exact FP32 outputs of every
selected k_proj/v_proj GEMV into a device table indexed by (prompt, position)
-- both read from device counters, so the mechanism is capturable. An oracle
graph then replaces those GEMV launches with a 1 KiB device-to-device table
load. Everything else in the captured token body is untouched.

Gates (preregistered in the Phase-16R handoff discussion):
- KV_ORACLE_PARITY: produced token ids (64 per prompt) and bit-exact
  final-position logits (10/10 prompts) identical to the graph parent.
- KV_ORACLE_3PCT_OPEN: oracle aggregate tok/s >= 1.03x graph reference AND
  p50 latency >= 1.03x faster.
- KV_ORACLE_S100_GAP_COVERAGE: savings_ms / gap_to_10ms (reportage).

Protocol: the first post-build workload pass is a known outlier and is
discarded (warm pass); a plain recapture (control A) is the parity and
timing baseline; a restored-parent recapture (control B) brackets drift.

Addendum (frozen Phase-17 plan): an overhead arm keeps the original K/V
GEMVs running while executing the same table-load copies into scratch, and
a prompt-clustered bootstrap (10,000 resamples) puts one-sided 95% bounds
on bracket-corrected savings. KV_ORACLE_DEFINITIVELY_CLOSED requires
parity and a corrected upper bound below 3% of the parent latency.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback

import numpy as np

from common import write_json_atomic, utc_now
from s100_phase10a_runtime import build
from diag_component_marginals_graph import _recapture
from s100_lightning16_common import assert_lightning, case_manifest
from s100_lightning16r_throughput import frozen_workload, metrics

RESULTS_DIRNAME = "s100_lightning17a"
WARMUP_TARGETS = 8
MIN_SPEEDUP = 1.03
MAX_POS = 512
TARGET_MS = 10.0  # 100 tok/s

# Heldout-green winner of Phase 16R: five K/V pairs plus attention_19_v.
ORACLE_CASES = [
    "attention_5_k", "attention_12_k", "attention_26_k",
    "attention_33_k", "attention_42_k",
    "attention_5_v", "attention_12_v", "attention_19_v",
    "attention_26_v", "attention_33_v", "attention_42_v",
]

KERNELS = r"""
extern "C" __global__ void kv_store(float* base, const float* src,
                                    const int* pos, const int* pctr,
                                    int max_pos, int n_prompts, int dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    int p = pctr[0];
    if (p >= n_prompts) p = n_prompts - 1;  // capture/warmup safety
    long long idx = ((long long)p * max_pos + pos[0]) * dim + t;
    base[idx] = src[t];
}
extern "C" __global__ void kv_load(float* dst, const float* base,
                                   const int* pos, const int* pctr,
                                   int max_pos, int n_prompts, int dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    int p = pctr[0];
    if (p >= n_prompts) p = n_prompts - 1;
    long long idx = ((long long)p * max_pos + pos[0]) * dim + t;
    dst[t] = base[idx];
}
extern "C" __global__ void ctr_inc(int* p) { p[0] += 1; }
"""

class OracleHook:
    """mv_bf16 replacement active only during graph (re)capture.

    record: run the original GEMV, then store its FP32 output into the table.
    oracle: skip the GEMV; load the recorded vector from the table.
    overhead: run the original GEMV AND execute the same table-load copy
        into scratch, so the replay-copy cost is measured without removing
        the real K/V kernels.
    Non-selected weights pass through untouched.
    """

    def __init__(self, rt, *, mode, original, slot_of, table,
                 slot_stride, pctr, n_prompts, kern_store, kern_load,
                 scratch=None):
        self.rt = rt
        self.mode = mode
        self.original = original
        self.slot_of = slot_of
        self.table = table
        self.slot_stride = int(slot_stride)
        self.pctr = pctr
        self.n_prompts = np.int32(n_prompts)
        self.kern_store = kern_store
        self.kern_load = kern_load
        self.scratch = scratch
        self.max_pos = np.int32(MAX_POS)
        self.oracle_calls = 0
        self.passthrough_calls = 0

    def __call__(self, out, weight, x, rows, cols):
        slot = self.slot_of.get(int(weight.data.ptr))
        if slot is None:
            self.passthrough_calls += 1
            return self.original(out, weight, x, rows, cols)
        dim = int(rows)
        base = self.table[slot * self.slot_stride:
                          (slot + 1) * self.slot_stride]
        if self.mode == "record":
            self.original(out, weight, x, rows, cols)
            self.kern_store(
                (1,), (256,),
                (base, out, self.rt._pos_dev, self.pctr,
                 self.max_pos, self.n_prompts, np.int32(dim)),
            )
        elif self.mode == "overhead":
            self.original(out, weight, x, rows, cols)
            dst = self.scratch[slot * dim:(slot + 1) * dim]
            self.kern_load(
                (1,), (256,),
                (dst, base, self.rt._pos_dev, self.pctr,
                 self.max_pos, self.n_prompts, np.int32(dim)),
            )
        else:
            self.kern_load(
                (1,), (256,),
                (out, base, self.rt._pos_dev, self.pctr,
                 self.max_pos, self.n_prompts, np.int32(dim)),
            )
        self.oracle_calls += 1
        return out

def run_workload(rt, workload, *, timed, pctr, kern_inc):
    """Drive the frozen teacher-forced workload through step_graph.

    Mirrors the Phase-16R graph arm: per token, time perf_counter around
    step_graph + ring_harvest (only when timed). Returns per-token produced
    ids, timing samples and the sha256 of the final-position logits per
    prompt.
    """
    samples = []
    samples_per_prompt = []
    token_ids = []
    logits_sha = []
    for row in workload:
        rt.reset()
        for token in row["prompt_ids"]:
            rt.step_graph(int(token))
        prompt_produced = []
        prompt_samples = []
        for index, token in enumerate(row["target_ids"]):
            slot = int(rt._ring_i)
            if (not timed) or index < WARMUP_TARGETS:
                rt.step_graph(int(token))
                produced = rt.ring_harvest(slot, 1)[0]
            else:
                started = time.perf_counter_ns()
                rt.step_graph(int(token))
                produced = rt.ring_harvest(slot, 1)[0]
                took = (time.perf_counter_ns() - started) / 1e6
                samples.append(took)
                prompt_samples.append(took)
            prompt_produced.append(int(produced))
        token_ids.append(prompt_produced)
        samples_per_prompt.append(prompt_samples)
        digest = hashlib.sha256(
            np.ascontiguousarray(rt.logits.get()).tobytes()
        ).hexdigest()
        logits_sha.append(digest)
        expected = int(rt._pos_dev.get()[0])
        wanted = len(row["prompt_ids"]) + len(row["target_ids"])
        if expected != wanted:
            raise RuntimeError(
                f"position drift on {row['id']}: pos_dev={expected} "
                f"expected={wanted}"
            )
        if wanted > MAX_POS:
            raise RuntimeError(
                f"prompt {row['id']} needs {wanted} positions > MAX_POS"
            )
        # Advance the device prompt counter AFTER the prompt completes, so
        # prompt i records/replays under pctr == i (0-based, in bounds).
        with rt._graph_stream:
            kern_inc((1,), (1,), (pctr,))
    return {
        "samples": samples,
        "samples_per_prompt": samples_per_prompt,
        "token_ids": token_ids,
        "logits_sha256": logits_sha,
    }

def main() -> int:
    import cupy as cp

    from common import REPO
    results = REPO / "pro_research" / "results" / RESULTS_DIRNAME
    results.mkdir(parents=True, exist_ok=True)
    out_path = results / "S100_LIGHTNING17A_KV_ORACLE.json"
    payload = {
        "kind": "s100_lightning17a_kv_oracle",
        "status": "started",
        "started_utc": utc_now(),
        "claim_boundary": (
            "exact-replay oracle ceiling: K/V vectors are the parent's own "
            "bit-exact outputs, delivered by table copy inside the captured "
            "graph. Measures the upper bound of removing the eleven "
            "heldout-green K/V GEMVs; says nothing about the cost of any "
            "real kernel."
        ),
        "oracle_cases": ORACLE_CASES,
        "max_pos": MAX_POS,
        "warmup_targets_per_prompt": WARMUP_TARGETS,
        "minimum_speedup": MIN_SPEEDUP,
        "target_ms_per_token": TARGET_MS,
    }
    bundle = None
    try:
        identity = assert_lightning()
        workload = frozen_workload()
        n_prompts = len(workload)

        bundle = build()
        rt = bundle.rt
        manifest = case_manifest(rt)
        slot_of = {}
        dims = set()
        for row in manifest:
            if row["case"] in ORACLE_CASES:
                slot_of[row["pointer"]] = len(slot_of)
                dims.add(int(row["rows"]))
        if len(slot_of) != len(ORACLE_CASES):
            raise RuntimeError(
                f"manifest resolved {len(slot_of)} of "
                f"{len(ORACLE_CASES)} oracle cases"
            )
        if dims != {256}:
            raise RuntimeError(f"unexpected oracle dims: {sorted(dims)}")

        module = cp.RawModule(code=KERNELS)
        kern_store = module.get_function("kv_store")
        kern_load = module.get_function("kv_load")
        kern_inc = module.get_function("ctr_inc")

        dim = 256
        slot_stride = n_prompts * MAX_POS * dim
        table = cp.zeros(len(ORACLE_CASES) * slot_stride, cp.float32)
        scratch = cp.zeros(len(ORACLE_CASES) * dim, cp.float32)
        pctr = cp.zeros(1, cp.int32)
        original_mv = rt.k.mv_bf16

        # Warm pass: the first post-build workload pass is a known outlier
        # (fresh-state carry-over; run 2 showed control, recorder and oracle
        # all diverge IDENTICALLY from it while being mutually identical).
        # Discard it so every measured arm runs in steady state.
        run_workload(
            rt, workload, timed=False, pctr=pctr, kern_inc=kern_inc
        )

        # Arm 1: plain recapture of the UNMODIFIED body, timed. This is the
        # steady-state graph reference (control A) and the parity baseline
        # for every later arm.
        pctr.fill(0)
        _recapture(rt)
        ctl_a = run_workload(
            rt, workload, timed=True, pctr=pctr, kern_inc=kern_inc
        )

        # Arm 2 (untimed): recorder graph fills the table.
        pctr.fill(0)
        rt.k.mv_bf16 = OracleHook(
            rt, mode="record", original=original_mv,
            slot_of=slot_of, table=table, slot_stride=slot_stride,
            pctr=pctr, n_prompts=n_prompts,
            kern_store=kern_store, kern_load=kern_load,
        )
        _recapture(rt)
        rec = run_workload(
            rt, workload, timed=False, pctr=pctr, kern_inc=kern_inc
        )
        recorder_tokens_equal = rec["token_ids"] == ctl_a["token_ids"]

        # Arm 3: oracle graph, timed.
        pctr.fill(0)
        hook_oracle = OracleHook(
            rt, mode="oracle", original=original_mv,
            slot_of=slot_of, table=table, slot_stride=slot_stride,
            pctr=pctr, n_prompts=n_prompts,
            kern_store=kern_store, kern_load=kern_load,
        )
        rt.k.mv_bf16 = hook_oracle
        _recapture(rt)
        oracle = run_workload(
            rt, workload, timed=True, pctr=pctr, kern_inc=kern_inc
        )
        # The hook only runs during capture (warmup + capture = 2 bodies);
        # replay is pure graph. Expect 2 * len(ORACLE_CASES) hook calls.
        capture_hook_calls = int(hook_oracle.oracle_calls)

        # Arm 3b: replay-overhead control, timed. The original K/V GEMVs
        # keep running; the same table-load copies execute into scratch, so
        # the replay-copy cost is measured without removing real work.
        pctr.fill(0)
        rt.k.mv_bf16 = OracleHook(
            rt, mode="overhead", original=original_mv,
            slot_of=slot_of, table=table, slot_stride=slot_stride,
            pctr=pctr, n_prompts=n_prompts,
            kern_store=kern_store, kern_load=kern_load, scratch=scratch,
        )
        _recapture(rt)
        overhead = run_workload(
            rt, workload, timed=True, pctr=pctr, kern_inc=kern_inc
        )
        overhead_tokens_equal = overhead["token_ids"] == ctl_a["token_ids"]

        # Arm 4: restored parent graph, timed reference B (drift bracket).
        pctr.fill(0)
        rt.k.mv_bf16 = original_mv
        _recapture(rt)
        ctl_b = run_workload(
            rt, workload, timed=True, pctr=pctr, kern_inc=kern_inc
        )
        control_b_tokens_equal = ctl_b["token_ids"] == ctl_a["token_ids"]

        graph_reference = metrics(ctl_a["samples"] + ctl_b["samples"])
        graph_reference["construction"] = (
            "combined steady-state control A/B bracket samples"
        )
        oracle_timing = metrics(oracle["samples"])

        tokens_equal = oracle["token_ids"] == ctl_a["token_ids"]
        logits_equal = oracle["logits_sha256"] == ctl_a["logits_sha256"]
        parity = bool(
            tokens_equal and logits_equal and recorder_tokens_equal
            and control_b_tokens_equal and overhead_tokens_equal
        )

        def first_divergences(a, b):
            rows = []
            for pid, (sa, sb) in enumerate(zip(a, b)):
                idx = next(
                    (j for j, (x, y) in enumerate(zip(sa, sb))
                     if x != y),
                    None,
                )
                if idx is not None:
                    rows.append({"prompt_index": pid,
                                 "first_divergence": idx,
                                 "parent": sa[idx], "other": sb[idx]})
            return rows

        divergence = {
            "control_b_vs_control_a": first_divergences(
                ctl_b["token_ids"], ctl_a["token_ids"]),
            "recorder_vs_control_a": first_divergences(
                rec["token_ids"], ctl_a["token_ids"]),
            "oracle_vs_control_a": first_divergences(
                oracle["token_ids"], ctl_a["token_ids"]),
        }

        savings_ms = float(
            graph_reference["mean_ms"] - oracle_timing["mean_ms"]
        )
        gap_ms = float(graph_reference["mean_ms"] - TARGET_MS)
        aggregate_speedup = (
            oracle_timing["aggregate_tok_s"]
            / graph_reference["aggregate_tok_s"]
        )
        p50_speedup = (
            graph_reference["p50_ms"] / oracle_timing["p50_ms"]
        )
        speed_open = bool(
            parity
            and aggregate_speedup >= MIN_SPEEDUP
            and p50_speedup >= MIN_SPEEDUP
            and oracle_timing["samples"] >= 500
        )

        # --- Statistical addendum (frozen Phase-17 plan, 17A section) ---
        # Prompt-clustered bootstrap of bracket-corrected savings, plus the
        # replay-copy overhead control. Arm wall-clock order is
        # A, recorder, oracle, overhead, B, so the linear bracket weights
        # are alpha_oracle = 2/4 and alpha_overhead = 3/4.
        def per_prompt_means(arm):
            return np.array(
                [float(np.mean(s)) for s in arm["samples_per_prompt"]]
            )
        ppm_a = per_prompt_means(ctl_a)
        ppm_b = per_prompt_means(ctl_b)
        ppm_o = per_prompt_means(oracle)
        ppm_h = per_prompt_means(overhead)
        alpha_oracle = 0.5
        alpha_overhead = 0.75
        ref_o = (1.0 - alpha_oracle) * ppm_a + alpha_oracle * ppm_b
        ref_h = (1.0 - alpha_overhead) * ppm_a + alpha_overhead * ppm_b
        savings_p = ref_o - ppm_o
        copycost_p = ppm_h - ref_h
        # measured saving S = K - C (true kernel time minus replay-copy
        # cost); overhead arm gives C directly, so K = S + C.
        corrected_p = savings_p + copycost_p

        def cluster_bounds(values, seed=20260819, resamples=10000):
            rng = np.random.default_rng(seed)
            values = np.asarray(values, dtype=np.float64)
            n = len(values)
            idx = rng.integers(0, n, size=(resamples, n))
            stat = values[idx].mean(axis=1)
            return (float(np.percentile(stat, 5)),
                    float(np.percentile(stat, 95)))

        s_lo95, s_hi95 = cluster_bounds(savings_p)
        c_lo95, c_hi95 = cluster_bounds(corrected_p)
        three_pct_ms = float(graph_reference["mean_ms"] * 0.03)
        definitively_closed = bool(parity and c_hi95 < three_pct_ms)

        payload.update({
            "status": "measured",
            "identity": identity,
            "workload": {
                "prompts": n_prompts,
                "target_tokens_per_prompt": len(
                    workload[0]["target_ids"]
                ),
                "measured_tokens_per_arm": len(ctl_a["samples"]),
                "prompt_ids": [row["id"] for row in workload],
            },
            "table": {
                "cases": len(ORACLE_CASES),
                "prompts": n_prompts,
                "max_pos": MAX_POS,
                "dim": dim,
                "bytes": int(table.nbytes),
            },
            "recorder": {
                "tokens_equal_control_a": bool(recorder_tokens_equal),
            },
            "control_b_recapture": {
                "tokens_equal_control_a": bool(control_b_tokens_equal),
            },
            "divergence": divergence,
            "parity": {
                "token_ids_equal": bool(tokens_equal),
                "final_logits_sha256_equal": bool(logits_equal),
                "positions_per_prompt": len(ctl_a["token_ids"][0]),
            },
            "graph_A": {"timing": metrics(ctl_a["samples"])},
            "oracle_graph": {
                "timing": oracle_timing,
                "oracle_loads_per_token": len(ORACLE_CASES),
                "capture_hook_calls": capture_hook_calls,
            },
            "graph_B": {"timing": metrics(ctl_b["samples"])},
            "overhead_arm": {
                "timing": metrics(overhead["samples"]),
                "tokens_equal_control_a": bool(overhead_tokens_equal),
            },
            "addendum": {
                "bracket_alpha": {"oracle": alpha_oracle,
                                  "overhead": alpha_overhead},
                "bootstrap": {"cluster": "prompt", "resamples": 10000,
                              "seed": 20260819},
                "savings_ms_per_prompt": [
                    round(float(v), 6) for v in savings_p
                ],
                "savings_ms_mean_bracketed": float(savings_p.mean()),
                "savings_ms_one_sided95": {
                    "lower": s_lo95, "upper": s_hi95,
                },
                "replay_copy_overhead_ms_mean": float(copycost_p.mean()),
                "corrected_savings_ms_mean": float(corrected_p.mean()),
                "corrected_savings_ms_one_sided95": {
                    "lower": c_lo95, "upper": c_hi95,
                },
                "true_kv_kernel_ms_estimate": float(corrected_p.mean()),
                "three_pct_gate_ms": three_pct_ms,
            },
            "graph_reference": graph_reference,
            "analysis": {
                "savings_ms_mean": savings_ms,
                "gap_to_100toks_ms": gap_ms,
                "aggregate_speedup_vs_graph": float(aggregate_speedup),
                "p50_speedup_vs_graph": float(p50_speedup),
                "KV_ORACLE_S100_GAP_COVERAGE": (
                    float(savings_ms / gap_ms) if gap_ms > 0 else None
                ),
            },
            "KV_ORACLE_PARITY": parity,
            "KV_ORACLE_3PCT_OPEN": speed_open,
            "KV_ORACLE_DEFINITIVELY_CLOSED": definitively_closed,
            "completed_utc": utc_now(),
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

    write_json_atomic(out_path, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "KV_ORACLE_PARITY": payload.get("KV_ORACLE_PARITY"),
        "KV_ORACLE_3PCT_OPEN": payload.get("KV_ORACLE_3PCT_OPEN"),
        "KV_ORACLE_DEFINITIVELY_CLOSED": payload.get(
            "KV_ORACLE_DEFINITIVELY_CLOSED"),
        "addendum": payload.get("addendum"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out_path),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
