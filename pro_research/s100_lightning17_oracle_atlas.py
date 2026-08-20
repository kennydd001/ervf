"""S100 Lightning Phase 17: exact component-oracle atlas inside the CUDA graph.

Frozen plan: agents/S100_LIGHTNING_PHASE17_ORACLE_ATLAS_PLAN.md.

Question: for oracle groups L (LM head), E (complete MoE layer), A (complete
attention compute, FP8 KV-cache append retained) and M_in/M_out/M_io (Mamba
dense projections only; conv/SSM/gating/state stay real), how many ms/token
does each group -- and each required subset-lattice combination -- cost the
production graph parent?

Method (same pattern as s100_lightning17a_kv_oracle.py): hooks that are only
active during graph (re)capture. A recorder arm stores the bit-exact FP32
output vectors of every group path into device tables indexed by
(prompt, position) via device counters (rt._pos_dev and a prompt counter), so
the mechanism is capturable. An oracle arm replaces the computed paths with
device-to-device table loads inside the captured graph. No host readback,
allocation or synchronisation inside graph execution.

Per arm the protocol is control A -> recorder (untimed, only when its tables
are not yet recorded) -> oracle -> copy-overhead control -> control B, with an
optional second oracle arm when the thermal slope over the A/B controls
exceeds 0.15 ms. The first post-build workload pass is discarded (known
outlier, same as 17A). Statistics: linear A/B bracket per prompt, prompt-
clustered bootstrap (10,000 resamples), one-sided 95% bounds, replay-overhead
correction (measured saving S = K - C, overhead arm measures C, so
K = S + C), decision bands from the frozen plan.

Design notes / deviations from the plan (all flagged in the output):

- L parity: the teacher-forced feed makes per-step logits semantically
  irrelevant (tok_dev is overwritten by the staged H2D copy on every
  step_graph(token) call, so the skipped argmax cannot feed back into state).
  After the timed positions of each prompt, one final UNTIMED probe position
  re-runs the original LM head + argmax eagerly (outside the graph) and final
  logits, produced probe token, hidden state, recurrent state (SSM+conv) and
  used-KV bytes are compared against the control. This is exactly the plan's
  "final untimed position with the original LM head re-enabled", done eagerly
  instead of by recapture. L is reported as a teacher-forced upper bound.
- Mamba table VRAM: full M_in tables for all 23 Mamba layers are ~4.5 GiB, so
  Mamba layers are partitioned into deterministic contiguous groups whose
  replay tables fit the budget (free VRAM minus a 256 MiB reserve, capped by
  --max-mamba-vram-mib). Every group gets an independent A/oracle/B bracket.
  Groups are summed only after an additivity check on one adjacent pair
  (group-pair union when it fits, otherwise the two layers straddling the
  first group boundary -- a documented fallback).
- Combination arms that do not fit in VRAM run as two complementary
  table-resident halves (M_io split in two, the rest of the combination
  complete). Only an interval is reported, never an additive point estimate.
- The A oracle appends the recorded FP32 K/V through the real kv_write_fp8_dp
  kernels at the normal device position, so used-KV bytes stay bit-identical;
  Q/K/V/O projections and the attention core are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import time
import traceback

import numpy as np

from common import write_json_atomic, write_text_atomic, utc_now

RESULTS_DIRNAME = "s100_lightning17"
WARMUP_TARGETS = 8
MAX_POS = 128  # frozen workload needs <= 111 positions (10 prompts, 64 targets)
TARGET_MS = 10.0  # 100 tok/s
BAND_TARGET_LOWER_MS = 0.58
BAND_CLOSE_UPPER_MS = 0.20
THERMAL_SLOPE_GATE_MS = 0.15
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260819
RESERVE_MIB = 256
ADDITIVITY_TOL_MS = 0.05
ADDITIVITY_TOL_FRAC = 0.10

SINGLE_ARMS = ("L", "E", "A", "M_IN", "M_OUT", "M_IO")
PLAIN_COMBOS = ("E+L", "A+L", "E+A")
MIO_COMBOS = ("M_IO+E", "M_IO+A", "M_IO+L", "M_IO+E+A+L")
ALL_ARM_TOKENS = SINGLE_ARMS + PLAIN_COMBOS + MIO_COMBOS

KERNELS = r"""
extern "C" __global__ void vec_store(float* base, const float* src,
                                     const int* pos, const int* pctr,
                                     int max_pos, int n_prompts, int dim) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    int p = pctr[0];
    if (p >= n_prompts) p = n_prompts - 1;  // capture/warmup safety
    long long idx = ((long long)p * max_pos + pos[0]) * dim + t;
    base[idx] = src[t];
}
extern "C" __global__ void vec_load(float* dst, const float* base,
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

# ---------------------------------------------------------------------------
# Pure helpers (no CUDA) -- unit-tested without a GPU.
# ---------------------------------------------------------------------------

def runtime_dims(rt) -> dict:
    """Table-relevant dimensions, read from the built runtime."""
    return {
        "hidden": int(rt.hidden),
        "kv_dim": int(rt.kv_dim),
        "proj_dim": int(rt.proj.size),
        "vocab": int(rt.vocab),
        "moe_layers": [int(i) for i in rt.moe_layers],
        "attn_layers": [int(i) for i in rt.attn_layers],
        "mamba_layers": [int(i) for i in rt.mamba_layers],
    }

def table_plan(dims: dict, n_prompts: int, max_pos: int) -> dict:
    """Per-layer replay-table bytes (FP32) for each group table."""
    slots = n_prompts * max_pos
    f32 = 4
    per = {
        "E": dims["hidden"] * f32 * slots,
        "A_K": dims["kv_dim"] * f32 * slots,
        "A_V": dims["kv_dim"] * f32 * slots,
        "A_O": dims["hidden"] * f32 * slots,
        "M_IN": dims["proj_dim"] * f32 * slots,
        "M_OUT": dims["hidden"] * f32 * slots,
    }
    per["A"] = per["A_K"] + per["A_V"] + per["A_O"]
    per["M_IO"] = per["M_IN"] + per["M_OUT"]
    return per

def partition_layers(layers, per_layer_bytes: int, budget_bytes: int):
    """Deterministic contiguous partition of `layers` so each group's tables
    fit `budget_bytes`. Raises when a single layer does not fit."""
    if per_layer_bytes <= 0:
        raise ValueError("per_layer_bytes must be positive")
    if per_layer_bytes > budget_bytes:
        raise ValueError(
            f"single layer table {per_layer_bytes} B exceeds budget "
            f"{budget_bytes} B"
        )
    groups, current, used = [], [], 0
    for layer in layers:
        if current and used + per_layer_bytes > budget_bytes:
            groups.append(current)
            current, used = [], 0
        current.append(int(layer))
        used += per_layer_bytes
    if current:
        groups.append(current)
    return groups

def split_halves(layers):
    """Complementary deterministic halves; the first half takes the extra
    layer when the count is odd."""
    mid = (len(layers) + 1) // 2
    return [int(v) for v in layers[:mid]], [int(v) for v in layers[mid:]]

def fits_budget(free_bytes: int, needed_bytes: int, reserve_bytes: int) -> bool:
    return free_bytes - needed_bytes >= reserve_bytes

def per_prompt_means(samples_per_prompt) -> np.ndarray:
    return np.array(
        [float(np.mean(s)) for s in samples_per_prompt], dtype=np.float64
    )

def bracket_reference(ppm_a: np.ndarray, ppm_b: np.ndarray, alpha: float):
    """Linear wall-clock bracket: reference(alpha) between control A and B."""
    return (1.0 - alpha) * ppm_a + alpha * ppm_b

def cluster_bounds(values, *, seed=BOOTSTRAP_SEED,
                   resamples=BOOTSTRAP_RESAMPLES):
    """Prompt-clustered bootstrap: resample prompts (not tokens), one-sided
    95% lower/upper bounds on the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    idx = rng.integers(0, n, size=(resamples, n))
    stat = values[idx].mean(axis=1)
    return float(np.percentile(stat, 5)), float(np.percentile(stat, 95))

def drift_and_slope(early_samples, late_samples,
                    gate_ms=THERMAL_SLOPE_GATE_MS) -> dict:
    """A/B drift plus a linear slope fitted over the concatenated control
    samples (approximation: the oracle/overhead passes sit between A and B in
    wall-clock, so the true per-sample spacing is wider; reported as such)."""
    early = np.asarray(early_samples, dtype=np.float64)
    late = np.asarray(late_samples, dtype=np.float64)
    seq = np.concatenate([early, late])
    idx = np.arange(seq.size, dtype=np.float64)
    slope_per = float(np.polyfit(idx, seq, 1)[0]) if seq.size > 1 else 0.0
    total = slope_per * seq.size
    return {
        "drift_ms": float(late.mean() - early.mean()),
        "slope_ms_per_sample": slope_per,
        "slope_ms_over_controls": total,
        "exceeds_gate": bool(abs(total) > gate_ms),
        "note": ("slope fitted over concatenated control A+B samples; "
                 "interleaved oracle/overhead wall-clock not modelled"),
    }

def decision_band(lower95: float, upper95: float) -> str:
    """Frozen decision bands on the corrected one-sided 95% bounds."""
    if lower95 >= BAND_TARGET_LOWER_MS:
        return "target"
    if upper95 < BAND_CLOSE_UPPER_MS:
        return "closed"
    return "secondary"

def interaction(saving_st: float, saving_s: float, saving_t: float) -> float:
    """interaction(S,T) = saving(S union T) - saving(S) - saving(T)."""
    return float(saving_st - saving_s - saving_t)

def combined_path_exists(ref_mean_ms: float, saving_lower95_ms: float,
                         target_ms=TARGET_MS) -> bool:
    """Conservative: even the lower-bound saving reaches <= target ms/token."""
    return bool(ref_mean_ms - saving_lower95_ms <= target_ms)

def additivity_check(combined: float, singles_sum: float,
                     tol_ms=ADDITIVITY_TOL_MS,
                     tol_frac=ADDITIVITY_TOL_FRAC) -> dict:
    diff = float(combined - singles_sum)
    tol = max(tol_ms, tol_frac * abs(singles_sum))
    return {
        "combined_ms": float(combined),
        "singles_sum_ms": float(singles_sum),
        "difference_ms": diff,
        "tolerance_ms": float(tol),
        "additive_ok": bool(abs(diff) <= tol),
    }

def arm_stats_from_passes(ctl_a, oracle, overhead, ctl_b, oracle2=None,
                          *, seed=BOOTSTRAP_SEED,
                          resamples=BOOTSTRAP_RESAMPLES) -> dict:
    """Bracketed savings statistics for one arm.

    Timed wall-clock order is A, oracle, overhead, B (recorder untimed), so
    alpha_oracle = 1/3 and alpha_overhead = 2/3. With the optional second
    oracle pass appended after B (order A, O, OH, B, O2) the two oracle
    passes are averaged with alpha = mean(1/4, 5/4) = 3/4 and the overhead
    control uses alpha = 1/2.
    """
    ppm_a = per_prompt_means(ctl_a["samples_per_prompt"])
    ppm_b = per_prompt_means(ctl_b["samples_per_prompt"])
    ppm_o = per_prompt_means(oracle["samples_per_prompt"])
    ppm_h = per_prompt_means(overhead["samples_per_prompt"])
    if oracle2 is not None:
        ppm_o = 0.5 * (ppm_o + per_prompt_means(
            oracle2["samples_per_prompt"]))
        alpha_oracle, alpha_overhead = 0.75, 0.5
    else:
        alpha_oracle, alpha_overhead = 1.0 / 3.0, 2.0 / 3.0
    ref_o = bracket_reference(ppm_a, ppm_b, alpha_oracle)
    ref_h = bracket_reference(ppm_a, ppm_b, alpha_overhead)
    savings_p = ref_o - ppm_o
    copycost_p = ppm_h - ref_h
    # measured saving S = K - C (true kernel time minus replay-copy cost);
    # the overhead arm measures C directly, so K = S + C.
    corrected_p = savings_p + copycost_p
    s_lo, s_hi = cluster_bounds(savings_p, seed=seed, resamples=resamples)
    c_lo, c_hi = cluster_bounds(corrected_p, seed=seed + 1,
                                resamples=resamples)
    ref_samples = np.asarray(
        ctl_a["samples"] + ctl_b["samples"], dtype=np.float64)
    oracle_samples = np.asarray(oracle["samples"], dtype=np.float64)
    ref_mean = float(ref_samples.mean())
    oracle_mean = float(oracle_samples.mean())
    gap_ms = ref_mean - TARGET_MS
    corrected_mean = float(corrected_p.mean())
    thermal = drift_and_slope(ctl_a["samples"], ctl_b["samples"])
    return {
        "bracket_alpha": {"oracle": alpha_oracle,
                          "overhead": alpha_overhead},
        "bootstrap": {"cluster": "prompt", "resamples": int(resamples),
                      "seed": int(seed)},
        "reference_mean_ms": ref_mean,
        "oracle_mean_ms": oracle_mean,
        "savings_ms_per_prompt": [
            round(float(v), 6) for v in savings_p],
        "savings_ms_mean": float(savings_p.mean()),
        "savings_ms_median": float(np.median(savings_p)),
        "savings_ms_one_sided95": {"lower": s_lo, "upper": s_hi},
        "replay_copy_overhead_ms_mean": float(copycost_p.mean()),
        "corrected_savings_ms_per_prompt": [
            round(float(v), 6) for v in corrected_p],
        "corrected_savings_ms_mean": corrected_mean,
        "corrected_savings_ms_median": float(np.median(corrected_p)),
        "corrected_savings_ms_one_sided95": {"lower": c_lo, "upper": c_hi},
        "aggregate_speedup": float(
            1000.0 * oracle_samples.size / oracle_samples.sum()
            / (1000.0 * ref_samples.size / ref_samples.sum())),
        "p50_speedup": float(
            np.percentile(ref_samples, 50)
            / np.percentile(oracle_samples, 50)),
        "gap_to_target_ms": gap_ms,
        "s100_gap_coverage_corrected": (
            float(corrected_mean / gap_ms) if gap_ms > 0 else None),
        "thermal": thermal,
        "second_oracle_used": oracle2 is not None,
        "decision_band": decision_band(c_lo, c_hi),
    }

# ---------------------------------------------------------------------------
# CUDA-side machinery (only used from main()).
# ---------------------------------------------------------------------------

class AtlasHooks:
    """Capture-time hooks for the oracle groups.

    The hooks only run while the graph body is (re)captured; replay executes
    whatever was captured without touching host code. Non-selected calls pass
    through to the original implementations untouched.

    modes:
      record  -- run the original path, then store its FP32 output into the
                 device table at (prompt counter, device position).
      oracle  -- skip the original path; load the recorded vector from the
                 table (for A: still append the recorded K/V through the real
                 FP8 cache-write kernels).
      overhead-- run the original path AND execute the same table-load copies
                 into scratch, measuring replay-copy cost while the real
                 work stays.
    """

    def __init__(self, rt, *, kern_store, kern_load, pctr, n_prompts,
                 max_pos):
        self.rt = rt
        self.kern_store = kern_store
        self.kern_load = kern_load
        self.pctr = pctr
        self.n_prompts32 = np.int32(n_prompts)
        self.max_pos32 = np.int32(max_pos)
        self.max_pos = int(max_pos)
        self.n_prompts = int(n_prompts)
        self.mode = None
        self.active = {"L": False, "E": set(), "A": set(),
                       "M_IN": set(), "M_OUT": set()}
        self.tables = {}   # name -> {arr, slot_of, dim, stride, recorded}
        self.scratch = {}  # name -> flat scratch (n_slots * dim)
        self.cur_mamba = None
        self.hook_calls = {"record": 0, "oracle": 0, "overhead": 0,
                           "passthrough": 0}
        self._orig = None

    # ------------------------------------------------------------- tables
    def add_table(self, name, layers, dim):
        cp = self.rt.cp
        slot_of = {int(layer): k for k, layer in enumerate(sorted(layers))}
        stride = self.n_prompts * self.max_pos * int(dim)
        arr = cp.zeros(len(slot_of) * stride, cp.float32)
        self.tables[name] = {
            "arr": arr, "slot_of": slot_of, "dim": int(dim),
            "stride": int(stride), "recorded": False,
        }
        self.scratch[name] = cp.zeros(max(len(slot_of), 1) * int(dim),
                                      cp.float32)
        return int(arr.nbytes + self.scratch[name].nbytes)

    def add_scratch_only(self, name, dim):
        cp = self.rt.cp
        self.scratch[name] = cp.zeros(int(dim), cp.float32)
        return int(self.scratch[name].nbytes)

    def drop_table(self, name):
        self.tables.pop(name, None)
        self.scratch.pop(name, None)

    def _slice(self, name, layer):
        t = self.tables[name]
        slot = t["slot_of"][int(layer)]
        return t["arr"][slot * t["stride"]:(slot + 1) * t["stride"]], t["dim"]

    def _launch(self, kern, dst_or_base, src_or_base, dim):
        kern(((dim + 255) // 256,), (256,),
             (dst_or_base, src_or_base, self.rt._pos_dev, self.pctr,
              self.max_pos32, self.n_prompts32, np.int32(dim)))

    def store(self, name, layer, src):
        base, dim = self._slice(name, layer)
        self._launch(self.kern_store, base, src, dim)
        self.hook_calls["record"] += 1

    def load(self, name, layer, dst):
        base, dim = self._slice(name, layer)
        self._launch(self.kern_load, dst, base, dim)
        self.hook_calls["oracle"] += 1

    def load_scratch(self, name, layer):
        t = self.tables[name]
        slot = t["slot_of"][int(layer)]
        dim = t["dim"]
        dst = self.scratch[name][slot * dim:(slot + 1) * dim]
        base = t["arr"][slot * t["stride"]:(slot + 1) * t["stride"]]
        self._launch(self.kern_load, dst, base, dim)
        self.hook_calls["overhead"] += 1

    # -------------------------------------------------------------- modes
    def set_mode(self, mode, active=None):
        assert mode in (None, "record", "oracle", "overhead")
        self.mode = mode
        if active is not None:
            self.active = {
                "L": bool(active.get("L", False)),
                "E": {int(v) for v in active.get("E", set())},
                "A": {int(v) for v in active.get("A", set())},
                "M_IN": {int(v) for v in active.get("M_IN", set())},
                "M_OUT": {int(v) for v in active.get("M_OUT", set())},
            }
        else:
            self.active = {"L": False, "E": set(), "A": set(),
                           "M_IN": set(), "M_OUT": set()}

    # -------------------------------------------------------------- hooks
    def install(self):
        rt = self.rt
        hooks = self
        orig = self._orig = {
            "moe": rt._moe,
            "attention": rt._attention,
            "mamba": rt._mamba,
            "gemv_into": rt.fused.gemv_into,
            "mv_bf16": rt.k.mv_bf16,
            "mv_fp8_tensor": rt.k.mv_fp8_tensor,
            "argmax_logits": rt.k.argmax_logits,
        }

        def moe_hook(i, out):
            mode = hooks.mode
            if mode is None or int(i) not in hooks.active["E"]:
                hooks.hook_calls["passthrough"] += 1
                return orig["moe"](i, out)
            if mode == "record":
                result = orig["moe"](i, out)
                hooks.store("E", i, out)
                return result
            if mode == "overhead":
                result = orig["moe"](i, out)
                hooks.load_scratch("E", i)
                return result
            hooks.load("E", i, out)
            return None, None

        def attention_hook(i, out):
            mode = hooks.mode
            if mode is None or int(i) not in hooks.active["A"]:
                hooks.hook_calls["passthrough"] += 1
                return orig["attention"](i, out)
            if mode == "record":
                # Original path first: kv_/vv then hold this layer's exact
                # FP32 K/V and `out` the exact attention-layer output before
                # the residual add.
                result = orig["attention"](i, out)
                hooks.store("A_K", i, rt.kv_)
                hooks.store("A_V", i, rt.vv)
                hooks.store("A_O", i, out)
                return result
            if mode == "overhead":
                result = orig["attention"](i, out)
                hooks.load_scratch("A_K", i)
                hooks.load_scratch("A_V", i)
                hooks.load_scratch("A_O", i)
                return result
            # Oracle: replay the recorded K/V through the REAL FP8 cache
            # append at the normal device position (semantic KV state stays
            # correct), replay the recorded attention output, and skip the
            # Q/K/V/O projections plus the attention core.
            hooks.load("A_K", i, rt.kv_)
            hooks.load("A_V", i, rt.vv)
            rt.k.kv_write_fp8_dp(rt.kc[i], rt.kv_, rt._pos_dev, rt.n_kv,
                                 rt.head_dim, rt.max_ctx)
            rt.k.kv_write_fp8_dp(rt.vc[i], rt.vv, rt._pos_dev, rt.n_kv,
                                 rt.head_dim, rt.max_ctx)
            hooks.load("A_O", i, out)
            return None

        def mamba_hook(i, out):
            # Shim only: tag the current layer so the projection hooks below
            # know which table slot to use. The body itself always runs.
            hooks.cur_mamba = int(i)
            try:
                return orig["mamba"](i, out)
            finally:
                hooks.cur_mamba = None

        def proj_dispatch(out, x, rows, call):
            layer = hooks.cur_mamba
            if layer is None:
                return call()
            if layer in hooks.active["M_IN"] and out is rt.proj:
                return hooks._proj("M_IN", layer, out, int(rows), call)
            if (layer in hooks.active["M_OUT"] and out is rt.acc
                    and x is rt.gn):
                return hooks._proj("M_OUT", layer, out, int(rows), call)
            return call()

        def gemv_into_hook(out, codes, scales, x, global_scale, rows, cols,
                           apply_relu2=False, out_scale=1.0):
            def call():
                return orig["gemv_into"](
                    out, codes, scales, x, global_scale, rows, cols,
                    apply_relu2=apply_relu2, out_scale=out_scale)
            mode = hooks.mode
            if mode is None:
                hooks.hook_calls["passthrough"] += 1
                return call()
            if hooks.active["L"] and out is rt.logits:
                return hooks._lm_head(mode, call)
            return proj_dispatch(out, x, rows, call)

        def mv_bf16_hook(out, weight, x, rows, cols):
            def call():
                return orig["mv_bf16"](out, weight, x, rows, cols)
            mode = hooks.mode
            if mode is None:
                hooks.hook_calls["passthrough"] += 1
                return call()
            if hooks.active["L"] and out is rt.logits:
                return hooks._lm_head(mode, call)
            return proj_dispatch(out, x, rows, call)

        def mv_fp8_tensor_hook(out, weight, x, wscale, rows, cols):
            def call():
                return orig["mv_fp8_tensor"](out, weight, x, wscale,
                                             rows, cols)
            mode = hooks.mode
            if mode is None:
                hooks.hook_calls["passthrough"] += 1
                return call()
            if hooks.active["L"] and out is rt.logits:
                return hooks._lm_head(mode, call)
            return proj_dispatch(out, x, rows, call)

        def argmax_hook(*args):
            if hooks.mode == "oracle" and hooks.active["L"]:
                # Timed positions skip sampling entirely; the teacher-forced
                # feed overwrites tok_dev before every launch, and the final
                # untimed probe position re-runs argmax eagerly.
                return None
            return orig["argmax_logits"](*args)

        rt._moe = moe_hook
        rt._attention = attention_hook
        rt._mamba = mamba_hook
        rt.fused.gemv_into = gemv_into_hook
        rt.k.mv_bf16 = mv_bf16_hook
        rt.k.mv_fp8_tensor = mv_fp8_tensor_hook
        rt.k.argmax_logits = argmax_hook

    def restore(self):
        if self._orig is None:
            return
        rt = self.rt
        rt._moe = self._orig["moe"]
        rt._attention = self._orig["attention"]
        rt._mamba = self._orig["mamba"]
        rt.fused.gemv_into = self._orig["gemv_into"]
        rt.k.mv_bf16 = self._orig["mv_bf16"]
        rt.k.mv_fp8_tensor = self._orig["mv_fp8_tensor"]
        rt.k.argmax_logits = self._orig["argmax_logits"]
        self._orig = None

    def _lm_head(self, mode, call):
        if mode == "oracle":
            return None  # LM head skipped on timed positions
        result = call()
        if mode == "overhead":
            # Same bytes the replay would have moved, copied to scratch while
            # the real LM head keeps running.
            self.scratch["L"][:] = self.rt.logits
            self.hook_calls["overhead"] += 1
        else:
            self.hook_calls["record"] += 1
        return result

    def _proj(self, table, layer, out, rows, call):
        dim = self.tables[table]["dim"]
        if rows != dim:
            raise RuntimeError(
                f"{table} layer {layer}: hook saw rows={rows}, "
                f"table dim={dim}"
            )
        if self.mode == "record":
            result = call()
            self.store(table, layer, out)
            return result
        if self.mode == "overhead":
            result = call()
            self.load_scratch(table, layer)
            return result
        self.load(table, layer, out)
        return None

def harvest_state(rt) -> dict:
    """Post-prompt semantic state parity bundle (readback outside the graph):
    hidden vector, recurrent state (SSM + conv per Mamba layer), used FP8 KV
    bytes per attention layer, and the used-KV byte count."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rt.h.get()).tobytes())
    rec = hashlib.sha256()
    for i in rt.mamba_layers:
        rec.update(np.ascontiguousarray(rt.ssm[i].get()).tobytes())
        rec.update(np.ascontiguousarray(rt.conv[i].get()).tobytes())
    pos = int(rt._pos_dev.get()[0])
    used = pos * int(rt.kv_dim)
    kv = hashlib.sha256()
    for i in rt.attn_layers:
        kv.update(np.ascontiguousarray(rt.kc[i][:used].get()).tobytes())
        kv.update(np.ascontiguousarray(rt.vc[i][:used].get()).tobytes())
    return {
        "hidden_sha256": h.hexdigest(),
        "recurrent_sha256": rec.hexdigest(),
        "kv_sha256": kv.hexdigest(),
        "kv_used_bytes_per_layer": int(used),
    }

def run_pass(rt, workload, hooks, pctr, kern_inc, *, timed,
             probe_lm_head, collect_state):
    """Drive the frozen teacher-forced workload through step_graph.

    Per prompt: feed prompt ids, then the frozen target ids (first
    WARMUP_TARGETS untimed per prompt when timed), then one final UNTIMED
    probe position that re-feeds the last target token. When probe_lm_head is
    set (arms containing L), the original LM head and argmax are re-run
    eagerly on that probe position -- the plan's parity requirement for the
    teacher-forced L ceiling. Returns per-token ids, timing samples,
    per-prompt logits sha256, probe ids and optional state hashes.
    """
    samples = []
    samples_per_prompt = []
    token_ids = []
    logits_sha = []
    probe_ids = []
    states = []
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
        # Final untimed probe position.
        rt.step_graph(int(row["target_ids"][-1]))
        rt._graph_stream.synchronize()
        if probe_lm_head:
            # Eager re-run of the ORIGINAL LM head + argmax outside the
            # graph; hooks are bypassed via mode=None so the call passes
            # through to the captured originals.
            saved_mode = hooks.mode
            hooks.mode = None
            try:
                if rt.lm_head_kind == "nvfp4":
                    rt.fused.gemv_into(rt.logits, rt.lm_head_codes,
                                       rt.lm_head_scales, rt.normed,
                                       rt.lm_head_g, rt.vocab, rt.hidden)
                else:
                    rt.k.mv_bf16(rt.logits, rt.lm_head, rt.normed,
                                 rt.vocab, rt.hidden)
                rt.k.argmax_logits(rt._tok_dev, rt.logits, rt.vocab,
                                   rt._am_max, rt._am_idx)
            finally:
                hooks.mode = saved_mode
            rt._graph_stream.synchronize()
            probe_ids.append(int(rt._tok_dev.get()[0]))
        else:
            probe_ids.append(
                int(rt.ring_harvest((int(rt._ring_i) - 1)
                                    % int(rt._ring_size), 1)[0]))
        token_ids.append(prompt_produced)
        samples_per_prompt.append(prompt_samples)
        digest = hashlib.sha256(
            np.ascontiguousarray(rt.logits.get()).tobytes()
        ).hexdigest()
        logits_sha.append(digest)
        if collect_state:
            states.append(harvest_state(rt))
        expected = int(rt._pos_dev.get()[0])
        wanted = (len(row["prompt_ids"]) + len(row["target_ids"]) + 1)
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
        "probe_ids": probe_ids,
        "states": states,
    }

def arm_needs(spec, dims, per):
    """Tables/scratch an arm needs: name -> dict(layers, dim, table_bytes,
    scratch_bytes, persistent)."""
    needs = {}
    if spec["E"]:
        layers = sorted(spec["E"])
        needs["E"] = {"layers": layers, "dim": dims["hidden"],
                      "table_bytes": per["E"] * len(layers),
                      "scratch_bytes": dims["hidden"] * 4 * len(layers),
                      "persistent": True}
    if spec["A"]:
        layers = sorted(spec["A"])
        for name, dim, per_key in (
                ("A_K", dims["kv_dim"], "A_K"),
                ("A_V", dims["kv_dim"], "A_V"),
                ("A_O", dims["hidden"], "A_O")):
            needs[name] = {"layers": layers, "dim": dim,
                           "table_bytes": per[per_key] * len(layers),
                           "scratch_bytes": dim * 4 * len(layers),
                           "persistent": True}
    if spec["M_IN"]:
        layers = sorted(spec["M_IN"])
        needs["M_IN"] = {"layers": layers, "dim": dims["proj_dim"],
                         "table_bytes": per["M_IN"] * len(layers),
                         "scratch_bytes": dims["proj_dim"] * 4 * len(layers),
                         "persistent": False}
    if spec["M_OUT"]:
        layers = sorted(spec["M_OUT"])
        needs["M_OUT"] = {"layers": layers, "dim": dims["hidden"],
                          "table_bytes": per["M_OUT"] * len(layers),
                          "scratch_bytes": dims["hidden"] * 4 * len(layers),
                          "persistent": False}
    if spec["L"]:
        needs["L"] = {"layers": [], "dim": dims["vocab"],
                      "table_bytes": 0,
                      "scratch_bytes": dims["vocab"] * 4,
                      "persistent": True}
    return needs

def replay_profile(spec, dims) -> dict:
    """Replay node count (table-load kernels per token) and replay bytes per
    token for an oracle arm."""
    nodes = 0
    bytes_per_token = 0
    nodes += len(spec["E"])
    bytes_per_token += len(spec["E"]) * dims["hidden"] * 4
    nodes += 3 * len(spec["A"])
    bytes_per_token += (len(spec["A"])
                        * (2 * dims["kv_dim"] + dims["hidden"]) * 4)
    nodes += len(spec["M_IN"])
    bytes_per_token += len(spec["M_IN"]) * dims["proj_dim"] * 4
    nodes += len(spec["M_OUT"])
    bytes_per_token += len(spec["M_OUT"]) * dims["hidden"] * 4
    return {"replay_nodes_per_token": int(nodes),
            "replay_bytes_per_token": int(bytes_per_token)}

def _log(arm, phase, started=None):
    stamp = time.strftime("%H:%M:%S")
    if started is None:
        print(f"[{stamp}] atlas | {arm} | {phase}", flush=True)
    else:
        print(f"[{stamp}] atlas | {arm} | {phase} | "
              f"{time.perf_counter() - started:.1f}s", flush=True)

def _first_divergences(a, b):
    rows = []
    for pid, (sa, sb) in enumerate(zip(a, b)):
        idx = next((j for j, (x, y) in enumerate(zip(sa, sb)) if x != y),
                   None)
        if idx is not None:
            rows.append({"prompt_index": pid, "first_divergence": idx,
                         "parent": sa[idx], "other": sb[idx]})
    return rows

def execute_arm(spec, *, rt, hooks, workload, kern_inc, pctr, results,
                dims, per, reserve_bytes, args, recapture):
    """One atlas arm: control A -> recorder (if needed) -> oracle ->
    overhead -> control B [-> oracle2 when the thermal gate trips]."""
    from s100_lightning16r_throughput import metrics

    cp = rt.cp
    name = spec["name"]
    payload = {
        "kind": "s100_lightning17_atlas_arm",
        "arm": name,
        "spec": {key: (sorted(int(v) for v in spec[key])
                       if isinstance(spec[key], (set, list))
                       else bool(spec[key]))
                 for key in ("L", "E", "A", "M_IN", "M_OUT")},
        "status": "started",
        "started_utc": utc_now(),
        "min_samples": int(spec.get("min_samples", 560)),
    }
    arm_started = time.perf_counter()
    try:
        needs = arm_needs(spec, dims, per)
        # Free non-persistent tables from earlier arms, then allocate.
        for tname in list(hooks.tables):
            if tname not in needs and tname in ("M_IN", "M_OUT"):
                hooks.drop_table(tname)
        cp.get_default_memory_pool().free_all_blocks()
        new_bytes = 0
        for tname, need in needs.items():
            existing = hooks.tables.get(tname)
            if existing is not None and existing["recorded"] and all(
                    layer in existing["slot_of"] for layer in need["layers"]):
                continue  # resident, immutable, reusable
            if existing is not None:
                hooks.drop_table(tname)
            new_bytes += need["table_bytes"] + need["scratch_bytes"]
        # Residency reporting, not a hard gate: on this 8 GiB WDDM laptop
        # GPU the driver reports ~0 B free after build while the CuPy pool
        # holds ~0.5 GiB reusable and larger allocations transparently page
        # to host. Skipping would make the whole atlas unexecutable, so we
        # run, record the residency class, and let the per-arm A/B brackets
        # plus the overhead-control arm absorb the paging pressure (the
        # brackets run under identical table residency as the oracle).
        free = int(cp.cuda.Device(0).mem_info[0])
        pool_free = int(cp.get_default_memory_pool().free_bytes())
        effective_free = free + pool_free
        absolute_cap = 3 * 2**30
        if new_bytes > absolute_cap:
            payload.update({
                "status": "skipped_vram",
                "required_new_bytes": int(new_bytes),
                "free_bytes": free,
                "pool_free_bytes": pool_free,
                "reserve_bytes": int(reserve_bytes),
                "completed_utc": utc_now(),
            })
            _log(name, "SKIPPED: above absolute VRAM cap", arm_started)
            return payload
        residency = ("pool_resident" if new_bytes <= effective_free
                     else "wddm_oversubscribed")
        if residency != "pool_resident":
            _log(name, f"WARNING: tables oversubscribed "
                       f"(new {new_bytes / 2**20:.0f} MiB, effective free "
                       f"{effective_free / 2**20:.0f} MiB)", arm_started)
        payload["table_residency"] = {
            "class": residency,
            "new_bytes": int(new_bytes),
            "driver_free_bytes": free,
            "pool_free_bytes": pool_free,
            "reserve_bytes_requested": int(reserve_bytes),
            "reserve_met": bool(new_bytes + reserve_bytes
                                <= effective_free),
        }
        for tname, need in needs.items():
            if tname in hooks.tables or tname in hooks.scratch:
                continue
            if need["table_bytes"] > 0:
                hooks.add_table(tname, need["layers"], need["dim"])
            else:
                hooks.add_scratch_only(tname, need["dim"])
        payload["tables"] = {
            tname: {"layers": need["layers"], "dim": need["dim"],
                    "table_bytes": need["table_bytes"],
                    "scratch_bytes": need["scratch_bytes"]}
            for tname, need in needs.items()
        }
        payload["replay_profile"] = replay_profile(spec, dims)

        def phase(label):
            _log(name, label)

        # Control A (steady-state reference, parity baseline).
        phase("control A recapture+pass")
        hooks.set_mode(None)
        recapture(rt)
        pctr.fill(0)
        ctl_a = run_pass(rt, workload, hooks, pctr, kern_inc, timed=True,
                         probe_lm_head=spec["L"], collect_state=True)

        # Recorder (untimed) for tables not yet recorded.
        rec = None
        unrecorded = {tname for tname in needs
                      if tname in hooks.tables
                      and not hooks.tables[tname]["recorded"]}
        if unrecorded:
            phase(f"record pass ({sorted(unrecorded)})")
            rec_active = {
                "L": False,
                "E": spec["E"] if "E" in unrecorded else set(),
                "A": spec["A"] if {"A_K", "A_V", "A_O"} & unrecorded
                    else set(),
                "M_IN": spec["M_IN"] if "M_IN" in unrecorded else set(),
                "M_OUT": spec["M_OUT"] if "M_OUT" in unrecorded else set(),
            }
            hooks.set_mode("record", rec_active)
            recapture(rt)
            pctr.fill(0)
            rec = run_pass(rt, workload, hooks, pctr, kern_inc, timed=False,
                           probe_lm_head=spec["L"], collect_state=False)
            for tname in unrecorded:
                hooks.tables[tname]["recorded"] = True

        # Oracle (timed).
        phase("oracle recapture+pass")
        hooks.set_mode("oracle", spec)
        recapture(rt)
        pctr.fill(0)
        oracle = run_pass(rt, workload, hooks, pctr, kern_inc, timed=True,
                          probe_lm_head=spec["L"], collect_state=True)

        # Copy-overhead control (timed): original paths keep running, the
        # same table loads execute into scratch.
        phase("overhead recapture+pass")
        hooks.set_mode("overhead", spec)
        recapture(rt)
        pctr.fill(0)
        overhead = run_pass(rt, workload, hooks, pctr, kern_inc, timed=True,
                            probe_lm_head=spec["L"], collect_state=False)

        # Control B (drift bracket).
        phase("control B recapture+pass")
        hooks.set_mode(None)
        recapture(rt)
        pctr.fill(0)
        ctl_b = run_pass(rt, workload, hooks, pctr, kern_inc, timed=True,
                         probe_lm_head=spec["L"], collect_state=False)

        thermal = drift_and_slope(ctl_a["samples"], ctl_b["samples"])
        oracle2 = None
        if thermal["exceeds_gate"] and not args.no_second_oracle:
            phase("thermal slope > gate: second oracle pass")
            hooks.set_mode("oracle", spec)
            recapture(rt)
            pctr.fill(0)
            oracle2 = run_pass(rt, workload, hooks, pctr, kern_inc,
                               timed=True, probe_lm_head=spec["L"],
                               collect_state=False)
            hooks.set_mode(None)

        stats = arm_stats_from_passes(
            ctl_a, oracle, overhead, ctl_b, oracle2,
            seed=args.seed)

        # ---------------- parity ----------------
        tokens_equal = (oracle["token_ids"] == ctl_a["token_ids"])
        logits_equal = (oracle["logits_sha256"] == ctl_a["logits_sha256"])
        probe_equal = (oracle["probe_ids"] == ctl_a["probe_ids"])
        states_equal = bool(
            oracle["states"] and ctl_a["states"]
            and all(
                o["hidden_sha256"] == c["hidden_sha256"]
                and o["recurrent_sha256"] == c["recurrent_sha256"]
                and o["kv_sha256"] == c["kv_sha256"]
                for o, c in zip(oracle["states"], ctl_a["states"]))
        )
        kv_used_bytes = [s["kv_used_bytes_per_layer"]
                         for s in ctl_a["states"]]
        control_b_equal = (ctl_b["token_ids"] == ctl_a["token_ids"])
        overhead_equal = (overhead["token_ids"] == ctl_a["token_ids"])
        rec_equal = (rec is None
                     or rec["token_ids"] == ctl_a["token_ids"])
        if spec["L"]:
            # Under L the timed-position ids are echoes of the staged
            # teacher-forced tokens by construction (argmax skipped); parity
            # is carried by state hashes and the eager probe position.
            parity = bool(states_equal and logits_equal and probe_equal
                          and control_b_equal and overhead_equal
                          and rec_equal)
        else:
            parity = bool(tokens_equal and logits_equal and probe_equal
                          and states_equal and control_b_equal
                          and overhead_equal and rec_equal)

        payload.update({
            "status": "measured" if parity else "parity_failed",
            "parity": {
                "token_ids_equal": (
                    "teacher_forced_echo" if spec["L"]
                    else bool(tokens_equal)),
                "final_logits_sha256_equal": bool(logits_equal),
                "probe_token_ids_equal": bool(probe_equal),
                "hidden_state_equal": states_equal,
                "recurrent_state_equal": states_equal,
                "kv_state_equal": states_equal,
                "kv_used_bytes_per_layer": kv_used_bytes,
                "control_b_tokens_equal": bool(control_b_equal),
                "overhead_tokens_equal": bool(overhead_equal),
                "recorder_tokens_equal": bool(rec_equal),
                "teacher_forced_upper_bound": bool(spec["L"]),
            },
            "bypassed_performance_state": (
                {"moe_cache_lru_metadata": "intentionally not updated by "
                 "the E oracle; semantic state unaffected"}
                if spec["E"] else None),
            "divergence": {
                "oracle_vs_control_a": (
                    [] if spec["L"] else _first_divergences(
                        oracle["token_ids"], ctl_a["token_ids"])),
                "control_b_vs_control_a": _first_divergences(
                    ctl_b["token_ids"], ctl_a["token_ids"]),
            },
            "workload": {
                "prompts": len(workload),
                "measured_positions": len(oracle["samples"]),
            },
            "timing": {
                "control_A": metrics(ctl_a["samples"]),
                "oracle": metrics(oracle["samples"]),
                "overhead": metrics(overhead["samples"]),
                "control_B": metrics(ctl_b["samples"]),
                "oracle2": (metrics(oracle2["samples"])
                            if oracle2 is not None else None),
            },
            "statistics": stats,
            "capture_hook_calls": dict(hooks.hook_calls),
            "completed_utc": utc_now(),
        })
        _log(name, f"done: status={payload['status']} "
                   f"corrected={stats['corrected_savings_ms_mean']:.4f} ms "
                   f"band={stats['decision_band']}", arm_started)
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "completed_utc": utc_now(),
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
        })
        _log(name, f"TECHNICAL FAILURE: {exc}", arm_started)
    return payload

def build_summary(arm_payloads, mamba_groups, additivity, dims, per,
                  n_prompts, identity, args) -> dict:
    """Lattice, interactions, decision bands and the final S100 flags."""
    by_name = {p["arm"]: p for p in arm_payloads}

    def corrected(name):
        p = by_name.get(name)
        if p is None or p.get("status") != "measured":
            return None
        return p["statistics"]

    singles = {}
    for group in ("L", "E", "A"):
        stats = corrected(group)
        if stats is not None:
            singles[group] = {
                "corrected_savings_ms_mean": stats[
                    "corrected_savings_ms_mean"],
                "corrected_one_sided95": stats[
                    "corrected_savings_ms_one_sided95"],
                "decision_band": stats["decision_band"],
                "reference_mean_ms": stats["reference_mean_ms"],
            }

    # Mamba rollups: sum of per-group corrected means, only valid after the
    # additivity check. Bounds are stacked per-group one-sided bounds
    # (conservative, flagged).
    mamba_rollup = {}
    for base in ("M_IN", "M_OUT", "M_IO"):
        group_stats = []
        for gi in range(len(mamba_groups)):
            stats = corrected(f"{base}_g{gi}")
            if stats is not None:
                group_stats.append(stats)
        if not group_stats:
            continue
        complete = len(group_stats) == len(mamba_groups)
        entry = {
            "groups_measured": len(group_stats),
            "groups_total": len(mamba_groups),
            "corrected_savings_ms_mean_sum": float(sum(
                s["corrected_savings_ms_mean"] for s in group_stats)),
            "corrected_one_sided95_stacked": {
                "lower": float(sum(
                    s["corrected_savings_ms_one_sided95"]["lower"]
                    for s in group_stats)),
                "upper": float(sum(
                    s["corrected_savings_ms_one_sided95"]["upper"]
                    for s in group_stats)),
            },
            "additive_sum_valid": bool(
                complete and (len(mamba_groups) == 1
                              or additivity.get("additive_ok"))),
            "note": ("sum of per-group arms; valid only after the "
                     "additivity check, bounds are stacked per-group "
                     "one-sided bounds"),
        }
        mamba_rollup[base] = entry

    def group_saving(group):
        """Point estimate of a group's corrected saving, or None."""
        if group in singles:
            return singles[group]["corrected_savings_ms_mean"]
        roll = mamba_rollup.get(group)
        if roll is not None and roll["additive_sum_valid"]:
            return roll["corrected_savings_ms_mean_sum"]
        return None

    # Combination arms (point or interval).
    combos = {}
    for p in arm_payloads:
        name = p["arm"]
        if "+" not in name or "__h" in name:
            continue
        stats = corrected(name)
        if stats is not None:
            combos[name] = {
                "type": "point",
                "corrected_savings_ms_mean": stats[
                    "corrected_savings_ms_mean"],
                "corrected_one_sided95": stats[
                    "corrected_savings_ms_one_sided95"],
                "reference_mean_ms": stats["reference_mean_ms"],
                "combined_latency_estimate_ms": float(
                    stats["reference_mean_ms"]
                    - stats["corrected_savings_ms_mean"]),
            }
    # Interval assembly for split combinations.
    split_names = [n for n in by_name if "__h" in n]
    split_bases = sorted({n.rsplit("__h", 1)[0] for n in split_names})
    for base in split_bases:
        halves = []
        for suffix in ("__h0", "__h1"):
            stats = corrected(base + suffix)
            if stats is not None:
                halves.append(stats["corrected_savings_ms_mean"])
        if halves:
            combos[base] = {
                "type": "interval",
                "measured_half_savings_ms": halves,
                "interval_ms": {"lower": max(halves), "upper": None},
                "note": ("combination did not fit in VRAM; two "
                         "complementary table-resident halves measured. "
                         "Lower edge = best measured half (the full "
                         "combination removes at least as much work, "
                         "assuming non-negative savings). No additive point "
                         "estimate per the frozen plan."),
            }

    # Interactions where every ingredient is a measured point.
    interactions = {}

    def maybe_interaction(s_name, t_name, st_name):
        s_s, s_t = group_saving(s_name), group_saving(t_name)
        st = combos.get(st_name)
        if s_s is None or s_t is None:
            return
        if st is None or st.get("type") != "point":
            interactions[f"{s_name}*{t_name}"] = {
                "status": "unavailable",
                "reason": f"{st_name} not measured as a point",
            }
            return
        interactions[f"{s_name}*{t_name}"] = {
            "status": "measured",
            "interaction_ms": interaction(
                st["corrected_savings_ms_mean"], s_s, s_t),
            "saving_union_ms": st["corrected_savings_ms_mean"],
            "saving_s_ms": s_s, "saving_t_ms": s_t,
        }

    maybe_interaction("E", "L", "E+L")
    maybe_interaction("A", "L", "A+L")
    maybe_interaction("E", "A", "E+A")
    maybe_interaction("M_IO", "E", "M_IO+E")
    maybe_interaction("M_IO", "A", "M_IO+A")
    maybe_interaction("M_IO", "L", "M_IO+L")

    # Decision-band classification per group.
    bands = {}
    for group, row in singles.items():
        bands[group] = row["decision_band"]
    for base, roll in mamba_rollup.items():
        if roll["additive_sum_valid"]:
            bands[base] = decision_band(
                roll["corrected_one_sided95_stacked"]["lower"],
                roll["corrected_one_sided95_stacked"]["upper"])
        else:
            bands[base] = "unclassified_additive_sum_not_validated"

    targets = sorted(g for g, b in bands.items() if b == "target")
    closed = sorted(g for g, b in bands.items() if b == "closed")

    # S100 path flag from the largest combination.
    full = combos.get("M_IO+E+A+L")
    if full is not None and full.get("type") == "point":
        ref = full["reference_mean_ms"]
        lower = full["corrected_one_sided95"]["lower"]
        s100 = {
            "basis": "M_IO+E+A+L point",
            "reference_mean_ms": ref,
            "combined_latency_estimate_ms": float(
                ref - full["corrected_savings_ms_mean"]),
            "combined_latency_conservative_ms": float(ref - lower),
            "S100_PATH_EXISTS_IN_MEASURED_SET": combined_path_exists(
                ref, lower),
        }
    elif full is not None:
        s100 = {
            "basis": "M_IO+E+A+L interval (VRAM split)",
            "interval": full["interval_ms"],
            "S100_PATH_EXISTS_IN_MEASURED_SET": "indeterminate_interval",
            "note": "no additive point estimate per the frozen plan",
        }
    else:
        s100 = {
            "basis": None,
            "S100_PATH_EXISTS_IN_MEASURED_SET": "not_measured",
        }

    return {
        "kind": "s100_lightning17_atlas_summary",
        "completed_utc": utc_now(),
        "identity": identity,
        "dimensions": dims,
        "table_bytes_per_layer": per,
        "workload": {"prompts": n_prompts, "max_pos": MAX_POS,
                     "warmup_targets_per_prompt": WARMUP_TARGETS},
        "mamba_partition": {"groups": mamba_groups,
                            "budget_mib": args.max_mamba_vram_mib,
                            "reserve_mib": args.reserve_mib},
        "additivity": additivity,
        "arms": {name: {"status": p.get("status"),
                        "statistics": p.get("statistics"),
                        "replay_profile": p.get("replay_profile"),
                        "parity": p.get("parity"),
                        "error": (p.get("error") or {}).get("message")}
                 for name, p in by_name.items()},
        "singles": singles,
        "mamba_rollup": mamba_rollup,
        "combinations": combos,
        "interactions": interactions,
        "decision_bands": bands,
        "flags": {
            "GROUPS_GE_0_58MS_LOWER": targets,
            "GROUPS_LT_0_20MS_UPPER_CLOSED": closed,
            **s100,
        },
    }

def summary_text(summary: dict) -> str:
    lines = []
    lines.append("S100 Lightning Phase 17 -- oracle atlas summary")
    lines.append(f"completed: {summary['completed_utc']}")
    lines.append("")
    lines.append("Arms:")
    for name, row in summary["arms"].items():
        stats = row["statistics"]
        if stats is None:
            lines.append(f"  {name:<22} {row['status']}")
            continue
        c95 = stats["corrected_savings_ms_one_sided95"]
        lines.append(
            f"  {name:<22} {row['status']:<8} corrected "
            f"{stats['corrected_savings_ms_mean']:+.4f} ms "
            f"[{c95['lower']:+.4f}, {c95['upper']:+.4f}] "
            f"band={stats['decision_band']}")
    lines.append("")
    lines.append("Mamba rollup (additive sums, gated on additivity check):")
    for base, roll in summary["mamba_rollup"].items():
        lines.append(
            f"  {base:<8} sum {roll['corrected_savings_ms_mean_sum']:+.4f} ms "
            f"valid={roll['additive_sum_valid']} "
            f"({roll['groups_measured']}/{roll['groups_total']} groups)")
    lines.append("")
    lines.append("Interactions (ms; positive = super-additive):")
    for pair, row in summary["interactions"].items():
        if row["status"] == "measured":
            lines.append(f"  {pair:<12} {row['interaction_ms']:+.4f}")
        else:
            lines.append(f"  {pair:<12} {row['status']} ({row['reason']})")
    lines.append("")
    lines.append("Decision bands:")
    for group, band in summary["decision_bands"].items():
        lines.append(f"  {group:<8} {band}")
    lines.append("")
    flags = summary["flags"]
    lines.append(f"Groups >= 0.58 ms lower bound (targets): "
                 f"{flags['GROUPS_GE_0_58MS_LOWER']}")
    lines.append(f"Groups < 0.20 ms upper bound (closed):  "
                 f"{flags['GROUPS_LT_0_20MS_UPPER_CLOSED']}")
    lines.append(f"S100_PATH_EXISTS_IN_MEASURED_SET: "
                 f"{flags['S100_PATH_EXISTS_IN_MEASURED_SET']}")
    return "\n".join(lines) + "\n"

def main() -> int:
    ap = argparse.ArgumentParser(
        description="S100 Lightning Phase 17 oracle atlas")
    ap.add_argument("--arms", default="ALL",
                    help="comma-separated arm tokens "
                         f"({','.join(ALL_ARM_TOKENS)}) or ALL")
    ap.add_argument("--max-mamba-vram-mib", type=float, default=None,
                    help="cap on total resident Mamba replay tables; "
                         "default: free VRAM minus the reserve")
    ap.add_argument("--reserve-mib", type=float, default=RESERVE_MIB)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--no-second-oracle", action="store_true")
    args = ap.parse_args()

    from common import REPO, require_gpu_free
    require_gpu_free()  # before any CUDA context is created
    import cupy as cp

    from s100_phase10a_runtime import build
    from diag_component_marginals_graph import _recapture
    from s100_lightning16_common import assert_lightning
    from s100_lightning16r_throughput import frozen_workload

    results = REPO / "pro_research" / "results" / RESULTS_DIRNAME
    results.mkdir(parents=True, exist_ok=True)
    reserve_bytes = int(args.reserve_mib * 2**20)

    requested = set(ALL_ARM_TOKENS if args.arms.strip().upper() == "ALL"
                    else [t.strip().upper() for t in args.arms.split(",")
                          if t.strip()])
    unknown = requested - set(ALL_ARM_TOKENS)
    if unknown:
        raise SystemExit(f"unknown arm tokens: {sorted(unknown)}")

    identity = assert_lightning()
    workload = frozen_workload()
    n_prompts = len(workload)

    _log("setup", "build runtime (phase10a graph parent)")
    bundle = build()
    rt = bundle.rt
    if not rt.fp8_kv:
        raise RuntimeError("atlas requires the FP8 KV path (fp8_kv=True)")
    dims = runtime_dims(rt)
    per = table_plan(dims, n_prompts, MAX_POS)

    module = cp.RawModule(code=KERNELS)
    kern_store = module.get_function("vec_store")
    kern_load = module.get_function("vec_load")
    kern_inc = module.get_function("ctr_inc")
    pctr = cp.zeros(1, cp.int32)

    hooks = AtlasHooks(rt, kern_store=kern_store, kern_load=kern_load,
                       pctr=pctr, n_prompts=n_prompts, max_pos=MAX_POS)
    hooks.install()

    # Warm pass: the first post-build workload pass is a known outlier
    # (same finding as 17A); discard it so every arm runs in steady state.
    _log("setup", "warm pass (discarded)")
    pctr.fill(0)
    run_pass(rt, workload, hooks, pctr, kern_inc, timed=False,
             probe_lm_head=False, collect_state=False)

    # ---------------- Mamba partition ----------------
    # Only computed when a Mamba arm is requested (the L/E/A smoke path must
    # not crash on partition bookkeeping). Budget comes from
    # --max-mamba-vram-mib: driver-free is ~0 on this WDDM box, so a
    # free-minus-reserve budget would be negative by construction.
    mamba_arms = {"M_IN", "M_OUT", "M_IO", "M_IO+E", "M_IO+A", "M_IO+L",
                  "M_IO+E+A+L"}
    mamba_groups = []
    if requested & mamba_arms:
        free0 = int(cp.cuda.Device(0).mem_info[0])
        pool0 = int(cp.get_default_memory_pool().free_bytes())
        budget = int((args.max_mamba_vram_mib or 512) * 2**20)
        mamba_groups = partition_layers(dims["mamba_layers"], per["M_IO"],
                                        budget)
        _log("setup", f"mamba partition: {[len(g) for g in mamba_groups]} "
                      f"layers/group, budget {budget / 2**20:.0f} MiB, "
                      f"driver-free {free0 / 2**20:.0f} MiB, "
                      f"pool-free {pool0 / 2**20:.0f} MiB")

    # ---------------- arm specs ----------------
    specs = []

    def spec(name, *, L=False, E=None, A=None, M_IN=None, M_OUT=None,
             min_samples=560):
        return {"name": name, "L": L, "E": set(E or []),
                "A": set(A or []), "M_IN": set(M_IN or []),
                "M_OUT": set(M_OUT or []), "min_samples": min_samples}

    if "L" in requested:
        specs.append(spec("L", L=True))
    if "E" in requested:
        specs.append(spec("E", E=dims["moe_layers"]))
    if "A" in requested:
        specs.append(spec("A", A=dims["attn_layers"]))
    if "E+L" in requested:
        specs.append(spec("E+L", L=True, E=dims["moe_layers"]))
    if "A+L" in requested:
        specs.append(spec("A+L", L=True, A=dims["attn_layers"]))
    if "E+A" in requested:
        specs.append(spec("E+A", E=dims["moe_layers"],
                          A=dims["attn_layers"]))
    for gi, group in enumerate(mamba_groups):
        if "M_IN" in requested:
            specs.append(spec(f"M_IN_g{gi}", M_IN=group, min_samples=256))
        if "M_OUT" in requested:
            specs.append(spec(f"M_OUT_g{gi}", M_OUT=group, min_samples=256))
        if "M_IO" in requested:
            specs.append(spec(f"M_IO_g{gi}", M_IN=group, M_OUT=group,
                              min_samples=256))

    # Additivity check on one adjacent pair. Preferred: union of the two
    # smallest adjacent groups when it fits; fallback: the two layers
    # straddling the first group boundary (documented deviation).
    additivity = {"status": "not_run"}
    add_specs = []
    if "M_IO" in requested and len(mamba_groups) >= 2:
        pair_union = None
        best = None
        for gi in range(len(mamba_groups) - 1):
            union = mamba_groups[gi] + mamba_groups[gi + 1]
            if best is None or len(union) < len(best):
                best = union
        # Full group-pair union when under the absolute cap (same rule as
        # the M_io combos; fits_budget on driver-free is meaningless here).
        if per["M_IO"] * len(best) <= 3 * 2**30:
            pair_union = best
        if pair_union is not None:
            gis = [gi for gi in range(len(mamba_groups) - 1)
                   if mamba_groups[gi] + mamba_groups[gi + 1] == best]
            add_specs.append(spec("ADDCHK_M_IO_PAIR", M_IN=pair_union,
                                  M_OUT=pair_union, min_samples=256))
            additivity = {"status": "scheduled", "kind": "group_pair",
                          "groups": gis, "layers": pair_union}
        else:
            a_layer = mamba_groups[0][-1]
            b_layer = mamba_groups[1][0]
            add_specs.append(spec("ADDCHK_M_IO_A", M_IN=[a_layer],
                                  M_OUT=[a_layer], min_samples=256))
            add_specs.append(spec("ADDCHK_M_IO_B", M_IN=[b_layer],
                                  M_OUT=[b_layer], min_samples=256))
            add_specs.append(spec("ADDCHK_M_IO_AB", M_IN=[a_layer, b_layer],
                                  M_OUT=[a_layer, b_layer],
                                  min_samples=256))
            additivity = {"status": "scheduled", "kind": "layer_pair",
                          "layers": [a_layer, b_layer],
                          "deviation": ("group-pair union exceeded the VRAM "
                                        "budget; checked on the two layers "
                                        "straddling the first group "
                                        "boundary instead")}
    specs.extend(add_specs)

    # ---------------- run arms ----------------
    arm_payloads = []
    for s in specs:
        payload = execute_arm(s, rt=rt, hooks=hooks, workload=workload,
                              kern_inc=kern_inc, pctr=pctr, results=results,
                              dims=dims, per=per, reserve_bytes=reserve_bytes,
                              args=args, recapture=_recapture)
        arm_payloads.append(payload)
        write_json_atomic(
            results / f"S100_LIGHTNING17_ATLAS_{s['name']}.json",
            payload, archive=True)

    # Evaluate the additivity check.
    by_name = {p["arm"]: p for p in arm_payloads}

    def corrected_mean(name):
        p = by_name.get(name)
        if p is None or p.get("status") != "measured":
            return None
        return p["statistics"]["corrected_savings_ms_mean"]

    if additivity.get("kind") == "group_pair":
        gis = additivity["groups"]
        combined = corrected_mean("ADDCHK_M_IO_PAIR")
        parts = [corrected_mean(f"M_IO_g{gi}") for gi in gis]
        if combined is not None and all(v is not None for v in parts):
            additivity.update(additivity_check(combined, sum(parts)))
        else:
            additivity["status"] = "inconclusive"
            additivity["additive_ok"] = False
    elif additivity.get("kind") == "layer_pair":
        a = corrected_mean("ADDCHK_M_IO_A")
        b = corrected_mean("ADDCHK_M_IO_B")
        ab = corrected_mean("ADDCHK_M_IO_AB")
        if None not in (a, b, ab):
            additivity.update(additivity_check(ab, a + b))
        else:
            additivity["status"] = "inconclusive"
            additivity["additive_ok"] = False

    # ---------------- M_io combinations (fit-dependent) ----------------
    combo_requested = requested & set(MIO_COMBOS)
    for combo in sorted(combo_requested):
        rest = combo.split("+", 1)[1]
        extra = {}
        if "E" in rest:
            extra["E"] = set(dims["moe_layers"])
        if "A" in rest:
            extra["A"] = set(dims["attn_layers"])
        if "L" in rest:
            extra["L"] = True
        mio_full_bytes = per["M_IO"] * len(dims["mamba_layers"])
        resident_extra = sum(
            need["table_bytes"] + need["scratch_bytes"]
            for need in arm_needs(
                {"L": extra.get("L", False), "E": extra.get("E", set()),
                 "A": extra.get("A", set()), "M_IN": set(),
                 "M_OUT": set()}, dims, per).values())
        # Full-vs-halves on the absolute cap: fits_budget against
        # driver-free is meaningless on this WDDM box (~0 B free after
        # build); the residency class is recorded per arm by execute_arm.
        if mio_full_bytes + resident_extra <= 3 * 2**30:
            combo_specs = [spec(combo, M_IN=dims["mamba_layers"],
                                M_OUT=dims["mamba_layers"], **extra)]
        else:
            h0, h1 = split_halves(dims["mamba_layers"])
            combo_specs = [
                spec(f"{combo}__h0", M_IN=h0, M_OUT=h0, **extra),
                spec(f"{combo}__h1", M_IN=h1, M_OUT=h1, **extra),
            ]
            _log(combo, "full M_io tables above absolute cap; running two "
                        "complementary halves (interval only)")
        for s in combo_specs:
            payload = execute_arm(
                s, rt=rt, hooks=hooks, workload=workload, kern_inc=kern_inc,
                pctr=pctr, results=results, dims=dims, per=per,
                reserve_bytes=reserve_bytes, args=args, recapture=_recapture)
            payload["resident_extra_bytes"] = int(resident_extra)
            arm_payloads.append(payload)
            write_json_atomic(
                results / f"S100_LIGHTNING17_ATLAS_{s['name']}.json",
                payload, archive=True)

    # ---------------- summary ----------------
    summary = build_summary(arm_payloads, mamba_groups, additivity, dims,
                            per, n_prompts, identity, args)
    write_json_atomic(results / "S100_LIGHTNING17_SUMMARY.json", summary,
                      archive=True)
    write_text_atomic(results / "S100_LIGHTNING17_SUMMARY.txt",
                      summary_text(summary), archive=True)
    hooks.restore()
    print(summary_text(summary), flush=True)
    measured = any(p.get("status") == "measured" for p in arm_payloads)
    return 0 if measured else 2

if __name__ == "__main__":
    raise SystemExit(main())
