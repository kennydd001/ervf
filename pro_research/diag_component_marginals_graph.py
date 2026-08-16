"""The component attribution again, but inside the captured graph this time.

`diag_component_marginals_v6` and `diag_moe_subkernel_marginals` were both
measured eager, via `rt.step()`. A no-op control kernel then showed that an
eager launch costs **~7.75 us of CPU issue time**, independent of grid size
(138 launches = 1.0698 ms at grid (21,8), 1.0805 ms at grid (1,1), 414 launches
= 3.0737 ms). Every eager marginal therefore carries 7.75 us times its own
launch count -- and the production stack runs in a captured graph, which does
not pay it.

The correction is not small. `down_masked` issues 138 launches per token, so
~1.07 ms of its 1.655 ms eager marginal was issue time: its real GPU work is
**0.431 ms against a 0.257 ms floor, i.e. 60% efficiency**, not the 15% the
eager number implied. MoE as a whole issues ~414 launches per token.

So the headroom table that has been steering this work is an eager table, and
no kernel should be rewritten off it. This measures the same marginals in the
regime production actually runs in.

## Method

Identical marginal probes to the eager version -- the real loop with exactly one
component called one extra time into a discarded buffer, so routing, cache,
residual stream and produced tokens stay bit-identical, which is the gate.
`_mamba` gets its own scratch recurrence state because `conv_step` writes
`conv[i]` and `ssm_step` writes `ssm[i]`; `_attention` and `_moe_dev` are
idempotent within a token and were verified so, not assumed.

The difference is that each arm re-captures the graph after installing its probe
(`_recapture` redoes only the capture, reusing every buffer `setup_graph()`
allocated -- calling `setup_graph()` again would early-return and, worse,
re-allocate the 0.656 GiB pinned embedding table each time). Timing is SYNC
semantics: one replay plus one ring harvest per token, the same regime the
21.0923 ms V6 record was measured in.

## Gates

  G1  every arm's token ids == BASE_A's, all prompts, all tokens
  G2  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms
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

OUT = REPO / "pro_research" / "diag_component_marginals_graph.json"

BYTES_PER_TOKEN = {"mamba": 892_000_000, "attn": 280_800_000, "moe": 741_300_000}
KERNEL_CEILING_GB_S = 249.0
EAGER_REFERENCE = {"mamba": 5.662, "attn": 1.917, "moe": 11.004}
LAUNCHES_PER_TOKEN = {"mamba": 23 * 2, "attn": 6 * 4, "moe": 414}
EAGER_LAUNCH_US = 7.75


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
    slot = (start + len(prompt_ids) - 1) % int(rt._ring_size)
    return int(rt.ring_harvest(slot, 1)[0])


def _run(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
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
        "kind": "diag_component_marginals_graph",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "why": "the eager marginals carry ~7.75 us of CPU launch-issue time per kernel launch (no-op control, grid-independent), which the captured production graph does not pay. down_masked alone: 138 launches = 1.07 ms of its 1.655 ms eager marginal. The headroom table steering this work is an eager table.",
        "claim_boundary": "SYNC semantics (one replay + one ring harvest per token), the regime the 21.0923 ms V6 record was measured in.",
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _e, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 24) if args.mode == "smoke" else max(n, 192)
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts)}
        payload["environment"] = environment_snapshot((
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",))

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, _ = _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)
        rt.setup_graph()

        scratch = cp.zeros(rt.hidden, dtype=cp.float32)
        orig_mamba, orig_attn, orig_moe = rt._mamba, rt._attention, rt._moe_dev
        scratch_conv = {j: cp.zeros_like(v) for j, v in rt.conv.items()}
        scratch_ssm = {j: cp.zeros_like(v) for j, v in rt.ssm.items()}

        def probe_mamba(self, i, out):
            r = orig_mamba(i, out)
            rc, rs = self.conv, self.ssm
            self.conv, self.ssm = scratch_conv, scratch_ssm
            try:
                orig_mamba(i, scratch)
            finally:
                self.conv, self.ssm = rc, rs
            return r

        def probe_attn(self, i, out):
            r = orig_attn(i, out)
            orig_attn(i, scratch)
            return r

        def probe_moe(self, i, out):
            r = orig_moe(i, out)
            orig_moe(i, scratch)
            return r

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
        results = [
            arm("BASE_A", noop, noop),
            arm("MARGINAL_MAMBA",
                lambda: setattr(rt, "_mamba", types.MethodType(probe_mamba, rt)),
                lambda: setattr(rt, "_mamba", orig_mamba)),
            arm("MARGINAL_ATTN",
                lambda: setattr(rt, "_attention", types.MethodType(probe_attn, rt)),
                lambda: setattr(rt, "_attention", orig_attn)),
            arm("MARGINAL_MOE",
                lambda: setattr(rt, "_moe_dev", types.MethodType(probe_moe, rt)),
                lambda: setattr(rt, "_moe_dev", orig_moe)),
            arm("BASE_B", noop, noop),
        ]

        by = {lab: (ids, pc) for lab, ids, pc in results}
        base_ids, a = by["BASE_A"]
        _, b = by["BASE_B"]
        drift = abs(float(a["p50"]) - float(b["p50"]))
        midpoint = (float(a["p50"]) + float(b["p50"])) / 2.0

        arms_out, marginals = {}, {}
        for lab, ids, pc in results:
            divs = {p["prompt"]: first_divergence(base_ids[p["prompt"]], ids[p["prompt"]])
                    for p in prompts}
            arms_out[lab] = {"percentiles": pc, "tok_s": 1000.0 / float(pc["p50"]),
                             "ids_match_base_a": all(v is None for v in divs.values()),
                             "first_divergence": divs}
            if lab.startswith("MARGINAL_"):
                key = lab.split("_", 1)[1].lower()
                marg = float(pc["p50"]) - midpoint
                byt = BYTES_PER_TOKEN.get(key)
                eager = EAGER_REFERENCE.get(key)
                overhead = LAUNCHES_PER_TOKEN.get(key, 0) * EAGER_LAUNCH_US / 1000.0
                marginals[key] = {
                    "graph_marginal_ms_per_token": marg,
                    "fraction_of_base_token": marg / midpoint,
                    "eager_marginal_ms_per_token": eager,
                    "eager_minus_graph_ms": (eager - marg) if eager else None,
                    "predicted_launch_overhead_ms": overhead,
                    "launches_per_token": LAUNCHES_PER_TOKEN.get(key),
                    "bytes_per_token": byt,
                    "floor_ms_at_249_GB_s": (byt / (KERNEL_CEILING_GB_S * 1e9) * 1e3) if byt else None,
                    "headroom_ms": (marg - byt / (KERNEL_CEILING_GB_S * 1e9) * 1e3) if byt else None,
                    "efficiency_vs_kernel_ceiling": ((byt / (marg * 1e-3) / 1e9) / KERNEL_CEILING_GB_S)
                                                    if (byt and marg > 0) else None,
                }

        gates = {
            "G1_all_arms_ids_match_base_a": all(v["ids_match_base_a"] for v in arms_out.values()),
            "G2_drift_le_1ms": drift <= 1.0,
        }
        payload.update({
            "arms": arms_out,
            "baseline_midpoint_ms": midpoint,
            "baseline_tok_s": 1000.0 / midpoint,
            "drift_ms": drift,
            "marginals": marginals,
            "sum_of_marginals_ms": sum(v["graph_marginal_ms_per_token"] for v in marginals.values()),
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
                      "baseline_tok_s": payload.get("baseline_tok_s"),
                      "drift_ms": payload.get("drift_ms"),
                      "marginals": payload.get("marginals"),
                      "sum_of_marginals_ms": payload.get("sum_of_marginals_ms"),
                      "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
