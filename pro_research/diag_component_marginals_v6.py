"""Unconfounded in-loop cost attribution for the V6 stack, by S12's marginal
method -- and a direct test of the one component the project has never
isolated: Mamba.

## Why the existing breakdown cannot answer this

`diag_v6_component_breakdown.json` stubs a component out (`rt._mamba ->
out.fill(0)`) and reads the end-to-end difference. Its Mamba arm came out at
21.288 ms against a 20.859 ms real arm -- a NEGATIVE upper bound of -0.429 ms,
i.e. "Mamba is free". That cannot be true, and the reason the arm is unusable is
structural, not statistical: zeroing a block's output changes the residual
stream, which changes MoE routing, which changes which experts miss the LRU,
which changes PCIe traffic. The stub arms are not the same workload. The note
in that file warns the STUB arms produce wrong tokens by design; what it does
not account for is that they also produce a different *cache* workload.

## Why Mamba specifically

Exact per-token active-byte accounting, computed from the safetensors headers
(not estimated):

    Mamba          892.0 MB/token   (23 layers x 38.78 MB)   43.6%
    routed up      387.3 MB/token   (23 x 6 x 2.806 MB)      18.9%
    shared+gate    290.0 MB/token   (23 x 12.61 MB)          14.2%
    attention      280.8 MB/token   (6 x 46.80 MB)           13.7%
    lm_head        198.2 MB/token                             9.7%
    ------------------------------------------------------ VRAM 2048 MB
    routed down     ~64 MB/token over PCIe (9% sparse + scales)

2048 MB at the machine's 338.4 GB/s is 6.05 ms/token = 165 tok/s, which
reproduces this project's own roofline figure exactly and therefore validates
the accounting. Under it, **Mamba is the single largest byte consumer in the
model at 43.6%, with a 2.64 ms/token floor** -- and every optimisation from
ERVF through V6 targeted MoE and the dense GEMVs.

## Method: marginal, not ablation

Run the REAL loop and call one component exactly ONE extra time per layer, into
a scratch buffer that is discarded. The residual stream, the routing, the cache
and the produced tokens are all bit-identical to the base arm -- the gate below
checks that -- so the end-to-end delta is that component's marginal in-loop
cost with nothing else changed.

The marginal of a SECOND execution is a lower bound on the first one's cost
where L2 reuse is possible. For Mamba (38.78 MB per layer) and attention
(46.80 MB per layer) the working set is far past L2, so the bound is tight;
for lm_head (198 MB) likewise. This is stated, not assumed away.

Arms, in order, with drift control by the V12 recipe (preheat to steady state,
one runtime, no reallocation between arms):

    BASE_A -> MARGINAL_MAMBA -> MARGINAL_ATTN -> MARGINAL_MOE -> BASE_B

Gates:
    G1  every arm produces token ids identical to BASE_A (the probe leaks
        nothing); a mismatch invalidates that arm.
    G2  |BASE_A.p50 - BASE_B.p50| <= 1.0 ms, else no attribution is reported.

Read-only diagnostic. runtime.py untouched; probes are installed on the live
instance with types.MethodType, the same pattern V3-V6 use.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import types
from typing import Any

from common import REPO, environment_snapshot, first_divergence, percentiles, require_gpu_free, utc_now
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

OUT = REPO / "pro_research" / "diag_component_marginals_v6.json"

# From the safetensors headers, for the efficiency column.
BYTES_PER_TOKEN = {
    "mamba": 892_000_000,
    "attn": 280_800_000,
    "moe": 741_300_000,   # routed up (387.3) + shared/gate (290.0) + sparse down (~64)
}
DEVICE_BW_GB_S = 338.4


def _reset_exact_state(rt) -> None:
    import cupy as cp

    rt.reset()
    for dev in getattr(rt, "_dev_cache", {}).values():
        for name in ("ids", "w", "slots", "need", "state2", "stats2"):
            if name in dev:
                dev[name].fill(0)
        for name, val in (("slot_of", -1), ("expert_of", -1), ("last_used", -1)):
            if name in dev:
                dev[name].fill(val)
    cp.cuda.Device(0).synchronize()


def _run(rt, prompt_ids: list[int], n: int) -> tuple[list[int], list[float]]:
    import cupy as cp

    _reset_exact_state(rt)
    nxt = None
    for tok in prompt_ids:
        nxt = int(rt.step(int(tok)))
    cp.cuda.Device(0).synchronize()
    ids, ms = [nxt], []
    for _ in range(n - 1):
        t0 = time.perf_counter_ns()
        nxt = int(rt.step(nxt))
        ms.append((time.perf_counter_ns() - t0) / 1e6)
        ids.append(nxt)
    return ids, ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "diag_component_marginals_v6",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "method": "S12 marginal: the real loop with exactly one extra call of one component into a discarded scratch buffer. Residual stream, routing, cache and produced tokens are bit-identical to the base arm, so the delta is unconfounded -- unlike the stub-ablation arms of diag_v6_component_breakdown.json, whose zeroed outputs change MoE routing and therefore the PCIe workload.",
        "caveat": "The marginal of a SECOND execution is a lower bound where L2 reuse is possible. Mamba (38.78 MB/layer), attention (46.80 MB/layer) and the MoE block all exceed L2 by a wide margin, so the bound is tight; it is still a bound.",
        "byte_accounting_source": "safetensors headers of models/nemotron_3_5_lightning_v35, per-layer, times the layer counts in config.layers_block_type",
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 32) if args.mode == "smoke" else max(n, 192)
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts),
                             "capacity": capacity}
        payload["environment"] = environment_snapshot((
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, _ = _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)

        scratch = cp.zeros(rt.hidden, dtype=cp.float32)
        orig_mamba, orig_attn, orig_moe = rt._mamba, rt._attention, rt._moe_dev

        # _mamba is STATEFUL: conv_step writes self.conv[i] and ssm_step writes
        # self.ssm[i]. A naive second call therefore advances the recurrence
        # twice and the token stream diverges -- which is exactly what the first
        # run of this diagnostic showed (MARGINAL_MAMBA diverged at generated
        # token 0/3/7 while ATTN and MOE stayed bit-exact). The probe call is
        # given its own scratch recurrence state instead, so it performs the
        # same work on the same weights while the real state is untouched.
        # (_attention and _moe_dev needed no such treatment: the KV append is
        # position-addressed and idempotent within a token, and a repeated
        # cache_assign re-hits the ids it just installed. Both were verified
        # bit-exact rather than assumed -- see the gate.)
        # conv/ssm are per-layer dicts, not one stacked array.
        scratch_conv = {j: cp.zeros_like(v) for j, v in rt.conv.items()}
        scratch_ssm = {j: cp.zeros_like(v) for j, v in rt.ssm.items()}

        def probe_mamba(self, i, out):
            r = orig_mamba(i, out)
            real_conv, real_ssm = self.conv, self.ssm
            self.conv, self.ssm = scratch_conv, scratch_ssm
            try:
                orig_mamba(i, scratch)
            finally:
                self.conv, self.ssm = real_conv, real_ssm
            return r

        def probe_attn(self, i, out):
            r = orig_attn(i, out)
            orig_attn(i, scratch)
            return r

        def probe_moe(self, i, out):
            r = orig_moe(i, out)
            orig_moe(i, scratch)
            return r

        # preheat to steady clocks before any timed arm
        _reset_exact_state(rt)
        nxt = None
        for tok in prompts[0]["prompt_ids"]:
            nxt = int(rt.step(int(tok)))
        for _ in range(32 if args.mode == "smoke" else 128):
            nxt = int(rt.step(nxt))

        def arm(label, install, restore):
            install()
            ids_by_prompt, ms_all = {}, []
            for p in prompts:
                ids, ms = _run(rt, p["prompt_ids"], n)
                ids_by_prompt[p["prompt"]] = ids
                ms_all.extend(ms)
            restore()
            return label, ids_by_prompt, percentiles(ms_all)

        noop = lambda: None
        results = []
        results.append(arm("BASE_A", noop, noop))
        results.append(arm("MARGINAL_MAMBA",
                           lambda: setattr(rt, "_mamba", types.MethodType(probe_mamba, rt)),
                           lambda: setattr(rt, "_mamba", orig_mamba)))
        results.append(arm("MARGINAL_ATTN",
                           lambda: setattr(rt, "_attention", types.MethodType(probe_attn, rt)),
                           lambda: setattr(rt, "_attention", orig_attn)))
        results.append(arm("MARGINAL_MOE",
                           lambda: setattr(rt, "_moe_dev", types.MethodType(probe_moe, rt)),
                           lambda: setattr(rt, "_moe_dev", orig_moe)))
        results.append(arm("BASE_B", noop, noop))

        by_label = {lab: (ids, pc) for lab, ids, pc in results}
        base_a_ids, base_a = by_label["BASE_A"]
        _, base_b = by_label["BASE_B"]
        drift = abs(float(base_a["p50"]) - float(base_b["p50"]))
        midpoint = (float(base_a["p50"]) + float(base_b["p50"])) / 2.0

        arms_out, marginals = {}, {}
        for lab, ids, pc in results:
            divs = {p["prompt"]: first_divergence(base_a_ids[p["prompt"]], ids[p["prompt"]])
                    for p in prompts}
            arms_out[lab] = {"percentiles": pc, "tok_s": 1000.0 / float(pc["p50"]),
                             "ids_match_base_a": all(v is None for v in divs.values()),
                             "first_divergence": divs}
            if lab.startswith("MARGINAL_"):
                key = lab.split("_", 1)[1].lower()
                marg = float(pc["p50"]) - midpoint
                b = BYTES_PER_TOKEN.get(key)
                marginals[key] = {
                    "marginal_ms_per_token": marg,
                    "fraction_of_base_token": marg / midpoint,
                    "bytes_per_token": b,
                    "roofline_ms_at_338_4_GB_s": (b / (DEVICE_BW_GB_S * 1e9) * 1e3) if b else None,
                    "achieved_GB_s": (b / (marg * 1e-3) / 1e9) if (b and marg > 0) else None,
                    "efficiency_vs_device_bw": ((b / (marg * 1e-3) / 1e9) / DEVICE_BW_GB_S)
                                               if (b and marg > 0) else None,
                }

        gates = {
            "G1_all_arms_ids_match_base_a": all(v["ids_match_base_a"] for v in arms_out.values()),
            "G2_drift_le_1ms": drift <= 1.0,
        }
        status = ("correctness_failed" if not gates["G1_all_arms_ids_match_base_a"]
                  else "measurement_unstable" if not gates["G2_drift_le_1ms"]
                  else "measured")

        payload.update({
            "arms": arms_out,
            "baseline_midpoint_ms": midpoint,
            "drift_ms": drift,
            "marginals": marginals,
            "gates": gates,
            "status": status,
            "completed_utc": utc_now(),
        })
        restore_sel()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload.get("status"),
                      "baseline_midpoint_ms": payload.get("baseline_midpoint_ms"),
                      "drift_ms": payload.get("drift_ms"),
                      "marginals": payload.get("marginals"),
                      "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
