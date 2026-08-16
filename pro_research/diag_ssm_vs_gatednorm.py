"""Which of ssm_step / gated_norm holds the 1.011 ms?

Today's fourth self-correction established, by subtraction, that Mamba's
in-graph marginal splits roughly as:

    in_proj + out_proj (isolated, measured)   3.448 ms   GEMV, ~259 GB/s
    conv_step/dt_activate/ssm_step/gated_norm 1.720 ms   never measured

The 1.720 ms is 33% of Mamba and 8% of the whole token -- larger than the entire
down_masked path -- and it fell outside every byte-based analysis because the
SSM state is small, so nothing bandwidth-shaped ever pointed at it. But it was
obtained by *subtracting* an isolated number from an in-loop number, which is
exactly the move that produced a phantom priority earlier today. It needs
measuring.

## Arms, deliberately grouped

An earlier 7-arm sweep showed ~0.39 ms of drift between intermediate arms, which
is the same size as the effects here. The rule filed from that: measure small
candidates in few arms. So three probe groups, not six:

    gemvs     in_proj + out_proj      -- should reproduce the 3.448 ms derivation
    conv_dt   conv_step + dt_activate
    ssm_gn    ssm_step + gated_norm

If `gemvs` comes back near 3.45 ms the derivation is confirmed and the split is
real; if it does not, the subtraction was wrong and so is the 1.72 ms.

## Statefulness

`conv_step` writes `conv[i]` and `ssm_step` writes `ssm[i]`, so their probes get
their own scratch recurrence state -- a naive second call advances the recurrence
twice and the token stream diverges, which already happened once today with the
whole-block `_mamba` probe. `dt_activate` and `gated_norm` are stateless but
their probes still write to scratch outputs, so no probe can perturb anything
downstream.

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
from diag_component_marginals_graph import _prefill, _recapture, _reset_exact_state, _run
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

OUT = REPO / "pro_research" / "diag_ssm_vs_gatednorm.json"
GROUPS = ("ssm_step", "gated_norm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "diag_ssm_vs_gatednorm",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "why": "Measured stage split put ssm_step + gated_norm at 1.011 ms/token = 4.9% of the token, the largest single unmeasured item left, while conv_step + dt_activate is negligible at 0.197. Neither is bandwidth-bound (2.1 MB of SSM state per layer), so the cause is compute or latency and the two must be separated before anything is built.",
        "pair_under_test_ms": 1.011,
    }

    try:
        require_gpu_free()
        import cupy as cp

        prompts, _e, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 24) if args.mode == "smoke" else max(n, 192)
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts),
                             "arms": ["BASE_A", *GROUPS, "BASE_B"]}
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

        orig_mamba = rt._mamba
        s_proj = cp.zeros_like(rt.proj)
        s_convo = cp.zeros_like(rt.convo)
        s_dt = cp.zeros_like(rt.dt)
        s_y = cp.zeros_like(rt.y)
        s_gn = cp.zeros_like(rt.gn)
        s_out = cp.zeros(rt.hidden, dtype=cp.float32)
        # conv_step and ssm_step mutate conv[i] / ssm[i]; give the probes their
        # own recurrence so the real state is untouched.
        s_conv = {j: cp.zeros_like(v) for j, v in rt.conv.items()}
        s_ssm = {j: cp.zeros_like(v) for j, v in rt.ssm.items()}

        def make_probe(group: str):
            def mamba(self, i, out):
                r = orig_mamba(i, out)
                k, d = self.k, self.layer[i]
                di, cd = self.d_inner, self.conv_dim
                ng, ns = self.n_groups, self.n_state
                if group == "ssm_step":
                    # writes ssm[i] -- scratch recurrence, or the token stream
                    # diverges (that already happened once today)
                    k.ssm_step(s_y, s_ssm[i], self.convo[:di],
                               self.convo[di:di + ng * ns], self.convo[di + ng * ns:],
                               self.dt, d["A_log"], d["D"],
                               self.m_heads, self.m_hdim, ns, self.hpg)
                elif group == "gated_norm":
                    k.gated_norm(s_gn, self.y, self.proj[:di], d["m_norm"],
                                 di, di // ng, self.eps)
                return r
            return mamba

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
        for g in GROUPS:
            results.append(arm(
                g,
                lambda g=g: setattr(rt, "_mamba", types.MethodType(make_probe(g), rt)),
                lambda: setattr(rt, "_mamba", orig_mamba)))
        results.append(arm("BASE_B", noop, noop))

        by = {lab: (ids, pc) for lab, ids, pc in results}
        base_ids, a = by["BASE_A"]
        _, b = by["BASE_B"]
        drift = abs(float(a["p50"]) - float(b["p50"]))
        mid = (float(a["p50"]) + float(b["p50"])) / 2.0

        arms_out, marg = {}, {}
        for lab, ids, pc in results:
            divs = {p["prompt"]: first_divergence(base_ids[p["prompt"]], ids[p["prompt"]])
                    for p in prompts}
            arms_out[lab] = {"percentiles": pc,
                             "ids_match_base_a": all(v is None for v in divs.values()),
                             "first_divergence": divs}
            if lab in GROUPS:
                marg[lab] = {"marginal_ms_per_token": float(pc["p50"]) - mid,
                             "fraction_of_token": (float(pc["p50"]) - mid) / mid}

        total = sum(v["marginal_ms_per_token"] for v in marg.values())
        gates = {
            "G1_all_arms_ids_match_base_a": all(v["ids_match_base_a"] for v in arms_out.values()),
            "G2_drift_le_1ms": drift <= 1.0,
        }
        payload.update({
            "arms": arms_out,
            "baseline_midpoint_ms": mid,
            "drift_ms": drift,
            "marginals": marg,
            "sum_of_group_marginals_ms": total,
            
            "pair_check": {
                "measured_sum_ms": total,
                "grouped_measurement_ms": 1.011,
                "abs_difference_ms": abs(total - 1.011),
                "note": "the two halves should re-sum to the grouped measurement; a large gap means the arms are not resolving at this magnitude",
            },
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
                      "pair_check": payload.get("pair_check"),
                      "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
