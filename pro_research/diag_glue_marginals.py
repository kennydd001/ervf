"""Close the last unmeasured region of the token: lm_head, the norms and the adds.

Everything else is now attributed in-graph, bit-exact, one measurement each:

    Mamba GEMVs        4.187      MoE gather        3.849
    attention          2.479      MoE up_proj       2.253
    MoE shared_expert  1.810      MoE down_masked   1.372
    MoE scan/red/acc   1.119      ssm_step          1.095
    gated_norm         0.273      conv+dt           0.197
                                                   ------
                                      sum           18.63
                                      token         21.24
                                      unattributed  ~2.6

That remainder is the only part of `_step_body_graph` never probed, and it is
not one thing:

  * `lm_head` -- an NVFP4 GEMV over 198.2 MB/token, which at the measured
    ~260 GB/s kernel rate should be ~0.76 ms;
  * **53 `norm` launches and 52 `add_` launches per token**. `rmsnorm_bf16w` is
    launched as `(1,) x block` -- **a single block** for a 2688-element
    reduction, so 25 of 26 SMs sit idle and the cost is pure launch latency, not
    work: 10.75 KB at 345.9 GB/s is 0.03 us of actual traffic.
  * embed_gather and argmax_logits, once each.

If the norms and adds turn out to cost real time, that is 105 launches of
near-zero work per token and a different kind of target from anything else
measured today -- one that fusion, not bandwidth, addresses.

## Arms

Probes wrap `_step_body_graph` rather than a component, since these kernels live
outside the layer bodies. Two probes, not four, per the rule that small
candidates do not belong in a sweep:

  lm_head      one extra lm_head GEMV into scratch
  norms_adds   53 extra norms + 52 extra adds into scratch

`argmax` and `pos_increment` are deliberately NOT probed: argmax writes
`_tok_dev` and pos_increment mutates the position, so a second call would move
the sequence. Everything probed writes to scratch.

Gates: G1 every arm bit-exact against BASE_A; G2 drift <= 1.0 ms.
"""

from __future__ import annotations

import argparse
import json
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

OUT = REPO / "pro_research" / "diag_glue_marginals.json"
GROUPS = ("lm_head", "norms_adds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "kind": "diag_glue_marginals",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "why": "~2.6 ms of the 21.24 ms token is unattributed: lm_head plus 53 norm launches and 52 add_ launches. rmsnorm_bf16w runs as a SINGLE block for a 2688-element reduction, so its cost is launch latency rather than work -- a fusion target, not a bandwidth one.",
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

        orig_body = rt._step_body_graph
        s_norm = cp.zeros_like(rt.normed)
        s_h = cp.zeros_like(rt.h)
        s_logits = cp.zeros_like(rt.logits)
        n_layers = len(rt.pattern)

        def make_probe(group: str):
            def body(self):
                orig_body()
                k = self.k
                if group == "lm_head":
                    if self.lm_head_kind == "nvfp4":
                        self.fused.gemv_into(s_logits, self.lm_head_codes,
                                             self.lm_head_scales, self.normed,
                                             self.lm_head_g, self.vocab, self.hidden)
                    else:
                        k.mv_bf16(s_logits, self.lm_head, self.normed,
                                  self.vocab, self.hidden)
                elif group == "norms_adds":
                    for i2, _ch in enumerate(self.pattern):
                        d2 = self.layer[i2]
                        k.norm(s_norm, self.h, d2["norm"], self.hidden, self.eps)
                        k.add_(s_h, self.acc, self.hidden)
                    k.norm(s_norm, self.h, self.norm_f, self.hidden, self.eps)
            return body

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
                lambda g=g: setattr(rt, "_step_body_graph",
                                    types.MethodType(make_probe(g), rt)),
                lambda: setattr(rt, "_step_body_graph", orig_body)))
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
                m = float(pc["p50"]) - mid
                marg[lab] = {"marginal_ms_per_token": m,
                             "fraction_of_token": m / mid}
        if "norms_adds" in marg:
            launches = n_layers * 2 + 1
            marg["norms_adds"]["launches_per_token"] = launches
            marg["norms_adds"]["us_per_launch"] = (
                marg["norms_adds"]["marginal_ms_per_token"] * 1000.0 / launches)
        if "lm_head" in marg:
            m = marg["lm_head"]["marginal_ms_per_token"]
            marg["lm_head"]["bytes_per_token"] = 198_200_000
            marg["lm_head"]["achieved_GB_s"] = (198_200_000 / (m * 1e-3) / 1e9
                                                if m > 0 else None)
            marg["lm_head"]["floor_ms_at_260_GB_s"] = 198_200_000 / 260e9 * 1e3

        gates = {
            "G1_all_arms_ids_match_base_a": all(v["ids_match_base_a"] for v in arms_out.values()),
            "G2_drift_le_1ms": drift <= 1.0,
        }
        payload.update({
            "arms": arms_out, "baseline_midpoint_ms": mid, "drift_ms": drift,
            "marginals": marg,
            "sum_of_group_marginals_ms": sum(v["marginal_ms_per_token"] for v in marg.values()),
            "unattributed_before_ms": 2.6,
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
                      "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
