"""Which of _moe_dev's sub-kernels holds the 5.81 ms of headroom?

`diag_component_marginals_v6` put MoE at **11.004 ms of a 23.141 ms token**
against a **5.19 ms floor** (677 MB of VRAM traffic at the honest 249 GB/s
kernel rate, plus 2.47 ms of down_proj PCIe). That 5.81 ms is the largest
single block of implementation inefficiency left, and single-kernel bandwidth is
not the explanation -- the dense GEMV gets 230-261 GB/s cold, 67-76% of the
device. MoE is not one GEMV; it is, per layer: a router GEMV, top-k, LRU assign,
miss fetch, two shared-expert GEMVs, six expert up-GEMVs, panel_scan, six
gathers, six masked GEMVs, a reduce and an accumulate.

Same marginal method as the component pass, one level deeper: run the real loop
and call exactly one sub-kernel one extra time, into a discarded buffer where
needed. Routing, cache, residual stream and produced tokens stay bit-identical,
and that is the gate.

## Idempotence, checked before relying on it

Most of these write a deterministic function of unchanged inputs to a fixed
buffer, so a second call rewrites the same bytes:

    shared_expert   gemv_into(_act_shared) then gemv_into(out) -- both overwrite
    up_proj         writes bs["act"]
    panel_scan      writes masks/plist/pcount/nz/nzc
    gather          writes the same columns and panels into the same mirror
    down_masked     writes bs["partials"]
    reduce          writes self.contrib

`accumulate` is the exception: it does `dst[i] = fmaf(src[i], w, dst[i])`, so a
second call would double-count the expert contributions. Its probe therefore
writes into a scratch destination. `cache_assign` is also excluded -- it
advances the LRU tick and would change later eviction order, the same class of
trap that made the naive `_mamba` probe diverge.

Every arm is still gated on bit-exact token ids rather than trusted on this
reasoning.
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
from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

OUT = REPO / "pro_research" / "diag_moe_subkernel_marginals.json"

PROBES = ("none", "shared_expert", "up_proj", "panel_scan", "gather",
          "down_masked", "reduce", "accumulate")

# VRAM bytes per token attributable to each step, from the safetensors headers.
BYTES = {
    "shared_expert": 290_000_000,
    "up_proj": 387_300_000,
    "gather": 64_000_000,          # PCIe, not VRAM
    "down_masked": 64_000_000,     # re-reads the gathered mirror from VRAM
}


def install_probed_moe_dev(rt, batch_kernels, up_kernels, probe: str):
    cp = rt.cp
    top_k, inter, hidden = rt.top_k, rt.moe_inter, rt.hidden
    npanel = inter // 16
    nchunks = rt.fused.nchunks
    orig = rt._moe_dev
    state: dict[int, dict] = {}
    scratch_out = cp.zeros(hidden, dtype=cp.float32)
    scratch_act = cp.zeros(rt.shared_inter, dtype=cp.float32)

    def _alloc() -> dict:
        return {
            "act": cp.zeros(top_k * inter, dtype=cp.float32),
            "masks": cp.zeros(top_k * npanel, dtype=cp.uint32),
            "plist": cp.zeros(top_k * npanel, dtype=cp.int32),
            "pcount": cp.zeros(top_k, dtype=cp.int32),
            "nz": cp.zeros(top_k * inter, dtype=cp.int32),
            "nzc": cp.zeros(top_k, dtype=cp.int32),
            "partials": cp.zeros(top_k * nchunks * hidden, dtype=cp.float32),
        }

    def probed(self, i, out):
        cp2, k, d, fused2 = self.cp, self.k, self.layer[i], self.fused
        bank, c = self.bank[i], self.cache[i]
        if not hasattr(self, "_dev_cache"):
            self._dev_cache = {}
        if i not in self._dev_cache:
            self._dev_cache[i] = fused2.alloc_device_cache(
                self.n_experts, c["cap"], self.top_k, bank["globals"])
        dev = self._dev_cache[i]
        if i not in state:
            state[i] = _alloc()
        bs = state[i]

        k.mv_f32(self.rlog, d["gate_w"], self.normed, self.n_experts, self.hidden)
        fused2.route_topk(self.rlog, d["gate_b"], dev["ids"], dev["w"],
                          self.n_experts, self.top_k, self.scaling,
                          bad_pick=self._bad_pick)
        fused2.cache_assign(dev, dev["ids"], c["cap"], self.top_k)
        self.evt[0].record()
        with self.copy_stream:
            self.copy_stream.wait_event(self.evt[0])
            fused2.cache_fetch(bank["up_codes"].ctypes.data,
                               bank["up_scales"].ctypes.data,
                               c["codes"], c["scales"], dev,
                               UP_CODE, UP_SCALE, self.top_k)
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused2.gemv_into(self._act_shared, d["sh_up_c"], d["sh_up_s"],
                         self.normed, d["sh_up_g"], self.shared_inter,
                         self.hidden, apply_relu2=True)
        fused2.gemv_into(out, d["sh_dn_c"], d["sh_dn_s"],
                         self._act_shared, d["sh_dn_g"],
                         self.hidden, self.shared_inter)
        if probe == "shared_expert":
            fused2.gemv_into(scratch_act, d["sh_up_c"], d["sh_up_s"],
                             self.normed, d["sh_up_g"], self.shared_inter,
                             self.hidden, apply_relu2=True)
            fused2.gemv_into(scratch_out, d["sh_dn_c"], d["sh_dn_s"],
                             self._act_shared, d["sh_dn_g"],
                             self.hidden, self.shared_inter)

        cp2.cuda.get_current_stream().wait_event(self.evt[1])

        def run_up(dst):
            up_kernels.run_batched(
                dst, c["codes"], c["scales"], dev["slots"], dev["ids"],
                dev["globals"], 1, fused2.e2m1, fused2.e4m3, self.normed,
                self.moe_inter, self.hidden, True, UP_CODE, UP_SCALE, self.top_k)

        run_up(bs["act"])
        if probe == "up_proj":
            run_up(bs["act"])

        def run_scan():
            batch_kernels.panel_scan_batched(
                (top_k,), (256,),
                (bs["act"], np.int32(inter), bs["masks"], bs["plist"],
                 bs["pcount"], bs["nz"], bs["nzc"]))

        run_scan()
        if probe == "panel_scan":
            run_scan()

        max_warps = inter + npanel
        gblocks = (max_warps * 32 + 255) // 256
        grid_dm = ((hidden + 127) // 128, nchunks)
        for s in range(self.top_k):
            plist_s = bs["plist"][s * npanel:(s + 1) * npanel]
            masks_s = bs["masks"][s * npanel:(s + 1) * npanel]
            pcount_s = bs["pcount"][s:s + 1]
            nz_s = bs["nz"][s * inter:(s + 1) * inter]
            nzc_s = bs["nzc"][s:s + 1]
            act_s = bs["act"][s * inter:(s + 1) * inter]
            partials_s = bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden]

            def do_gather():
                fused2.gather_ind_k(
                    (gblocks,), (256,),
                    (np.uint64(bank["down_base_ptr"]), dev["ids"][s:],
                     np.uint64(DOWN_PANEL_BYTES), self.mstate["mirror"],
                     plist_s, pcount_s, nz_s, nzc_s, np.int32(hidden)))

            def do_masked():
                fused2.down_masked_ind_k(
                    grid_dm, (128,),
                    (self.mstate["mirror"], dev["ids"][s:], dev["globals"],
                     act_s, plist_s, masks_s, pcount_s,
                     fused2.e2m1, fused2.e4m3, partials_s,
                     np.int32(hidden), np.int32(inter)))

            do_gather()
            if probe == "gather":
                do_gather()
            do_masked()
            if probe == "down_masked":
                do_masked()

        blocks_x = (hidden + 255) // 256

        def run_reduce(dst):
            batch_kernels.reduce_partials_batched(
                (blocks_x, top_k), (256,),
                (bs["partials"], dst, np.int32(hidden), np.int32(nchunks)))

        run_reduce(self.contrib)
        if probe == "reduce":
            run_reduce(self.contrib)

        batch_kernels.run_accumulate_batched(out, self.contrib, dev["w"],
                                             self.hidden, self.top_k)
        if probe == "accumulate":
            # NOT idempotent -- accumulates -- so it writes to scratch.
            batch_kernels.run_accumulate_batched(scratch_out, self.contrib,
                                                 dev["w"], self.hidden, self.top_k)
        return None, None

    rt._moe_dev = types.MethodType(probed, rt)
    return lambda: setattr(rt, "_moe_dev", orig)


def _reset(rt) -> None:
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


def _run(rt, prompt_ids, n):
    import cupy as cp

    _reset(rt)
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
        "kind": "diag_moe_subkernel_marginals",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "method": "S12 marginal, one level below diag_component_marginals_v6: the real loop with exactly one _moe_dev sub-kernel called one extra time. accumulate writes to scratch because it accumulates; cache_assign is excluded because it advances the LRU tick.",
        "opens_from": "MoE is 11.004 ms of a 23.141 ms token against a 5.19 ms floor -- 5.81 ms of headroom, the largest block left, and not explained by single-kernel bandwidth (dense GEMV gets 230-261 GB/s cold).",
    }

    try:
        require_gpu_free()
        prompts, _e, n, capacity = _load_prompt_set(args.mode)
        n = min(n, 24) if args.mode == "smoke" else max(n, 128)
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts)}
        payload["environment"] = environment_snapshot()

        rt = _new_runtime(capacity)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        rt.enable_cache(capacity)
        apply_nonuniform_capacity(rt)
        rt.device_cache = True
        rt.deterministic_accum = True
        restore_sel, _ = _install_selective(rt, dense)

        # preheat
        restore = install_probed_moe_dev(rt, down, up, "none")
        _reset(rt)
        nxt = None
        for tok in prompts[0]["prompt_ids"]:
            nxt = int(rt.step(int(tok)))
        for _ in range(32 if args.mode == "smoke" else 128):
            nxt = int(rt.step(nxt))
        restore()

        order = ["none"] + [p for p in PROBES if p != "none"] + ["none"]
        arms, base_ids = [], None
        for idx, probe in enumerate(order):
            restore = install_probed_moe_dev(rt, down, up, probe)
            ids_by, ms_all = {}, []
            for p in prompts:
                ids, ms = _run(rt, p["prompt_ids"], n)
                ids_by[p["prompt"]] = ids
                ms_all.extend(ms)
            restore()
            if base_ids is None:
                base_ids = ids_by
            divs = {p["prompt"]: first_divergence(base_ids[p["prompt"]], ids_by[p["prompt"]])
                    for p in prompts}
            arms.append({
                "label": f"{'BASE_A' if idx == 0 else 'BASE_B' if idx == len(order) - 1 else probe}",
                "probe": probe,
                "percentiles": percentiles(ms_all),
                "ids_match_base_a": all(v is None for v in divs.values()),
                "first_divergence": divs,
            })

        a, b = arms[0]["percentiles"], arms[-1]["percentiles"]
        drift = abs(float(a["p50"]) - float(b["p50"]))
        midpoint = (float(a["p50"]) + float(b["p50"])) / 2.0

        marginals = {}
        for arm in arms[1:-1]:
            marg = float(arm["percentiles"]["p50"]) - midpoint
            byt = BYTES.get(arm["probe"])
            marginals[arm["probe"]] = {
                "marginal_ms_per_token": marg,
                "fraction_of_token": marg / midpoint,
                "ids_match_base_a": arm["ids_match_base_a"],
                "bytes_per_token": byt,
                "achieved_GB_s": (byt / (marg * 1e-3) / 1e9) if (byt and marg > 0) else None,
            }
        total = sum(v["marginal_ms_per_token"] for v in marginals.values()
                    if v["marginal_ms_per_token"] > 0)

        gates = {
            "G1_all_arms_ids_match_base_a": all(x["ids_match_base_a"] for x in arms),
            "G2_drift_le_1ms": drift <= 1.0,
        }
        payload.update({
            "arms": arms,
            "baseline_midpoint_ms": midpoint,
            "drift_ms": drift,
            "marginals": marginals,
            "sum_of_positive_marginals_ms": total,
            "moe_total_marginal_ms_reference": 11.004,
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
                      "sum_of_positive_marginals_ms": payload.get("sum_of_positive_marginals_ms"),
                      "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
