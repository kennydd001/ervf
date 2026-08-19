from __future__ import annotations

import gc
import json
import statistics
import traceback
import types

import numpy as np

from common import REPO, require_model_dir, write_json_atomic, utc_now
from s100_phase17_kernels import Phase17Kernels, HS

OUT = (
    REPO / "pro_research" / "results" / "s100_phase17"
    / "S100_PHASE17_MAMBA_BLOCK.json"
)
HMAX = 8
CORR_SSM = 5e-5
CORR_CONV = 5e-6
CORR_FULL = 1e-4

def nrmse(a, b):
    aa = np.asarray(a, np.float64)
    bb = np.asarray(b, np.float64)
    return float(
        np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-30)
    )

def pct(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))

def timed(cp, fn, reset=None, reps=16):
    for _ in range(3):
        if reset is not None:
            reset()
        fn()
    cp.cuda.get_current_stream().synchronize()
    vals = []
    for _ in range(reps):
        if reset is not None:
            reset()
        cp.cuda.get_current_stream().synchronize()
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        fn()
        b.record()
        b.synchronize()
        vals.append(float(cp.cuda.get_elapsed_time(a, b)))
    return {
        "median_ms": statistics.median(vals),
        "p10_ms": pct(vals, 10),
        "p90_ms": pct(vals, 90),
        "raw_ms": vals,
    }

def capture_sequences(rt, layers, prompt_ids):
    import cupy as cp

    captures = {
        int(layer): {
            "normed": [],
            "out": [],
            "conv_post": [],
            "ssm_post": [],
            "conv0": None,
            "ssm0": None,
        }
        for layer in layers
    }
    enabled = {"value": False}
    original = rt._mamba

    def wrapped(self, i, out):
        i = int(i)
        take = (
            enabled["value"]
            and i in captures
            and len(captures[i]["normed"]) < HMAX
        )
        if take:
            rec = captures[i]
            if rec["conv0"] is None:
                rec["conv0"] = cp.asnumpy(self.conv[i]).astype(
                    np.float32, copy=True
                )
                rec["ssm0"] = cp.asnumpy(self.ssm[i]).astype(
                    np.float32, copy=True
                )
            rec["normed"].append(
                cp.asnumpy(self.normed).astype(np.float32, copy=True)
            )
        result = original(i, out)
        if take:
            rec = captures[i]
            rec["out"].append(
                cp.asnumpy(out).astype(np.float32, copy=True)
            )
            rec["conv_post"].append(
                cp.asnumpy(self.conv[i]).astype(np.float32, copy=True)
            )
            rec["ssm_post"].append(
                cp.asnumpy(self.ssm[i]).astype(np.float32, copy=True)
            )
        return result

    rt._mamba = types.MethodType(wrapped, rt)
    try:
        rt.reset()
        nxt = None
        for token in prompt_ids:
            nxt = int(rt.step(int(token)))
        if nxt is None:
            raise RuntimeError("empty capture prompt")
        enabled["value"] = True
        cur = int(nxt)
        for _ in range(HMAX):
            cur = int(rt.step(cur))
        enabled["value"] = False
        cp.cuda.get_current_stream().synchronize()
    finally:
        rt._mamba = original

    for layer, rec in captures.items():
        if len(rec["normed"]) != HMAX:
            raise RuntimeError(
                f"layer {layer}: captured {len(rec['normed'])}/{HMAX}"
            )
        rec["normed"] = np.stack(rec["normed"])
        rec["out"] = np.stack(rec["out"])
    return captures

def in_proj(rt, d, out, x):
    if d["in_k"] == "nvfp4":
        rt.fused.gemv_into(
            out, d["in_codes"], d["in_scales"], x, d["in_g"],
            int(rt.proj.size), int(rt.hidden),
        )
    elif d["in_k"] == "fp8_tensor":
        rt.k.mv_fp8_tensor(
            out, d["in_w8"], x, d["in_s"],
            int(rt.proj.size), int(rt.hidden),
        )
    else:
        rt.k.mv_bf16(
            out, d["in_w"], x, int(rt.proj.size), int(rt.hidden)
        )

def out_proj(rt, d, out, x):
    if d["out_k"] == "nvfp4":
        rt.fused.gemv_into(
            out, d["out_codes"], d["out_scales"], x, d["out_g"],
            int(rt.hidden), int(rt.d_inner),
        )
    elif d["out_k"] == "fp8_tensor":
        rt.k.mv_fp8_tensor(
            out, d["out_w8"], x, d["out_s"],
            int(rt.hidden), int(rt.d_inner),
        )
    else:
        rt.k.mv_bf16(
            out, d["out_w"], x, int(rt.hidden), int(rt.d_inner)
        )

def benchmark_layer(rt, kernels, layer, cap):
    import cupy as cp

    d = rt.layer[int(layer)]
    Hh = int(rt.m_heads)
    P = int(rt.m_hdim)
    N = int(rt.n_state)
    hpg = int(rt.hpg)
    G = Hh // hpg
    conv_dim = int(rt.conv_dim)
    d_inner = int(rt.d_inner)
    proj_size = int(rt.proj.size)
    group_size = d_inner // int(rt.n_groups)

    result = {
        "layer": int(layer),
        "in_kind": d["in_k"],
        "out_kind": d["out_k"],
        "dimensions": {
            "Hh": Hh, "P": P, "N": N, "hpg": hpg, "G": G,
            "conv_dim": conv_dim, "d_inner": d_inner,
            "proj_size": proj_size, "conv_k": int(rt.conv_k),
        },
        "per_H": {},
    }

    normed_all = cp.asarray(cap["normed"])
    exact_out_all = cap["out"]
    conv0_all = cp.asarray(cap["conv0"])
    ssm0_all = cp.asarray(cap["ssm0"])

    for T in HS:
        normed = normed_all[:T]
        exact_out = exact_out_all[:T]
        exact_conv_final = cap["conv_post"][T - 1]
        exact_ssm_final = cap["ssm_post"][T - 1]

        proj = cp.empty((T, proj_size), cp.float32)
        base_out = cp.empty((T, int(rt.hidden)), cp.float32)
        cand_out = cp.empty_like(base_out)

        # Compute exact projection oracle once for isolated core tests.
        for t in range(T):
            in_proj(rt, d, proj[t], normed[t])
        cp.cuda.get_current_stream().synchronize()

        # ---------------------------------------------------------- conv only
        base_conv_state = cp.empty_like(conv0_all)
        base_convo = cp.empty((T, conv_dim), cp.float32)
        cand_convo = cp.empty_like(base_convo)
        cand_conv_final = cp.empty_like(conv0_all)

        xbc_offset = d_inner

        def reset_conv():
            cp.copyto(base_conv_state, conv0_all)

        def baseline_conv():
            for t in range(T):
                rt.k.conv_step(
                    base_convo[t], base_conv_state,
                    proj[t, xbc_offset:xbc_offset + conv_dim],
                    d["conv_w"], d["conv_b"],
                    conv_dim, int(rt.conv_k),
                )

        def candidate_conv():
            kernels.block_conv(
                T, conv0_all, proj, d["conv_w"], d["conv_b"],
                cand_convo, cand_conv_final, conv_dim, int(rt.conv_k),
                x_stride=proj_size, x_offset=xbc_offset,
            )

        reset_conv()
        baseline_conv()
        candidate_conv()
        cp.cuda.get_current_stream().synchronize()
        conv_corr = {
            "output_nrmse": nrmse(
                cp.asnumpy(cand_convo), cp.asnumpy(base_convo)
            ),
            "final_state_nrmse": nrmse(
                cp.asnumpy(cand_conv_final), cp.asnumpy(base_conv_state)
            ),
        }
        conv_corr["pass"] = (
            conv_corr["output_nrmse"] <= CORR_CONV
            and conv_corr["final_state_nrmse"] <= CORR_CONV
        )
        conv_base_t = timed(cp, baseline_conv, reset_conv)
        conv_cand_t = timed(cp, candidate_conv)
        conv_speed = conv_base_t["median_ms"] / conv_cand_t["median_ms"]

        # --------------------------------------------------- prepare SSM oracle
        # Use the exact sequential conv and dt kernels to feed SSM-only test.
        reset_conv()
        baseline_conv()
        dt_seq = cp.empty((T, Hh), cp.float32)
        dtr_offset = d_inner + conv_dim
        for t in range(T):
            rt.k.dt_activate(
                dt_seq[t],
                proj[t, dtr_offset:dtr_offset + Hh],
                d["dt_bias"], Hh, 0.0, 3.4e38,
            )
        cp.cuda.get_current_stream().synchronize()

        # ------------------------------------------------------------ SSM only
        base_ssm_state = cp.empty_like(ssm0_all)
        base_y = cp.empty((T, d_inner), cp.float32)
        base_states = cp.empty((T, int(ssm0_all.size)), cp.float32)
        dx = cp.empty((T, d_inner), cp.float32)
        decay = cp.empty((T, Hh), cp.float32)
        cand_states_prefix = cp.empty_like(base_states)
        cand_states_serial = cp.empty_like(base_states)
        cand_y_prefix = cp.empty_like(base_y)
        cand_y_serial = cp.empty_like(base_y)

        B_offset = d_inner
        C_offset = d_inner + int(rt.n_groups) * N

        def reset_ssm():
            cp.copyto(base_ssm_state, ssm0_all)

        def baseline_ssm(store_states=False):
            for t in range(T):
                row = base_convo[t]
                rt.k.ssm_step(
                    base_y[t], base_ssm_state,
                    row[:d_inner],
                    row[B_offset:B_offset + G*N],
                    row[C_offset:C_offset + G*N],
                    dt_seq[t], d["A_log"], d["D"],
                    Hh, P, N, hpg,
                )
                if store_states:
                    cp.copyto(base_states[t], base_ssm_state)

        def candidate_ssm(kind, statebuf, ybuf):
            kernels.prepare(
                T, base_convo, dt_seq, d["A_log"], dx, decay,
                Hh, P, x_stride=conv_dim, x_offset=0,
            )
            kernels.scan(
                kind, T, ssm0_all, dx, base_convo, decay, statebuf,
                Hh, P, N, hpg,
                b_stride=conv_dim, b_offset=B_offset,
            )
            kernels.y(
                T, statebuf, base_convo, base_convo, d["D"], ybuf,
                Hh, P, N, hpg,
                c_stride=conv_dim, c_offset=C_offset,
                x_stride=conv_dim, x_offset=0,
            )

        reset_ssm()
        baseline_ssm(store_states=True)
        candidate_ssm("prefix", cand_states_prefix, cand_y_prefix)
        candidate_ssm("serial", cand_states_serial, cand_y_serial)
        cp.cuda.get_current_stream().synchronize()

        ssm_candidates = {}
        for kind, states, yy in (
            ("prefix", cand_states_prefix, cand_y_prefix),
            ("serial", cand_states_serial, cand_y_serial),
        ):
            corr = {
                "y_nrmse": nrmse(cp.asnumpy(yy), cp.asnumpy(base_y)),
                "final_state_nrmse": nrmse(
                    cp.asnumpy(states[T - 1]),
                    cp.asnumpy(base_states[T - 1]),
                ),
                "all_states_nrmse": nrmse(
                    cp.asnumpy(states), cp.asnumpy(base_states)
                ),
            }
            corr["pass"] = (
                corr["y_nrmse"] <= CORR_SSM
                and corr["final_state_nrmse"] <= CORR_SSM
            )
            tm = timed(
                cp,
                lambda kind=kind, states=states, yy=yy:
                    candidate_ssm(kind, states, yy),
            )
            ssm_candidates[kind] = {
                "correctness": corr,
                "timing": tm,
            }

        ssm_base_t = timed(
            cp, lambda: baseline_ssm(store_states=False), reset_ssm
        )
        for row in ssm_candidates.values():
            row["speedup"] = (
                ssm_base_t["median_ms"] / row["timing"]["median_ms"]
            )

        valid = [
            (k, v) for k, v in ssm_candidates.items()
            if v["correctness"]["pass"]
        ]
        selected_kind = (
            min(valid, key=lambda kv: kv[1]["timing"]["median_ms"])[0]
            if valid else None
        )

        # ------------------------------------------------------------- core all
        base_core_conv = cp.empty_like(conv0_all)
        base_core_ssm = cp.empty_like(ssm0_all)
        base_core_convo = cp.empty((T, conv_dim), cp.float32)
        base_core_dt = cp.empty((T, Hh), cp.float32)
        base_core_y = cp.empty((T, d_inner), cp.float32)
        base_gn = cp.empty((T, d_inner), cp.float32)

        cand_dt = cp.empty((T, Hh), cp.float32)
        cand_states = cp.empty((T, int(ssm0_all.size)), cp.float32)
        cand_y = cp.empty((T, d_inner), cp.float32)
        cand_gn = cp.empty((T, d_inner), cp.float32)
        cand_conv_final2 = cp.empty_like(conv0_all)
        cand_convo2 = cp.empty((T, conv_dim), cp.float32)
        dx2 = cp.empty((T, d_inner), cp.float32)
        decay2 = cp.empty((T, Hh), cp.float32)

        def reset_core():
            cp.copyto(base_core_conv, conv0_all)
            cp.copyto(base_core_ssm, ssm0_all)

        def baseline_core():
            for t in range(T):
                rt.k.conv_step(
                    base_core_convo[t], base_core_conv,
                    proj[t, xbc_offset:xbc_offset + conv_dim],
                    d["conv_w"], d["conv_b"], conv_dim, int(rt.conv_k),
                )
                rt.k.dt_activate(
                    base_core_dt[t],
                    proj[t, dtr_offset:dtr_offset + Hh],
                    d["dt_bias"], Hh, 0.0, 3.4e38,
                )
                row = base_core_convo[t]
                rt.k.ssm_step(
                    base_core_y[t], base_core_ssm,
                    row[:d_inner],
                    row[B_offset:B_offset + G*N],
                    row[C_offset:C_offset + G*N],
                    base_core_dt[t], d["A_log"], d["D"],
                    Hh, P, N, hpg,
                )
                rt.k.gated_norm(
                    base_gn[t], base_core_y[t], proj[t, :d_inner],
                    d["m_norm"], d_inner, group_size, float(rt.eps),
                )

        def candidate_core():
            if selected_kind is None:
                raise RuntimeError("no correctness-green SSM candidate")
            kernels.block_conv(
                T, conv0_all, proj, d["conv_w"], d["conv_b"],
                cand_convo2, cand_conv_final2, conv_dim, int(rt.conv_k),
                x_stride=proj_size, x_offset=xbc_offset,
            )
            kernels.block_dt(
                T, proj, d["dt_bias"], cand_dt, Hh,
                dtr_stride=proj_size, dtr_offset=dtr_offset,
            )
            kernels.prepare(
                T, cand_convo2, cand_dt, d["A_log"], dx2, decay2,
                Hh, P, x_stride=conv_dim, x_offset=0,
            )
            kernels.scan(
                selected_kind, T, ssm0_all, dx2, cand_convo2,
                decay2, cand_states, Hh, P, N, hpg,
                b_stride=conv_dim, b_offset=B_offset,
            )
            kernels.y(
                T, cand_states, cand_convo2, cand_convo2, d["D"],
                cand_y, Hh, P, N, hpg,
                c_stride=conv_dim, c_offset=C_offset,
                x_stride=conv_dim, x_offset=0,
            )
            kernels.gated(
                T, cand_y, proj, d["m_norm"], cand_gn,
                d_inner, group_size, float(rt.eps),
                z_stride=proj_size, z_offset=0,
            )

        reset_core()
        baseline_core()
        if selected_kind is not None:
            candidate_core()
        cp.cuda.get_current_stream().synchronize()

        if selected_kind is not None:
            core_corr = {
                "gn_nrmse": nrmse(
                    cp.asnumpy(cand_gn), cp.asnumpy(base_gn)
                ),
                "conv_final_nrmse": nrmse(
                    cp.asnumpy(cand_conv_final2),
                    cp.asnumpy(base_core_conv),
                ),
                "ssm_final_nrmse": nrmse(
                    cp.asnumpy(cand_states[T - 1]),
                    cp.asnumpy(base_core_ssm),
                ),
            }
            core_corr["pass"] = all(
                core_corr[x] <= CORR_SSM
                for x in ("gn_nrmse", "ssm_final_nrmse")
            ) and core_corr["conv_final_nrmse"] <= CORR_CONV
            core_base_t = timed(cp, baseline_core, reset_core)
            core_cand_t = timed(cp, candidate_core)
            core_speed = (
                core_base_t["median_ms"] / core_cand_t["median_ms"]
            )
        else:
            core_corr = {"pass": False}
            core_base_t = timed(cp, baseline_core, reset_core)
            core_cand_t = None
            core_speed = None

        # -------------------------------------------------------- full layer
        base_full_conv = cp.empty_like(conv0_all)
        base_full_ssm = cp.empty_like(ssm0_all)
        base_proj = cp.empty((T, proj_size), cp.float32)
        base_convo_full = cp.empty((T, conv_dim), cp.float32)
        base_dt_full = cp.empty((T, Hh), cp.float32)
        base_y_full = cp.empty((T, d_inner), cp.float32)
        base_gn_full = cp.empty((T, d_inner), cp.float32)

        cand_proj = cp.empty((T, proj_size), cp.float32)
        cand_out_full = cp.empty((T, int(rt.hidden)), cp.float32)
        cand_conv_full = cp.empty((T, conv_dim), cp.float32)
        cand_conv_final_full = cp.empty_like(conv0_all)
        cand_dt_full = cp.empty((T, Hh), cp.float32)
        cand_states_full = cp.empty((T, int(ssm0_all.size)), cp.float32)
        cand_y_full = cp.empty((T, d_inner), cp.float32)
        cand_gn_full = cp.empty((T, d_inner), cp.float32)
        cand_dx_full = cp.empty((T, d_inner), cp.float32)
        cand_decay_full = cp.empty((T, Hh), cp.float32)

        def reset_full():
            cp.copyto(base_full_conv, conv0_all)
            cp.copyto(base_full_ssm, ssm0_all)

        def baseline_full():
            for t in range(T):
                in_proj(rt, d, base_proj[t], normed[t])
                rt.k.conv_step(
                    base_convo_full[t], base_full_conv,
                    base_proj[t, xbc_offset:xbc_offset + conv_dim],
                    d["conv_w"], d["conv_b"], conv_dim, int(rt.conv_k),
                )
                rt.k.dt_activate(
                    base_dt_full[t],
                    base_proj[t, dtr_offset:dtr_offset + Hh],
                    d["dt_bias"], Hh, 0.0, 3.4e38,
                )
                row = base_convo_full[t]
                rt.k.ssm_step(
                    base_y_full[t], base_full_ssm,
                    row[:d_inner],
                    row[B_offset:B_offset + G*N],
                    row[C_offset:C_offset + G*N],
                    base_dt_full[t], d["A_log"], d["D"],
                    Hh, P, N, hpg,
                )
                rt.k.gated_norm(
                    base_gn_full[t], base_y_full[t],
                    base_proj[t, :d_inner], d["m_norm"],
                    d_inner, group_size, float(rt.eps),
                )
                out_proj(rt, d, base_out[t], base_gn_full[t])

        def candidate_full():
            if selected_kind is None:
                raise RuntimeError("no correctness-green SSM candidate")
            for t in range(T):
                in_proj(rt, d, cand_proj[t], normed[t])
            kernels.block_conv(
                T, conv0_all, cand_proj, d["conv_w"], d["conv_b"],
                cand_conv_full, cand_conv_final_full,
                conv_dim, int(rt.conv_k),
                x_stride=proj_size, x_offset=xbc_offset,
            )
            kernels.block_dt(
                T, cand_proj, d["dt_bias"], cand_dt_full, Hh,
                dtr_stride=proj_size, dtr_offset=dtr_offset,
            )
            kernels.prepare(
                T, cand_conv_full, cand_dt_full, d["A_log"],
                cand_dx_full, cand_decay_full, Hh, P,
                x_stride=conv_dim, x_offset=0,
            )
            kernels.scan(
                selected_kind, T, ssm0_all, cand_dx_full,
                cand_conv_full, cand_decay_full, cand_states_full,
                Hh, P, N, hpg,
                b_stride=conv_dim, b_offset=B_offset,
            )
            kernels.y(
                T, cand_states_full, cand_conv_full, cand_conv_full,
                d["D"], cand_y_full, Hh, P, N, hpg,
                c_stride=conv_dim, c_offset=C_offset,
                x_stride=conv_dim, x_offset=0,
            )
            kernels.gated(
                T, cand_y_full, cand_proj, d["m_norm"],
                cand_gn_full, d_inner, group_size, float(rt.eps),
                z_stride=proj_size, z_offset=0,
            )
            for t in range(T):
                out_proj(rt, d, cand_out_full[t], cand_gn_full[t])

        reset_full()
        baseline_full()
        if selected_kind is not None:
            candidate_full()
        cp.cuda.get_current_stream().synchronize()

        baseline_capture_corr = {
            "output_vs_captured_nrmse": nrmse(
                cp.asnumpy(base_out), exact_out
            ),
            "conv_final_vs_captured_nrmse": nrmse(
                cp.asnumpy(base_full_conv), exact_conv_final
            ),
            "ssm_final_vs_captured_nrmse": nrmse(
                cp.asnumpy(base_full_ssm), exact_ssm_final
            ),
        }

        if selected_kind is not None:
            full_corr = {
                "output_nrmse": nrmse(
                    cp.asnumpy(cand_out_full), cp.asnumpy(base_out)
                ),
                "conv_final_nrmse": nrmse(
                    cp.asnumpy(cand_conv_final_full),
                    cp.asnumpy(base_full_conv),
                ),
                "ssm_final_nrmse": nrmse(
                    cp.asnumpy(cand_states_full[T - 1]),
                    cp.asnumpy(base_full_ssm),
                ),
            }
            full_corr["pass"] = (
                full_corr["output_nrmse"] <= CORR_FULL
                and full_corr["conv_final_nrmse"] <= CORR_CONV
                and full_corr["ssm_final_nrmse"] <= CORR_SSM
            )
            full_base_t = timed(cp, baseline_full, reset_full)
            full_cand_t = timed(cp, candidate_full)
            full_speed = (
                full_base_t["median_ms"] / full_cand_t["median_ms"]
            )
        else:
            full_corr = {"pass": False}
            full_base_t = timed(cp, baseline_full, reset_full)
            full_cand_t = None
            full_speed = None

        result["per_H"][str(T)] = {
            "conv": {
                "correctness": conv_corr,
                "baseline": conv_base_t,
                "candidate": conv_cand_t,
                "speedup": conv_speed,
            },
            "ssm": {
                "baseline": ssm_base_t,
                "candidates": ssm_candidates,
                "selected": selected_kind,
            },
            "core": {
                "correctness": core_corr,
                "baseline": core_base_t,
                "candidate": core_cand_t,
                "speedup": core_speed,
            },
            "full_layer": {
                "baseline_reproduction": baseline_capture_corr,
                "correctness": full_corr,
                "baseline": full_base_t,
                "candidate": full_cand_t,
                "speedup": full_speed,
            },
        }

        # Release the large per-H state arrays before H=8 allocation grows.
        del (
            proj, base_out, cand_out, base_conv_state, base_convo,
            cand_convo, cand_conv_final, dt_seq, base_ssm_state,
            base_y, base_states, dx, decay, cand_states_prefix,
            cand_states_serial, cand_y_prefix, cand_y_serial,
            base_core_conv, base_core_ssm, base_core_convo,
            base_core_dt, base_core_y, base_gn, cand_dt,
            cand_states, cand_y, cand_gn, cand_conv_final2,
            cand_convo2, dx2, decay2, base_full_conv, base_full_ssm,
            base_proj, base_convo_full, base_dt_full, base_y_full,
            base_gn_full, cand_proj, cand_out_full, cand_conv_full,
            cand_conv_final_full, cand_dt_full, cand_states_full,
            cand_y_full, cand_gn_full, cand_dx_full, cand_decay_full,
        )
        cp.get_default_memory_pool().free_all_blocks()

    return result

def main():
    payload = {
        "kind": "s100_phase17_mamba_block",
        "status": "started",
        "started_utc": utc_now(),
        "horizons": list(HS),
        "claim_boundary": (
            "real Mamba-layer block microkernel on exact captured layer "
            "inputs; not end-to-end block decode"
        ),
    }
    try:
        # No Torch in this script: avoids the Torch/CuPy import-order issue
        # that affected prior native-BF16 experiments.
        import cupy as cp
        from transformers import AutoTokenizer
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

        kernels = Phase17Kernels()
        rt = LightningRuntime(
            require_model_dir(), contexts_max=512,
            embed_on_host=True, fp8_kv=True, verbose=False,
        )
        rt.load_routed_bank()
        rt.deterministic_accum = True

        layers = [int(x) for x in rt.mamba_layers]
        chosen = sorted({
            layers[0], layers[len(layers)//2], layers[-1]
        })

        tok = AutoTokenizer.from_pretrained(
            str(require_model_dir()),
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
        prompt_ids = tok.encode(
            "The history of computing and artificial intelligence",
            add_special_tokens=False,
        )

        captures = capture_sequences(rt, chosen, prompt_ids)

        # Mamba-only benchmarks no longer need the huge pinned routed bank.
        rt.bank = {}
        gc.collect()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        results = []
        for layer in chosen:
            row = benchmark_layer(rt, kernels, layer, captures[layer])
            results.append(row)
            print(
                f"Phase17 layer {layer}: "
                f"H4 full={row['per_H']['4']['full_layer']['speedup']} "
                f"core={row['per_H']['4']['core']['speedup']} "
                f"ssm={row['per_H']['4']['ssm']['selected']}",
                flush=True,
            )

        # Gates at H=4.
        h4 = [row["per_H"]["4"] for row in results]
        ssm_green = all(
            x["ssm"]["selected"] is not None
            and x["ssm"]["candidates"][x["ssm"]["selected"]][
                "correctness"
            ]["pass"]
            and x["ssm"]["candidates"][x["ssm"]["selected"]][
                "speedup"
            ] >= 1.50
            for x in h4
        )
        core_green = all(
            x["core"]["correctness"]["pass"]
            and x["core"]["speedup"] is not None
            and x["core"]["speedup"] >= 1.35
            for x in h4
        )
        full_green = all(
            x["full_layer"]["correctness"]["pass"]
            and x["full_layer"]["speedup"] is not None
            and x["full_layer"]["speedup"] >= 1.10
            for x in h4
        )

        payload.update({
            "status": "measured",
            "sampled_layers": chosen,
            "results": results,
            "SSM_SCAN_MICROKERNEL_OPEN": bool(ssm_green),
            "MAMBA_CORE_BLOCK_OPEN": bool(core_green),
            "MAMBA_LAYER_B4_CEILING_OPEN": bool(full_green),
            "PHASE18_FULL_BLOCK_VERIFIER_OPEN": bool(full_green),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "sampled_layers": payload.get("sampled_layers"),
        "SSM_SCAN_MICROKERNEL_OPEN": payload.get(
            "SSM_SCAN_MICROKERNEL_OPEN"
        ),
        "MAMBA_CORE_BLOCK_OPEN": payload.get("MAMBA_CORE_BLOCK_OPEN"),
        "MAMBA_LAYER_B4_CEILING_OPEN": payload.get(
            "MAMBA_LAYER_B4_CEILING_OPEN"
        ),
        "PHASE18_FULL_BLOCK_VERIFIER_OPEN": payload.get(
            "PHASE18_FULL_BLOCK_VERIFIER_OPEN"
        ),
        "layer_H4": [
            {
                "layer": r["layer"],
                "ssm_selected": r["per_H"]["4"]["ssm"]["selected"],
                "ssm_speedup": (
                    r["per_H"]["4"]["ssm"]["candidates"][
                        r["per_H"]["4"]["ssm"]["selected"]
                    ]["speedup"]
                    if r["per_H"]["4"]["ssm"]["selected"] else None
                ),
                "core_speedup": r["per_H"]["4"]["core"]["speedup"],
                "full_speedup": r["per_H"]["4"]["full_layer"]["speedup"],
                "full_nrmse": r["per_H"]["4"]["full_layer"][
                    "correctness"
                ].get("output_nrmse"),
            }
            for r in payload.get("results", [])
        ],
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
