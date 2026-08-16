"""S100-KVERIFY K1: prove exact Mamba rollback from one state snapshot + proj log.

Correctness diagnostic only.  It captures real target-layer in_proj outputs from
a causal greedy rollout, then reconstructs each accepted-prefix state using
only the production conv/dt/SSM transition kernels.
"""
from __future__ import annotations

import gc
import json
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, require_gpu_free, utc_now
from graph_e1f22 import _load_prompt_set, _new_runtime

RESULT_DIR = REPO / "pro_research" / "results" / "s100_kverify"
OUT = RESULT_DIR / "PRO_S100_KVERIFY_K1_MAMBA_ROLLBACK.json"
PREREG = REPO / "pro_research" / "S100_KVERIFY_PREREGISTRATION.md"
K = 4


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _mismatch_count(cp, a, b) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    # Device-side compare, one scalar sync per state comparison.
    return int(cp.count_nonzero(a.view(cp.uint32) != b.view(cp.uint32)).get())


def _replay_transition(rt, layer: int, proj) -> None:
    """Replay exactly the state-mutating subset of LightningRuntime._mamba."""
    d = rt.layer[layer]
    z_end = rt.d_inner
    xbc_end = rt.d_inner + rt.conv_dim
    xbc = proj[z_end:xbc_end]
    dtr = proj[xbc_end:]

    rt.k.conv_step(
        rt.convo, rt.conv[layer], xbc, d["conv_w"], d["conv_b"],
        rt.conv_dim, rt.conv_k,
    )
    x = rt.convo[:rt.d_inner]
    Bv = rt.convo[rt.d_inner:rt.d_inner + rt.n_groups * rt.n_state]
    Cv = rt.convo[rt.d_inner + rt.n_groups * rt.n_state:]
    rt.k.dt_activate(rt.dt, dtr, d["dt_bias"], rt.m_heads, 0.0, 3.4e38)
    rt.k.ssm_step(
        rt.y, rt.ssm[layer], x, Bv, Cv, rt.dt, d["A_log"], d["D"],
        rt.m_heads, rt.m_hdim, rt.n_state, rt.hpg,
    )


def main() -> int:
    payload: dict[str, Any] = {
        "kind": "pro_s100_kverify_k1_mamba_rollback",
        "status": "started",
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "k": K,
        "claim_boundary": "single-layer persistent-state rollback correctness only; no verifier throughput or MTP acceptance claim",
    }
    try:
        require_gpu_free()
        import cupy as cp

        payload["environment_start"] = environment_snapshot((
            Path(__file__),
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py",
        ))
        prompts, _expected, _n, capacity = _load_prompt_set("smoke")
        p = prompts[0]
        rt = _new_runtime(capacity)
        if not rt.mamba_layers:
            raise RuntimeError("target checkpoint has no Mamba layers")
        layer = int(rt.mamba_layers[len(rt.mamba_layers) // 2])

        # Warm a real causal state with the preregistered anchor prompt.
        rt.reset()
        cur = None
        for tok in p["prompt_ids"]:
            cur = int(rt.step(int(tok)))
        if cur is None:
            raise RuntimeError("anchor prompt is empty")
        cp.cuda.get_current_stream().synchronize()

        start_conv = rt.conv[layer].copy()
        start_ssm = rt.ssm[layer].copy()
        cp.cuda.get_current_stream().synchronize()

        proj_log = []
        ref_conv = [start_conv.copy()]
        ref_ssm = [start_ssm.copy()]
        input_tokens = []
        output_tokens = []
        orig_mamba = rt._mamba

        def hooked(self, i, out):
            orig_mamba(i, out)
            if int(i) == layer:
                # Same-stream copies are ordered before later layers can reuse
                # self.proj/scratch. No host readback is required here.
                proj_log.append(self.proj.copy())
                ref_conv.append(self.conv[i].copy())
                ref_ssm.append(self.ssm[i].copy())

        rt._mamba = types.MethodType(hooked, rt)
        try:
            for _ in range(K):
                input_tokens.append(int(cur))
                cur = int(rt.step(int(cur)))
                output_tokens.append(int(cur))
        finally:
            rt._mamba = orig_mamba
        cp.cuda.get_current_stream().synchronize()

        if len(proj_log) != K or len(ref_conv) != K + 1 or len(ref_ssm) != K + 1:
            raise RuntimeError(
                f"capture cardinality mismatch proj={len(proj_log)} conv={len(ref_conv)} ssm={len(ref_ssm)}"
            )

        prefix_results = []
        for j in range(K + 1):
            rt.conv[layer][...] = start_conv
            rt.ssm[layer][...] = start_ssm
            for t in range(j):
                _replay_transition(rt, layer, proj_log[t])
            cp.cuda.get_current_stream().synchronize()
            cm = _mismatch_count(cp, rt.conv[layer], ref_conv[j])
            sm = _mismatch_count(cp, rt.ssm[layer], ref_ssm[j])
            prefix_results.append({
                "accepted_prefix": j,
                "conv_mismatch_count": cm,
                "ssm_mismatch_count": sm,
                "bit_exact": cm == 0 and sm == 0,
            })

        # Frozen sabotage: omit transition 1 from a four-token reconstruction.
        rt.conv[layer][...] = start_conv
        rt.ssm[layer][...] = start_ssm
        for t in range(K):
            if t == 1:
                continue
            _replay_transition(rt, layer, proj_log[t])
        cp.cuda.get_current_stream().synchronize()
        sab_conv = _mismatch_count(cp, rt.conv[layer], ref_conv[K])
        sab_ssm = _mismatch_count(cp, rt.ssm[layer], ref_ssm[K])
        sabotage_diverged = sab_conv != 0 or sab_ssm != 0

        all_prefix_exact = all(x["bit_exact"] for x in prefix_results)
        status = "rollback_exact" if all_prefix_exact and sabotage_diverged else "correctness_failed"
        payload.update({
            "status": status,
            "config": {
                "capacity": int(capacity),
                "layer": layer,
                "layer_kind": rt.layer[layer]["kind"],
                "prompt_kind": p["kind"],
                "prompt_ids": [int(x) for x in p["prompt_ids"]],
                "mamba_in_proj_width": int(rt.proj.size),
                "conv_state_bytes": int(rt.conv[layer].nbytes),
                "ssm_state_bytes": int(rt.ssm[layer].nbytes),
                "one_layer_snapshot_bytes": int(rt.conv[layer].nbytes + rt.ssm[layer].nbytes),
                "stored_proj_bytes": int(sum(x.nbytes for x in proj_log)),
            },
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "prefix_results": prefix_results,
            "sabotage": {
                "kind": "omit_transition_index_1",
                "conv_mismatch_count": sab_conv,
                "ssm_mismatch_count": sab_ssm,
                "diverged": sabotage_diverged,
            },
            "gates": {
                "all_prefix_states_bit_exact": all_prefix_exact,
                "sabotage_diverged": sabotage_diverged,
            },
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        del rt, start_conv, start_ssm, proj_log, ref_conv, ref_ssm
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "prefix_results": payload.get("prefix_results"),
        "sabotage": payload.get("sabotage"),
        "gates": payload.get("gates"),
    }, indent=2))
    return 0 if payload.get("status") == "rollback_exact" else 2


if __name__ == "__main__":
    raise SystemExit(main())
