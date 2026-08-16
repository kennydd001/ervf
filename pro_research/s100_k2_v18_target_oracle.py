"""S100-K2 V18: correctness-first layer-major two-position target oracle.

This harness answers one narrow question: can the *current V18 target* verify
/consume two already-known correct consecutive tokens in a layer-major block
cheaply enough to leave a physical path to 100 tok/s with one native MTP depth?

The second input token is oracle-provided from the frozen target sequence.  No
MTP prediction quality, acceptance or speculative-delivery claim is made here.
See S100_K2_V18_TARGET_ORACLE_PREREGISTRATION.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import REPO, environment_snapshot, percentiles, require_gpu_free, utc_now
from diag_component_marginals_graph import _prefill, _recapture, _reset_exact_state
from down_proj_batch_kernels import DownProjBatchKernels
from ervf_dense import DenseERVF
from graph_e1f22 import _load_prompt_set, _new_runtime
from layer_capacity import apply_nonuniform_capacity
from moe_dev_batched import install_batched_moe_dev
from moe_dev_combined import install_combined_moe_dev
from moe_dev_scale_resident import planned_plane_bytes
from scale_resident_kernels import ScaleResidentKernels
from selective_ervf_v3 import _install_selective
from up_proj_batch_kernels import UpProjBatchKernels

RESULT_DIR = REPO / "pro_research" / "results" / "s100_k2_v18"
OUT = RESULT_DIR / "PRO_S100_K2_V18_TARGET_ORACLE.json"
PREREG = REPO / "pro_research" / "S100_K2_V18_TARGET_ORACLE_PREREGISTRATION.md"


def _write(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def _build_v18(capacity: int):
    """Build the adopted V18 target stack without modifying production files."""
    import cupy as cp

    rt = _new_runtime(capacity)
    dense = DenseERVF()
    down = DownProjBatchKernels()
    up = UpProjBatchKernels()

    # Mirror combined_v18.py exactly through the V6 base before H-SCALE+B3.
    rt.enable_cache(capacity)
    apply_nonuniform_capacity(rt)
    rt.device_cache = True
    rt.deterministic_accum = True
    restore_sel, _ = _install_selective(rt, dense)
    install_batched_moe_dev(rt, down, up)
    rt.setup_graph()

    # V18 candidate: resident scale planes + graph-captured B3 overlap.
    cp.get_default_memory_pool().free_all_blocks()
    planned = int(planned_plane_bytes(rt))
    free_before = int(cp.cuda.Device(0).mem_info[0])
    if planned > free_before:
        raise RuntimeError(
            f"V18 plane VRAM gate failed: planned={planned} free={free_before}"
        )
    sres = ScaleResidentKernels()
    install_combined_moe_dev(rt, down, up, sres)
    _recapture(rt)
    return rt, {"dense": dense, "down": down, "up": up, "sres": sres,
                "restore_sel": restore_sel, "planned_plane_bytes": planned,
                "free_before_planes": free_before}


def _stage_int32_pair(rt, dev, host_mem, host_np, a: int, b: int, stream) -> None:
    host_np[0] = np.int32(a)
    host_np[1] = np.int32(b)
    rt.cp.cuda.runtime.memcpyAsync(
        dev.data.ptr, host_mem.ptr, 8,
        rt.cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
    )


def _copy_int32_pair_to_host(rt, dev, host_mem, stream) -> None:
    rt.cp.cuda.runtime.memcpyAsync(
        host_mem.ptr, dev.data.ptr, 8,
        rt.cp.cuda.runtime.memcpyDeviceToHost, stream.ptr,
    )


class K2Graph:
    """Captured layer-major K=2 body over the already-installed V18 kernels."""

    def __init__(self, rt):
        import cupy as cp

        self.rt = rt
        self.cp = cp
        self.stream = rt._graph_stream
        self.orig_pos_dev = rt._pos_dev

        # Two independent logical hidden streams; all operator scratch can be
        # reused because position 0 then position 1 are ordered on the main
        # stream. Persistent conv/SSM/KV/cache intentionally remain shared.
        self.h0 = cp.zeros(rt.hidden, dtype=cp.float32)
        self.h1 = cp.zeros(rt.hidden, dtype=cp.float32)
        self.a0 = cp.zeros(rt.hidden, dtype=cp.float32)
        self.a1 = cp.zeros(rt.hidden, dtype=cp.float32)
        self.logits0 = cp.zeros(rt.vocab, dtype=cp.float32)
        self.logits1 = cp.zeros(rt.vocab, dtype=cp.float32)
        self.tok_in = cp.zeros(2, dtype=cp.int32)
        self.tok_out = cp.zeros(2, dtype=cp.int32)
        self.pos0 = cp.zeros(1, dtype=cp.int32)
        self.pos1 = cp.ones(1, dtype=cp.int32)

        self.stage_mem = cp.cuda.alloc_pinned_memory(8)
        self.stage_np = np.frombuffer(self.stage_mem, dtype=np.int32, count=2)
        self.out_mem = cp.cuda.alloc_pinned_memory(8)
        self.out_np = np.frombuffer(self.out_mem, dtype=np.int32, count=2)

        self.graph = None
        self._capture()

    def _embed(self, out, tok_slice):
        self.rt.k.embed_gather(out, self.rt._embed_tbl_ptr, tok_slice, self.rt.hidden)

    def _one_layer_position(self, i: int, ch: str, h, acc, pos_dev) -> None:
        rt, k, d = self.rt, self.rt.k, self.rt.layer[i]
        k.norm(rt.normed, h, d["norm"], rt.hidden, rt.eps)
        if ch == "M":
            rt._mamba(i, acc)
        elif ch == "*":
            saved = rt._pos_dev
            rt._pos_dev = pos_dev
            try:
                rt._attention(i, acc)
            finally:
                rt._pos_dev = saved
        else:
            rt._moe(i, acc)
        k.add_(h, acc, rt.hidden)

    def _head(self, h, logits, out_tok) -> None:
        rt, k = self.rt, self.rt.k
        k.norm(rt.normed, h, rt.norm_f, rt.hidden, rt.eps)
        if rt.lm_head_kind == "nvfp4":
            rt.fused.gemv_into(
                logits, rt.lm_head_codes, rt.lm_head_scales, rt.normed,
                rt.lm_head_g, rt.vocab, rt.hidden,
            )
        else:
            k.mv_bf16(logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden)
        k.argmax_logits(out_tok, logits, rt.vocab, rt._am_max, rt._am_idx)

    def _body(self) -> None:
        rt, k = self.rt, self.rt.k
        self._embed(self.h0, self.tok_in[0:1])
        self._embed(self.h1, self.tok_in[1:2])

        # Layer-major causal order. Per-layer persistent state/cache ordering is
        # exactly position0 -> position1, matching two sequential target steps.
        for i, ch in enumerate(rt.pattern):
            self._one_layer_position(i, ch, self.h0, self.a0, self.pos0)
            self._one_layer_position(i, ch, self.h1, self.a1, self.pos1)

        self._head(self.h0, self.logits0, self.tok_out[0:1])
        self._head(self.h1, self.logits1, self.tok_out[1:2])

        # Keep both private attention positions and the ordinary V18 graph's
        # position coherent so an exact ordinary continuation can follow K2.
        for _ in range(2):
            k.pos_increment(self.pos0)
            k.pos_increment(self.pos1)
            k.pos_increment(self.orig_pos_dev)

    def _capture(self) -> None:
        s = self.stream
        # Compile/warm every path before capture. State corruption is erased by
        # the caller's exact reset before any scientific arm.
        with s:
            self._body()
        s.synchronize()
        s.begin_capture()
        with s:
            self._body()
        self.graph = s.end_capture()
        s.synchronize()
        # Pointer identity used by the ordinary V18 graph must remain unchanged.
        self.rt._pos_dev = self.orig_pos_dev

    def set_positions(self, p: int) -> None:
        self.pos0.fill(np.int32(p))
        self.pos1.fill(np.int32(p + 1))
        self.orig_pos_dev.fill(np.int32(p))
        self.cp.cuda.Device(0).synchronize()

    def launch(self, t0: int, t1: int, timed: bool = True) -> tuple[list[int], float]:
        s = self.stream
        tstart = time.perf_counter_ns() if timed else 0
        _stage_int32_pair(self.rt, self.tok_in, self.stage_mem, self.stage_np, t0, t1, s)
        self.graph.launch(s)
        _copy_int32_pair_to_host(self.rt, self.tok_out, self.out_mem, s)
        s.synchronize()
        ms = (time.perf_counter_ns() - tstart) / 1e6 if timed else 0.0
        return [int(self.out_np[0]), int(self.out_np[1])], ms


def _prefill_clean(rt, prompt_ids: list[int]) -> int:
    _reset_exact_state(rt)
    return int(_prefill(rt, prompt_ids))


def _freeze_target(rt, prompt_ids: list[int], n_future: int) -> list[int]:
    """Return g0..gN where g0 is target output after the prompt."""
    g0 = _prefill_clean(rt, prompt_ids)
    out = [g0]
    for _ in range(n_future):
        slot = int(rt._ring_i)
        rt.step_graph(None)
        out.append(int(rt.ring_harvest(slot, 1)[0]))
    return out


def _run_seq_blocks(rt, prompt_ids: list[int], nblocks: int, timed: bool = True):
    first = _prefill_clean(rt, prompt_ids)
    ids = [first]
    block_ms: list[float] = []
    for _ in range(nblocks):
        start = int(rt._ring_i)
        t0 = time.perf_counter_ns() if timed else 0
        rt.step_graph(None)
        rt.step_graph(None)
        pair = [int(x) for x in rt.ring_harvest(start, 2)]
        if timed:
            block_ms.append((time.perf_counter_ns() - t0) / 1e6)
        ids.extend(pair)
    return ids, block_ms


def _run_k2_blocks(rt, k2: K2Graph, prompt_ids: list[int], frozen: list[int],
                   nblocks: int, timed: bool = True):
    first = _prefill_clean(rt, prompt_ids)
    if first != int(frozen[0]):
        raise RuntimeError(f"frozen/prefill mismatch {first} != {frozen[0]}")
    k2.set_positions(len(prompt_ids))
    ids = [first]
    block_ms: list[float] = []
    for b in range(nblocks):
        t0 = int(frozen[2 * b])
        t1 = int(frozen[2 * b + 1])
        pair, ms = k2.launch(t0, t1, timed=timed)
        ids.extend(pair)
        if timed:
            block_ms.append(ms)
    return ids, block_ms


def _snapshot(rt, used_positions: int, final_logits=None) -> dict[str, Any]:
    cp = rt.cp
    rt._graph_stream.synchronize()
    kv_elems = int(used_positions * rt.kv_dim)
    snap = {
        "pos": int(cp.asnumpy(rt._pos_dev)[0]),
        "conv": {str(i): cp.asnumpy(v) for i, v in rt.conv.items()},
        "ssm": {str(i): cp.asnumpy(v) for i, v in rt.ssm.items()},
        "kc": {str(i): cp.asnumpy(v[:kv_elems]) for i, v in rt.kc.items()},
        "vc": {str(i): cp.asnumpy(v[:kv_elems]) for i, v in rt.vc.items()},
        "logits": cp.asnumpy(final_logits if final_logits is not None else rt.logits),
    }
    return snap


def _compare_snap(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"pos_equal": a["pos"] == b["pos"], "groups": {}}
    total = 0
    for group in ("conv", "ssm", "kc", "vc"):
        gm = 0
        for key in a[group]:
            aa, bb = a[group][key], b[group][key]
            if aa.shape != bb.shape or aa.dtype != bb.dtype:
                gm += max(aa.size, bb.size, 1)
            else:
                gm += int(np.count_nonzero(aa.view(np.uint8) != bb.view(np.uint8)))
        out["groups"][group] = gm
        total += gm
    la, lb = a["logits"], b["logits"]
    lm = (max(la.size, lb.size, 1) if la.shape != lb.shape or la.dtype != lb.dtype
          else int(np.count_nonzero(la.view(np.uint8) != lb.view(np.uint8))))
    out["groups"]["logits"] = lm
    total += lm
    out["total_byte_mismatches"] = int(total)
    out["bitexact"] = bool(out["pos_equal"] and total == 0)
    return out


def _finite_candidate(rt, k2: K2Graph) -> bool:
    cp = rt.cp
    checks = [k2.logits0, k2.logits1]
    checks.extend(rt.conv.values())
    checks.extend(rt.ssm.values())
    return all(bool(cp.all(cp.isfinite(x)).get()) for x in checks)


def _continuation(rt, next_input: int, n: int) -> list[int]:
    rt._tok_dev.fill(np.int32(next_input))
    rt._graph_stream.synchronize()
    ids = []
    for _ in range(n):
        slot = int(rt._ring_i)
        rt.step_graph(None)
        ids.append(int(rt.ring_harvest(slot, 1)[0]))
    return ids


def _sabotage(rt, k2: K2Graph, prompt_ids: list[int], frozen: list[int]):
    _prefill_clean(rt, prompt_ids)
    k2.set_positions(len(prompt_ids))
    wrong = (int(frozen[1]) + 1) % int(rt.vocab)
    pair, _ = k2.launch(int(frozen[0]), wrong, timed=False)
    # The first output may remain equal because its hidden state does not depend
    # on the sabotaged second input; the second output and/or recurrent state
    # must carry the control signal.
    state_checksum = 0
    for v in list(rt.conv.values()) + list(rt.ssm.values()):
        state_checksum ^= int(rt.cp.sum(v.view(rt.cp.uint32), dtype=rt.cp.uint64).get())
    return {"wrong_second_input": wrong, "outputs": pair,
            "expected": [int(frozen[1]), int(frozen[2])],
            "token_diverged": pair != [int(frozen[1]), int(frozen[2])],
            "state_checksum": int(state_checksum)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {
        "kind": "s100_k2_v18_target_oracle",
        "status": "started",
        "mode": args.mode,
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "V18 layer-major K=2 target-verification oracle only; no MTP acceptance or user-visible S100 claim",
    }
    try:
        require_gpu_free()
        import cupy as cp

        prompts, _expected, _n, capacity = _load_prompt_set(args.mode)
        nblocks = 4 if args.mode == "smoke" else 64
        continuation_n = 32
        payload["config"] = {
            "blocks_per_prompt": nblocks,
            "positions_per_block": 2,
            "continuation_steps": continuation_n,
            "capacity": int(capacity),
            "prompt_count": len(prompts),
        }
        payload["environment_start"] = environment_snapshot((
            Path(__file__), PREREG,
            REPO / "pro_research" / "combined_v18.py",
            REPO / "pro_research" / "moe_dev_combined.py",
            REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",
        ))

        rt, keep = _build_v18(int(capacity))
        k2 = K2Graph(rt)
        payload["v18_stack"] = {
            "planned_plane_bytes": keep["planned_plane_bytes"],
            "free_before_planes": keep["free_before_planes"],
        }

        all_seq_a_ms: list[float] = []
        all_k2_ms: list[float] = []
        all_seq_b_ms: list[float] = []
        per_prompt = []
        all_cand_equal = True
        all_ref_equal = True
        all_states_exact = True
        all_det = True
        all_cont = True
        all_finite = True
        any_control_diverged = False

        for p in prompts:
            plen = len(p["prompt_ids"])
            # Freeze enough exact target tokens for timed blocks and the later
            # continuation check. This opening is untimed.
            frozen = _freeze_target(rt, p["prompt_ids"], 2 * nblocks + continuation_n + 2)

            seq_a_ids, seq_a_ms = _run_seq_blocks(rt, p["prompt_ids"], nblocks, timed=True)
            used = plen + 2 * nblocks
            seq_snap = _snapshot(rt, used)

            k2_ids, k2_ms = _run_k2_blocks(rt, k2, p["prompt_ids"], frozen, nblocks, timed=True)
            k2_snap = _snapshot(rt, used, final_logits=k2.logits1)
            state_cmp = _compare_snap(seq_snap, k2_snap)
            finite = _finite_candidate(rt, k2)
            # Exact continuation begins with g[2*nblocks], the second K2 output.
            cont = _continuation(rt, int(frozen[2 * nblocks]), continuation_n)
            cont_expected = [int(x) for x in frozen[2 * nblocks + 1:2 * nblocks + 1 + continuation_n]]
            cont_ok = cont == cont_expected

            # Deterministic repeat, untimed and short but same full K2 topology.
            det_ids, _ = _run_k2_blocks(rt, k2, p["prompt_ids"], frozen, nblocks, timed=False)
            det_ok = det_ids == k2_ids

            seq_b_ids, seq_b_ms = _run_seq_blocks(rt, p["prompt_ids"], nblocks, timed=True)
            ctl = _sabotage(rt, k2, p["prompt_ids"], frozen)
            control_diverged = bool(ctl["token_diverged"])
            if not control_diverged:
                # Compare against a correct one-block state checksum so the
                # sabotage can still prove causality even if argmax is stable.
                _prefill_clean(rt, p["prompt_ids"])
                k2.set_positions(plen)
                k2.launch(int(frozen[0]), int(frozen[1]), timed=False)
                good_checksum = 0
                for v in list(rt.conv.values()) + list(rt.ssm.values()):
                    good_checksum ^= int(cp.sum(v.view(cp.uint32), dtype=cp.uint64).get())
                ctl["good_state_checksum"] = int(good_checksum)
                ctl["state_diverged"] = int(good_checksum) != int(ctl["state_checksum"])
                control_diverged = bool(ctl["state_diverged"])
            else:
                ctl["state_diverged"] = None

            cand_equal = k2_ids == seq_a_ids
            ref_equal = seq_b_ids == seq_a_ids
            all_cand_equal &= cand_equal
            all_ref_equal &= ref_equal
            all_states_exact &= bool(state_cmp["bitexact"])
            all_det &= det_ok
            all_cont &= cont_ok
            all_finite &= finite
            any_control_diverged |= control_diverged

            all_seq_a_ms.extend(seq_a_ms)
            all_k2_ms.extend(k2_ms)
            all_seq_b_ms.extend(seq_b_ms)
            per_prompt.append({
                "prompt": p["prompt"], "kind": p["kind"], "prompt_tokens": plen,
                "candidate_token_parity": cand_equal,
                "reference_a_b_parity": ref_equal,
                "deterministic": det_ok,
                "continuation_32": cont_ok,
                "finite": finite,
                "state_compare": state_cmp,
                "control": ctl,
                "seq_a_ids": seq_a_ids,
                "k2_ids": k2_ids,
                "seq_b_ids": seq_b_ids,
            })

        a = percentiles(all_seq_a_ms)
        c = percentiles(all_k2_ms)
        b = percentiles(all_seq_b_ms)
        drift = abs(float(a["p50"]) - float(b["p50"]))
        mid = (float(a["p50"]) + float(b["p50"])) / 2.0
        k2_ms = float(c["p50"])
        speedup = mid / k2_ms if k2_ms else 0.0
        verified_tps = 2000.0 / k2_ms if k2_ms else 0.0

        gates = {
            "G1_reference_A_B_token_parity": bool(all_ref_equal),
            "G2_candidate_token_parity": bool(all_cand_equal),
            "G3_deterministic": bool(all_det),
            "G4_state_bitexact": bool(all_states_exact),
            "G5_continuation_32": bool(all_cont),
            "G6_control_diverges": bool(any_control_diverged),
            "G7_no_nan_inf": bool(all_finite),
            "D1_seq_A_B_drift_le_1ms": drift <= 1.0,
            "P1_K2_block_lt_19_285ms": k2_ms < 19.285,
            "P2_K2_block_lt_17_500ms": k2_ms < 17.500,
            "P3_effective_verified_ge_110tps": verified_tps >= 110.0,
            "P4_speedup_vs_seq_mid_ge_1_50x": speedup >= 1.50,
        }
        correctness = all(gates[k] for k in (
            "G1_reference_A_B_token_parity", "G2_candidate_token_parity",
            "G3_deterministic", "G4_state_bitexact", "G5_continuation_32",
            "G6_control_diverges", "G7_no_nan_inf"))
        stable = bool(gates["D1_seq_A_B_drift_le_1ms"])
        if not correctness:
            status = "correctness_failed"
        elif not stable:
            status = "measurement_unstable"
        elif gates["P1_K2_block_lt_19_285ms"]:
            status = "k2_v18_feasible_candidate"
        elif k2_ms >= mid:
            status = "layer_major_v18_negative"
        else:
            status = "k2_v18_below_s100_gate"

        payload.update({
            "arms": {"SEQ_A": a, "K2": c, "SEQ_B": b},
            "summary": {
                "seq_mid_p50_ms_per_2tok": mid,
                "k2_p50_ms_per_2tok": k2_ms,
                "k2_effective_verified_tok_s": verified_tps,
                "k2_speedup_vs_seq_mid": speedup,
                "seq_drift_ms": drift,
            },
            "per_prompt": per_prompt,
            "gates": gates,
            "status": status,
            "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })

        del k2, rt
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
            "completed_utc": utc_now(),
        })

    _write(payload)
    print(json.dumps({
        "status": payload.get("status"),
        "output": str(OUT),
        "summary": payload.get("summary"),
        "gates": payload.get("gates"),
        "error": (payload.get("error") or {}).get("message"),
    }, indent=2))
    return 0 if payload.get("status") not in {"technical_failure", "correctness_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
