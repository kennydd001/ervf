"""Measure the replaceable RTX QFAST routed-down pipeline on one real causal snapshot.

The snapshot index is recreated deterministically from the same full prompt used by the
Arc exporter.  We time the actual H-SCALE+B3 gather -> masked-down -> reduce -> route
accumulate path using the already-populated live activations, route ids, masks, resident
scale planes, and pinned expert bank.  A second conservative arm serializes the scale-plane
miss fetch before the same down path; this is an upper bound because production overlaps
that fetch with routed-up staging.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

import numpy as np


def add_repo(repo: Path) -> None:
    sys.path.insert(0, str(repo / "pro_research"))
    sys.path.insert(0, str(repo / "src"))
    os.chdir(repo)


def closure_dict(fn):
    out = {}
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            out[name] = cell.cell_contents
        except ValueError:
            pass
    return out


def pcts(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(a.size),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "mean_ms": float(a.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-tokens", type=int, default=0)
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    payload = {
        "kind": "s100_phase8_overnight_rtx_live_downflow",
        "status": "started",
        "skip_tokens": int(args.skip_tokens),
        "reps": int(args.reps),
        "records": [],
    }
    try:
        add_repo(repo)
        import cupy as cp
        from s100_phase5_runtime import build_phase5_runtime
        from graph_e1f22 import _load_prompt_set
        from diag_component_marginals_graph import _reset_exact_state, _prefill
        from moe_dev_batched import DOWN_PANEL_BYTES

        prompts, _e, _n, capacity = _load_prompt_set("full")
        b = build_phase5_runtime(int(capacity), layer_k={}, alpha=0.0)
        rt = b.rt
        _reset_exact_state(rt)
        _prefill(rt, prompts[0]["prompt_ids"])
        for _ in range(int(args.skip_tokens) + 1):
            rt.step_graph(None)
            rt._graph_stream.synchronize()

        free = closure_dict(rt._moe_dev.__func__)
        state = free.get("state")
        sres = free.get("sres")
        batch = free.get("batch_kernels")
        nchunks = int(free.get("nchunks", rt.fused.nchunks))
        if not isinstance(state, dict):
            for v in free.values():
                if isinstance(v, dict) and v:
                    first = next(iter(v.values()))
                    if isinstance(first, dict) and "act" in first and "masks" in first:
                        state = v
                        break
        if not isinstance(state, dict):
            raise RuntimeError(f"phase5 batched state not found; closure={list(free)}")
        if sres is None or batch is None:
            raise RuntimeError(f"phase5 kernels not found; closure={list(free)}")

        topk = int(rt.top_k)
        inter = int(rt.moe_inter)
        hidden = int(rt.hidden)
        npanel = inter // 16
        gather_blocks = (inter * 32 + 255) // 256
        blocks_x = (hidden + 255) // 256
        grid_dm = ((hidden + 127) // 128, nchunks)
        payload["shape_contract"] = {
            "hidden": hidden,
            "moe_inter": inter,
            "top_k": topk,
            "layer_count": len(rt.moe_layers),
            "down_panel_bytes": int(DOWN_PANEL_BYTES),
            "nchunks": nchunks,
        }

        for li, layer in enumerate(rt.moe_layers, 1):
            layer = int(layer)
            bs = state[layer]
            dev = rt._dev_cache[layer]
            bank = rt.bank[layer]
            planes = sres.planes[layer]
            ids = cp.asnumpy(dev["ids"][:topk]).astype(np.int32)
            route_w = cp.asnumpy(dev["w"][:topk]).astype(np.float32)
            nzc = cp.asnumpy(bs["nzc"][:topk]).astype(np.int32)
            pcount = cp.asnumpy(bs["pcount"][:topk]).astype(np.int32)
            need = cp.asnumpy(dev["need"][:topk]).astype(np.int32)

            # Scratch intentionally separate from production output/state.
            mirrors = [cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8),
                       cp.zeros(DOWN_PANEL_BYTES, dtype=cp.uint8)]
            scratch = cp.zeros(hidden, dtype=cp.float32)
            gather_stream = cp.cuda.Stream(non_blocking=True)
            g_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(topk + 1)]
            m_done = [cp.cuda.Event(block=False, disable_timing=True) for _ in range(topk)]
            main_stream = cp.cuda.get_current_stream()

            def down_once(include_plane_fetch: bool) -> float:
                # Do not include zero-fill in the replacement timing: ADE returns a routed
                # contribution that production adds to the already-computed shared expert.
                scratch.fill(0)
                main_stream.synchronize()
                start = cp.cuda.Event()
                stop = cp.cuda.Event()
                start.record(main_stream)

                if include_plane_fetch:
                    # Conservative serialization. Production normally overlaps this with
                    # up-cache staging, so this arm is reported separately as an upper bound.
                    sres.fetch_planes(bank["down_base_ptr"], planes, dev, topk)
                    main_stream.synchronize()

                def issue(s: int):
                    sres.gather_cols(
                        gather_blocks, bank["down_base_ptr"], dev["ids"][s:],
                        mirrors[s & 1],
                        bs["nz"][s * inter:(s + 1) * inter], bs["nzc"][s:s + 1],
                        hidden,
                    )

                main_stream.record(g_done[topk])
                gather_stream.wait_event(g_done[topk])
                with gather_stream:
                    issue(0)
                    g_done[0].record(gather_stream)
                for s in range(topk):
                    if s + 1 < topk:
                        with gather_stream:
                            if s >= 1:
                                gather_stream.wait_event(m_done[s - 1])
                            issue(s + 1)
                            g_done[s + 1].record(gather_stream)
                    main_stream.wait_event(g_done[s])
                    sres.down_masked_sres(
                        grid_dm, mirrors[s & 1], planes, dev["slots"][s:],
                        dev["ids"][s:], dev["globals"],
                        bs["act"][s * inter:(s + 1) * inter],
                        bs["plist"][s * npanel:(s + 1) * npanel],
                        bs["masks"][s * npanel:(s + 1) * npanel],
                        bs["pcount"][s:s + 1], rt.fused.e2m1, rt.fused.e4m3,
                        bs["partials"][s * nchunks * hidden:(s + 1) * nchunks * hidden],
                        hidden, inter,
                    )
                    m_done[s].record(main_stream)
                batch.reduce_partials_batched(
                    (blocks_x, topk), (256,),
                    (bs["partials"], rt.contrib, np.int32(hidden), np.int32(nchunks)),
                )
                batch.run_accumulate_batched(scratch, rt.contrib, dev["w"], hidden, topk)
                stop.record(main_stream)
                stop.synchronize()
                return float(cp.cuda.get_elapsed_time(start, stop))

            # Warm both paths, then interleave to reduce temperature/order bias.
            for _ in range(8):
                down_once(False)
            vals_down = []
            vals_serial = []
            reps = max(10, int(args.reps))
            for r in range(reps):
                if r & 1:
                    vals_serial.append(down_once(True))
                    vals_down.append(down_once(False))
                else:
                    vals_down.append(down_once(False))
                    vals_serial.append(down_once(True))

            rec = {
                "layer": layer,
                "ids": ids.tolist(),
                "route_w": route_w.tolist(),
                "need": need.tolist(),
                "miss_count": int(need.sum()),
                "nzc": nzc.tolist(),
                "pcount": pcount.tolist(),
                "down_only": pcts(vals_down),
                "serial_plane_fetch_plus_down": pcts(vals_serial),
            }
            payload["records"].append(rec)
            print(
                f"RTX layer {layer:02d} {li:02d}/{len(rt.moe_layers)}: "
                f"down={rec['down_only']['median_ms']:.4f} ms "
                f"serial={rec['serial_plane_fetch_plus_down']['median_ms']:.4f} ms",
                flush=True,
            )
            del mirrors, scratch
            cp.get_default_memory_pool().free_all_blocks()

        payload["all_layer_sum_down_only_ms"] = float(sum(
            r["down_only"]["median_ms"] for r in payload["records"]
        ))
        payload["all_layer_sum_serial_ms"] = float(sum(
            r["serial_plane_fetch_plus_down"]["median_ms"] for r in payload["records"]
        ))
        payload["status"] = "measured"
        b.restore_combined()
        b.restore_selective()
        del rt, b
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload.get("status"),
        "skip_tokens": payload.get("skip_tokens"),
        "all_layer_sum_down_only_ms": payload.get("all_layer_sum_down_only_ms"),
        "all_layer_sum_serial_ms": payload.get("all_layer_sum_serial_ms"),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
