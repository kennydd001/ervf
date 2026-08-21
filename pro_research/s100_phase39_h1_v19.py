"""Phase39 arm runner: exact context-1024 H1 V6 versus V19."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import traceback

import numpy as np

from common import (
    REPO,
    environment_snapshot,
    gpu_processes,
    utc_now,
    write_json_atomic,
)
from s100_phase21_common import identity_gate, load_trace, make_rt, prefill_to, release

RESULTS = REPO / "pro_research" / "results" / "s100_phase39"


def _require_gpu_compute_free() -> list[str]:
    """Ignore WDDM graphics-only rows whose CUDA memory is unavailable."""
    rows = gpu_processes()
    compute_rows = [row for row in rows if "[N/A]" not in row]
    if compute_rows:
        raise RuntimeError(
            "Another process currently owns measurable CUDA memory:\n  "
            + "\n  ".join(compute_rows)
        )
    return rows


def _percentiles(values: list[float]) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}

    def q(p: float) -> float:
        i = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
        return vals[i]

    return {
        "count": len(vals),
        "median_ms": statistics.median(vals),
        "mean_ms": sum(vals) / len(vals),
        "p10_ms": q(0.10),
        "p90_ms": q(0.90),
        "min_ms": vals[0],
        "max_ms": vals[-1],
    }


class GraphH1Verifier:
    """One exact position per replay using a device-side FP32-KV position."""

    def __init__(self, rt):
        import cupy as cp
        from s100_phase22_common import Phase22GraphKernels

        self.cp = cp
        self.rt = rt
        self.gk = Phase22GraphKernels(cp, int(rt.max_ctx))
        self.pos_dev = cp.zeros(1, cp.int32)
        self.tok_dev = cp.zeros(1, cp.int32)
        self.nparts = 256
        self.am_max = cp.zeros(self.nparts, cp.float32)
        self.am_idx = cp.zeros(self.nparts, cp.int32)

        ns = int(self.gk.max_splits)
        self.part_acc = cp.zeros(
            int(rt.n_heads) * ns * 4 * int(rt.head_dim), cp.float32
        )
        self.part_ml = cp.zeros(int(rt.n_heads) * ns * 4 * 2, cp.float32)

        nbytes = int(rt.embed_host.nbytes)
        self.embed_pm = cp.cuda.alloc_pinned_memory(nbytes)
        np.frombuffer(self.embed_pm, dtype=np.uint8, count=nbytes)[:] = (
            rt.embed_host.view(np.uint8)
        )
        self.embed_ptr = int(self.embed_pm.ptr)

        self.stage_pm = cp.cuda.alloc_pinned_memory(4)
        self.stage_np = np.frombuffer(self.stage_pm, dtype=np.int32, count=1)
        self.out_pm = cp.cuda.alloc_pinned_memory(4)
        self.out_np = np.frombuffer(self.out_pm, dtype=np.int32, count=1)
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.graph = None

    def _attention(self, i: int) -> None:
        import math

        rt, k, d = self.rt, self.rt.k, self.rt.layer[int(i)]
        k.mv_bf16(rt.qv, d["q_proj"], rt.normed, rt.n_heads * rt.head_dim, rt.hidden)
        k.mv_bf16(rt.kv_, d["k_proj"], rt.normed, rt.kv_dim, rt.hidden)
        k.mv_bf16(rt.vv, d["v_proj"], rt.normed, rt.kv_dim, rt.hidden)
        self.gk.kv_write(
            rt.kc[int(i)], rt.kv_, self.pos_dev, 0,
            rt.n_kv, rt.head_dim, rt.max_ctx,
        )
        self.gk.kv_write(
            rt.vc[int(i)], rt.vv, self.pos_dev, 0,
            rt.n_kv, rt.head_dim, rt.max_ctx,
        )
        self.gk.attention(
            rt.ctx, rt.qv, rt.kc[int(i)], rt.vc[int(i)], self.pos_dev, 0,
            rt.n_heads, rt.head_dim, rt.groups, rt.max_ctx,
            1.0 / math.sqrt(float(rt.head_dim)), self.part_acc, self.part_ml,
        )
        k.mv_bf16(
            rt.acc, d["o_proj"], rt.ctx, rt.hidden, rt.n_heads * rt.head_dim
        )

    def body(self) -> None:
        rt, k = self.rt, self.rt.k
        k.embed_gather(rt.h, self.embed_ptr, self.tok_dev, rt.hidden)
        for i, ch in enumerate(rt.pattern):
            d = rt.layer[i]
            k.norm(rt.normed, rt.h, d["norm"], rt.hidden, rt.eps)
            if ch == "M":
                rt._mamba(i, rt.acc)
            elif ch == "*":
                self._attention(i)
            else:
                rt._moe(i, rt.acc)
            k.add_(rt.h, rt.acc, rt.hidden)
        k.norm(rt.normed, rt.h, rt.norm_f, rt.hidden, rt.eps)
        if rt.lm_head_kind == "nvfp4":
            rt.fused.gemv_into(
                rt.logits, rt.lm_head_codes, rt.lm_head_scales,
                rt.normed, rt.lm_head_g, rt.vocab, rt.hidden,
            )
        else:
            k.mv_bf16(rt.logits, rt.lm_head, rt.normed, rt.vocab, rt.hidden)
        k.argmax_logits(self.tok_dev, rt.logits, rt.vocab, self.am_max, self.am_idx)
        k.pos_increment(self.pos_dev)

    def setup_graph(self) -> dict:
        cp, rt, stream = self.cp, self.rt, self.stream
        self.tok_dev.fill(0)
        self.pos_dev.fill(0)
        with stream:
            self.body()
        stream.synchronize()
        rt.copy_stream.synchronize()
        rt.reset()
        self.pos_dev.fill(0)
        cp.cuda.Device(0).synchronize()

        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        free_before = int(cp.cuda.Device(0).mem_info[0])
        stream.begin_capture()
        with stream:
            self.body()
        self.graph = stream.end_capture()
        stream.synchronize()
        rt.copy_stream.synchronize()
        rt.reset()
        self.pos_dev.fill(0)
        cp.cuda.Device(0).synchronize()
        free_after = int(cp.cuda.Device(0).mem_info[0])
        return {
            "free_before_capture_bytes": free_before,
            "free_after_capture_bytes": free_after,
            "graph_extra_vram_bytes": max(0, free_before - free_after),
            "cache_capacity": 72,
        }

    def prepare_after_prefill(self) -> None:
        self.pos_dev.fill(np.int32(self.rt.pos))
        self.cp.cuda.Device(0).synchronize()

    def launch(self, token: int) -> int:
        if self.graph is None:
            raise RuntimeError("call setup_graph first")
        rtapi, stream = self.cp.cuda.runtime, self.stream
        self.stage_np[0] = np.int32(token)
        rtapi.memcpyAsync(
            self.tok_dev.data.ptr, self.stage_pm.ptr, 4,
            rtapi.memcpyHostToDevice, stream.ptr,
        )
        self.graph.launch(stream)
        rtapi.memcpyAsync(
            self.out_pm.ptr, self.tok_dev.data.ptr, 4,
            rtapi.memcpyDeviceToHost, stream.ptr,
        )
        stream.synchronize()
        self.rt.pos += 1
        return int(self.out_np[0])


def _window(trace: list[int], context: int, count: int) -> tuple[list[int], list[int]]:
    feed = [int(v) for v in trace[context:context + count]]
    expected = [int(v) for v in trace[context + 1:context + count + 1]]
    if len(feed) != count or len(expected) != count:
        raise RuntimeError("canonical trace is too short")
    return feed, expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("base_a", "v19", "base_b"), required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    args = parser.parse_args()

    out = RESULTS / f"S100_PHASE39_{args.arm.upper()}_CTX{args.context}.json"
    payload = {
        "kind": "s100_phase39_h1_v19",
        "status": "started",
        "arm": args.arm,
        "context": int(args.context),
        "tokens": int(args.tokens),
        "warmup": int(args.warmup),
        "started_utc": utc_now(),
        "preregistration": "pro_research/S100_PHASE39_H1_V19_PREREGISTRATION.md",
    }
    rt = None
    graph = None
    try:
        payload["gpu_process_rows_preflight"] = _require_gpu_compute_free()
        identity_gate()
        trace = [int(v) for v in load_trace()["tokens"]]
        feed, expected = _window(
            trace, int(args.context), int(args.warmup) + int(args.tokens)
        )

        runtime_arm = "v19_device_rows" if args.arm == "v19" else "v6_device_rows"
        # make_rt reserves `requested_context + 128` KV positions. The frozen
        # window is 16 warmup + 128 measured positions, so pass the end of the
        # window as the allocation request while still prefilling/measuring at
        # the canonical context-1024 position.
        allocation_context = int(args.context) + int(args.warmup) + int(args.tokens)
        rt, keep = make_rt(allocation_context, "v6_device_rows")
        if args.arm == "v19":
            import cupy as cp
            from down_proj_batch_kernels import DownProjBatchKernels
            from moe_dev_combined import install_combined_moe_dev
            from moe_dev_scale_resident import planned_plane_bytes
            from scale_resident_kernels import ScaleResidentKernels
            from ssm_block_install import install_ssm_block
            from up_proj_batch_kernels import UpProjBatchKernels

            # The stock v18 builder checks physical free bytes while CuPy may
            # retain hundreds of MiB of *free* blocks. Restore V6, release only
            # those reusable pool blocks, then install the same V19 machinery.
            restore_v6 = keep.pop()
            restore_v6()
            cp.get_default_memory_pool().free_all_blocks()
            planned = int(planned_plane_bytes(rt))
            free_before_v19 = int(cp.cuda.Device(0).mem_info[0])
            if planned > free_before_v19:
                raise RuntimeError(
                    "V19 resident planes do not fit at cache72: "
                    f"planned={planned / 2**20:.1f}MiB "
                    f"free={free_before_v19 / 2**20:.1f}MiB"
                )
            down = next(obj for obj in keep if isinstance(obj, DownProjBatchKernels))
            up = next(obj for obj in keep if isinstance(obj, UpProjBatchKernels))
            sres = ScaleResidentKernels()
            keep.extend((
                sres,
                install_combined_moe_dev(rt, down, up, sres),
            ))
            keep.append(install_ssm_block(rt))
            payload["v19_preallocation"] = {
                "planned_plane_bytes": planned,
                "free_before_v19_bytes": free_before_v19,
                "cache_capacity": 72,
            }

        graph = GraphH1Verifier(rt)
        capture = graph.setup_graph()
        prefill_to(rt, trace, int(args.context))
        graph.prepare_after_prefill()

        generated: list[int] = []
        elapsed: list[float] = []
        for token in feed:
            t0 = time.perf_counter_ns()
            generated.append(graph.launch(token))
            elapsed.append((time.perf_counter_ns() - t0) / 1e6)

        first_divergence = next(
            (i for i, (got, want) in enumerate(zip(generated, expected)) if got != want),
            None,
        )
        measured = elapsed[int(args.warmup):]
        timing = _percentiles(measured)
        timing["tok_s"] = 1000.0 / float(timing["median_ms"])
        payload.update({
            "status": "measured" if first_divergence is None else "exactness_failed",
            "runtime_arm": runtime_arm,
            "allocation_context": allocation_context,
            "tokens_exact": first_divergence is None,
            "first_divergence": first_divergence,
            "generated_head": generated[:8],
            "expected_head": expected[:8],
            "capture": capture,
            "timing": timing,
            "raw_ms": measured,
            "keep_objects": len(keep),
            "environment": environment_snapshot((
                REPO / "pro_research" / "s100_phase39_h1_v19.py",
                REPO / "pro_research" / "moe_dev_combined.py",
                REPO / "pro_research" / "ssm_block_install.py",
            )),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        message = str(exc).lower()
        payload.update({
            "status": "infeasible_vram" if "out of memory" in message else "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    finally:
        graph = None
        gc.collect()
        if rt is not None:
            try:
                release(rt)
            except Exception:
                pass

    write_json_atomic(out, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "arm": args.arm,
        "tokens_exact": payload.get("tokens_exact"),
        "median_ms": (payload.get("timing") or {}).get("median_ms"),
        "tok_s": (payload.get("timing") or {}).get("tok_s"),
        "free_after_capture_mib": (
            (payload.get("capture") or {}).get("free_after_capture_bytes", 0) / 2**20
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(out),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
