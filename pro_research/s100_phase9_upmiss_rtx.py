from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from common import REPO

sys.path.insert(0, str(REPO / "src"))

UP_CODE = 2_494_464
UP_SCALE = 311_808


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    payload = {
        "kind": "s100_phase9_rtx_upmiss",
        "status": "started",
        "rows": [],
        "direct_pointer_contract": (
            "CUDA-pinned UVA host pointer, same contract already consumed "
            "by cache_fetch"
        ),
    }

    try:
        import cupy as cp
        from moe_lab.lightningstream_nemotron.fused_nvfp4 import (
            FusedNVFP4,
        )
        from up_proj_batch_kernels import UpProjBatchKernels

        with np.load(args.sample) as sample:
            codes = sample["codes"].copy()
            scales = sample["scales"].copy()
            globals_up = sample["globals"].astype(np.float32).copy()
            x_host = sample["x"].astype(np.float32).copy()
            rows = int(sample["inter"])
            cols = int(sample["hidden"])

        expert_count = len(codes)
        x = cp.asarray(x_host)

        requested_codes = expert_count * UP_CODE
        requested_scales = expert_count * UP_SCALE
        pinned_codes_owner = cp.cuda.alloc_pinned_memory(requested_codes)
        pinned_scales_owner = cp.cuda.alloc_pinned_memory(
            requested_scales
        )
        # alloc_pinned_memory rounds the allocation up (observed: 14,271 KiB
        # -> 16 MiB), so the raw buffer is larger than requested; reshape
        # only the requested prefix. Slicing keeps the pinned base pointer.
        host_codes = np.frombuffer(
            pinned_codes_owner, np.uint8
        )[:requested_codes].reshape(expert_count, UP_CODE)
        host_scales = np.frombuffer(
            pinned_scales_owner, np.uint8
        )[:requested_scales].reshape(expert_count, UP_SCALE)
        host_codes[:] = codes
        host_scales[:] = scales

        fused = FusedNVFP4()
        up = UpProjBatchKernels()

        globals_host = np.zeros((expert_count, 2), np.float32)
        globals_host[:, 1] = globals_up
        dev = fused.alloc_device_cache(
            expert_count,
            expert_count,
            expert_count,
            globals_host,
        )
        dev["ids"].set(np.arange(expert_count, dtype=np.int32))
        dev["slots"].set(np.arange(expert_count, dtype=np.int32))
        dev["need"].fill(0)

        cached_codes = cp.zeros(
            expert_count * UP_CODE, dtype=cp.uint8
        )
        cached_scales = cp.zeros(
            expert_count * UP_SCALE, dtype=cp.uint8
        )
        output_buffer = cp.zeros(
            expert_count * rows, dtype=cp.float32
        )

        def timed(function, repetitions=80):
            for _ in range(8):
                function()
                cp.cuda.Stream.null.synchronize()
            values = []
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                function()
                cp.cuda.Stream.null.synchronize()
                values.append(
                    (time.perf_counter_ns() - start) / 1e6
                )
            return {
                "median_ms": statistics.median(values),
                "p95_ms": float(np.percentile(values, 95)),
                "min_ms": min(values),
                "max_ms": max(values),
            }

        final_reference = None
        for n_experts in (1, 2, 3):
            if n_experts > expert_count:
                continue

            dev["need"].fill(0)
            dev["need"][:n_experts].fill(1)

            def fetch():
                fused.cache_fetch(
                    int(host_codes.ctypes.data),
                    int(host_scales.ctypes.data),
                    cached_codes,
                    cached_scales,
                    dev,
                    UP_CODE,
                    UP_SCALE,
                    n_experts,
                )

            def cached_up():
                up.run_batched(
                    output_buffer,
                    cached_codes,
                    cached_scales,
                    dev["slots"],
                    dev["ids"],
                    dev["globals"],
                    1,
                    fused.e2m1,
                    fused.e4m3,
                    x,
                    rows,
                    cols,
                    True,
                    UP_CODE,
                    UP_SCALE,
                    n_experts,
                )

            def staged():
                fetch()
                cached_up()

            fetch()
            cp.cuda.Stream.null.synchronize()
            cached_up()
            cp.cuda.Stream.null.synchronize()
            reference = cp.asnumpy(
                output_buffer[: n_experts * rows]
            ).copy()
            final_reference = reference

            # The runtime already lets cache_fetch dereference these same
            # CUDA-pinned host allocations. This arm feeds the same UVA
            # pointers directly into the identical batched ERVF kernel.
            def direct():
                up.run_batched(
                    output_buffer,
                    np.uint64(host_codes.ctypes.data),
                    np.uint64(host_scales.ctypes.data),
                    dev["slots"],
                    dev["ids"],
                    dev["globals"],
                    1,
                    fused.e2m1,
                    fused.e4m3,
                    x,
                    rows,
                    cols,
                    True,
                    UP_CODE,
                    UP_SCALE,
                    n_experts,
                )

            direct()
            cp.cuda.Stream.null.synchronize()
            direct_output = cp.asnumpy(
                output_buffer[: n_experts * rows]
            ).copy()
            bitexact = bool(
                np.array_equal(reference, direct_output)
            )
            max_abs = float(
                np.max(np.abs(reference - direct_output))
            )

            payload["rows"].append(
                {
                    "nexperts": n_experts,
                    "fetch_only": timed(fetch),
                    "warm_up_only": timed(cached_up),
                    "staged_fetch_plus_up": timed(staged),
                    "direct_host_up": timed(direct),
                    "direct_bitexact": bitexact,
                    "direct_max_abs": max_abs,
                }
            )

        if final_reference is None:
            raise RuntimeError("no N=1/2/3 miss geometry was measured")

        np.savez_compressed(
            output.with_suffix(".ref.npz"),
            ref=final_reference,
            rows=np.int32(rows),
            cols=np.int32(cols),
        )
        payload["status"] = "measured"
        payload["all_direct_bitexact"] = all(
            row["direct_bitexact"] for row in payload["rows"]
        )
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
