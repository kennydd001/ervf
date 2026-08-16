"""Which is wrong: the V18 stack build, or the HTTP layer around it?

serve_openai.py measured 24.7-24.9 tok/s on a clean GPU where
bench_current_toks.py (bare runtime + graph, no V18) gets 38.69. Slower than the
bare path is a fault, not noise, and it has to be attributed before the server
can be trusted for anything but chatting.

This strips the HTTP layer away entirely and times the three stacks through the
identical decode loop:

    bare   runtime + device cache + graph                       (what bench_current_toks builds)
    v6     + selective ERVF + batched MoE
    v18    + H-SCALE and B3 overlap (combined)                  (the 51.0 tok/s record path)

If v18 here is fast, the fault is in the server. If v18 here is slow too, the
fault is in build_v18_runtime() and the server is merely reporting it.

One stack per process on purpose: building several full runtimes in one process
exhausts pinned host memory (load_routed_bank's pinned allocation is not fully
returned between cupy pool frees), which diag_v6_component_breakdown already
had to work around the same way.

    for s in bare v6 v18; do .venv-nemotron/Scripts/python.exe \
        scripts/lightningstream_nemotron/bench_stacks.py --stack $s; done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "pro_research"))

MODEL = REPO / "models" / "nemotron_3_5_lightning_v35"
OUT = REPO / "pro_research" / "results" / "bench_stacks"


def build(stack: str, capacity: int):
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(str(MODEL), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(capacity)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True
    info = {"stack": stack, "capacity": capacity}

    if stack != "bare":
        from down_proj_batch_kernels import DownProjBatchKernels
        from ervf_dense import DenseERVF
        from layer_capacity import apply_nonuniform_capacity
        from moe_dev_batched import install_batched_moe_dev
        from selective_ervf_v3 import _install_selective
        from up_proj_batch_kernels import UpProjBatchKernels

        apply_nonuniform_capacity(rt)
        dense, down, up = DenseERVF(), DownProjBatchKernels(), UpProjBatchKernels()
        _install_selective(rt, dense)
        install_batched_moe_dev(rt, down, up)

        if stack == "v18":
            from moe_dev_combined import install_combined_moe_dev
            from moe_dev_scale_resident import planned_plane_bytes
            from scale_resident_kernels import ScaleResidentKernels

            # combined_v18.py gates on VRAM before allocating the scale planes,
            # after returning unused pool blocks. The server did neither, and a
            # 492 MiB allocation that does not fit is exactly the kind of thing
            # that degrades into pool thrash instead of a clean error.
            cp.get_default_memory_pool().free_all_blocks()
            planned = planned_plane_bytes(rt)
            free_b = int(cp.cuda.Device(0).mem_info[0])
            info["planned_plane_mib"] = planned / 2**20
            info["free_before_mib"] = free_b / 2**20
            info["vram_gate_fits"] = planned <= free_b
            install_combined_moe_dev(rt, down, up, ScaleResidentKernels())

    rt.setup_graph()
    info["graph_extra_vram_mib"] = rt.graph_extra_vram_bytes / 2**20
    return rt, info


def bench(rt, prompt_ids, warmup: int, tokens: int):
    rt.reset()
    for tid in prompt_ids:
        rt.step_graph(int(tid))
        rt._graph_stream.synchronize()
    for _ in range(warmup):
        rt.step_graph()
    rt.ring_harvest((rt._ring_i - 1) % rt._ring_size, 1)

    # per-token harvest, the same SYNC semantics the record uses
    t0 = time.perf_counter()
    for _ in range(tokens):
        rt.step_graph()
        rt.ring_harvest((rt._ring_i - 1) % rt._ring_size, 1)
    dt_sync = time.perf_counter() - t0

    # queued: harvest once at the end, what bench_current_toks.py does
    t0 = time.perf_counter()
    for _ in range(tokens):
        rt.step_graph()
    rt.ring_harvest((rt._ring_i - 1) % rt._ring_size, 1)
    dt_queued = time.perf_counter() - t0
    return dt_sync, dt_queued


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", choices=["bare", "v6", "v18"], required=True)
    ap.add_argument("--capacity", type=int, default=72)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=128)
    args = ap.parse_args()

    # nvidia-smi, not cp.cuda.Device().mem_info: mem_info is measured from
    # inside our own CUDA context, which by itself accounts for ~1.1 GiB here,
    # so total-minus-free reads "1107 MiB busy" on a completely idle GPU.
    import subprocess
    used = float(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15).stdout.split("\n")[0].strip())
    if used > 1024:
        print(json.dumps({"status": "gpu_busy_refusing", "used_mib": used}))
        return 2

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    prompt_ids = tok.encode("The history of computing began when", add_special_tokens=False)

    t0 = time.perf_counter()
    rt, info = build(args.stack, args.capacity)
    info["build_s"] = round(time.perf_counter() - t0, 1)

    dt_sync, dt_q = bench(rt, prompt_ids, args.warmup, args.tokens)
    info.update({
        "tokens": args.tokens,
        "sync_ms_per_token": 1000.0 * dt_sync / args.tokens,
        "sync_tok_s": args.tokens / dt_sync,
        "queued_ms_per_token": 1000.0 * dt_q / args.tokens,
        "queued_tok_s": args.tokens / dt_q,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.stack}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
