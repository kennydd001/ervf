"""Read-only diagnostic: split down_masked_into_indirect into its four
sub-kernels (panel_scan, gather_down_sparse_ind, down_masked_ind, reduce
partials) to find out whether the 9.57 ms/token measured in
diag_component_timing_v4.py is the PCIe host-mapped gather, the compute
kernel, or overhead -- before proposing a specific fix.

Not a gated PRO experiment. Wraps the four RawKernel function objects on
`rt.fused` with cp.cuda.Event timers; eager mode only (device_cache=True).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, require_model_dir, utc_now, write_json_atomic


def main() -> int:
    require_gpu_free()
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    import cupy as cp

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True

    fused = rt.fused
    names = ["panel_scan_k", "gather_ind_k", "down_masked_ind_k", "reduce_partials_k"]
    originals = {name: getattr(fused, name) for name in names}
    events: dict[str, list] = {name: [] for name in names}

    def make_wrapper(name, orig):
        def wrapped(*args, **kwargs):
            e0 = cp.cuda.Event()
            e1 = cp.cuda.Event()
            e0.record()
            r = orig(*args, **kwargs)
            e1.record()
            events[name].append((e0, e1))
            return r
        return wrapped

    for name in names:
        setattr(fused, name, make_wrapper(name, originals[name]))

    prompt = "The history of computing began when"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)
    ids = tok.encode(prompt, add_special_tokens=False)

    rt.reset()
    nxt = None
    for t in ids:
        nxt = int(rt.step(int(t)))
    cp.cuda.Device(0).synchronize()
    for name in names:
        events[name].clear()

    n = 96
    cur = nxt
    import time
    token_ms = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        cur = int(rt.step(cur))
        cp.cuda.Device(0).synchronize()
        token_ms.append((time.perf_counter_ns() - t0) / 1e6)

    for name in names:
        setattr(fused, name, originals[name])

    result = {}
    for name in names:
        ms = [cp.cuda.get_elapsed_time(a, b) for a, b in events[name]]
        result[name] = {
            "calls": len(ms),
            "ms_per_call": percentiles(ms),
            "ms_per_token_total": sum(ms) / n,
        }

    total_down_ms = sum(result[n_]["ms_per_token_total"] for n_ in names)
    payload = {
        "kind": "diag_down_subkernels_v4",
        "created_utc": utc_now(),
        "note": "read-only diagnostic, not a gated PRO experiment; eager device_cache path",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",)),
        "prompt": prompt,
        "generated_tokens": n,
        "token_ms": percentiles(token_ms),
        "subkernels": result,
        "down_pipeline_ms_per_token_total": total_down_ms,
        "down_pipeline_fraction_of_token": (total_down_ms * n / sum(token_ms)) if sum(token_ms) else None,
    }
    out = REPO / "pro_research" / "diag_down_subkernels_v4.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
