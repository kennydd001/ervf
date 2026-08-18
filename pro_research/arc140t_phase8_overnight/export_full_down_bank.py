"""Export one complete real routed-down expert bank plus one real activation snapshot.

This is generated locally for cache/residency pressure testing and is intentionally NOT
published to GitHub.  A complete layer bank is hundreds of MB, large enough to defeat any
misleading tiny-six-expert cache-hot benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-tokens", type=int, default=4)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.out_dir).resolve()
    meta = {"kind": "s100_p8_overnight_full_down_bank", "status": "started"}
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
        layers = [int(x) for x in rt.moe_layers]
        layer = layers[len(layers) // 2]
        _reset_exact_state(rt)
        _prefill(rt, prompts[0]["prompt_ids"])
        for _ in range(int(args.skip_tokens) + 1):
            rt.step_graph(None)
            rt._graph_stream.synchronize()

        free = closure_dict(rt._moe_dev.__func__)
        state = free.get("state")
        if not isinstance(state, dict):
            for v in free.values():
                if isinstance(v, dict) and v:
                    first = next(iter(v.values()))
                    if isinstance(first, dict) and "act" in first:
                        state = v
                        break
        if not isinstance(state, dict):
            raise RuntimeError("phase5 state not found")
        bs = state[layer]
        bank = rt.bank[layer]
        dev = rt._dev_cache[layer]
        topk = int(rt.top_k)
        inter = int(rt.moe_inter)
        hidden = int(rt.hidden)
        npanel = inter // 16
        n_experts = int(rt.n_experts)

        out.mkdir(parents=True, exist_ok=True)
        records = np.memmap(
            out / "records.npy.tmp", mode="w+", dtype=np.uint8,
            shape=(n_experts, int(DOWN_PANEL_BYTES))
        )
        # bank['down_pm'] is already the exact panel-major pinned host corpus.
        src = np.asarray(bank["down_pm"], dtype=np.uint8).reshape(n_experts, int(DOWN_PANEL_BYTES))
        records[:] = src
        records.flush()
        del records
        # Convert raw memmap to canonical .npy without creating a second in-RAM copy.
        raw = np.memmap(out / "records.npy.tmp", mode="r", dtype=np.uint8,
                        shape=(n_experts, int(DOWN_PANEL_BYTES)))
        dst_mm = np.lib.format.open_memmap(out / "records.npy", mode="w+", dtype=np.uint8,
                                           shape=raw.shape)
        dst_mm[:] = raw
        dst_mm.flush()
        del dst_mm, raw
        (out / "records.npy.tmp").unlink(missing_ok=True)

        gg = bank["globals"]
        gg = cp.asnumpy(gg) if hasattr(gg, "get") else np.asarray(gg)
        gg = np.asarray(gg, dtype=np.float32).reshape(n_experts, 2)
        np.save(out / "globals.npy", gg[:, 0])
        np.save(out / "act.npy", cp.asnumpy(bs["act"][:topk * inter]).reshape(topk, inter).astype(np.float32))
        np.save(out / "masks.npy", cp.asnumpy(bs["masks"][:topk * npanel]).reshape(topk, npanel).astype(np.uint32))
        np.save(out / "route_w.npy", cp.asnumpy(dev["w"][:topk]).astype(np.float32))
        np.save(out / "actual_ids.npy", cp.asnumpy(dev["ids"][:topk]).astype(np.int32))
        np.save(out / "e2m1.npy", cp.asnumpy(rt.fused.e2m1).astype(np.float32))
        np.save(out / "e4m3.npy", cp.asnumpy(rt.fused.e4m3).astype(np.float32))

        meta.update({
            "status": "measured",
            "layer": layer,
            "skip_tokens": int(args.skip_tokens),
            "hidden": hidden,
            "moe_inter": inter,
            "top_k": topk,
            "n_experts": n_experts,
            "npanel": npanel,
            "down_panel_bytes": int(DOWN_PANEL_BYTES),
            "records_bytes": int(n_experts * int(DOWN_PANEL_BYTES)),
            "actual_ids": cp.asnumpy(dev["ids"][:topk]).astype(np.int32).tolist(),
        })
        b.restore_combined()
        b.restore_selective()
    except Exception as exc:
        meta.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": traceback.format_exc()},
        })
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if meta.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
