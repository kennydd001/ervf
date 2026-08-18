"""Diagnostic: where does the 12A block graph diverge from sequential decode?

Runs one prompt, 8 sequential steps vs 4x B=2 block cycles, and compares
every state component (ssm/conv per Mamba layer, KV per attention layer,
pos, logits) after each cycle to localise the first divergence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pro_research"))
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    import cupy as cp
    from diag_component_marginals_graph import _prefill, _reset_exact_state
    from s100_phase10a_runtime import build
    from s100_phase9_trace import load_prompts
    from s100_phase12a_block_verifier import setup_block_graph, step_block

    prompts = load_prompts(REPO)
    rt = build().rt
    B = 2
    setup_block_graph(rt, B)

    prompt_ids = prompts[0]["prompt_ids"]
    N = 8

    def snap(rt):
        out = {"pos": int(rt._pos_dev.get()[0])}
        for i in sorted(rt.ssm):
            out[f"ssm{i}"] = rt.ssm[i].get().copy()
            out[f"conv{i}"] = rt.conv[i].get().copy()
        used = out["pos"]
        for i in sorted(rt.kc):
            out[f"kc{i}"] = rt.kc[i].reshape(rt.n_kv, rt.max_ctx, rt.head_dim)[:, :used].get().copy()
            out[f"vc{i}"] = rt.vc[i].reshape(rt.n_kv, rt.max_ctx, rt.head_dim)[:, :used].get().copy()
        out["logits"] = rt.logits.get().copy()
        return out

    def diff(a, b):
        bad = []
        for k in a:
            if isinstance(a[k], (int, np.integer)):
                if a[k] != b[k]:
                    bad.append((k, int(a[k]), int(b[k])))
                continue
            if a[k].shape != b[k].shape:
                bad.append((k, "shape", a[k].shape, b[k].shape))
                continue
            neq = np.any(a[k] != b[k])
            if neq:
                af = a[k].reshape(-1, a[k].shape[-1])
                bf = b[k].reshape(-1, b[k].shape[-1])
                rows = np.nonzero(np.any(af != bf, axis=1))[0]
                bad.append((k, int(rows.size),
                            f"rows={rows[:6].tolist()}...{rows[-3:].tolist()}",
                            f"shape={a[k].shape}"))
        return bad

    # Baseline with per-step snapshots.
    _reset_exact_state(rt)
    _prefill(rt, prompt_ids)
    base_ids, base_snaps = [], []
    for _ in range(N):
        slot = int(rt._ring_i)
        rt.step_graph(None)
        base_ids.append(int(rt.ring_harvest(slot, 1)[0]))
        base_snaps.append(snap(rt))

    # Block run with per-cycle snapshots.
    _reset_exact_state(rt)
    _prefill(rt, prompt_ids)
    blk_ids = []
    for c in range(N // B):
        draft = base_ids[c * B : (c + 1) * B]
        got = step_block(rt, draft)
        blk_ids.extend(got)
        bs = snap(rt)
        # Compare against the baseline snapshot at the same token count.
        bb = base_snaps[(c + 1) * B - 1]
        bad = diff(bb, bs)
        print(f"cycle {c}: ids ok={got == draft} divergent:", flush=True)
        for row in bad[:8]:
            print("   ", row, flush=True)
        if not bad:
            print("    none", flush=True)
    print("ids equal overall:", blk_ids == base_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
