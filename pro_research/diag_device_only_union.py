"""First step of PATH_TO_100_TOKS.md's roadmap item 1 (device-only union-
routing kernel): verifies, in isolation, that the EXISTING production
cache_assign/cache_fetch kernels -- unmodified, no new CUDA code -- can
replace proto_multi_seq_moe_shared.py's host-side Python union computation
(cp.asnumpy + set() + dict) for the up_proj fetch step entirely.

Read directly from fused_nvfp4.py's kernel source (not assumed):
  cache_assign: for duplicate ids within ONE call, the second+ occurrence
    finds slot_of[e] already set (by the first occurrence, earlier in the
    same sequential single-threaded loop) and marks need[s]=0 while still
    recording the correct slots[s] -- i.e. it already deduplicates a RAW,
    non-unioned id list on its own.
  cache_fetch: `if (!need[s]) return;` skips the fetch for any position
    whose expert is already resident (including duplicates within this same
    call), and writes to `cache_c + slots[s]*code_bytes` -- so multiple
    positions safely sharing one physical slot is exactly what it already
    supports, no new indexing scheme needed.

This test: capture REAL per-sequence routing for N=2 sequences at one real
MoE layer, then compare the CURRENT host-side approach (cp.asnumpy + set +
dict, as proto_multi_seq_moe_shared.py currently does) against calling
cache_assign directly on the RAW concatenated N*top_k id list (no host
union computation at all) -- checking that the two produce equivalent
fetch-buffer contents and equivalent (slot, need) semantics per position.

Not a gated PRO experiment -- an isolated mechanism-verification diagnostic,
read-only, no runtime modification.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic

PROMPTS = [
    "The history of computing began when",
    "Write a correct Python function that computes the longest increasing subsequence length in O(n log n), then explain its invariant.\n",
]

UP_CODE = 2_494_464
UP_SCALE = 311_808
N = 2
TOP_K = 6


def main() -> int:
    require_gpu_free()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()), local_files_only=True,
                                        trust_remote_code=True, use_fast=True)

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]
    target_layer = moe_layers[10]
    bank = rt.bank[target_layer]
    fused = rt.fused
    n_experts = rt.n_experts

    # capture N real per-sequence route_topk outputs at target_layer (same
    # kernel _moe_dev uses, matching proto_multi_seq_moe_shared.py exactly).
    captured_ids = []
    d = rt.layer[target_layer]
    orig_route = rt._route_device

    def capture_route(i):
        packed = orig_route(i)
        if i == target_layer:
            ids_dev = cp.zeros(TOP_K, dtype=cp.int32)
            w_dev = cp.zeros(TOP_K, dtype=cp.float32)
            rt.k.mv_f32(rt.rlog, d["gate_w"], rt.normed, n_experts, rt.hidden)
            fused.route_topk(rt.rlog, d["gate_b"], ids_dev, w_dev, n_experts, TOP_K,
                             rt.scaling, bad_pick=rt._bad_pick)
            captured_ids.append(cp.asnumpy(ids_dev).copy())
        return packed

    rt._route_device = types.MethodType(lambda self, i: capture_route(i), rt)
    for prompt in PROMPTS[:N]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)
    rt._route_device = orig_route
    cp.cuda.Device(0).synchronize()

    captured_ids = captured_ids[-N:]
    seq_ids = [c.tolist() for c in captured_ids]

    # ---- CURRENT approach: host-side Python union (as proto_multi_seq_moe_shared.py does).
    union_experts = sorted(set(e for ids_s in seq_ids for e in ids_s))
    expert_to_slot_host = {e: idx for idx, e in enumerate(union_experts)}
    u = len(union_experts)
    batched_c_host = cp.zeros(u * UP_CODE, dtype=cp.uint8)
    batched_s_host = cp.zeros(u * UP_SCALE, dtype=cp.uint8)
    ids_dev_b = cp.asarray(union_experts, dtype=cp.int32)
    slots_dev_b = cp.arange(u, dtype=cp.int32)
    need_dev_b = cp.ones(u, dtype=cp.int32)
    fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                      batched_c_host, batched_s_host,
                      {"ids": ids_dev_b, "slots": slots_dev_b, "need": need_dev_b},
                      UP_CODE, UP_SCALE, u)
    cp.cuda.Device(0).synchronize()

    # ---- NEW approach: device-only, raw (non-deduplicated) N*TOP_K ids fed
    # directly to cache_assign -- no host union computation at all.
    all_ids_flat = [e for ids_s in seq_ids for e in ids_s]
    P = len(all_ids_flat)
    globals_host = bank["globals"]
    dev_union = fused.alloc_device_cache(n_experts, P, P, globals_host)
    # cache_fetch later reads dev["ids"] specifically (not whatever array was
    # passed to cache_assign as its `ids` argument) -- production's _moe_dev
    # writes routing output directly into dev["ids"] via route_topk for
    # exactly this reason. A first version of this script passed a SEPARATE
    # all_ids_dev array to cache_assign without populating dev_union["ids"],
    # so cache_fetch read stale zeros and fetched expert 0 for every
    # position -- 12/12 byte mismatches. Fixed by writing into dev_union["ids"]
    # directly, matching the production pattern.
    dev_union["ids"][:] = cp.asarray(all_ids_flat, dtype=cp.int32)
    fused.cache_assign(dev_union, dev_union["ids"], P, P)
    batched_c_dev = cp.zeros(P * UP_CODE, dtype=cp.uint8)
    batched_s_dev = cp.zeros(P * UP_SCALE, dtype=cp.uint8)
    fused.cache_fetch(bank["up_codes"].ctypes.data, bank["up_scales"].ctypes.data,
                      batched_c_dev, batched_s_dev, dev_union, UP_CODE, UP_SCALE, P)
    cp.cuda.Device(0).synchronize()

    slots_result = cp.asnumpy(dev_union["slots"])
    need_result = cp.asnumpy(dev_union["need"])

    # ---- verify: for every position pi in the flat pair order, the bytes
    # cache_fetch wrote via the device-only path at dev_union slot
    # slots_result[pi] must bit-exact match what the host-union path wrote
    # at expert_to_slot_host[all_ids_flat[pi]] for the SAME expert id.
    mismatches = 0
    checked = 0
    for pi, e in enumerate(all_ids_flat):
        dev_slot = int(slots_result[pi])
        host_slot = expert_to_slot_host[e]
        c_dev = batched_c_dev[dev_slot * UP_CODE:(dev_slot + 1) * UP_CODE]
        s_dev = batched_s_dev[dev_slot * UP_SCALE:(dev_slot + 1) * UP_SCALE]
        c_host = batched_c_host[host_slot * UP_CODE:(host_slot + 1) * UP_CODE]
        s_host = batched_s_host[host_slot * UP_SCALE:(host_slot + 1) * UP_SCALE]
        checked += 1
        if not (bool((c_dev == c_host).all()) and bool((s_dev == s_host).all())):
            mismatches += 1

    # also verify the dedup bookkeeping itself: need[] should be 1 exactly
    # once per unique expert (at its first occurrence in the flat list) and
    # 0 for every repeat -- and the number of 1s should equal len(union_experts).
    seen = set()
    expected_need = []
    for e in all_ids_flat:
        expected_need.append(0 if e in seen else 1)
        seen.add(e)
    need_matches_expected = (list(int(x) for x in need_result) == expected_need)
    total_needed = int(need_result.sum())

    correctness_pass = (mismatches == 0) and need_matches_expected and (total_needed == u)

    payload = {
        "kind": "diag_device_only_union",
        "created_utc": utc_now(),
        "note": "verifies cache_assign+cache_fetch (unmodified production kernels) can replace the host-side Python union computation (cp.asnumpy+set+dict) in proto_multi_seq_moe_shared.py's up_proj fetch step, entirely device-side -- first step of PATH_TO_100_TOKS.md roadmap item 1",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "fused_nvfp4.py",)),
        "target_layer": target_layer,
        "n_sequences": N,
        "top_k": TOP_K,
        "flat_pair_count": P,
        "union_expert_count_host": u,
        "total_needed_device": total_needed,
        "need_matches_expected_dedup_pattern": need_matches_expected,
        "pairs_checked": checked,
        "byte_mismatches": mismatches,
        "correctness_pass": correctness_pass,
    }
    out = REPO / "pro_research" / "diag_device_only_union.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0 if correctness_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
