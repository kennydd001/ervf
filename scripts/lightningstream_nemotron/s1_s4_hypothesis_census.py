"""S1-S4: hypothesis census for the 50 tok/s question.

Preregistered in S1_S4_HYPOTHESIS_CENSUS_PREREGISTRATION_2026-08-14.md.
This runner produces STATISTICS ONLY -- no timing claims, no tok/s.

  S1  route predictability (temporal bigram recall@C, frozen predictor)
  S2  ReLU2 sparsity census (exact zeros, zero 16-/64-column blocks)
  S3  NVFP4 code / scale entropy, cross-expert conditional entropy
  S4  draft-weight (MTP/eagle/nextn) key scan

Raw artifacts are dumped so the independent verifier can recompute every
gated number without importing this runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN_TOKENS = 256
TRAIN_ROWS = 128                      # per prompt: generated rows [0,128) train
C_GRID = [6, 8, 12, 16, 24]
S3_EXPERT_SAMPLE = list(range(0, 128, 16))   # frozen in the preregistration
S3_PAIRS = [(0, 16), (32, 48), (64, 80), (96, 112)]

CODE_BYTES = 4_988_928
SCALE_BYTES = 623_616
HALF_CODE = CODE_BYTES // 2
HALF_SCALE = SCALE_BYTES // 2


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------- S1 analysis
def build_bigram_counts(route_rows):
    """route_rows: list over steps of list per layer of 6 ids (generated only).

    Returns counts[layer_ord] = dict c -> Counter(e). Train rows only.
    """
    from collections import Counter, defaultdict
    n_layers = len(route_rows[0])
    counts = [defaultdict(Counter) for _ in range(n_layers)]
    for t in range(len(route_rows) - 1):
        cur, nxt = route_rows[t], route_rows[t + 1]
        for l in range(n_layers):
            for c in cur[l]:
                for e in nxt[l]:
                    counts[l][c][e] += 1
    return counts


def predict_candidates(counts_l, cur_set, c_budget):
    score = {}
    for c in cur_set:
        for e, n in counts_l.get(c, {}).items():
            score[e] = score.get(e, 0) + n
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return [e for e, _ in ranked[:c_budget]]


def s1_analysis(routes_by_prompt, moe_layers):
    """routes_by_prompt: list per prompt of {layer_str: [[6 ids] per step]}.

    Uses only the last GEN_TOKENS rows of each prompt (the generated
    continuation). Train rows 0..127, test rows 128..255 per prompt.
    """
    n_layers = len(moe_layers)
    gen_rows = []
    for cap in routes_by_prompt:
        rows = []
        for s in range(len(cap[str(moe_layers[0])]) - GEN_TOKENS,
                       len(cap[str(moe_layers[0])])):
            rows.append([cap[str(L)][s] for L in moe_layers])
        gen_rows.append(rows)

    recalls = {C: [[] for _ in range(n_layers)] for C in C_GRID}
    temporal_overlap = [[] for _ in range(n_layers)]
    adjacent_overlap = [[] for _ in range(n_layers - 1)]

    for rows in gen_rows:
        train, test = rows[:TRAIN_ROWS], rows[TRAIN_ROWS:]
        counts = build_bigram_counts(train)
        for t in range(len(test) - 1):
            cur, nxt = test[t], test[t + 1]
            for l in range(n_layers):
                cset, nset = set(cur[l]), set(nxt[l])
                temporal_overlap[l].append(len(cset & nset) / 6.0)
                for C in C_GRID:
                    cand = set(predict_candidates(counts[l], cur[l], C))
                    recalls[C][l].append(len(cand & nset) / 6.0)
        for t in range(len(rows)):
            for l in range(n_layers - 1):
                a, b = set(rows[t][l]), set(rows[t][l + 1])
                adjacent_overlap[l].append(len(a & b) / 6.0)

    per_layer = {}
    for l in range(n_layers):
        per_layer[str(moe_layers[l])] = {
            f"recall@{C}": float(np.mean(recalls[C][l])) for C in C_GRID
        } | {"temporal_identity_overlap": float(np.mean(temporal_overlap[l]))}
    mean_recall = {f"recall@{C}": float(np.mean([per_layer[str(moe_layers[l])][f"recall@{C}"]
                                                 for l in range(n_layers)]))
                   for C in C_GRID}
    min_recall = {f"recall@{C}": float(np.min([per_layer[str(moe_layers[l])][f"recall@{C}"]
                                               for l in range(n_layers)]))
                  for C in C_GRID}
    gates = {
        "G-S1-PASS": bool(mean_recall["recall@12"] >= 0.90
                          and min_recall["recall@12"] >= 0.75),
        "G-S1-CLOSE": bool(mean_recall["recall@24"] < 0.80),
    }
    return {
        "per_layer": per_layer,
        "mean_over_layers": mean_recall,
        "min_over_layers": min_recall,
        "mean_temporal_identity_overlap": float(np.mean([v for l in temporal_overlap for v in l])),
        "mean_adjacent_layer_overlap": float(np.mean([v for l in adjacent_overlap for v in l])),
        "n_train_pairs": sum(len(r[:TRAIN_ROWS]) - 1 for r in gen_rows),
        "n_test_pairs": sum(len(r[TRAIN_ROWS:]) - 1 for r in gen_rows),
        "gates": gates,
    }


# --------------------------------------------------------------- S3 analysis
def s3_analysis(bank, moe_layers):
    e1, e2, e3 = {}, {}, {}
    for L in moe_layers:
        codes, scales = bank[L]["codes"], bank[L]["scales"]
        nib_counts = np.zeros(16, dtype=np.int64)
        scale_counts = np.zeros(256, dtype=np.int64)
        joint = np.zeros((16, 16), dtype=np.int64)
        for e in S3_EXPERT_SAMPLE:
            c0 = e * CODE_BYTES
            cb = codes[c0:c0 + CODE_BYTES]
            nib_counts += np.bincount(cb & 15, minlength=16)
            nib_counts += np.bincount(cb >> 4, minlength=16)
            s0 = e * SCALE_BYTES
            scale_counts += np.bincount(scales[s0:s0 + SCALE_BYTES], minlength=256)
        for a, b in S3_PAIRS:
            # up matrix of each expert = second half of its record
            ab = codes[a * CODE_BYTES + HALF_CODE:(a + 1) * CODE_BYTES]
            bb = codes[b * CODE_BYTES + HALF_CODE:(b + 1) * CODE_BYTES]
            joint += np.bincount((ab & 15) * 16 + (bb & 15), minlength=256).reshape(16, 16)
            joint += np.bincount((ab >> 4) * 16 + (bb >> 4), minlength=256).reshape(16, 16)
        e1[str(L)] = entropy_bits(nib_counts)
        e2[str(L)] = entropy_bits(scale_counts)
        p_ab = joint / joint.sum()
        p_a = p_ab.sum(axis=1, keepdims=True)
        p_b_given_a = np.where(p_a > 0, p_ab / np.where(p_a > 0, p_a, 1), 0)
        nz = p_ab > 0
        h_b_given_a = float(-(p_ab[nz] * np.log2(p_b_given_a[nz])).sum())
        e3[str(L)] = h_b_given_a
    gates = {
        "G-S3-LOSSLESS_open": bool(min(e1.values()) <= 3.5),
        "G-S3-DELTA_open": bool(float(np.mean(list(e3.values()))) <= 2.5),
    }
    return {"nibble_entropy_bits_per_layer": e1,
            "min_nibble_entropy": float(min(e1.values())),
            "mean_nibble_entropy": float(np.mean(list(e1.values()))),
            "scale_byte_entropy_bits_per_layer": e2,
            "cond_entropy_H_B_given_A_per_layer": e3,
            "mean_cond_entropy": float(np.mean(list(e3.values()))),
            "sample_experts": S3_EXPERT_SAMPLE, "pairs": S3_PAIRS,
            "gates": gates}


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        foreign = [l for l in o.stdout.strip().splitlines()
                   if l.strip() and int(l.split(",")[0]) != os.getpid()]
    except Exception:
        foreign = ["query failed"]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------ S4: draft-weight scan
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    keys = list(idx["weight_map"].keys())
    draft_hits = [k for k in keys if re.search(r"mtp|nextn|eagle|draft|spec", k, re.I)]
    print(f"S4: {len(keys)} keys scanned, draft-like hits: {len(draft_hits)}", flush=True)

    rt = LightningRuntime(MODEL_DIR, contexts_max=4096, verbose=False)
    rt.enable_cache(31)
    rt.load_routed_bank()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    # --------------------------------------- S2: device-side zero counting
    max_calls = 2 * (16 + GEN_TOKENS) * 138 + 1024
    z_buf = cp.zeros(max_calls, dtype=cp.int32)
    z16_buf = cp.zeros(max_calls, dtype=cp.int32)
    z64_buf = cp.zeros(max_calls, dtype=cp.int32)
    call_meta = []
    state = {"call": 0, "prompt": -1, "step": -1}

    orig_expert = rt.fused.expert

    def wrapped_expert(*args, **kwargs):
        orig_expert(*args, **kwargs)
        act = rt.act[: rt.moe_inter]
        i = state["call"]
        z = act == 0
        z_buf[i] = z.sum()
        z16_buf[i] = z.reshape(116, 16).all(axis=1).sum()
        z64_buf[i] = z.reshape(29, 64).all(axis=1).sum()
        call_meta.append((state["prompt"], state["step"]))
        state["call"] += 1

    rt.fused.expert = wrapped_expert

    # ------------------------------------------------ rollout (S1 + S2)
    routes_by_prompt = []
    for p_id, prompt in enumerate(PROMPTS):
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        rt.cache_stats = {"hits": 0, "misses": 0}
        cap: dict = {}
        cur = ids[0]
        total = len(ids) + GEN_TOKENS
        for s in range(total):
            state["prompt"], state["step"] = p_id, s
            nxt = rt.step(cur, capture_routes=cap)
            cur = nxt if s >= len(ids) - 1 else ids[s + 1]
        routes_by_prompt.append(cap)
        print(f"prompt {p_id}: {total} steps, cache {rt.cache_stats}", flush=True)

    cp.cuda.Device(0).synchronize()
    n_calls = state["call"]
    zeros = cp.asnumpy(z_buf[:n_calls]).astype(np.int64)
    z16 = cp.asnumpy(z16_buf[:n_calls]).astype(np.int64)
    z64 = cp.asnumpy(z64_buf[:n_calls]).astype(np.int64)

    # per-call layer ordinal: 138 calls per step, 6 per MoE layer, in pattern
    # order -- a structural property of LightningRuntime.step, asserted here.
    assert n_calls % 138 == 0, f"call count {n_calls} not a multiple of 138"
    layer_ord = np.tile(np.repeat(np.arange(23), 6), n_calls // 138)

    s2 = {
        "n_expert_calls": int(n_calls),
        "mean_zero_fraction": float(zeros.sum() / (n_calls * 1856)),
        "mean_zero16_block_fraction": float(z16.sum() / (n_calls * 116)),
        "mean_zero64_block_fraction": float(z64.sum() / (n_calls * 29)),
        "per_layer_mean_zero_fraction": {
            str(rt.moe_layers[l]): float(zeros[layer_ord == l].sum()
                                         / max(1, (layer_ord == l).sum() * 1856))
            for l in range(23)},
        "gates": {},  # filled after raw dump so gate == recomputed statistic
    }
    s2["gates"]["G-S2-PASS"] = bool(s2["mean_zero_fraction"] >= 0.45)
    s2["gates"]["G-S2-CLOSE"] = bool(s2["mean_zero_fraction"] < 0.30)
    print(f"S2: mean zero fraction {s2['mean_zero_fraction']:.4f} "
          f"(16-block {s2['mean_zero16_block_fraction']:.4f}, "
          f"64-block {s2['mean_zero64_block_fraction']:.4f})", flush=True)

    # ------------------------------------------------ S1
    s1 = s1_analysis(routes_by_prompt, rt.moe_layers)
    print(f"S1: mean recall@12 {s1['mean_over_layers']['recall@12']:.4f} "
          f"min {s1['min_over_layers']['recall@12']:.4f} "
          f"recall@24 {s1['mean_over_layers']['recall@24']:.4f}", flush=True)

    # ------------------------------------------------ S3 (pinned bank, RO)
    s3 = s3_analysis(rt.bank, rt.moe_layers)
    print(f"S3: min nibble entropy {s3['min_nibble_entropy']:.4f} bits, "
          f"mean H(B|A) {s3['mean_cond_entropy']:.4f} bits", flush=True)

    # ------------------------------------------------ dumps
    raw_routes = {
        "kind": "lightningstream_nemotron_s1_routes_raw",
        "prompts": PROMPTS, "gen_tokens": GEN_TOKENS, "train_rows": TRAIN_ROWS,
        "moe_layers": rt.moe_layers,
        "routes_by_prompt": [
            {layer: rows for layer, rows in cap.items()}
            for cap in routes_by_prompt],
    }
    (OUT_DIR / "s1_routes_raw.json").write_text(
        json.dumps(raw_routes) + "\n", encoding="utf-8")

    raw_s2 = {
        "kind": "lightningstream_nemotron_s2_sparsity_raw",
        "moe_intermediate": 1856, "n_expert_calls": int(n_calls),
        "call_meta": call_meta,
        "zeros": zeros.tolist(), "zero16_blocks": z16.tolist(),
        "zero64_blocks": z64.tolist(),
    }
    (OUT_DIR / "s2_sparsity_raw.json").write_text(
        json.dumps(raw_s2) + "\n", encoding="utf-8")

    result = {
        "kind": "lightningstream_nemotron_s1_s4_hypothesis_census",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "S1_S4_HYPOTHESIS_CENSUS",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_path(Path(__file__)),
        "preregistration": "reports/lightningstream_nemotron/S1_S4_HYPOTHESIS_CENSUS_PREREGISTRATION_2026-08-14.md",
        "s1_route_predictability": s1,
        "s2_relu2_sparsity": s2,
        "s3_code_entropy": s3,
        "s4_draft_scan": {"keys_scanned": len(keys), "draft_hits": draft_hits,
                          "gate": "closed_no_draft_weights" if not draft_hits else "open"},
        "timing_claims": False,
        "claim_boundary": (
            "Statistics and gate verdicts only: route predictability, ReLU2 "
            "sparsity, code entropy, draft-weight presence, on this checkpoint "
            "and this sample. NO tok/s, timing, quality, or build-success "
            "claims. A passing gate justifies a build phase, nothing more."),
    }
    (OUT_DIR / "s1_s4_hypothesis_census.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("written s1_s4_hypothesis_census.json (+ s1_routes_raw, s2_sparsity_raw)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
