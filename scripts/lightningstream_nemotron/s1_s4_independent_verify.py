"""S1-S4 independent verifier.

Recomputes every gated number of the hypothesis census from the dumped raw
artifacts and the safetensors shards, WITHOUT importing the census runner.
Shares only the frozen preregistration parameters (hard-coded here, checked
against the preregistration text by the input lock).

Checks:
  V1  raw artifact shapes and structural invariants
  V2  S1 recall@C, temporal/adjacent overlap, gates  (from s1_routes_raw.json)
  V3  S2 zero fractions and gates                    (from s2_sparsity_raw.json)
  V4  S3 entropies and gates                         (re-read from shards)
  V5  S4 draft-weight scan                           (re-scan of the index)
  V6  runner-reported numbers match recomputed numbers
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"

GEN_TOKENS = 256
TRAIN_ROWS = 128
C_GRID = [6, 8, 12, 16, 24]
S3_EXPERT_SAMPLE = list(range(0, 128, 16))
S3_PAIRS = [(0, 16), (32, 48), (64, 80), (96, 112)]

checks = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    census = json.loads((OUT_DIR / "s1_s4_hypothesis_census.json").read_text())
    routes_raw = json.loads((OUT_DIR / "s1_routes_raw.json").read_text())
    s2_raw = json.loads((OUT_DIR / "s2_sparsity_raw.json").read_text())

    # ------------------------------------------------------------- V1 shapes
    moe_layers = routes_raw["moe_layers"]
    n_layers = len(moe_layers)
    check("V1.1 23 MoE layers", n_layers == 23, f"n={n_layers}")
    ok_rows = all(
        len(cap[str(L)]) >= GEN_TOKENS for cap in routes_raw["routes_by_prompt"]
        for L in moe_layers)
    check("V1.2 route rows per layer >= 256", ok_rows)
    ok_sets = all(
        len(set(row)) == 6 for cap in routes_raw["routes_by_prompt"]
        for L in moe_layers for row in cap[str(L)])
    check("V1.3 every route row is 6 unique ids", ok_sets)
    n_calls = s2_raw["n_expert_calls"]
    check("V1.4 sparsity call count multiple of 138", n_calls % 138 == 0,
          f"n={n_calls}")
    check("V1.5 sparsity arrays consistent",
          len(s2_raw["zeros"]) == n_calls
          and len(s2_raw["zero16_blocks"]) == n_calls
          and len(s2_raw["zero64_blocks"]) == n_calls)

    # ------------------------------------------------------------- V2 S1
    gen_rows = []
    for cap in routes_raw["routes_by_prompt"]:
        total = len(cap[str(moe_layers[0])])
        rows = [[cap[str(L)][s] for L in moe_layers]
                for s in range(total - GEN_TOKENS, total)]
        gen_rows.append(rows)

    recalls = {C: [] for C in C_GRID}
    per_layer_r12 = []
    temporal, adjacent = [], []
    for rows in gen_rows:
        train, test = rows[:TRAIN_ROWS], rows[TRAIN_ROWS:]
        counts = [defaultdict(Counter) for _ in range(n_layers)]
        for t in range(len(train) - 1):
            for l in range(n_layers):
                for c in train[t][l]:
                    for e in train[t + 1][l]:
                        counts[l][c][e] += 1
        per_layer_r12.append([])
        for t in range(len(test) - 1):
            for l in range(n_layers):
                cset, nset = set(test[t][l]), set(test[t + 1][l])
                temporal.append(len(cset & nset) / 6.0)
                score = {}
                for c in test[t][l]:
                    for e, n in counts[l].get(c, {}).items():
                        score[e] = score.get(e, 0) + n
                ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
                for C in C_GRID:
                    cand = {e for e, _ in ranked[:C]}
                    r = len(cand & nset) / 6.0
                    recalls[C].append(r)
                    if C == 12:
                        per_layer_r12[-1].append((l, r))
        for t in range(len(rows)):
            for l in range(n_layers - 1):
                adjacent.append(len(set(rows[t][l]) & set(rows[t][l + 1])) / 6.0)

    mean_recall = {C: float(np.mean(recalls[C])) for C in C_GRID}
    # min over layers: group the pooled (layer, recall) rows by layer
    layer_vals = defaultdict(list)
    for prompt_rows in per_layer_r12:
        for l, r in prompt_rows:
            layer_vals[l].append(r)
    min_r12 = float(np.min([np.mean(v) for v in layer_vals.values()]))
    mean_r12 = mean_recall[12]

    runner_s1 = census["s1_route_predictability"]
    check("V2.1 recall@12 recomputed matches runner",
          close(mean_r12, runner_s1["mean_over_layers"]["recall@12"], 1e-6),
          f"verifier {mean_r12:.6f} runner {runner_s1['mean_over_layers']['recall@12']:.6f}")
    check("V2.2 min-layer recall@12 matches",
          close(min_r12, runner_s1["min_over_layers"]["recall@12"], 1e-6),
          f"verifier {min_r12:.6f}")
    check("V2.3 recall@24 matches",
          close(mean_recall[24], runner_s1["mean_over_layers"]["recall@24"], 1e-6),
          f"verifier {mean_recall[24]:.6f}")
    g_pass = mean_r12 >= 0.90 and min_r12 >= 0.75
    g_close = mean_recall[24] < 0.80
    check("V2.4 G-S1-PASS verdict matches runner",
          g_pass == runner_s1["gates"]["G-S1-PASS"], f"PASS={g_pass}")
    check("V2.5 G-S1-CLOSE verdict matches runner",
          g_close == runner_s1["gates"]["G-S1-CLOSE"], f"CLOSE={g_close}")

    # ------------------------------------------------------------- V3 S2
    zeros = np.asarray(s2_raw["zeros"], dtype=np.int64)
    z16 = np.asarray(s2_raw["zero16_blocks"], dtype=np.int64)
    z64 = np.asarray(s2_raw["zero64_blocks"], dtype=np.int64)
    mz = zeros.sum() / (n_calls * 1856)
    mz16 = z16.sum() / (n_calls * 116)
    mz64 = z64.sum() / (n_calls * 29)
    runner_s2 = census["s2_relu2_sparsity"]
    check("V3.1 mean zero fraction matches",
          close(mz, runner_s2["mean_zero_fraction"], 1e-12), f"verifier {mz:.6f}")
    check("V3.2 zero-16 block fraction matches",
          close(mz16, runner_s2["mean_zero16_block_fraction"], 1e-12),
          f"verifier {mz16:.6f}")
    check("V3.3 zero-64 block fraction matches",
          close(mz64, runner_s2["mean_zero64_block_fraction"], 1e-12),
          f"verifier {mz64:.6f}")
    check("V3.4 G-S2-PASS verdict matches",
          (mz >= 0.45) == runner_s2["gates"]["G-S2-PASS"], f"PASS={mz >= 0.45}")
    check("V3.5 G-S2-CLOSE verdict matches",
          (mz < 0.30) == runner_s2["gates"]["G-S2-CLOSE"], f"CLOSE={mz < 0.30}")

    # ------------------------------------------------------------- V4 S3
    idx = ShardIndex(MODEL_DIR)
    e1, e3 = {}, {}
    scale_ent = {}
    for L in moe_layers:
        nib = np.zeros(16, dtype=np.int64)
        sc = np.zeros(256, dtype=np.int64)
        joint = np.zeros((16, 16), dtype=np.int64)
        up_parts = {}
        for e in S3_EXPERT_SAMPLE:
            pre = f"backbone.layers.{L}.mixer.experts.{e}"
            for mat in ("down_proj", "up_proj"):
                cb = idx.read_raw(f"{pre}.{mat}.weight")
                nib += np.bincount(cb & 15, minlength=16)
                nib += np.bincount(cb >> 4, minlength=16)
                sc += np.bincount(idx.read_raw(f"{pre}.{mat}.weight_scale"),
                                  minlength=256)
                if mat == "up_proj":
                    up_parts[e] = cb
        for a, b in S3_PAIRS:
            ab, bb = up_parts[a], up_parts[b]
            joint += np.bincount((ab & 15).astype(np.int64) * 16 + (bb & 15),
                                 minlength=256).reshape(16, 16)
            joint += np.bincount((ab >> 4).astype(np.int64) * 16 + (bb >> 4),
                                 minlength=256).reshape(16, 16)
        e1[str(L)] = entropy_bits(nib)
        scale_ent[str(L)] = entropy_bits(sc)
        p = joint / joint.sum()
        pa = p.sum(axis=1, keepdims=True)
        pba = np.where(pa > 0, p / np.where(pa > 0, pa, 1), 0)
        nz = p > 0
        e3[str(L)] = float(-(p[nz] * np.log2(pba[nz])).sum())

    runner_s3 = census["s3_code_entropy"]
    min_e1 = float(min(e1.values()))
    mean_h = float(np.mean(list(e3.values())))
    max_e1_diff = max(abs(e1[k] - runner_s3["nibble_entropy_bits_per_layer"][k])
                      for k in e1)
    max_h_diff = max(abs(e3[k] - runner_s3["cond_entropy_H_B_given_A_per_layer"][k])
                     for k in e3)
    check("V4.1 per-layer nibble entropy matches runner", max_e1_diff < 1e-9,
          f"max diff {max_e1_diff:.2e}")
    check("V4.2 per-layer conditional entropy matches runner", max_h_diff < 1e-9,
          f"max diff {max_h_diff:.2e}")
    check("V4.3 G-S3-LOSSLESS verdict matches",
          (min_e1 <= 3.5) == runner_s3["gates"]["G-S3-LOSSLESS_open"],
          f"open={min_e1 <= 3.5} (min {min_e1:.4f} bits)")
    check("V4.4 G-S3-DELTA verdict matches",
          (mean_h <= 2.5) == runner_s3["gates"]["G-S3-DELTA_open"],
          f"open={mean_h <= 2.5} (mean {mean_h:.4f} bits)")

    # ------------------------------------------------------------- V5 S4
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    hits = [k for k in index["weight_map"]
            if re.search(r"mtp|nextn|eagle|draft|spec", k, re.I)]
    check("V5.1 draft scan matches runner",
          hits == census["s4_draft_scan"]["draft_hits"],
          f"{len(hits)} hits")

    # ------------------------------------------------------------- verdict
    n_pass = sum(1 for c in checks if c["pass"])
    out = {
        "kind": "lightningstream_nemotron_s1_s4_independent_verification",
        "verifier": "scripts/lightningstream_nemotron/s1_s4_independent_verify.py",
        "checks_pass": n_pass, "checks_total": len(checks),
        "all_pass": n_pass == len(checks),
        "recomputed": {
            "s1_mean_recall": {f"recall@{C}": mean_recall[C] for C in C_GRID},
            "s1_min_layer_recall12": min_r12,
            "s1_temporal_identity_overlap": float(np.mean(temporal)),
            "s1_adjacent_layer_overlap": float(np.mean(adjacent)),
            "s2_mean_zero_fraction": float(mz),
            "s2_zero16_block_fraction": float(mz16),
            "s2_zero64_block_fraction": float(mz64),
            "s3_min_nibble_entropy_bits": min_e1,
            "s3_mean_nibble_entropy_bits": float(np.mean(list(e1.values()))),
            "s3_mean_scale_byte_entropy_bits": float(np.mean(list(scale_ent.values()))),
            "s3_mean_cond_entropy_bits": mean_h,
            "s4_draft_hits": hits,
        },
        "checks": checks,
    }
    (OUT_DIR / "s1_s4_independent_verification.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n{n_pass}/{len(checks)} checks passed")
    return 0 if n_pass == len(checks) else 3


if __name__ == "__main__":
    sys.exit(main())
