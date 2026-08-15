"""C1-R1 + O1: a sharper sound bound, and OrbitANS measured instead of inherited.

Preregistered in W1R2_C1R1_O1_PREREGISTRATION_2026-08-15.md.

C1-R1  C1's Cauchy-Schwarz bound ran 9-31x wider than the preactivation because
       it adds 168 groups as if every residual pushed the same way as x. The core
       keeps the SIGN bit, and bit truncation can only raise the magnitude, so
       the direction of every residual term is known without reading the tail:

           dy_j <= sum_k dmax(core_jk) * max(s_jk * x_k, 0)

       Only the adverse half of the terms counts, each with its exact maximum
       magnitude. Still mechanical, still sound.

O1     Entropy of the real NVFP4 codes and FP8 scales in this checkpoint,
       marginal and under the conditional model the pack names (codes given their
       group's scale exponent), plus scale deltas. A lower bound on record size,
       hence an upper bound on what an exact codec could save.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402
from moe_lab.lightningstream_nemotron import nvfp4  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"

CORE_MASK = {2: 0b1100, 3: 0b1110}
GATE_TAIL = 0.30
GATE_ZERO_SHARE = 0.30
GATE_O1_PASS = 0.12
GATE_O1_STRONG = 0.20
GROUP = nvfp4.GROUP_SIZE


class CapturingRuntime(LightningRuntime):
    capture = None

    def _moe_cached(self, i, out):
        idx, w = super()._moe_cached(i, out)
        if self.capture is not None:
            self.capture.append((i, self.cp.asnumpy(self.normed).astype(np.float64),
                                 np.asarray(idx, dtype=np.int64).copy()))
        return idx, w


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def entropy_bits(symbols: np.ndarray, alphabet: int) -> float:
    counts = np.bincount(symbols.ravel(), minlength=alphabet).astype(np.float64)
    n = counts.sum()
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def conditional_entropy_bits(symbols: np.ndarray, cond: np.ndarray,
                             alphabet: int, cond_alphabet: int) -> float:
    """H(S | C), in bits per symbol."""
    s = symbols.ravel()
    c = cond.ravel()
    joint = np.bincount(c * alphabet + s,
                        minlength=cond_alphabet * alphabet).astype(np.float64)
    joint = joint.reshape(cond_alphabet, alphabet)
    n = joint.sum()
    total = 0.0
    for row in joint:
        m = row.sum()
        if m == 0:
            continue
        p = row[row > 0] / m
        total += (m / n) * float(-(p * np.log2(p)).sum())
    return total


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=48)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=240)
    ap.add_argument("--entropy-experts", type=int, default=24)
    args = ap.parse_args()

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    corpus = json.loads((OUT_DIR / "s10a_corpus.json").read_text(encoding="utf-8"))
    rt = CapturingRuntime(MODEL_DIR, contexts_max=args.max_ctx,
                          embed_on_host=True, fp8_kv=True)
    rt.enable_cache(args.capacity)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    ids = tokenizer.encode(corpus["gate_prompts"][0]["text"], add_special_tokens=False)
    rt.reset()
    rt.capture = []
    nxt = None
    for t in ids:
        nxt = rt.step(t)
    cur = int(nxt)
    for _ in range(args.gen_tokens):
        cur = int(rt.step(cur))
    cp.cuda.Device(0).synchronize()
    cap, rt.capture = rt.capture, None
    print(f"captured {len(cap)} layer visits", flush=True)

    e2m1 = np.asarray(nvfp4.E2M1_TABLE, dtype=np.float64)
    e4m3 = np.nan_to_num(np.asarray(nvfp4.E4M3_TABLE, dtype=np.float64), nan=0.0)
    hidden, inter = rt.hidden, rt.moe_inter
    ngroup = hidden // GROUP

    # dmax[core_masked_code] = largest |E2M1(full)| - |E2M1(core)| over the codes
    # that share that core. Sign is carried separately.
    dmax_tbl = {}
    for mask in CORE_MASK.values():
        t = np.zeros(16, dtype=np.float64)
        for code in range(16):
            cm = code & mask
            t[cm] = max(t[cm], abs(e2m1[code]) - abs(e2m1[cm]))
        dmax_tbl[mask] = t

    rng = np.random.default_rng(17)
    picks = []
    for j in rng.permutation(len(cap)):
        layer, x, idx = cap[j]
        picks.append((layer, x, int(idx[int(rng.integers(0, len(idx)))])))
        if len(picks) >= args.pairs:
            break

    stats = {str(c): {"rows": 0, "true_zero": 0, "certified": 0, "false_cert": 0,
                      "cert_true_zero": 0, "bound_sum": 0.0, "y0_sum": 0.0,
                      "pairs": 0}
             for c in CORE_MASK}

    def expert_arrays(layer, e):
        bank = rt.bank[layer]
        codes_raw = bank["up_codes"][e * (inter * hidden // 2):
                                     (e + 1) * (inter * hidden // 2)]
        scales_raw = bank["up_scales"][e * (inter * ngroup):(e + 1) * (inter * ngroup)]
        packed = codes_raw.reshape(inter, hidden // 2)
        code = np.empty((inter, hidden), dtype=np.uint8)
        code[:, 0::2] = packed & 15
        code[:, 1::2] = packed >> 4
        return code, scales_raw.reshape(inter, ngroup), bank["g_up"][e]

    print("\nC1-R1: sign-aware sound bound", flush=True)
    for n, (layer, x, e) in enumerate(picks):
        code, sc, gs = expert_arrays(layer, e)
        blk = e4m3[sc] * gs
        blk_full = np.repeat(blk, GROUP, axis=1)
        w = e2m1[code] * blk_full
        y_full = w @ x
        true_zero = y_full <= 0.0
        for c, mask in CORE_MASK.items():
            core = code & mask
            w0 = e2m1[core] * blk_full
            y0 = w0 @ x
            # Sign comes from BIT 3 of the core code, not from the sign of the
            # decoded core value: E2M1 code 8 decodes to -0.0, and -0.0 < 0.0 is
            # False, so reading the sign off the value silently flips it for
            # every negative small-magnitude weight and the bound stops being
            # sound. Both masks preserve bit 3.
            sgn = np.where((core & 0b1000) != 0, -1.0, 1.0)
            dmax = dmax_tbl[mask][core] * blk_full          # >= 0
            adverse = np.maximum(sgn * x[None, :], 0.0)
            bound = (dmax * adverse).sum(axis=1)
            certified = (y0 + bound) <= 0.0
            s = stats[str(c)]
            s["pairs"] += 1
            s["rows"] += inter
            s["true_zero"] += int(true_zero.sum())
            s["certified"] += int(certified.sum())
            s["false_cert"] += int((certified & ~true_zero).sum())
            s["cert_true_zero"] += int((certified & true_zero).sum())
            s["bound_sum"] += float(bound.mean())
            s["y0_sum"] += float(np.abs(y0).mean())
        if (n + 1) % 60 == 0:
            print(f"  {n + 1}/{len(picks)} pairs", flush=True)

    c1r1 = {}
    for c, s in stats.items():
        cert = s["certified"] / s["rows"]
        c1r1[c] = {
            "core_bits": int(c), "pairs": s["pairs"], "rows": s["rows"],
            "true_zero_fraction": s["true_zero"] / s["rows"],
            "certified_fraction": cert,
            "certified_share_of_true_zero": s["cert_true_zero"] / max(1, s["true_zero"]),
            "false_certificates": s["false_cert"],
            "mean_bound": s["bound_sum"] / s["pairs"],
            "mean_abs_y0": s["y0_sum"] / s["pairs"],
            "bound_over_y0": (s["bound_sum"] / s["pairs"]) / (s["y0_sum"] / s["pairs"]),
        }
        print(f"  core {c}: certified {cert * 100:6.2f}% | "
              f"{c1r1[c]['certified_share_of_true_zero'] * 100:6.2f}% of true zeros | "
              f"false {s['false_cert']} | bound/|y0| {c1r1[c]['bound_over_y0']:.2f}",
              flush=True)

    # ------------------------------------------------------------------- O1
    print("\nO1: entropy of the real codes and scales", flush=True)
    codes_all, sc_all = [], []
    seen = set()
    for layer, _, e in picks:
        if (layer, e) in seen:
            continue
        seen.add((layer, e))
        code, sc, _ = expert_arrays(layer, e)
        codes_all.append(code)
        sc_all.append(sc)
        if len(seen) >= args.entropy_experts:
            break
    code_cat = np.concatenate([c.ravel() for c in codes_all])
    sc_cat = np.concatenate([s.ravel() for s in sc_all])
    # each code's group scale, repeated to code resolution
    cond_cat = np.concatenate([np.repeat(s, GROUP, axis=1).ravel() for s in sc_all])
    sc_exp = (sc_cat >> 3).astype(np.int64)          # E4M3 exponent field
    cond_exp = (cond_cat >> 3).astype(np.int64)

    h_code = entropy_bits(code_cat.astype(np.int64), 16)
    h_code_given_exp = conditional_entropy_bits(code_cat.astype(np.int64), cond_exp, 16, 32)
    h_scale = entropy_bits(sc_cat.astype(np.int64), 256)
    h_scale_exp = entropy_bits(sc_exp, 32)
    sc_delta = np.concatenate([np.diff(s.astype(np.int64), axis=1).ravel() % 256
                               for s in sc_all])
    h_scale_delta = entropy_bits(sc_delta, 256)

    n_codes = inter * hidden
    n_scales = inter * ngroup
    bytes_now = n_codes / 2 + n_scales
    best_code_bits = min(h_code, h_code_given_exp)
    best_scale_bits = min(h_scale, h_scale_delta)
    bytes_entropy = (n_codes * best_code_bits + n_scales * best_scale_bits) / 8.0
    reduction = 1.0 - bytes_entropy / bytes_now

    o1 = {
        "experts_sampled": len(seen), "symbols_codes": int(code_cat.size),
        "symbols_scales": int(sc_cat.size),
        "code_entropy_bits": h_code,
        "code_entropy_given_scale_exponent_bits": h_code_given_exp,
        "scale_entropy_bits": h_scale,
        "scale_exponent_entropy_bits": h_scale_exp,
        "scale_delta_entropy_bits": h_scale_delta,
        "record_bytes_now": bytes_now,
        "record_bytes_at_entropy": bytes_entropy,
        "pack_reduction": reduction,
        "gemv_time_saving_via_y2r1": reduction * 0.684,
    }
    print(f"  codes  H={h_code:.4f} b, H|scale-exp={h_code_given_exp:.4f} b (of 4)",
          flush=True)
    print(f"  scales H={h_scale:.4f} b, delta H={h_scale_delta:.4f} b (of 8)", flush=True)
    print(f"  record {bytes_now:,.0f} -> {bytes_entropy:,.0f} B = "
          f"{reduction * 100:.2f}% reduction", flush=True)

    best = max(c1r1, key=lambda k: c1r1[k]["certified_fraction"])
    sound = all(v["false_certificates"] == 0 for v in c1r1.values())
    gates = {
        "G_C1R1_S1_soundness": {"false_by_core": {k: v["false_certificates"]
                                                  for k, v in c1r1.items()},
                                "passed": bool(sound)},
        "G_C1R1_R1_tail_yield": {"required": GATE_TAIL,
                                 "measured": c1r1[best]["certified_fraction"],
                                 "best_core_bits": int(best),
                                 "passed": bool(c1r1[best]["certified_fraction"] >= GATE_TAIL)},
        "G_C1R1_B1_bound_useful": {"required": GATE_ZERO_SHARE,
                                   "measured": c1r1[best]["certified_share_of_true_zero"],
                                   "passed": bool(c1r1[best]["certified_share_of_true_zero"]
                                                  >= GATE_ZERO_SHARE)},
        "G_O1_1_pass": {"required": GATE_O1_PASS, "measured": reduction,
                        "passed": bool(reduction >= GATE_O1_PASS)},
        "G_O1_2_strong": {"required": GATE_O1_STRONG, "measured": reduction,
                          "passed": bool(reduction >= GATE_O1_STRONG)},
    }

    payload = {
        "kind": "lightningstream_nemotron_c1r1_o1_bound_and_entropy",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "C1_R1_AND_O1",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"pairs": len(picks), "core_masks": CORE_MASK,
                   "entropy_experts": args.entropy_experts, "group_size": GROUP,
                   "hidden": hidden, "inter": inter},
        "c1_reference": {"bound_over_y0": {"2": 31.09, "3": 9.06},
                         "certified_fraction": 0.0033},
        "c1r1": c1r1,
        "o1": o1,
        "gates": gates,
        "claim_boundary": (
            "C1-R1 is the same numerical oracle as C1 with a strictly sharper "
            "SOUND bound: the core carries the sign bit and bit truncation can "
            "only raise a magnitude, so every residual term's direction is known "
            "without reading the tail and only the adverse half is counted. It "
            "still uses the exact dmax table, which a built system would have to "
            "derive from the core alone -- which it can, since dmax depends only "
            "on the core code -- so unlike C1's exact residual norms this bound "
            "needs no stored metadata. O1 is an ENTROPY LOWER BOUND on record "
            "size, not a codec: a real ANS coder does not reach entropy and pays "
            "decode overhead, so the measured reduction is an UPPER BOUND on "
            "what OrbitANS could save. Neither is a time or throughput "
            "measurement; the GEMV-time figure applies Y2-R1's measured slope "
            "and is arithmetic."),
    }
    (OUT_DIR / "c1r1_o1_bound_and_entropy.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\ngates")
    print(f"  G-C1R1-S1 soundness : {sound}")
    print(f"  G-C1R1-R1 tail yield: {gates['G_C1R1_R1_tail_yield']['passed']} "
          f"({c1r1[best]['certified_fraction'] * 100:.2f}% vs 30%)")
    print(f"  G-C1R1-B1 useful    : {gates['G_C1R1_B1_bound_useful']['passed']}")
    print(f"  G-O1-1 (>=12%)      : {gates['G_O1_1_pass']['passed']} "
          f"({reduction * 100:.2f}%)")
    print(f"  G-O1-2 (>=20%)      : {gates['G_O1_2_strong']['passed']}")
    print("\nwritten c1r1_o1_bound_and_entropy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
