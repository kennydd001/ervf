"""N3: can a cheap SOUND bound prove ReLU^2(y_j) = 0 and skip that row entirely?

Preregistered in N1_N5_OWN_HYPOTHESES_PREREGISTRATION_2026-08-15.md.

~91% of an expert's ReLU^2 outputs are zero (S5) and all of them are computed in
full. A rank-r approximation of the weight matrix gives a cheap estimate plus a
sound residual bound:

    yhat = (W V_r)(V_r^T x)      |y_j - yhat_j| <= ||w_j - what_j||_2 * ||x||_2

and y_j <= 0 is PROVEN once yhat_j + ||w_j - what_j||_2 ||x||_2 <= 0. Unlike a
low-rank surrogate this never replaces the weight: rows that are not certified
are computed exactly, so the output stays bit-identical.
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
RANKS = [8, 16, 32, 64]
GATE_CERT = 0.30
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


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", type=int, default=48)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--gen-tokens", type=int, default=20)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--acts-per-expert", type=int, default=8)
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

    rng = np.random.default_rng(23)
    chosen, seen = [], set()
    for j in rng.permutation(len(cap)):
        layer, x, idx = cap[j]
        e = int(idx[int(rng.integers(0, len(idx)))])
        if (layer, e) in seen:
            continue
        seen.add((layer, e))
        acts = [cap[int(k)][1] for k in
                rng.choice(len(cap), size=args.acts_per_expert, replace=False)]
        chosen.append((layer, e, acts))
        if len(chosen) >= args.experts:
            break

    stats = {str(r): {"rows": 0, "true_zero": 0, "cert": 0, "false": 0,
                      "cert_true_zero": 0, "bound_sum": 0.0, "yh_sum": 0.0,
                      "cases": 0}
             for r in RANKS}
    per_expert = []

    for n, (layer, e, acts) in enumerate(chosen):
        bank = rt.bank[layer]
        codes_raw = bank["up_codes"][e * (inter * hidden // 2):
                                     (e + 1) * (inter * hidden // 2)]
        scales_raw = bank["up_scales"][e * (inter * ngroup):(e + 1) * (inter * ngroup)]
        packed = codes_raw.reshape(inter, hidden // 2)
        code = np.empty((inter, hidden), dtype=np.uint8)
        code[:, 0::2] = packed & 15
        code[:, 1::2] = packed >> 4
        blk = np.repeat(e4m3[scales_raw.reshape(inter, ngroup)] * bank["g_up"][e],
                        GROUP, axis=1)
        W = e2m1[code] * blk                                  # [inter, hidden]

        # one SVD per expert, reused for every rank
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        rec = {"layer": layer, "expert": e,
               "spectrum_head": [float(v) for v in S[:8]],
               "energy_fraction": {}}
        tot = float((S ** 2).sum())
        for r in RANKS:
            Wr = (U[:, :r] * S[:r]) @ Vt[:r]
            resid = W - Wr
            rnorm = np.sqrt((resid * resid).sum(axis=1))      # [inter]
            rec["energy_fraction"][str(r)] = float((S[:r] ** 2).sum() / tot)
            for x in acts:
                y = W @ x
                yh = Wr @ x
                bound = rnorm * float(np.linalg.norm(x))
                cert = (yh + bound) <= 0.0
                tz = y <= 0.0
                s = stats[str(r)]
                s["cases"] += 1
                s["rows"] += inter
                s["true_zero"] += int(tz.sum())
                s["cert"] += int(cert.sum())
                s["false"] += int((cert & ~tz).sum())
                s["cert_true_zero"] += int((cert & tz).sum())
                s["bound_sum"] += float(bound.mean())
                s["yh_sum"] += float(np.abs(yh).mean())
        per_expert.append(rec)
        print(f"  {n + 1}/{len(chosen)} experts", flush=True)

    summary = {}
    for r, s in stats.items():
        cert = s["cert"] / s["rows"]
        summary[r] = {
            "rank": int(r), "cases": s["cases"], "rows": s["rows"],
            "true_zero_fraction": s["true_zero"] / s["rows"],
            "certified_fraction": cert,
            "certified_share_of_true_zero": s["cert_true_zero"] / max(1, s["true_zero"]),
            "false_certificates": s["false"],
            "mean_bound": s["bound_sum"] / s["cases"],
            "mean_abs_yhat": s["yh_sum"] / s["cases"],
            "bound_over_yhat": (s["bound_sum"] / s["cases"]) / (s["yh_sum"] / s["cases"]),
            "mean_energy_fraction": float(np.mean(
                [p["energy_fraction"][r] for p in per_expert])),
            # projection costs r*hidden MACs; skipping saves cert*inter*hidden
            "projection_pays_off": bool(int(r) < cert * inter),
        }
        print(f"  rank {r:>3}: certified {cert * 100:6.2f}% | "
              f"{summary[r]['certified_share_of_true_zero'] * 100:6.2f}% of zeros | "
              f"false {s['false']} | bound/|yhat| {summary[r]['bound_over_yhat']:.2f} | "
              f"energy {summary[r]['mean_energy_fraction'] * 100:.1f}%", flush=True)

    best = max(summary, key=lambda k: summary[k]["certified_fraction"])
    sound = all(v["false_certificates"] == 0 for v in summary.values())
    r1 = summary[best]["certified_fraction"] >= GATE_CERT
    gates = {
        "G_N3_S1_soundness": {"passed": bool(sound),
                              "false_by_rank": {k: v["false_certificates"]
                                                for k, v in summary.items()}},
        "G_N3_R1_certified": {"required": GATE_CERT, "best_rank": int(best),
                              "measured": summary[best]["certified_fraction"],
                              "passed": bool(r1)},
        "G_N3_C1_projection_pays": {k: v["projection_pays_off"]
                                    for k, v in summary.items()},
    }

    payload = {
        "kind": "lightningstream_nemotron_n3_relu2_prefilter_oracle",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N3_RELU2_PREFILTER",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "config": {"ranks": RANKS, "experts": len(chosen),
                   "acts_per_expert": args.acts_per_expert,
                   "hidden": hidden, "inter": inter},
        "summary": summary,
        "per_expert": per_expert,
        "gates": gates,
        "claim_boundary": (
            "Numerical oracle on real expert weights and real activations. The "
            "bound is SOUND: rows it certifies are provably zero after ReLU^2, so "
            "skipping them leaves the output bit-identical, and rows it does not "
            "certify are still computed exactly. This is NOT a low-rank surrogate "
            "and never replaces a weight. The residual row norms are treated as "
            "free here; a built version would have to store them (one float per "
            "row per expert) and pay the rank-r projection, which the "
            "projection_pays_off field accounts for but does not time. No kernel "
            "was written and nothing here is a time or throughput measurement."),
    }
    (OUT_DIR / "n3_relu2_prefilter_oracle.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n  G-N3-S1 soundness : {sound}")
    print(f"  G-N3-R1 certified : {r1} "
          f"({summary[best]['certified_fraction'] * 100:.2f}% vs {GATE_CERT * 100:.0f}%)")
    print("\nwritten n3_relu2_prefilter_oracle.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
