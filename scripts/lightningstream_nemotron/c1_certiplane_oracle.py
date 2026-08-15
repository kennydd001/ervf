"""C1: the CertiPlane oracle -- how much of the NVFP4 tail is provably skippable?

Preregistered in C1_CERTIPLANE_ORACLE_PREREGISTRATION_2026-08-15.md.

Split the 4-bit NVFP4 code into a low-bit core and an exact residual. Compute the
preactivation from the core alone, bound the residual's contribution per
group-of-16 with Cauchy-Schwarz, and count how often the bound proves ReLU^2 is
exactly zero -- in which case that neuron's residual bits never need reading.

Real experts, real activations captured from a real greedy generation. No kernel,
no runtime change: the pack itself says the oracle comes before any kernel.
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

# core keeps the sign bit plus the top (c-1) magnitude bits of the 4-bit code
CORE_MASK = {2: 0b1100, 3: 0b1110}
GATE_TAIL_FRACTION = 0.30
GATE_ZERO_SHARE = 0.30
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
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=240)
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

    rng = np.random.default_rng(17)
    picks = []
    order = rng.permutation(len(cap))
    for j in order:
        layer, x, idx = cap[j]
        picks.append((layer, x, int(idx[int(rng.integers(0, len(idx)))])))
        if len(picks) >= args.pairs:
            break

    stats = {str(c): {"pairs": 0, "rows": 0, "true_zero": 0, "certified": 0,
                      "false_cert": 0, "certified_of_true_zero": 0}
             for c in CORE_MASK}
    per_pair = []

    for n, (layer, x, e) in enumerate(picks):
        bank = rt.bank[layer]
        codes_raw = bank["up_codes"][e * (inter * hidden // 2):
                                     (e + 1) * (inter * hidden // 2)]
        scales_raw = bank["up_scales"][e * (inter * ngroup):(e + 1) * (inter * ngroup)]
        gscale = bank["g_up"][e]

        packed = codes_raw.reshape(inter, hidden // 2)
        code = np.empty((inter, hidden), dtype=np.uint8)
        code[:, 0::2] = packed & 15
        code[:, 1::2] = packed >> 4
        blk = e4m3[scales_raw.reshape(inter, ngroup)] * gscale
        w = e2m1[code] * np.repeat(blk, GROUP, axis=1)
        y_full = w @ x
        true_zero = y_full <= 0.0

        xg = x.reshape(ngroup, GROUP)
        xnorm = np.sqrt((xg * xg).sum(axis=1))              # [ngroup]

        rec = {"layer": layer, "expert": e,
               "true_zero_fraction": float(true_zero.mean())}
        for c, mask in CORE_MASK.items():
            w0 = e2m1[code & mask] * np.repeat(blk, GROUP, axis=1)
            y0 = w0 @ x
            dw = w - w0
            dg = np.sqrt((dw.reshape(inter, ngroup, GROUP) ** 2).sum(axis=2))
            bound = dg @ xnorm                               # [inter]
            certified = (y0 + bound) <= 0.0
            false_cert = int((certified & ~true_zero).sum())
            s = stats[str(c)]
            s["pairs"] += 1
            s["rows"] += inter
            s["true_zero"] += int(true_zero.sum())
            s["certified"] += int(certified.sum())
            s["false_cert"] += false_cert
            s["certified_of_true_zero"] += int((certified & true_zero).sum())
            rec[f"c{c}"] = {"certified_fraction": float(certified.mean()),
                            "false_cert": false_cert,
                            "mean_bound": float(bound.mean()),
                            "mean_abs_y0": float(np.abs(y0).mean())}
        per_pair.append(rec)
        if (n + 1) % 40 == 0:
            print(f"  {n + 1}/{len(picks)} pairs", flush=True)

    summary = {}
    for c, s in stats.items():
        cert_frac = s["certified"] / s["rows"]
        summary[c] = {
            "core_bits": int(c), "mask": CORE_MASK[int(c)],
            "pairs": s["pairs"], "rows": s["rows"],
            "true_zero_fraction": s["true_zero"] / s["rows"],
            "certified_fraction": cert_frac,
            "certified_share_of_true_zero":
                s["certified_of_true_zero"] / max(1, s["true_zero"]),
            "false_certificates": s["false_cert"],
            "tail_bits_per_weight_saved": cert_frac * (4 - int(c)),
        }
        print(f"  core {c} bits: certified {cert_frac * 100:6.2f}% of rows | "
              f"{summary[c]['certified_share_of_true_zero'] * 100:6.2f}% of true zeros | "
              f"false {s['false_cert']}", flush=True)

    best = max(summary, key=lambda k: summary[k]["certified_fraction"])
    sound = all(v["false_certificates"] == 0 for v in summary.values())
    r1 = summary[best]["certified_fraction"] >= GATE_TAIL_FRACTION
    b1 = summary[best]["certified_share_of_true_zero"] >= GATE_ZERO_SHARE

    gates = {
        "G_C1_S1_soundness": {"required": "zero false certificates",
                              "false_by_core": {k: v["false_certificates"]
                                                for k, v in summary.items()},
                              "passed": bool(sound)},
        "G_C1_R1_tail_yield": {"required_fraction": GATE_TAIL_FRACTION,
                               "best_core_bits": int(best),
                               "measured": summary[best]["certified_fraction"],
                               "passed": bool(r1)},
        "G_C1_B1_bound_useful": {"required_share_of_true_zeros": GATE_ZERO_SHARE,
                                 "measured": summary[best]["certified_share_of_true_zero"],
                                 "passed": bool(b1)},
    }

    payload = {
        "kind": "lightningstream_nemotron_c1_certiplane_oracle",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "C1_CERTIPLANE_ORACLE",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacity": args.capacity, "max_ctx": args.max_ctx,
                   "gen_tokens": args.gen_tokens, "pairs": len(picks),
                   "core_masks": CORE_MASK, "group_size": GROUP,
                   "hidden": hidden, "inter": inter,
                   "reference_dtype": "float64"},
        "summary": summary,
        "per_pair": per_pair,
        "gates": gates,
        "claim_boundary": (
            "Numerical oracle on real expert records and real activations "
            "captured from a real greedy generation. The residual bound uses the "
            "EXACT per-group residual norms, which a real system could not store "
            "for free -- the pack budgets 0.15 bit/weight for certificate "
            "metadata -- so the certification rate measured here is an UPPER "
            "BOUND on what a built version would achieve. The reference "
            "preactivation is computed in float64 on the host, not by the "
            "runtime's float32 kernel, so neurons whose true value sits within "
            "float32 noise of zero are classified by the float64 value. No "
            "kernel was written and nothing here is a time or throughput "
            "measurement; byte savings are not time savings on this runtime "
            "(S11, S12, X1, Y2-R1)."),
    }
    (OUT_DIR / "c1_certiplane_oracle.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n  G-C1-S1 soundness   : {sound}")
    print(f"  G-C1-R1 tail yield  : {r1} "
          f"({summary[best]['certified_fraction'] * 100:.2f}% vs {GATE_TAIL_FRACTION * 100:.0f}%)")
    print(f"  G-C1-B1 bound useful: {b1} "
          f"({summary[best]['certified_share_of_true_zero'] * 100:.2f}% vs "
          f"{GATE_ZERO_SHARE * 100:.0f}%)")
    print("\nwritten c1_certiplane_oracle.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
