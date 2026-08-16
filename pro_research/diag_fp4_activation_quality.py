"""The gate before anything gets built: what does FP4-quantising the real
activations actually cost in output quality?

FP4_W4A8_RECIPES.json closed the milder option -- exactly one pairing works in
the whole matrix, FP4 x FP4 with BlockWise1x16 -- so adopting native FP4 forces
the activation to 4 bits. Every speed number measured so far (2.52x on lm_head,
1.68x on shared_down, M free to M=8) was taken with exact +1 or synthetic
activations and says nothing about that.

So measure the cost before writing a kernel for it. No Torch needed: the
question is what NVFP4 quantisation does to a real activation vector, and that
is a round trip we can do in CuPy against the production lm_head.

## Method

Run a real generation on the production stack. At every decode step, capture
`rt.normed` -- the final-norm output that feeds lm_head -- and compute the
logits twice through the SAME production NVFP4 lm_head kernel:

    reference   gemv_into(lm_head, normed)
    candidate   gemv_into(lm_head, dequant(quant_nvfp4(normed)))

Only the activation differs, so the delta is exactly the activation
quantisation, with no kernel, layout or accumulation-order change mixed in.

Reported per token:
  * top-1 agreement -- does the generated token change at all?
  * rank of the reference argmax under the candidate
  * cross-entropy of the candidate distribution against the reference argmax,
    and the CE delta, which is the number a quality gate would be set on
  * max |logit| deviation, for scale

## NVFP4 activation quantisation, as ModelOpt/vLLM do it

  global scale  s_g = amax(x) / (6 * 448)        448 = e4m3 max, 6 = e2m1 max
  per 16-block  bs  = amax(block) / 6 / s_g,  stored e4m3
  codes         round-to-nearest onto the e2m1 grid of x / (bs * s_g)

The e2m1 grid is exactly {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives -- eight
magnitudes. That is the whole representable set an activation gets mapped onto.

Read-only. Changes nothing in the runtime; the quantisation is applied to a copy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, gpu_processes, require_model_dir, run_text, utc_now, write_json_atomic

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E4M3_MAX = 448.0
BLOCK = 16


def _e4m3_round(v: np.ndarray) -> np.ndarray:
    """Round positive floats onto the e4m3 grid (what a stored block scale is)."""
    out = np.zeros_like(v)
    nz = v > 0
    if not np.any(nz):
        return out
    x = np.clip(v[nz], 2.0 ** -9, E4M3_MAX)
    e = np.floor(np.log2(x))
    e = np.clip(e, -6, 8)                     # e4m3 normal exponent range
    step = 2.0 ** (e - 3)                     # 3 mantissa bits
    out[nz] = np.clip(np.round(x / step) * step, 0.0, E4M3_MAX)
    return out


def _e4m3_ceil(v: np.ndarray) -> np.ndarray:
    """Smallest e4m3 value >= v.

    The block scale must be rounded UP, not to-nearest. diag_verify_nvfp4_
    quantizer.py caught this: rounding to nearest lets the scale come out just
    below what the block needs, so the largest elements land above 6.0 on the
    e2m1 grid and get clipped -- 4.35% of real lm_head weights did. Clipping is
    the quantiser's own error, not the format's, and it would have inflated the
    activation-quality number this file exists to produce.
    """
    r = _e4m3_round(v)
    lo = r < v - 1e-12
    if np.any(lo):
        x = np.clip(v[lo], 2.0 ** -9, E4M3_MAX)
        e = np.clip(np.floor(np.log2(x)), -6, 8)
        step = 2.0 ** (e - 3)
        up = np.clip(np.ceil(x / step) * step, 0.0, E4M3_MAX)
        # ceil can cross an exponent boundary; renormalise onto the grid
        r[lo] = _e4m3_round(up)
        still = r[lo] < v[lo] - 1e-12
        if np.any(still):
            r_lo = r[lo]
            r_lo[still] = E4M3_MAX
            r[lo] = r_lo
    return r


def quant_nvfp4(x: np.ndarray, s_g: float | None = None) -> tuple[np.ndarray, dict]:
    """NVFP4 round trip of one activation vector, ModelOpt/vLLM convention.

    ``s_g`` overrides the per-tensor global scale. The idempotence check needs
    it: real checkpoint weights were encoded against lm_head's own
    ``weight_scale_2``, so recomputing a global scale from a 4096-element slice
    gives a different grid and exact round-tripping is impossible by
    construction. Passing the source scale makes that test fair.
    """
    n = x.size
    pad = (-n) % BLOCK
    xp = np.concatenate([x, np.zeros(pad, dtype=np.float32)]) if pad else x
    blocks = xp.reshape(-1, BLOCK)

    amax = float(np.max(np.abs(xp)))
    if amax == 0.0:
        return x.copy(), {"global_scale": 0.0, "clipped_fraction": 0.0}
    if s_g is None:
        s_g = amax / (6.0 * E4M3_MAX)

    bamax = np.max(np.abs(blocks), axis=1)
    bs = _e4m3_ceil(bamax / 6.0 / s_g)
    eff = (bs * s_g)[:, None]
    eff = np.where(eff == 0.0, 1.0, eff)

    scaled = blocks / eff
    clipped = float(np.mean(np.abs(scaled) > 6.0))
    scaled = np.clip(scaled, -6.0, 6.0)

    idx = np.abs(np.abs(scaled)[..., None] - E2M1[None, None, :]).argmin(axis=-1)
    deq = (np.sign(scaled) * E2M1[idx] * eff).reshape(-1)[:n].astype(np.float32)
    return deq, {"global_scale": float(s_g), "clipped_fraction": clipped}


def _require_gpu_idle_wddm() -> dict:
    """Block on real competing work; ignore the idle ChatGPT WDDM GUI context."""
    raw = gpu_processes()
    ignored, blockers = [], []
    for line in raw:
        (ignored if ("chatgpt.exe" in line.lower() and "[n/a]" in line.lower())
         else blockers).append(line)
    if blockers:
        raise RuntimeError(
            "Another process currently owns a CUDA context; this runner will not "
            "kill it: " + "; ".join(blockers))
    snap = run_text(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"])
    if snap.startswith("ERROR") or snap.startswith("rc="):
        raise RuntimeError(f"Unable to query GPU idle state: {snap}")
    parts = [x.strip() for x in snap.splitlines()[0].split(",")]
    used, util = int(parts[0]), int(parts[1])
    if used > 1024:
        raise RuntimeError(f"GPU memory already busy: {used} MiB")
    if util > 10:
        raise RuntimeError(f"GPU utilization already busy: {util}%")
    return {"compute_app_lines": raw, "ignored_wddm_gui_contexts": ignored,
            "gpu_memory_used_mib": used, "gpu_utilization_percent": util}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args()

    # Same narrow exception Kimi established for C3A-v2 (commit 555be02) rather
    # than a new one: on this Windows/WDDM box the ChatGPT/Codex GUI shows up in
    # --query-compute-apps with used_memory=[N/A], and common.require_gpu_free
    # reads that 0 MiB graphics context as competing compute, which blocks every
    # run. Only ChatGPT.exe with non-numeric memory is ignored; any other
    # compute app still blocks, and total memory/utilisation are gated too.
    preflight = _require_gpu_idle_wddm()
    import cupy as cp
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    rt = LightningRuntime(str(require_model_dir()), contexts_max=4096,
                          embed_on_host=True, fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.device_cache = True
    rt.deterministic_accum = True
    if rt.lm_head_kind != "nvfp4":
        print(json.dumps({"status": "lm_head_not_nvfp4", "kind": rt.lm_head_kind}))
        return 2

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(require_model_dir()))
    prompt = tok.encode("The history of computing began when", add_special_tokens=False)

    fused = rt.fused
    ref_logits = cp.zeros(rt.vocab, dtype=cp.float32)
    cand_logits = cp.zeros(rt.vocab, dtype=cp.float32)

    rt.reset()
    nxt = None
    for t in prompt:
        nxt = int(rt.step(int(t)))

    per_token = []
    clip_fracs = []
    for _ in range(args.tokens):
        # rt.normed currently holds the final-norm output for this step, i.e.
        # exactly the vector lm_head consumes.
        normed = cp.asnumpy(rt.normed[:rt.hidden]).astype(np.float32)
        deq, info = quant_nvfp4(normed)
        clip_fracs.append(info["clipped_fraction"])

        fused.gemv_into(ref_logits, rt.lm_head_codes, rt.lm_head_scales,
                        rt.normed, rt.lm_head_g, rt.vocab, rt.hidden)
        fused.gemv_into(cand_logits, rt.lm_head_codes, rt.lm_head_scales,
                        cp.asarray(deq), rt.lm_head_g, rt.vocab, rt.hidden)
        cp.cuda.Device(0).synchronize()

        r = cp.asnumpy(ref_logits).astype(np.float64)
        c = cp.asnumpy(cand_logits).astype(np.float64)
        r_arg = int(r.argmax())
        c_arg = int(c.argmax())
        # rank of the reference token under the candidate distribution
        rank = int((c > c[r_arg]).sum())
        cm = c - c.max()
        ce = float(-(cm[r_arg] - np.log(np.exp(cm).sum())))
        rm = r - r.max()
        ce_ref = float(-(rm[r_arg] - np.log(np.exp(rm).sum())))
        per_token.append({
            "top1_agree": r_arg == c_arg,
            "ref_rank_under_cand": rank,
            "ce_ref": ce_ref, "ce_cand": ce, "ce_delta": ce - ce_ref,
            "max_abs_logit_dev": float(np.max(np.abs(c - r))),
        })
        nxt = int(rt.step(nxt))

    agree = float(np.mean([p["top1_agree"] for p in per_token]))
    ce_ref_m = float(np.mean([p["ce_ref"] for p in per_token]))
    ce_cand_m = float(np.mean([p["ce_cand"] for p in per_token]))
    payload = {
        "kind": "diag_fp4_activation_quality",
        "created_utc": utc_now(),
        "note": "isolates ONLY the activation quantisation: identical production NVFP4 lm_head kernel, identical weights, only the input vector differs. No kernel, layout or accumulation-order change is mixed in.",
        "environment": environment_snapshot(),
        "gpu_idle_preflight": preflight,
        "tokens": args.tokens,
        "quantisation": "NVFP4 per-16 block e4m3 scales with a per-tensor global scale, ModelOpt/vLLM convention; e2m1 grid {0,.5,1,1.5,2,3,4,6}",
        "summary": {
            "top1_agreement": agree,
            "top1_changed_fraction": 1.0 - agree,
            "mean_ce_reference": ce_ref_m,
            "mean_ce_candidate": ce_cand_m,
            "mean_ce_delta": ce_cand_m - ce_ref_m,
            "relative_ce_increase_pct": 100.0 * (ce_cand_m - ce_ref_m) / ce_ref_m if ce_ref_m else None,
            "mean_ref_rank_under_cand": float(np.mean([p["ref_rank_under_cand"] for p in per_token])),
            "worst_ref_rank": int(np.max([p["ref_rank_under_cand"] for p in per_token])),
            "mean_max_abs_logit_dev": float(np.mean([p["max_abs_logit_dev"] for p in per_token])),
            "mean_clipped_fraction": float(np.mean(clip_fracs)),
        },
        "per_token": per_token,
        "claim_boundary": "lm_head activation only, greedy decode on one prompt; not a full-model quality result and not a tok/s claim",
    }
    write_json_atomic(REPO / "pro_research" / "diag_fp4_activation_quality.json",
                      payload, archive=False)
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
