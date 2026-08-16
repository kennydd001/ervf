"""Follow-up to the N-scaling synthesis (agents/RESEARCH_NOTEBOOK.md
2026-08-16, "Synthese van de vier N-schalingstests"): expensive kernels
(Mamba, lm_head) scale worse than cheap ones (attention, shared-expert) under
repeated back-to-back calls -- correlation established, mechanism not. Two
candidate explanations were named but not distinguished: GPU clock/power
throttling under sustained load, or memory-controller contention that grows
with kernel working-set size.

This tests the throttling hypothesis directly: poll nvidia-smi clocks.sm/
power.draw/temperature.gpu at ~5 Hz while running a SUSTAINED lm_head
workload (repeated N=16 batches for several seconds, not just the few dozen
rounds the original scaling test used), and check whether clocks.sm actually
drops over the course of the run. If it does not drop, throttling is ruled
out as the explanation and memory contention becomes the more likely
candidate (not tested here -- would need Nsight Compute).

Not a gated PRO experiment -- a root-cause diagnostic, read-only.
"""

from __future__ import annotations

import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, require_gpu_free, require_model_dir, utc_now, write_json_atomic

PROMPTS = [
    "The history of computing began when",
    "Write a correct Python function that computes the longest increasing subsequence length in O(n log n), then explain its invariant.\n",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "In the quiet village, the old fisherman noticed something strange about the tide",
    "The quarterly earnings report showed a significant increase in revenue driven by",
    "Photosynthesis is the process by which plants convert light energy into",
    "The defendant's attorney argued that the evidence presented by the prosecution was",
    "To configure the network firewall, first navigate to the settings panel and",
    "The ancient Roman aqueducts were engineering marvels that transported water using",
    "She picked up the violin, tucked it under her chin, and began to play a melody that",
    "The stock market experienced significant volatility today as investors reacted to",
    "According to the latest climate research, rising ocean temperatures are causing",
    "The chess grandmaster studied the board carefully before deciding to sacrifice his",
    "In object-oriented programming, inheritance allows a class to acquire properties from",
    "The archaeologists uncovered pottery fragments dating back to",
    "Machine learning models require large amounts of training data to",
]

N = 16
SUSTAIN_SECONDS = 6.0


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

    if rt.lm_head_kind != "nvfp4":
        print(f"lm_head_kind is {rt.lm_head_kind}, expected nvfp4")
        return 1

    captured = []
    fused = rt.fused
    orig_gemv_into = fused.gemv_into

    def capture_gemv_into(out, codes, scales, x, g, rows, cols, **kwargs):
        if rows == rt.vocab:
            captured.append(cp.asarray(x).copy())
        return orig_gemv_into(out, codes, scales, x, g, rows, cols, **kwargs)

    fused.gemv_into = capture_gemv_into
    for prompt in PROMPTS[:N]:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = int(rt.step(int(t)))
        rt.step(nxt)
    fused.gemv_into = orig_gemv_into
    cp.cuda.Device(0).synchronize()

    if len(captured) < N:
        print(f"only captured {len(captured)}, expected {N}")
        return 1
    vecs = captured[-N:]
    vocab, hidden = rt.vocab, rt.hidden
    out_bufs = [cp.zeros(vocab, dtype=cp.float32) for _ in range(N)]

    # ---- start nvidia-smi polling in the background at ~5 Hz.
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,pstate",
         "--format=csv,noheader", "-lms", "200"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )

    # ---- sustained lm_head load: repeated N=16 batches for SUSTAIN_SECONDS.
    t_start = time.perf_counter()
    batches = 0
    while time.perf_counter() - t_start < SUSTAIN_SECONDS:
        for x, out in zip(vecs, out_bufs):
            orig_gemv_into(out, rt.lm_head_codes, rt.lm_head_scales, x, rt.lm_head_g, vocab, hidden)
        cp.cuda.Device(0).synchronize()
        batches += 1
    elapsed = time.perf_counter() - t_start

    time.sleep(0.3)  # let the last poll lines flush
    smi.terminate()
    try:
        out, _ = smi.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        smi.kill()
        out, _ = smi.communicate()

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    samples = []
    for l in lines:
        parts = [p.strip() for p in l.split(",")]
        try:
            samples.append({
                "clocks_sm_mhz": float(parts[0].split()[0]),
                "power_draw_w": float(parts[1].split()[0]),
                "temperature_c": float(parts[2].split()[0]),
                "pstate": parts[3],
            })
        except (ValueError, IndexError):
            continue

    if len(samples) < 4:
        print(f"too few nvidia-smi samples captured ({len(samples)}), inconclusive")
        clocks_first_half_mean = clocks_second_half_mean = None
        throttle_detected = None
    else:
        half = len(samples) // 2
        first = samples[:half]
        second = samples[half:]
        clocks_first_half_mean = sum(s["clocks_sm_mhz"] for s in first) / len(first)
        clocks_second_half_mean = sum(s["clocks_sm_mhz"] for s in second) / len(second)
        # a real throttle shows a clear downward step, not just sample noise.
        throttle_detected = (clocks_first_half_mean - clocks_second_half_mean) > 50.0

    payload = {
        "kind": "diag_lmhead_throttle_check",
        "created_utc": utc_now(),
        "note": "root-cause follow-up to the N-scaling synthesis: polls nvidia-smi clocks.sm during a sustained lm_head workload to test whether clock throttling explains the supra-linear penalty found for Mamba/lm_head",
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "n_per_batch": N,
        "sustain_seconds_requested": SUSTAIN_SECONDS,
        "sustain_seconds_actual": elapsed,
        "batches_run": batches,
        "nvidia_smi_sample_count": len(samples),
        "nvidia_smi_samples": samples,
        "clocks_sm_first_half_mean_mhz": clocks_first_half_mean,
        "clocks_sm_second_half_mean_mhz": clocks_second_half_mean,
        "throttle_detected_gt_50mhz_drop": throttle_detected,
    }
    out_path = REPO / "pro_research" / "diag_lmhead_throttle_check.json"
    write_json_atomic(out_path, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
