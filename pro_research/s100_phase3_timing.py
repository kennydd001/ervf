"""Matched V18 A/C/C/B for phase-3 Q-only and routed-top-k profiles."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, first_divergence, percentiles, utc_now, write_json_atomic
from diag_component_marginals_graph import _prefill, _recapture, _reset_exact_state, _run
from diag_fp4_activation_quality import _require_gpu_idle_wddm
from s100_phase3_profiles import TIMING_PROFILES, apply_phase3_profile, public_profile_record, restore_phase3_profile
from s100_phase3_runtime import build_v18_runtime, install_combined_current, rebuild_cache

PREREG = REPO / "pro_research" / "S100_PHASE3_TIMING_PREREGISTRATION.md"
GAIN_GATES = {
    "qfast": {"kind": "gain", "value": 0.75},
    "k5": {"kind": "gain", "value": 0.75},
    "k4": {"kind": "gain", "value": 1.50},
    "fast_k5": {"kind": "absolute", "value": 16.50},
    "fast_k4": {"kind": "absolute", "value": 15.50},
}


def _smi() -> dict[str, Any]:
    p = subprocess.run([
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=False)
    if p.returncode:
        return {"error": (p.stderr or p.stdout).strip()}
    v = [x.strip() for x in (p.stdout or "").splitlines()[0].split(",")]
    return {
        "memory_used_mib": int(v[0]), "utilization_percent": int(v[1]),
        "sm_mhz": float(v[2]), "mem_mhz": float(v[3]), "power_w": float(v[4]),
        "temperature_c": float(v[5]), "pstate": v[6],
    }


def _preheat(rt, prompt_ids: list[int], count: int) -> None:
    _reset_exact_state(rt)
    _prefill(rt, prompt_ids)
    for _ in range(count):
        rt.step_graph(None)
    rt._graph_stream.synchronize()


def _arm(rt, prompts: list[dict[str, Any]], n: int, label: str) -> dict[str, Any]:
    ids_by, samples = {}, []
    before = _smi()
    for p in prompts:
        ids, ms = _run(rt, p["prompt_ids"], n)
        ids_by[p["prompt"]] = ids
        samples.extend(ms)
    rt._graph_stream.synchronize()
    return {"label": label, "ids": ids_by, "timing": percentiles(samples),
            "smi_before": before, "smi_after": _smi()}


def _divergence(a, b):
    return {name: first_divergence(a[name], b[name]) for name in a}


def _information_gate(profile: str, gain: float, candidate_ms: float):
    rule = GAIN_GATES[profile]
    if rule["kind"] == "gain":
        passed = gain >= float(rule["value"])
        return passed, {"kind": "gain_ms_at_least", "threshold": float(rule["value"]), "observed": gain}
    passed = candidate_ms <= float(rule["value"])
    return passed, {"kind": "candidate_ms_at_most", "threshold": float(rule["value"]), "observed": candidate_ms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--profile", choices=TIMING_PROFILES, required=True)
    args = ap.parse_args()
    out = REPO / "pro_research" / "results" / f"S100_PHASE3_TIMING_{args.profile.upper()}.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase3_timing", "status": "started", "profile": args.profile,
        "mode": args.mode, "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": "matched full V18 graph timing for an approximate profile; no fidelity, external quality or S100 claim",
    }
    bundle = None
    active_restore_combined = None
    try:
        payload["gpu_idle_preflight"] = _require_gpu_idle_wddm()
        import cupy as cp
        from graph_e1f22 import _load_prompt_set

        prompts, _expected, n, capacity = _load_prompt_set(args.mode)
        n = min(int(n), 32) if args.mode == "smoke" else max(int(n), 256)
        preheat = 48 if args.mode == "smoke" else 128
        payload["config"] = {"tokens_per_prompt": n, "prompt_count": len(prompts),
                             "capacity": int(capacity), "preheat_tokens": preheat}
        payload["environment_start"] = environment_snapshot((
            Path(__file__), PREREG,
            REPO / "pro_research" / "s100_phase3_profiles.py",
            REPO / "pro_research" / "s100_phase3_runtime.py",
            REPO / "pro_research" / "d4_mixed_runtime.py",
            REPO / "pro_research" / "moe_dev_combined.py",
        ))

        bundle = build_v18_runtime(int(capacity), profile=None)
        rt = bundle.rt
        active_restore_combined = bundle.restore_combined
        _preheat(rt, prompts[0]["prompt_ids"], preheat)
        base_a = _arm(rt, prompts, n, "BASE_A")

        # H-SCALE+B3 captures top_k in its closure. Reinstall for candidate.
        active_restore_combined(); active_restore_combined = None
        applied = apply_phase3_profile(rt, args.profile)
        if applied["top_k_changed"]:
            rebuild_cache(rt, int(capacity))
        cand_sres, active_restore_combined = install_combined_current(rt, bundle.down, bundle.up)
        _recapture(rt)
        installed_candidate_top_k = int(rt.top_k)
        attention_capture_calls = int(getattr(rt, "_d4_attention_capture_calls", 0))

        _preheat(rt, prompts[0]["prompt_ids"], preheat)
        cand_a = _arm(rt, prompts, n, "CAND_A")
        finite_a = bool(cp.isfinite(rt.logits).all().item())
        _preheat(rt, prompts[0]["prompt_ids"], preheat)
        cand_b = _arm(rt, prompts, n, "CAND_B")
        finite_b = bool(cp.isfinite(rt.logits).all().item())
        candidate_vram = int(_smi().get("memory_used_mib", 10**9))

        active_restore_combined(); active_restore_combined = None
        del cand_sres
        restore_phase3_profile(rt, applied)
        if applied["top_k_changed"]:
            rebuild_cache(rt, int(capacity))
        base_sres, active_restore_combined = install_combined_current(rt, bundle.down, bundle.up)
        _recapture(rt)
        installed_base_b_top_k = int(rt.top_k)
        _preheat(rt, prompts[0]["prompt_ids"], preheat)
        base_b = _arm(rt, prompts, n, "BASE_B")

        pa, pb = float(base_a["timing"]["p50"]), float(base_b["timing"]["p50"])
        pca, pcb = float(cand_a["timing"]["p50"]), float(cand_b["timing"]["p50"])
        base_mid, cand_mid = 0.5*(pa+pb), 0.5*(pca+pcb)
        base_drift, cand_drift = abs(pa-pb), abs(pca-pcb)
        saving = base_mid - cand_mid
        info_pass, info_rule = _information_gate(args.profile, saving, cand_mid)

        base_parity = _divergence(base_a["ids"], base_b["ids"])
        cand_parity = _divergence(cand_a["ids"], cand_b["ids"])
        cand_vs_base = _divergence(base_a["ids"], cand_a["ids"])
        public_applied = public_profile_record(applied)
        weights_changed = int((public_applied.get("weights") or {}).get("changed_matrix_count", 0))
        requires_q = args.profile in {"qfast", "fast_k5", "fast_k4"}
        expected_weight_count = 6 if args.profile == "qfast" else (52 if args.profile in {"fast_k5", "fast_k4"} else 0)

        gates = {
            "G1_baseline_A_B_token_parity": all(v is None for v in base_parity.values()),
            "G2_candidate_A_B_token_parity": all(v is None for v in cand_parity.values()),
            "G3_candidate_finite": finite_a and finite_b,
            "G4_expected_weight_count": weights_changed == expected_weight_count,
            "G5_candidate_top_k_exact": installed_candidate_top_k == int(applied["candidate_top_k"]),
            "G6_base_b_top_k_restored_6": installed_base_b_top_k == 6,
            "G7_attention_dispatch_exercised": (not requires_q) or attention_capture_calls >= len(rt.attn_layers),
            "M1_baseline_drift_le_1ms": base_drift <= 1.0,
            "M2_candidate_drift_le_1ms": cand_drift <= 1.0,
            "M3_candidate_vram_le_7987MiB": candidate_vram <= 7987,
            "M4_full_samples_ge_765": int(cand_a["timing"]["count"]) >= 765 if args.mode == "full" else True,
            "P1_profile_information_gate": info_pass,
        }
        instrument_ok = all(gates[k] for k in (
            "G1_baseline_A_B_token_parity", "G2_candidate_A_B_token_parity",
            "G3_candidate_finite", "G4_expected_weight_count", "G5_candidate_top_k_exact",
            "G6_base_b_top_k_restored_6", "G7_attention_dispatch_exercised"))
        measurement_ok = all(gates[k] for k in (
            "M1_baseline_drift_le_1ms", "M2_candidate_drift_le_1ms",
            "M3_candidate_vram_le_7987MiB", "M4_full_samples_ge_765"))
        status = "instrument_failed" if not instrument_ok else (
            "measurement_unstable" if not measurement_ok else (
                "timing_candidate" if info_pass else "timing_below_gate"))

        payload.update({
            "profile_install": public_applied,
            "candidate_combined_closure_top_k": installed_candidate_top_k,
            "base_b_combined_closure_top_k": installed_base_b_top_k,
            "attention_mixed_capture_calls": attention_capture_calls,
            "arms": {"BASE_A": base_a, "CAND_A": cand_a, "CAND_B": cand_b, "BASE_B": base_b},
            "divergence": {"baseline_A_B": base_parity, "candidate_A_B": cand_parity,
                           "candidate_vs_baseline_report_only": cand_vs_base},
            "summary": {
                "base_a_p50_ms": pa, "base_b_p50_ms": pb, "baseline_midpoint_ms": base_mid,
                "candidate_a_p50_ms": pca, "candidate_b_p50_ms": pcb,
                "candidate_midpoint_ms": cand_mid, "baseline_drift_ms": base_drift,
                "candidate_drift_ms": cand_drift, "saving_ms_per_token": saving,
                "speedup": base_mid/cand_mid if cand_mid else None,
                "candidate_tok_s": 1000.0/cand_mid if cand_mid else None,
                "remaining_ms_to_s100": cand_mid-10.0,
                "candidate_vram_mib": candidate_vram, "information_gate": info_rule,
            },
            "gates": gates, "status": status, "environment_end": environment_snapshot(),
            "completed_utc": utc_now(),
        })
        active_restore_combined(); active_restore_combined = None
        del base_sres
        bundle.restore_selective()
        del rt, bundle
        gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        payload.update({"status": "technical_failure", "completed_utc": utc_now(),
                        "error": {"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()}})
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out, payload, archive=True)
    print(json.dumps({"status": payload.get("status"), "profile": args.profile,
                      "summary": payload.get("summary"), "gates": payload.get("gates"),
                      "divergence": payload.get("divergence"),
                      "error": (payload.get("error") or {}).get("message"),
                      "output": str(out)}, indent=2, allow_nan=False))
    return 2 if payload.get("status") in {"technical_failure", "instrument_failed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
