"""PV2-13: physically compose only preregistered, individually adopted candidates."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shared import (
    capture_v6, compare_arms, environment, graph_dot, load_json,
    new_v6_bundle, prompt_set, result_path, run_arm, status_from_gates,
    utc_now, write_json,
)
from addnorm_v7 import AddNorm, install as install_addnorm
from qkv_v8 import QKV, install as install_qkv
from lmhead_argmax_v9 import LMArgmax, install as install_lm

OUT = result_path("PV2_13_FINALE.json")
CANDIDATES = {
    "addnorm": result_path("PV2_10_ADDNORM.json"),
    "qkv": result_path("PV2_11_QKV.json"),
    "lmhead_argmax": result_path("PV2_12_LMHEAD_ARGMAX.json"),
}


def adopted() -> dict[str, dict[str, Any]]:
    out = {}
    for name, path in CANDIDATES.items():
        if not path.exists():
            continue
        d = load_json(path)
        if bool(d.get("adopt")):
            out[name] = {
                "source": str(path), "status": d.get("status"),
                "micro": d.get("micro"), "summary": d.get("summary"),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = ap.parse_args()
    payload: dict[str, Any] = {
        "kind": "pv2_finale", "status": "started", "mode": args.mode,
        "started_utc": utc_now(), "preregistration": "PREREGISTRATION.md",
    }
    bundle = None
    restores = []
    try:
        from shared import require_gpu_free
        require_gpu_free()
        prompts, _expected, n, _capacity = prompt_set(args.mode)
        selected = adopted()
        payload["selected_candidates"] = selected
        payload["environment"] = environment((Path(__file__), HERE / "PREREGISTRATION.md"))
        bundle = new_v6_bundle(); rt = bundle.rt

        capture_v6(bundle)
        base_a = run_arm(rt, prompts, n)

        objects: dict[str, Any] = {}
        if "addnorm" in selected:
            objects["addnorm"] = AddNorm()
            restores.append(install_addnorm(rt, objects["addnorm"]))
        if "qkv" in selected:
            objects["qkv"] = QKV()
            restores.append(install_qkv(rt, objects["qkv"]))
        if "lmhead_argmax" in selected:
            objects["lmhead_argmax"] = LMArgmax(rt.vocab)
            restores.append(install_lm(rt, objects["lmhead_argmax"]))

        extra = capture_v6(bundle)
        dot = graph_dot(rt).lower()
        candidate = run_arm(rt, prompts, n)
        det_a = run_arm(rt, prompts, min(n, 64 if args.mode == "full" else n))
        det_b = run_arm(rt, prompts, min(n, 64 if args.mode == "full" else n))

        # Existing sabotage control, captured under the exact same candidate stack.
        rt._bad_pick = 1
        capture_v6(bundle)
        control = run_arm(rt, prompts, min(n, 64))
        rt._bad_pick = 0

        for restore in reversed(restores):
            restore()
        restores.clear()
        capture_v6(bundle)
        base_b = run_arm(rt, prompts, n)

        par_a = compare_arms(base_a, candidate)
        par_b = compare_arms(base_b, candidate)
        det = compare_arms(det_a, det_b)
        ctl_reference = {"ids": {name: ids[:min(n, 64)] for name, ids in base_a["ids"].items()}}
        ctl = compare_arms(ctl_reference, control)
        pa = float(base_a["timing_ms"]["p50"])
        pc = float(candidate["timing_ms"]["p50"])
        pb = float(base_b["timing_ms"]["p50"])
        mid, drift = (pa + pb) / 2.0, abs(pa - pb)
        structural = {
            "addnorm": "pv2_add_rmsnorm_bf16w" in dot if "addnorm" in selected else None,
            "qkv": "pv2_qkv_mixed_fused" in dot if "qkv" in selected else None,
            "lmhead_main": "pv2_lmhead_ervf_block_argmax" in dot if "lmhead_argmax" in selected else None,
            "lmhead_final": "pv2_argmax_final_serial" in dot if "lmhead_argmax" in selected else None,
        }
        gates = {
            "at_least_one_candidate_selected": bool(selected),
            "all_selected_structurally_captured": all(v is not False for v in structural.values()),
            "causal_parity": all(x["identical"] for x in par_a.values()) and all(x["identical"] for x in par_b.values()),
            "deterministic": all(x["identical"] for x in det.values()),
            "bad_pick_control_diverges": any(not x["identical"] for x in ctl.values()),
            "base_drift_le_1ms": drift <= 1.0,
            "extra_vram_lt_64MiB": extra < 64 * 1024 * 1024,
            "samples_ge_500": int(candidate["timing_ms"]["count"]) >= 500 if args.mode == "full" else None,
        }
        required = ("at_least_one_candidate_selected",
                    "all_selected_structurally_captured", "causal_parity",
                    "deterministic", "bad_pick_control_diverges",
                    "base_drift_le_1ms", "extra_vram_lt_64MiB")
        if args.mode == "full": required += ("samples_ge_500",)
        milestones = {
            "E50_single_stream": pc <= 20.0,
            "E75_single_stream": pc <= 1000.0 / 75.0,
            "E100_single_stream": pc <= 10.0,
        }
        payload.update({
            "arms": {"BASE_A": base_a, "V10": candidate, "BASE_B": base_b,
                     "DET_A": det_a, "DET_B": det_b, "CONTROL": control},
            "parity": {"v10_vs_base_a": par_a, "v10_vs_base_b": par_b,
                       "determinism": det, "control_vs_base_a": ctl},
            "graph_structural": structural, "gates": gates,
            "milestones": milestones,
            "summary": {
                "base_a_p50_ms": pa, "v10_p50_ms": pc,
                "base_b_p50_ms": pb, "baseline_mid_p50_ms": mid,
                "gain_ms": mid - pc, "gain_fraction": (mid - pc) / mid,
                "v10_tok_s": 1000.0 / pc, "base_drift_ms": drift,
                "remaining_to_50_ms": pc - 20.0,
                "remaining_to_75_ms": pc - 1000.0 / 75.0,
                "remaining_to_100_ms": pc - 10.0,
            },
            "status": status_from_gates(gates, required),
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update(status="technical_failure", completed_utc=utc_now(),
                       error={"type": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
    finally:
        for restore in reversed(restores):
            try: restore()
            except Exception: pass
        if bundle is not None: bundle.close()
    write_json(OUT, payload)
    print(json.dumps({"status": payload.get("status"),
                      "selected": list(payload.get("selected_candidates", {})),
                      "summary": payload.get("summary"),
                      "milestones": payload.get("milestones"),
                      "output": str(OUT)}, indent=2))
    return 0 if payload.get("status") in {"pass", "gate_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
