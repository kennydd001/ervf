"""Independent post-V6 verifier. It imports no experimental runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "pro_max_v2"
OUT = RESULTS / "PV2_VERIFICATION.json"


def read(name: str) -> dict[str, Any] | None:
    p = RESULTS / name
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8"))


def all_identical(section: dict[str, Any] | None) -> bool:
    if not section: return False
    return all(bool(x.get("identical")) for x in section.values())


def candidate(name: str, prefix: str) -> dict[str, Any]:
    d = read(name)
    if d is None: return {"status": "missing"}
    m, s = d.get("micro", {}), d.get("summary", {})
    pa, pc, pb = s.get("base_a_p50_ms"), s.get("candidate_p50_ms"), s.get("base_b_p50_ms")
    if None in (pa, pc, pb): return {"status": "unverifiable", "source_status": d.get("status")}
    mid, drift = (float(pa) + float(pb)) / 2.0, abs(float(pa) - float(pb))
    par = d.get("parity", {})
    exact_micro = False
    if prefix == "addnorm":
        exact_micro = bool(m.get("exact_hidden") and m.get("exact_normed"))
    elif prefix == "qkv":
        exact_micro = bool(m.get("exact") and all(m["exact"].values()))
    else:
        exact_micro = bool(m.get("all_logits_bitexact") and m.get("top1_exact"))
    causal = all_identical(par.get("candidate_vs_base_a")) and all_identical(par.get("candidate_vs_base_b"))
    graph = d.get("graph", {})
    sample_count = int(d.get("arms", {}).get({"addnorm":"ADDNORM","qkv":"QKV","lmhead":"LMHEAD_ARGMAX"}[prefix], {}).get("timing_ms", {}).get("count", 0))
    recomputed = {
        "exact_micro": exact_micro,
        "micro_speedup_ge_1_02": float(m.get("speedup_p50", 0)) >= 1.02,
        "graph_candidate_present": bool(graph.get("candidate_name_present")),
        "extra_vram_lt_64MiB": int(graph.get("extra_vram_bytes", 1 << 60)) < 64 * 1024 * 1024,
        "causal_parity": causal,
        "base_drift_le_1ms": drift <= 1.0,
        "no_regression_gt_0_2pct": float(pc) <= mid * 1.002,
        "full_samples_gate": True if d.get("mode") != "full" else sample_count >= 500,
    }
    expected_adopt = all(recomputed.values())
    return {"status": "verified" if expected_adopt == bool(d.get("adopt")) else "mismatch",
            "source_status": d.get("status"), "expected_adopt": expected_adopt,
            "recorded_adopt": bool(d.get("adopt")), "recomputed": recomputed,
            "p50_ms": float(pc), "tok_s": 1000.0 / float(pc)}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "kind": "pv2_independent_verification",
        "candidates": {
            "addnorm": candidate("PV2_10_ADDNORM.json", "addnorm"),
            "qkv": candidate("PV2_11_QKV.json", "qkv"),
            "lmhead_argmax": candidate("PV2_12_LMHEAD_ARGMAX.json", "lmhead"),
        },
    }
    final = read("PV2_13_FINALE.json")
    if final is None:
        out["finale"] = {"status": "missing"}
    else:
        s = final.get("summary", {})
        pc = s.get("v10_p50_ms")
        parity = final.get("parity", {})
        gates = {
            "causal": all_identical(parity.get("v10_vs_base_a")) and all_identical(parity.get("v10_vs_base_b")),
            "deterministic": all_identical(parity.get("determinism")),
            "control_diverges": any(not x.get("identical", True) for x in parity.get("control_vs_base_a", {}).values()),
            "drift_le_1ms": float(s.get("base_drift_ms", 999)) <= 1.0,
            "p50_present": pc is not None,
        }
        out["finale"] = {
            "status": "verified" if all(gates.values()) else "failed",
            "gates": gates, "p50_ms": pc,
            "tok_s": None if pc is None else 1000.0 / float(pc),
            "E50": False if pc is None else float(pc) <= 20.0,
            "E75": False if pc is None else float(pc) <= 1000.0 / 75.0,
            "E100_single": False if pc is None else float(pc) <= 10.0,
        }
    epoch = read("PV2_20_CHILD_EPOCH.json")
    if epoch is None:
        out["child_epoch"] = {"status": "missing"}
    else:
        checks = {}
        for k, rec in epoch.get("epochs", {}).items():
            if rec.get("status") != "measured":
                checks[k] = {"status": rec.get("status")}; continue
            ss, ps = rec["separate_per_token_ms"]["p50"], rec["parent_per_token_ms"]["p50"]
            speed = float(ss) / float(ps)
            checks[k] = {"status": "verified" if rec.get("identical") else "failed",
                         "identical": bool(rec.get("identical")),
                         "speedup_recomputed": speed,
                         "speedup_matches": abs(speed - float(rec.get("speedup_p50", 0))) < 1e-9}
        out["child_epoch"] = {"source_status": epoch.get("status"), "epochs": checks}
    bad = []
    for group in (out["candidates"],):
        for name, rec in group.items():
            if rec.get("status") == "mismatch": bad.append(name)
    if out["finale"].get("status") == "failed": bad.append("finale")
    out["verdict"] = "verified_no_internal_mismatch" if not bad else "verification_failed"
    out["failures"] = bad
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": out["verdict"], "output": str(OUT)}, indent=2))
    return 0 if not bad else 2

if __name__ == "__main__":
    raise SystemExit(main())
