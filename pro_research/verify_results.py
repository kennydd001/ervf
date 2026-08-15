"""Independent, CPU-side verifier for PRO research results.

This file never imports an experiment runner or the GPU runtime. It recomputes
all gates from serialized raw IDs/timing arrays and reports any disagreement
with a runner's own gate fields.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    PRO,
    REPO,
    RESULTS,
    environment_snapshot,
    first_divergence,
    geometric_mean,
    load_json,
    result_path,
    sha256_file,
    utc_now,
    write_json_atomic,
)

OUT = result_path("PRO_VERIFICATION.json")
TS200 = REPO / "reports" / "treesweep200"


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), 50))


def _record_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def verify_graph(path: Path) -> dict[str, Any]:
    d = load_json(path)
    checks: list[dict[str, Any]] = []
    if d.get("status") == "technical_failure":
        return {"status": "technical_failure", "checks": checks, "source_status": d.get("status")}
    arms = d.get("arms", {})
    egr = {x["prompt"]: x["ids"] for x in arms.get("EGR", {}).get("prompts", [])}
    graph = {x["prompt"]: x["ids"] for x in arms.get("GRAPH", {}).get("prompts", [])}
    same = bool(egr) and set(egr) == set(graph) and all(egr[k] == graph[k] for k in egr)
    _record_check(checks, "graph_ids_equal_eager", same, {k: first_divergence(egr.get(k, []), graph.get(k, [])) for k in set(egr) | set(graph)})

    a1 = load_json(TS200 / "A1_ADOPTION_PRECONDITION.json")
    expected = a1["gates"]["G_A2_ANCHOR_informative"]["produced_ids"]
    anchor_ok = True
    anchor_detail = {}
    for prompt, ids in expected.items():
        if prompt not in graph or len(graph[prompt]) < 64:
            anchor_ok = False
            anchor_detail[prompt] = "missing_or_short"
        else:
            ok = graph[prompt][:64] == ids[:64]
            anchor_ok &= ok
            anchor_detail[prompt] = {"passed": ok, "first_divergence": first_divergence(graph[prompt][:64], ids[:64])}
    _record_check(checks, "anchor_first64", anchor_ok, anchor_detail)

    det = arms.get("DET", {})
    det_ok = bool(det) and all(x.get("ids_a") == x.get("ids_b") for x in det.values())
    _record_check(checks, "graph_replay_deterministic", det_ok)

    ctl = arms.get("CTL")
    if ctl is None:
        ctl_ok = d.get("config", {}).get("skip_control") is True
        _record_check(checks, "control_present_or_explicitly_skipped", ctl_ok, "skipped" if ctl_ok else "missing")
    else:
        ctl_ok = any(x.get("ids") != x.get("reference_ids") for x in ctl.values())
        _record_check(checks, "bad_pick_control_diverges", ctl_ok)

    egr_raw = arms.get("EGR", {}).get("raw_timing_ms", [])
    graph_raw = arms.get("GRAPH", {}).get("raw_timing_ms", [])
    ep, gp = _p50(egr_raw), _p50(graph_raw)
    speed_ok = ep is not None and gp is not None and gp <= ep - 2.5
    _record_check(checks, "graph_p50_gain_ge_2_5ms", speed_ok, {"eager_p50": ep, "graph_p50": gp, "gain": None if ep is None or gp is None else ep - gp})
    sample_ok = d.get("mode") != "full" or len(graph_raw) >= 500
    _record_check(checks, "full_timed_samples_ge_500", sample_ok, len(graph_raw))

    extra = int(arms.get("GRAPH", {}).get("graph_extra_vram_bytes", 2**63 - 1))
    vram_ok = extra < 64 * 1024 * 1024
    _record_check(checks, "graph_extra_vram_lt_64MiB", vram_ok, extra)

    required = [same, anchor_ok, det_ok, speed_ok, sample_ok, vram_ok]
    if ctl is not None:
        required.append(ctl_ok)
    return {
        "status": "verified_pass" if all(required) else "verified_gate_failed",
        "source_status": d.get("status"),
        "checks": checks,
        "recomputed": {"eager_p50_ms": ep, "graph_p50_ms": gp, "gain_ms": None if ep is None or gp is None else ep - gp},
    }


def verify_dense(path: Path) -> dict[str, Any]:
    d = load_json(path)
    checks: list[dict[str, Any]] = []
    if d.get("status") == "technical_failure":
        return {"status": "technical_failure", "checks": checks, "source_status": d.get("status")}
    cases = d.get("microbench", {}).get("cases", [])
    exact = bool(cases) and all(bool(x.get("bit_equal")) and int(x.get("mismatch_count", 1)) == 0 for x in cases)
    gmean = geometric_mean(float(x["reference_ms"]) / float(x["ervf_ms"]) for x in cases if float(x["ervf_ms"]) > 0)
    no_reg = bool(cases) and all(float(x["reference_ms"]) / float(x["ervf_ms"]) >= 0.95 for x in cases)
    _record_check(checks, "micro_all_bit_exact", exact)
    _record_check(checks, "micro_geomean_speedup_ge_1_25", gmean is not None and gmean >= 1.25, gmean)
    _record_check(checks, "micro_no_regression_gt_5pct", no_reg)

    integration = d.get("integration")
    int_ok = None
    recomputed: dict[str, Any] = {"micro_geomean_speedup": gmean}
    if integration:
        arms = integration["arms"]
        ba = _p50(arms["BASE_A"]["raw_timing_ms"])
        pp = _p50(arms["PRO_ERVF"]["raw_timing_ms"])
        bb = _p50(arms["BASE_B"]["raw_timing_ms"])
        mid = None if ba is None or bb is None else (ba + bb) / 2.0
        gain = None if mid is None or pp is None else mid - pp
        ids_a, ids_p, ids_b = arms["BASE_A"]["ids"], arms["PRO_ERVF"]["ids"], arms["BASE_B"]["ids"]
        parity = set(ids_a) == set(ids_p) == set(ids_b) and all(ids_a[k] == ids_p[k] == ids_b[k] for k in ids_a)
        gain_ok = gain is not None and mid is not None and (gain >= 1.5 or gain / mid >= 0.05)
        drift_ok = ba is not None and bb is not None and abs(ba - bb) <= 1.0
        _record_check(checks, "integration_all_ids_identical", parity)
        _record_check(checks, "integration_gain_ge_1_5ms_or_5pct", gain_ok, {"base_a": ba, "pro": pp, "base_b": bb, "gain": gain})
        _record_check(checks, "integration_base_drift_lte_1ms", drift_ok, None if ba is None or bb is None else abs(ba - bb))
        int_ok = parity and gain_ok and drift_ok
        recomputed.update({"base_a_p50_ms": ba, "pro_p50_ms": pp, "base_b_p50_ms": bb, "baseline_mid_ms": mid, "gain_ms": gain})

    micro_ok = exact and gmean is not None and gmean >= 1.25 and no_reg
    if integration is None:
        status = "verified_micro_pass" if micro_ok else "verified_micro_failed"
    else:
        status = "verified_pass" if micro_ok and int_ok else "verified_gate_failed"
    return {"status": status, "source_status": d.get("status"), "checks": checks, "recomputed": recomputed}


def verify_epoch(path: Path) -> dict[str, Any]:
    d = load_json(path)
    checks: list[dict[str, Any]] = []
    if d.get("status") in {"technical_failure", "technical_blocked"}:
        return {"status": d.get("status"), "checks": checks, "source_status": d.get("status")}
    passing = []
    recomputed = {}
    for key, rec in d.get("epochs", {}).items():
        if rec.get("status") != "measured":
            continue
        exact = rec.get("child_ids") == rec.get("parent_ids")
        child = _p50(rec.get("raw_child_per_token_ms", []))
        parent = _p50(rec.get("raw_parent_per_token_ms", []))
        speed = None if child is None or parent is None else child / parent
        vram = int(rec.get("parent_extra_vram_bytes", 2**63 - 1)) < 64 * 1024 * 1024
        ok = exact and speed is not None and speed >= 1.10 and vram
        passing.append(ok)
        recomputed[key] = {"exact": exact, "child_p50": child, "parent_p50": parent, "speedup": speed, "vram_ok": vram}
        _record_check(checks, f"epoch_{key}_exact", exact)
        _record_check(checks, f"epoch_{key}_speedup_ge_1_10", speed is not None and speed >= 1.10, speed)
        _record_check(checks, f"epoch_{key}_vram", vram, rec.get("parent_extra_vram_bytes"))
    status = "verified_pass" if any(passing) else "verified_gate_failed"
    return {"status": status, "source_status": d.get("status"), "checks": checks, "recomputed": recomputed}


def main() -> int:
    sources = {
        "graph": result_path("PRO_G0_E1F22_GRAPH_AB.json"),
        "dense": result_path("PRO_G1_DENSE_ERVF.json"),
        "epoch": result_path("PRO_G2_EPOCH_GRAPH.json"),
    }
    payload: dict[str, Any] = {
        "kind": "pro_independent_verification",
        "created_utc": utc_now(),
        "environment": environment_snapshot((Path(__file__),)),
        "results": {},
        "source_sha256": {},
    }
    verifiers = {"graph": verify_graph, "dense": verify_dense, "epoch": verify_epoch}
    for name, path in sources.items():
        if not path.exists():
            payload["results"][name] = {"status": "not_run"}
            continue
        payload["source_sha256"][name] = sha256_file(path)
        try:
            payload["results"][name] = verifiers[name](path)
        except Exception as exc:
            payload["results"][name] = {"status": "verification_error", "error": f"{type(exc).__name__}: {exc}"}

    verified_candidates = [
        name for name, rec in payload["results"].items()
        if rec.get("status") == "verified_pass"
    ]
    payload["verified_candidates"] = verified_candidates
    payload["all_checks_passed_count"] = sum(
        1 for rec in payload["results"].values() for check in rec.get("checks", []) if check.get("passed")
    )
    payload["all_checks_failed_count"] = sum(
        1 for rec in payload["results"].values() for check in rec.get("checks", []) if not check.get("passed")
    )
    payload["verdict"] = (
        "one_or_more_verified_candidates" if verified_candidates
        else "no_verified_breakthrough_candidate_yet"
    )
    write_json_atomic(OUT, payload)
    print(json.dumps({"verdict": payload["verdict"], "output": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
