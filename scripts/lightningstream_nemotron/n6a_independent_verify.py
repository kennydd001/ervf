"""Independent verifier for N6-A.

Separate program from the runner.  Re-tokenises the frozen prompts, re-decodes
the recorded top-1 ids, and re-audits the natural-route capture from the raw
arrays -- validity, uniqueness, weight sums and usage statistics.

It does NOT recompute the 52-layer forward: that costs ~90 s per prompt and
would re-run the same code rather than check it.  What it verifies is that the
recorded artifacts are internally consistent, that the coherence claim rests on
a correctly decoded token, and that the route capture downstream phases will
consume is well formed.

Imports nothing from the runner.  Opens no GPU.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
RESULT = OUT_DIR / "n6a_full_depth_forward.json"
ROUTES = OUT_DIR / "n6a_natural_routes.json"
OUT = OUT_DIR / "n6a_independent_verification.json"

EXPECTED_MOE_LAYERS = [1, 3, 6, 8, 10, 13, 15, 17, 20, 22, 24, 27, 29, 31, 34,
                       36, 38, 40, 43, 45, 47, 49, 51]
TOP_K = 6
N_EXPERTS = 128
SCALING = 2.5
GIB = 1024 ** 3


def sha256_path(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    from transformers import AutoTokenizer

    result = json.loads(RESULT.read_text(encoding="utf-8"))
    routes_doc = json.loads(ROUTES.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}

    # -- provenance ---------------------------------------------------------
    checks["phase_correct"] = result.get("phase") == "N6_A_FULL_DEPTH_FORWARD"
    checks["claim_boundary_present"] = bool(result.get("claim_boundary"))
    checks["no_tok_s_promotion"] = "never promoted to tok/s" in result["claim_boundary"]
    checks["gpu_not_used"] = result.get("gpu_used") is False
    checks["no_timing_claim"] = result.get("timing_claim") is False
    checks["no_bf16_model_materialised"] = result.get("bf16_model_materialised") is False
    runner = REPO_ROOT / "scripts/lightningstream_nemotron/n6a_full_depth_forward.py"
    checks["runner_hash_matches_recorded"] = sha256_path(runner) == result.get("runner_sha256")
    checks["fifty_two_layers"] = result.get("layers") == 52

    # -- tokenisation and decoding, independently ---------------------------
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    tokens_ok, decode_ok = True, True
    for pid, row in result["prompts"].items():
        if tok.encode(row["prompt"], add_special_tokens=False) != row["token_ids"]:
            tokens_ok = False
        if tok.decode([row["top1_id"]]) != row["top1_text"]:
            decode_ok = False
    checks["prompts_retokenise_identically"] = tokens_ok
    checks["top1_ids_redecode_identically"] = decode_ok

    primary = result["prompts"][result["primary_prompt"]]
    checks["primary_top1_contains_paris"] = (
        result["primary_expectation"] in primary["top1_text"].strip().lower())
    checks["primary_coherence_flag_consistent"] = result["primary_coherent"] is checks[
        "primary_top1_contains_paris"]
    detail["primary_top1"] = primary["top1_text"]
    detail["primary_top1_prob"] = primary["top1_prob"]

    # -- logits sanity ------------------------------------------------------
    checks["all_logits_finite_flagged"] = all(
        r["logits_finite"] and r["hidden_last_finite"] for r in result["prompts"].values())
    checks["vocab_size_correct"] = all(
        r["logits_shape"] == [131072] for r in result["prompts"].values())
    checks["top1_prob_in_unit_interval"] = all(
        0.0 < r["top1_prob"] <= 1.0 for r in result["prompts"].values())
    checks["entropy_non_negative"] = all(
        r["entropy_nats"] >= 0.0 for r in result["prompts"].values())
    checks["top5_logits_descending"] = all(
        all(r["top5"][i]["logit"] >= r["top5"][i + 1]["logit"] for i in range(4))
        for r in result["prompts"].values())
    checks["top1_matches_top5_head"] = all(
        r["top1_id"] == r["top5"][0]["id"] for r in result["prompts"].values())

    # -- natural route capture, audited from the raw arrays -----------------
    routes = routes_doc["routes"]
    checks["route_capture_covers_all_prompts"] = set(routes) == set(result["prompts"])
    checks["route_provenance_marked_non_representative"] = (
        "NOT a representative" in routes_doc["provenance"])

    layers_ok = ids_ok = unique_ok = sums_ok = margins_ok = True
    usage = Counter()
    total_rows = 0
    for pid, layers in routes.items():
        if sorted(int(k) for k in layers) != EXPECTED_MOE_LAYERS:
            layers_ok = False
        for lay in layers.values():
            for row in lay["indices"]:
                total_rows += 1
                if len(row) != TOP_K:
                    ids_ok = False
                if not all(0 <= int(e) < N_EXPERTS for e in row):
                    ids_ok = False
                if len(set(int(e) for e in row)) != TOP_K:
                    unique_ok = False
                usage.update(int(e) for e in row)
            for row in lay["weights"]:
                arr = np.asarray(row, dtype=np.float64)
                if not np.isfinite(arr).all() or (arr <= 0).any():
                    sums_ok = False
                if abs(float(arr.sum()) - SCALING) > 1e-6:
                    sums_ok = False
            if not np.isfinite(np.asarray(lay["tie_margin"], dtype=np.float64)).all():
                margins_ok = False

    checks["routes_cover_expected_moe_layers"] = layers_ok
    checks["route_ids_valid"] = ids_ok
    checks["route_ids_unique_within_token"] = unique_ok
    checks["route_weights_sum_to_scaling_factor"] = sums_ok
    checks["tie_margins_finite"] = margins_ok
    checks["route_rows_recomputed"] = total_rows > 0

    distinct = len(usage)
    detail["route_rows"] = total_rows
    detail["distinct_experts_used"] = distinct
    detail["distinct_expert_fraction"] = distinct / N_EXPERTS
    detail["most_common_experts"] = usage.most_common(5)
    detail["expert_usage_max"] = max(usage.values())
    detail["expert_usage_min"] = min(usage.values())
    # A degenerate router that always picked the same experts would be a red flag.
    checks["router_uses_many_distinct_experts"] = distinct >= 32

    # -- memory -------------------------------------------------------------
    proc = result["process_memory"]
    checks["process_memory_measured"] = "error" not in proc and proc.get("peak_commit_bytes", 0) > 0
    checks["process_commit_under_32gib"] = proc.get("peak_commit_bytes", 1 << 62) <= 32 * GIB
    detail["peak_commit_gib"] = round(proc.get("peak_commit_bytes", 0) / GIB, 4)

    # -- gates and terminal state -------------------------------------------
    gates = result["gates"]
    checks["gates_all_pass_consistent"] = result["gates_all_pass"] == all(gates.values())
    checks["terminal_state_consistent"] = result["terminal_state"] == "n6a_full_depth_coherent"
    checks["joint_confirmation_note_present"] = "JOINT" in result["settles"]["note"]

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    payload = {
        "kind": "lightningstream_nemotron_n6a_independent_verification",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N6_A_FULL_DEPTH_FORWARD",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_result_sha256": sha256_path(RESULT),
        "verified_routes_sha256": sha256_path(ROUTES),
        "gpu_opened": False,
        "runner_imported": False,
        "forward_recomputed": False,
        "forward_recompute_note": (
            "Re-running the forward would re-execute the same code rather than "
            "check it; what is verified here is artifact consistency, correct "
            "token decoding behind the coherence claim, and route well-formedness."),
        "checks": checks,
        "passed": passed,
        "total": total,
        "pass": passed == total,
        "detail": detail,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for name, value in checks.items():
        print(f"  {'OK  ' if value else 'FAIL'} {name}")
    print()
    print(f"primary top1        : {detail['primary_top1']!r} p={detail['primary_top1_prob']:.4f}")
    print(f"route rows          : {detail['route_rows']}")
    print(f"distinct experts    : {detail['distinct_experts_used']}/{N_EXPERTS} "
          f"({detail['distinct_expert_fraction'] * 100:.1f}%)")
    print(f"usage max/min       : {detail['expert_usage_max']}/{detail['expert_usage_min']}")
    print(f"peak commit         : {detail['peak_commit_gib']} GiB")
    print(f"\n{passed}/{total} verification checks passed")
    return 0 if payload["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
