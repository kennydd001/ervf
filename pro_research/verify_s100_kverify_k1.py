"""Independent CPU verifier for S100-KVERIFY K1 rollback evidence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "pro_research" / "results" / "s100_kverify" / "PRO_S100_KVERIFY_K1_MAMBA_ROLLBACK.json"
OUT = ROOT / "pro_research" / "results" / "s100_kverify" / "PRO_S100_KVERIFY_K1_VERIFICATION.json"


def main() -> int:
    if not SRC.exists():
        print(f"missing result: {SRC}")
        return 2
    p = json.loads(SRC.read_text(encoding="utf-8"))
    errors = []
    prefixes = p.get("prefix_results") or []
    expected_js = list(range(int(p.get("k", -1)) + 1))
    got_js = [int(x.get("accepted_prefix", -99)) for x in prefixes]
    if got_js != expected_js:
        errors.append(f"prefix cardinality/order mismatch: got={got_js} expected={expected_js}")
    all_exact = bool(prefixes) and all(
        int(x.get("conv_mismatch_count", -1)) == 0
        and int(x.get("ssm_mismatch_count", -1)) == 0
        and bool(x.get("bit_exact"))
        for x in prefixes
    )
    sab = p.get("sabotage") or {}
    sabotage_diverged = bool(sab.get("diverged")) and (
        int(sab.get("conv_mismatch_count", 0)) != 0
        or int(sab.get("ssm_mismatch_count", 0)) != 0
    )
    expected_status = "rollback_exact" if all_exact and sabotage_diverged else "correctness_failed"
    if p.get("status") != expected_status:
        errors.append(f"status mismatch file={p.get('status')} recomputed={expected_status}")
    cfg = p.get("config") or {}
    if int(cfg.get("stored_proj_bytes", 0)) <= 0:
        errors.append("stored proj evidence missing/zero")
    if int(cfg.get("one_layer_snapshot_bytes", 0)) <= 0:
        errors.append("state snapshot byte evidence missing/zero")

    out = {
        "kind": "pro_s100_kverify_k1_independent_verification",
        "source": str(SRC.relative_to(ROOT)),
        "source_status": p.get("status"),
        "recomputed_status": expected_status,
        "all_prefix_states_bit_exact": all_exact,
        "sabotage_diverged": sabotage_diverged,
        "errors": errors,
        "passed": not errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(json.dumps(out, indent=2))
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
