"""Independent, dependency-light verifier for C3B result/capture integrity and frozen gates."""
from __future__ import annotations

import array
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CAPTURE = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_CAPTURE.json"
RESULT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_W4A4_REAL_ACT.json"
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3B_W4A4_VERIFY.json"

EXPECTED_THRESHOLDS = {
    "activation_cosine_min": 0.995, "activation_nrmse_max": 0.120,
    "output_cosine_min": 0.995, "output_nrmse_max": 0.100,
    "output_normalized_max_abs_max": 0.250, "lm_top1_min": 0.95,
    "lm_top5_overlap_min": 0.80, "lm_mean_kl_max": 0.020,
    "lm_max_kl_max": 0.100, "cold_l2_multiple_min": 4.0,
    "M8_total_over_M1_max_engineering_signal": 2.0,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def baseline_top1(path: Path, rows: int, cols: int) -> list[int]:
    vals = array.array("f")
    with path.open("rb") as fh:
        vals.fromfile(fh, rows * cols)
    if sys.byteorder != "little": vals.byteswap()
    out = []
    for r in range(rows):
        start = r * cols; best_i = 0; best_v = vals[start]
        for c in range(1, cols):
            v = vals[start + c]
            if v > best_v:
                best_v, best_i = v, c
        out.append(best_i)
    return out


def main() -> int:
    failures = []
    try:
        cap = json.loads(CAPTURE.read_text(encoding="utf-8"))
        res = json.loads(RESULT.read_text(encoding="utf-8"))
        if cap.get("status") != "capture_pass": failures.append(f"capture status={cap.get('status')}")
        rows = cap.get("rows") or []
        if len(rows) != 24: failures.append(f"capture rows={len(rows)} != 24")
        by_prompt = {}
        for r in rows:
            by_prompt.setdefault(int(r.get("prompt_index", -1)), []).append(r)
        if set(by_prompt) != {0, 1, 2} or any(
            len(v) != 8 or sorted(int(x.get("position", -1)) for x in v) != list(range(8))
            for v in by_prompt.values()
        ):
            failures.append("capture prompt/position partition is not exactly 3 x 8")
        expected_shapes = {
            "moe_normed": [24, 2688], "shared_up_ref": [24, 3712],
            "shared_down_input": [24, 3712], "shared_down_ref": [24, 2688],
            "routed_up_ref": [24, 1856], "lm_head_input": [24, 2688],
            "lm_head_ref": [24, 131072],
        }
        if set(cap.get("arrays") or {}) != set(expected_shapes):
            failures.append("capture array set mismatch")
        for name, e in (cap.get("arrays") or {}).items():
            if name in expected_shapes and [int(x) for x in e.get("shape", [])] != expected_shapes[name]:
                failures.append(f"capture shape mismatch: {name}={e.get('shape')}")
            p = REPO / e["path"]
            if not p.exists() or sha(p) != e.get("sha256"):
                failures.append(f"capture file hash mismatch: {name}")
            expected_bytes = math.prod(int(x) for x in e.get("shape", [])) * 4
            if p.exists() and p.stat().st_size != expected_bytes:
                failures.append(f"capture file size mismatch: {name}")
        if res.get("capture_manifest_sha256") != sha(CAPTURE):
            failures.append("capture manifest SHA mismatch")
        for k, v in EXPECTED_THRESHOLDS.items():
            if k not in (res.get("thresholds") or {}) or not close(res["thresholds"][k], v):
                failures.append(f"threshold drift {k}")
        gates = res.get("gates") or {}
        for i, suffix in enumerate(("capture_integrity", "parents_green", "activation_reuse_identity",
            "native_executes", "activation_quant_quality", "projection_quality", "projection_max_error",
            "lm_top1", "lm_top5_overlap", "lm_distribution"), 1):
            k = f"C3B_G{i}_{suffix}"
            if gates.get(k) is not True:
                failures.append(f"quality gate not green: {k}={gates.get(k)}")

        # Independently reconstruct baseline lm-head argmax from the captured W4A32 logits
        # and compare it to the reference ids recorded by each M arm.
        lm_e = cap["arrays"]["lm_head_ref"]
        base_ids = baseline_top1(REPO / lm_e["path"], int(lm_e["shape"][0]), int(lm_e["shape"][1]))
        lm = ((res.get("summary") or {}).get("lm_head_by_M") or {})
        for m in ("M1", "M2", "M4", "M8"):
            d = lm.get(m) or {}
            if [int(x) for x in d.get("reference_top1_ids", [])] != base_ids:
                failures.append(f"{m}: recorded reference top1 ids do not match captured W4A32 logits")
            nat = [int(x) for x in d.get("native_top1_ids", [])]
            if len(nat) != 24:
                failures.append(f"{m}: native top1 id count {len(nat)} !=24")
            agree = sum(a == b for a, b in zip(nat, base_ids))
            frac = agree / 24.0
            if not close(frac, d.get("top1_agreement_fraction", -1)):
                failures.append(f"{m}: top1 agreement arithmetic mismatch")

        fams = res.get("families") or {}
        if set(fams) != {"lm_head", "shared_up", "shared_down", "routed_up"}:
            failures.append("family set mismatch")
        for label, f in fams.items():
            for m in ("M1", "M2", "M4", "M8"):
                q = (f.get("quality_by_M") or {}).get(m) or {}
                if not q.get("groups") or not q.get("aggregate"):
                    failures.append(f"{label}/{m}: missing quality groups")
            p = f.get("performance") or {}
            if p.get("status") == "measured" and float(p.get("working_set_over_l2", 0)) < 4.0:
                failures.append(f"{label}: dishonest cold working set")

        status = "PASS" if not failures else "FAIL"
        out = {"kind": "s100_native_nvfp4_c3b_verification", "status": status,
               "result_status": res.get("status"), "quality_green": not failures,
               "baseline_lm_top1_ids": base_ids, "failures": failures}
    except Exception as exc:
        out = {"kind": "s100_native_nvfp4_c3b_verification", "status": "FAIL",
               "failures": [f"technical verifier failure: {type(exc).__name__}: {exc}"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
