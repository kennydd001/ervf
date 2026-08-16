"""Independent CPU-side verifier for C3A real-weight result.

This verifier intentionally does not import the diagnostic module. It re-reads
checkpoint bytes, recomputes SHA256 values and recomputes the stdlib dequantized
reference samples recorded by C3A.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "nemotron_3_5_lightning_v35"
INDEX = MODEL_DIR / "model.safetensors.index.json"
RESULT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_REAL_WEIGHT.json"

NRMSE_MAX = 0.020
COSINE_MIN = 0.9990
NMAX_MAX = 0.050
M8_ROW_NMAX_MAX = 0.005
COLD_L2_MULTIPLE = 4.0
M8_OVER_M1_MAX = 1.15
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_index_headers() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    entries = idx["weight_map"]
    headers: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(entries.values())):
        p = MODEL_DIR / shard
        with p.open("rb") as fh:
            hlen = int.from_bytes(fh.read(8), "little")
            headers[shard] = json.loads(fh.read(hlen))
    return entries, headers


def tensor_raw(name: str, entries: dict[str, str], headers: dict[str, dict[str, Any]]) -> bytes:
    shard = entries[name]
    rec = headers[shard][name]
    a, b = (int(x) for x in rec["data_offsets"])
    p = MODEL_DIR / shard
    with p.open("rb") as fh:
        hlen = int.from_bytes(fh.read(8), "little")
        fh.seek(8 + hlen + a)
        raw = fh.read(b - a)
    if len(raw) != b - a:
        raise IOError(f"short read {name}")
    return raw


def e4m3(raw: int) -> float:
    sign = -1.0 if (raw >> 7) & 1 else 1.0
    exp = (raw >> 3) & 0xF
    man = raw & 0x7
    if exp == 0:
        return sign * (2.0 ** -6) * (man / 8.0)
    if exp == 0xF and man == 0x7:
        return math.nan
    return sign * (2.0 ** (exp - 7)) * (1.0 + man / 8.0)


def reference_row(weight_raw: bytes, scale_raw: bytes, global_scale: float,
                  row: int, n: int, k: int) -> float:
    packed_k = k // 2
    sfk = k // 16
    wb = memoryview(weight_raw)[row * packed_k:(row + 1) * packed_k]
    sb = memoryview(scale_raw)[row * sfk:(row + 1) * sfk]
    terms: list[float] = []
    for j, byte in enumerate(wb):
        s = e4m3(sb[(2 * j) // 16])
        terms.append(E2M1[byte & 0xF] * s * global_scale)
        terms.append(E2M1[(byte >> 4) & 0xF] * s * global_scale)
    return math.fsum(terms)


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def main() -> int:
    if not RESULT.exists():
        print(f"FAIL: missing {RESULT}")
        return 2
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    failures: list[str] = []
    entries, headers = load_index_headers()

    allowed_status = {
        "real_weight_representation_and_geometry_candidate",
        "real_weight_representation_green_perf_miss",
    }
    if data.get("status") not in allowed_status:
        failures.append(f"result status not correctness-green: {data.get('status')}")

    th = data.get("thresholds") or {}
    expected_thresholds = {
        "normalized_rmse_max": NRMSE_MAX,
        "cosine_min": COSINE_MIN,
        "normalized_max_abs_error_max": NMAX_MAX,
        "M8_identical_rows_normalized_max_diff_max": M8_ROW_NMAX_MAX,
        "cold_working_set_over_l2_min": COLD_L2_MULTIPLE,
        "M8_over_M1_p50_max": M8_OVER_M1_MAX,
    }
    for k, v in expected_thresholds.items():
        if k not in th or not close(th[k], v):
            failures.append(f"threshold drift {k}: got {th.get(k)} expected {v}")

    fams = data.get("families") or []
    if len(fams) != 4 or {f.get("label") for f in fams} != {"lm_head", "shared_up", "shared_down", "routed_up"}:
        failures.append("expected exactly four frozen representative labels")

    hashes_ok = True
    references_ok = True
    numerics_ok = True
    cold_honest_ok = True
    ratio_pass = 0
    measured = 0
    lm_ratio_ok = True

    for f in fams:
        label = f.get("label", "?")
        sel = f.get("selected") or {}
        try:
            wraw = tensor_raw(sel["weight"], entries, headers)
            sraw = tensor_raw(sel["scale"], entries, headers)
            graw = tensor_raw(sel["global"], entries, headers)
        except Exception as exc:
            failures.append(f"{label}: checkpoint reread failed: {exc}")
            hashes_ok = False
            continue
        p = f.get("payload") or {}
        if sha(wraw) != p.get("weight_sha256") or sha(sraw) != p.get("scale_sha256") or sha(graw) != p.get("global_sha256"):
            failures.append(f"{label}: checkpoint SHA256 mismatch")
            hashes_ok = False
        if len(graw) != 4:
            failures.append(f"{label}: global scale byte count != 4")
            references_ok = False
            continue
        g = float(struct.unpack("<f", graw)[0])
        if not close(g, p.get("global_scale_f32")):
            failures.append(f"{label}: global F32 value mismatch")
            references_ok = False

        rs = f.get("reference_samples") or {}
        rows = [int(x) for x in rs.get("rows") or []]
        vals = [float(x) for x in rs.get("values") or []]
        if not rows or len(rows) != len(vals):
            failures.append(f"{label}: malformed reference sample vectors")
            references_ok = False
        else:
            n, k = int(sel["N"]), int(sel["K"])
            fresh = [reference_row(wraw, sraw, g, r, n, k) for r in rows]
            for i, (a, b) in enumerate(zip(vals, fresh)):
                if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                    failures.append(f"{label}: independent reference mismatch at sample {i}, row {rows[i]}")
                    references_ok = False
                    break

        nat = f.get("native") or {}
        m1 = ((nat.get("M1") or {}).get("reference_metrics_first_row") or {})
        if not (
            (nat.get("M1") or {}).get("finite") is True
            and (nat.get("M8") or {}).get("finite") is True
            and float(m1.get("normalized_rmse", math.inf)) <= NRMSE_MAX
            and float(m1.get("cosine", -math.inf)) >= COSINE_MIN
            and float(m1.get("normalized_max_abs_error", math.inf)) <= NMAX_MAX
            and float(f.get("M8_identical_rows_normalized_max_diff", math.inf)) <= M8_ROW_NMAX_MAX
        ):
            failures.append(f"{label}: numerical gate failed")
            numerics_ok = False

        ct = f.get("cold_timing") or {}
        if ct.get("status") == "measured":
            measured += 1
            ws = float(ct.get("working_set_over_l2", 0.0))
            if ws < COLD_L2_MULTIPLE:
                failures.append(f"{label}: cold working set only {ws:.3f}x L2")
                cold_honest_ok = False
            ratio = float(ct.get("M8_over_M1", math.inf))
            if ratio <= M8_OVER_M1_MAX:
                ratio_pass += 1
            if label == "lm_head" and ratio > M8_OVER_M1_MAX:
                lm_ratio_ok = False
        elif label == "lm_head":
            # P3 is conditional if measured. P1/P2 remain fail-closed through counts.
            lm_ratio_ok = True

    gates = data.get("gates") or {}
    for k in (
        "C3A_G1_environment_and_api", "C3A_G2_C1_parent_green",
        "C3A_G3_four_real_triples_match_contract", "C3A_G4_two_level_known_value_smoke",
        "C3A_G5_real_M1_M8_execute_finite", "C3A_G6_reference_nrmse_and_cosine",
        "C3A_G7_reference_normalized_max_abs", "C3A_G8_M8_identical_rows_agree",
    ):
        if gates.get(k) is not True:
            failures.append(f"recorded correctness gate not green: {k}={gates.get(k)}")

    verifier_gates = {
        "C3A_G9_checkpoint_hashes_reverified": hashes_ok and references_ok,
        "independent_reference_recomputed": references_ok,
        "recorded_numerics_within_frozen_thresholds": numerics_ok,
        "cold_rotation_measured_sets_ge_4x_L2": cold_honest_ok and measured > 0,
        "M8_ratio_le_1_15_at_least_3_of_4": ratio_pass >= 3,
        "lm_head_ratio_le_1_15_if_measured": lm_ratio_ok,
    }
    if not verifier_gates["C3A_G9_checkpoint_hashes_reverified"]:
        failures.append("C3A_G9 independent checkpoint/reference re-verification failed")

    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "result_status": data.get("status"),
        "verifier_gates": verifier_gates,
        "measured_families": measured,
        "M8_ratio_pass_count": ratio_pass,
        "failures": failures,
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
