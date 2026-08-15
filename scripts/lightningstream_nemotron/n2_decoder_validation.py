"""N2 step 10: bit-exact decoder validation for one quantized matrix.

Implements the four validation rules frozen in the N2 preregistration §4.  No
BF16 reference model is downloaded, so validation is self-consistency plus
conformance to the published NVFP4 semantics:

  1. range invariance   - every decoded value lies in the representable set
  2. round trip         - re-encoding reproduces the code bytes bit-exactly
  3. two implementations - table-driven and bit-arithmetic agree everywhere
  4. structural          - shapes, scale counts, finiteness

Only the byte ranges of the selected tensors are read; the shard is never loaded
whole and no BF16 model is materialized.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron import nvfp4  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT = REPO_ROOT / "reports" / "lightningstream_nemotron" / "n2_decoder_validation.json"

# Preregistered target: the first routed expert of the first MoE layer.
TARGET_LAYER = 1
TARGET_EXPERT = 0
MIN_SAMPLE_CODES = 1_048_576

HIDDEN = 2688
MOE_INTERMEDIATE = 1856


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        raw = handle.read(header_len)
    return json.loads(raw.decode("utf-8")), header_len


def read_tensor(path: Path, entry: dict, header_len: int) -> np.ndarray:
    """Read exactly one tensor's byte range."""
    start, end = entry["data_offsets"]
    base = 8 + header_len
    with path.open("rb") as handle:
        handle.seek(base + start)
        raw = handle.read(end - start)
    if len(raw) != end - start:
        raise IOError(f"short read: {len(raw)} != {end - start}")
    return np.frombuffer(raw, dtype=np.uint8)


def read_f32_scalar(path: Path, entry: dict, header_len: int) -> float:
    raw = read_tensor(path, entry, header_len)
    return float(np.frombuffer(raw.tobytes(), dtype="<f4")[0])


def validate_matrix(name: str, codes_raw: np.ndarray, scales_raw: np.ndarray,
                    weight_scale_2: float, input_scale: float,
                    rows: int, cols: int) -> dict:
    n_weights = rows * cols
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}

    # -- rule 4: structural -------------------------------------------------
    checks["code_bytes_equal_half_n"] = codes_raw.size == n_weights // 2
    checks["scale_count_equal_n_over_group"] = scales_raw.size == n_weights // nvfp4.GROUP_SIZE
    checks["weight_scale_2_finite"] = bool(np.isfinite(weight_scale_2))
    checks["input_scale_finite"] = bool(np.isfinite(input_scale))

    scales_table = nvfp4.decode_e4m3_table(scales_raw)
    scales_bits = nvfp4.decode_e4m3_bits(scales_raw)
    checks["no_nan_block_scales"] = not bool(np.isnan(scales_table).any())
    checks["block_scales_finite"] = bool(np.isfinite(scales_table).all())
    detail["block_scale_min"] = float(np.nanmin(scales_table))
    detail["block_scale_max"] = float(np.nanmax(scales_table))
    detail["weight_scale_2"] = weight_scale_2
    detail["input_scale"] = input_scale

    # -- rule 3: two implementations agree ---------------------------------
    codes = nvfp4.unpack_nibbles(codes_raw)
    elements_table = nvfp4.decode_e2m1_table(codes)
    elements_bits = nvfp4.decode_e2m1_bits(codes)
    checks["e2m1_implementations_agree"] = bool(np.array_equal(elements_table, elements_bits))
    checks["e4m3_implementations_agree"] = bool(np.array_equal(scales_table, scales_bits))

    dequant_table = nvfp4.dequantize(codes_raw, scales_raw, weight_scale_2,
                                     implementation="table")
    dequant_bits = nvfp4.dequantize(codes_raw, scales_raw, weight_scale_2,
                                    implementation="bits")
    checks["dequant_implementations_agree"] = bool(np.array_equal(dequant_table, dequant_bits))
    checks["dequant_all_finite"] = bool(np.isfinite(dequant_table).all())

    # -- rule 1: range invariance ------------------------------------------
    # Every decoded element must be an E2M1 magnitude; every dequantized value
    # must equal element * block_scale * global_scale by construction, so the
    # decisive test is that the elements themselves are in the closed set.
    magnitudes = np.unique(np.abs(elements_table))
    allowed = np.array(nvfp4.E2M1_MAGNITUDES, dtype=np.float64)
    checks["elements_in_representable_set"] = bool(np.isin(magnitudes, allowed).all())
    detail["distinct_element_magnitudes"] = magnitudes.tolist()
    detail["code_histogram"] = np.bincount(codes, minlength=16).tolist()

    # -- rule 2: bit-exact round trip --------------------------------------
    checks["sample_meets_minimum"] = codes.size >= MIN_SAMPLE_CODES
    recoded = nvfp4.encode_e2m1(elements_table)
    # Code 8 is negative zero and collapses onto 0; treat both as a match.
    normalized = np.where(codes == 8, 0, codes).astype(np.uint8)
    checks["code_round_trip_bit_exact"] = bool(np.array_equal(recoded, normalized))
    detail["codes_examined"] = int(codes.size)
    detail["negative_zero_codes"] = int((codes == 8).sum())

    repacked = nvfp4.pack_nibbles(recoded)
    without_negzero = np.where(codes == 8, 0, codes).astype(np.uint8)
    expected_bytes = nvfp4.pack_nibbles(without_negzero)
    checks["packed_bytes_round_trip_bit_exact"] = bool(np.array_equal(repacked, expected_bytes))

    return {
        "tensor": name,
        "rows": rows,
        "cols_logical": cols,
        "n_weights": n_weights,
        "checks": checks,
        "pass": all(checks.values()),
        "detail": detail,
    }


def main() -> int:
    index_path = MODEL_DIR / "model.safetensors.index.json"
    if not index_path.is_file():
        print("index missing")
        return 3
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    prefix = f"backbone.layers.{TARGET_LAYER}.mixer.experts.{TARGET_EXPERT}"
    wanted = {
        "up": (f"{prefix}.up_proj", MOE_INTERMEDIATE, HIDDEN),
        "down": (f"{prefix}.down_proj", HIDDEN, MOE_INTERMEDIATE),
    }

    headers: dict[str, tuple[dict, int]] = {}
    results = []
    for label, (base, rows, cols) in wanted.items():
        shard = weight_map[f"{base}.weight"]
        path = MODEL_DIR / shard
        if not path.is_file():
            print(f"shard {shard} not present yet")
            return 3
        if shard not in headers:
            headers[shard] = load_header(path)
        header, header_len = headers[shard]

        codes_raw = read_tensor(path, header[f"{base}.weight"], header_len)
        scales_raw = read_tensor(path, header[f"{base}.weight_scale"], header_len)
        ws2 = read_f32_scalar(path, header[f"{base}.weight_scale_2"], header_len)
        ins = read_f32_scalar(path, header[f"{base}.input_scale"], header_len)

        results.append(validate_matrix(base, codes_raw, scales_raw, ws2, ins, rows, cols))
        print(f"{label:<5} {base}: pass={results[-1]['pass']} "
              f"codes={results[-1]['detail']['codes_examined']:,}")

    all_pass = all(r["pass"] for r in results)
    result = {
        "kind": "lightningstream_nemotron_n2_decoder_validation",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS",
        "completed_utc": utc_now(),
        "runner_sha256": sha256_path(Path(__file__)),
        "codec_source_sha256": sha256_path(
            REPO_ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / "nvfp4.py"
        ),
        "target": {"layer": TARGET_LAYER, "expert": TARGET_EXPERT},
        "minimum_sample_codes": MIN_SAMPLE_CODES,
        "assumptions_recorded_not_proven": {
            "nibble_order": nvfp4.DEFAULT_NIBBLE_ORDER,
            "dequant_grouping": "w = e2m1(code) * e4m3(weight_scale) * f32(weight_scale_2)",
            "note": (
                "Nibble order and the role of input_scale versus weight_scale_2 "
                "are conventions taken from the published format. They are not "
                "falsifiable by self-consistency alone and must be confirmed in "
                "N3 against the official modeling code and one real forward."
            ),
        },
        "matrices": results,
        "all_pass": all_pass,
        "bf16_model_materialized": False,
        "gpu_used": False,
        "claim_boundary": (
            "One routed expert's two matrices decode self-consistently under "
            "published NVFP4 semantics. No model quality, latency or throughput "
            "claim, and no proof that these values equal the publisher's BF16 "
            "source weights."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print()
    for row in results:
        print(f"--- {row['tensor']} ---")
        for key, value in row["checks"].items():
            print(f"  {'OK ' if value else 'FAIL'} {key}")
    print(f"\nall pass : {all_pass}")
    print(f"written  : {OUT}")
    return 0 if all_pass else 3


if __name__ == "__main__":
    sys.exit(main())
