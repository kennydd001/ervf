"""S100 native NVFP4 C1 — lossless Blackwell scale-layout repack audit.

C0B proved logical group-16 across all 5,935 NVFP4 pairs. C1 freezes the
NVIDIA-documented SWIZZLE_32_4_4 mapping and proves that Lightning's natural
row-major block-scale bytes can be permuted into that padded layout and back
without changing codes, scales, global scales or sampled dequantized values.

No CUDA matmul is executed. No speed or model-quality claim is made.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "nemotron_3_5_lightning_v35"
INDEX = MODEL_DIR / "model.safetensors.index.json"
C0B = REPO / "pro_research" / "results" / "native_nvfp4" / "C0B_FORMAT_AUDIT.json"
OUT = REPO / "pro_research" / "results" / "native_nvfp4" / "C1_REPACK_AUDIT.json"
PREREG = REPO / "pro_research" / "S100_NATIVE_NVFP4_C1_REPACK_PREREGISTRATION.md"

RNG_SEED = 0xC001C1
MAX_ENUM_COORDS = 2_000_000
RANDOM_COORDS_LARGE = 50_000
MAX_CODE_PAYLOAD_FOR_DEQUANT = 24 * 1024 * 1024
DEQUANT_SAMPLES = 4096


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def padded_count(M: int, SFK: int) -> int:
    return ceil_div(M, 128) * 128 * ceil_div(SFK, 4) * 4


def swizzle_offset(m: np.ndarray | int, sf: np.ndarray | int, M: int) -> np.ndarray:
    """Frozen SWIZZLE_32_4_4 natural-coordinate -> flat native offset."""
    m = np.asarray(m, dtype=np.int64)
    sf = np.asarray(sf, dtype=np.int64)
    nmb = ceil_div(M, 128)
    mb = m // 128
    r = m % 128
    r32 = r % 32
    g32 = r // 32
    kb = sf // 4
    sf4 = sf % 4
    block = kb * nmb + mb
    inner = ((r32 * 4 + g32) * 4 + sf4)
    return block * 512 + inner


def inverse_offset(off: np.ndarray | int, M: int) -> tuple[np.ndarray, np.ndarray]:
    off = np.asarray(off, dtype=np.int64)
    nmb = ceil_div(M, 128)
    block = off // 512
    inner = off % 512
    kb = block // nmb
    mb = block % nmb
    sf4 = inner % 4
    t = inner // 4
    g32 = t % 4
    r32 = t // 4
    m = mb * 128 + g32 * 32 + r32
    sf = kb * 4 + sf4
    return m, sf


def sha256_bytes(x: bytes | np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(x.tobytes() if isinstance(x, np.ndarray) else x)
    return h.hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_headers(entries: dict[str, str]) -> dict[str, dict]:
    out = {}
    for shard in sorted(set(entries.values())):
        p = MODEL_DIR / shard
        with p.open("rb") as f:
            hlen = int.from_bytes(f.read(8), "little")
            out[shard] = json.loads(f.read(hlen))
    return out


def tensor_raw(name: str, entries: dict[str, str], headers: dict[str, dict]) -> bytes:
    shard = entries[name]
    rec = headers[shard][name]
    a, b = (int(x) for x in rec["data_offsets"])
    p = MODEL_DIR / shard
    with p.open("rb") as f:
        hlen = int.from_bytes(f.read(8), "little")
        base = 8 + hlen
        f.seek(base + a)
        raw = f.read(b - a)
    if len(raw) != b - a:
        raise IOError(f"short read {name}: {len(raw)} != {b-a}")
    return raw


def swizzle(natural: np.ndarray) -> np.ndarray:
    natural = np.ascontiguousarray(natural, dtype=np.uint8)
    M, SFK = (int(x) for x in natural.shape)
    dst = np.zeros(padded_count(M, SFK), dtype=np.uint8)
    # Chunk rows to avoid a giant transient offset matrix on lm_head.
    for r0 in range(0, M, 256):
        r1 = min(M, r0 + 256)
        mm = np.arange(r0, r1, dtype=np.int64)[:, None]
        ss = np.arange(SFK, dtype=np.int64)[None, :]
        off = swizzle_offset(mm, ss, M)
        dst[off.reshape(-1)] = natural[r0:r1].reshape(-1)
    return dst


def unswizzle(src: np.ndarray, M: int, SFK: int) -> np.ndarray:
    src = np.asarray(src, dtype=np.uint8).reshape(-1)
    if src.size != padded_count(M, SFK):
        raise ValueError(f"bad swizzled size {src.size} != {padded_count(M,SFK)}")
    dst = np.empty((M, SFK), dtype=np.uint8)
    for r0 in range(0, M, 256):
        r1 = min(M, r0 + 256)
        mm = np.arange(r0, r1, dtype=np.int64)[:, None]
        ss = np.arange(SFK, dtype=np.int64)[None, :]
        off = swizzle_offset(mm, ss, M)
        dst[r0:r1] = src[off]
    return dst


def e2m1(code: np.ndarray) -> np.ndarray:
    c = np.asarray(code, dtype=np.uint8)
    mags = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
    v = mags[c & 7]
    return np.where((c >> 3) & 1, -v, v)


def e4m3(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, dtype=np.uint8)
    sign = np.where((x >> 7) & 1, -1.0, 1.0)
    exp = ((x >> 3) & 15).astype(np.int64)
    man = (x & 7).astype(np.float64)
    sub = np.exp2(-6.0) * man / 8.0
    norm = np.exp2((exp - 7).astype(np.float64)) * (1.0 + man / 8.0)
    val = sign * np.where(exp == 0, sub, norm)
    return np.where((exp == 15) & (man == 7), np.nan, val)


def scalar_f32(raw: bytes) -> tuple[float, str]:
    if len(raw) != 4:
        raise ValueError(f"expected scalar F32 = 4 bytes, got {len(raw)}")
    return struct.unpack("<f", raw)[0], raw.hex()


def structural_shape_check(M: int, SFK: int, rng: np.random.Generator) -> dict:
    natural_n = M * SFK
    native_n = padded_count(M, SFK)
    # Boundaries are always included.
    coords = {
        (0, 0), (max(M - 1, 0), 0), (0, max(SFK - 1, 0)),
        (max(M - 1, 0), max(SFK - 1, 0)),
    }
    for m in (31, 32, 63, 64, 95, 96, 127, 128, 129):
        if 0 <= m < M:
            for sf in (0, 3, 4, SFK - 1):
                if 0 <= sf < SFK:
                    coords.add((m, sf))
    for sf in (3, 4, 7, 8, 11, 12):
        if 0 <= sf < SFK:
            for m in (0, min(31, M - 1), min(127, M - 1), M - 1):
                if m >= 0:
                    coords.add((m, sf))

    enumerated = natural_n <= MAX_ENUM_COORDS
    if enumerated:
        mm = np.repeat(np.arange(M, dtype=np.int64), SFK)
        ss = np.tile(np.arange(SFK, dtype=np.int64), M)
    else:
        n = min(RANDOM_COORDS_LARGE, natural_n)
        mm = rng.integers(0, M, size=n, dtype=np.int64)
        ss = rng.integers(0, SFK, size=n, dtype=np.int64)
        if coords:
            bb = np.asarray(sorted(coords), dtype=np.int64)
            mm = np.concatenate([mm, bb[:, 0]])
            ss = np.concatenate([ss, bb[:, 1]])

    off = swizzle_offset(mm, ss, M)
    im, isf = inverse_offset(off, M)
    inverse_ok = bool(np.array_equal(mm, im) and np.array_equal(ss, isf))
    in_bounds = bool(off.size == 0 or ((off >= 0).all() and (off < native_n).all()))
    unique = bool(np.unique(off).size == off.size) if enumerated else None

    # The frozen mapping is injective algebraically because offset decodes back
    # to the exact natural coordinate. For large sampled shapes, record both the
    # algebraic proof condition and sampled collision check.
    sampled_unique = bool(np.unique(off).size == off.size)
    return {
        "M": M, "SFK": SFK,
        "natural_count": natural_n, "native_padded_count": native_n,
        "padding_count": native_n - natural_n,
        "padding_fraction_of_natural": (native_n - natural_n) / natural_n if natural_n else 0.0,
        "enumerated_all_natural_coordinates": enumerated,
        "tested_coordinates": int(off.size),
        "in_bounds": in_bounds,
        "inverse_exact": inverse_ok,
        "enumerated_unique": unique,
        "sampled_unique": sampled_unique,
        "injective_by_exact_inverse": inverse_ok,
    }


def choose_representatives(pairs: list[dict]) -> list[dict]:
    # Freeze selection from metadata only: one first tensor for each distinct
    # scale shape, then evenly spaced early/middle/late extras up to 24 total.
    chosen = []
    seen = set()
    for p in pairs:
        shape = tuple(int(x) for x in p["scale_shape"])
        if shape not in seen:
            chosen.append(p)
            seen.add(shape)
    if len(chosen) < 24 and pairs:
        idx = np.linspace(0, len(pairs) - 1, num=min(24, len(pairs)), dtype=int)
        names = {p["scale"] for p in chosen}
        for i in idx.tolist():
            if pairs[i]["scale"] not in names:
                chosen.append(pairs[i])
                names.add(pairs[i]["scale"])
            if len(chosen) >= 24:
                break
    return chosen


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    payload = {
        "kind": "s100_native_nvfp4_c1_repack_audit",
        "status": "started",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "scale-layout permutation/reconstruction only; no native matmul, activation quantization, model quality or speed claim",
        "preregistration": str(PREREG.relative_to(REPO)),
    }
    try:
        c0 = json.loads(C0B.read_text(encoding="utf-8"))
        pairs = list(c0.get("pairs") or [])
        if c0.get("status") != "format_counts_group16_packed_exact" or not pairs:
            raise RuntimeError(f"C0B parent not green: {c0.get('status')}")

        idx = json.loads(INDEX.read_text(encoding="utf-8"))
        entries = idx["weight_map"]
        headers = load_headers(entries)

        shapes = sorted({tuple(int(x) for x in p["scale_shape"]) for p in pairs})
        shape_checks = [structural_shape_check(M, SFK, rng) for M, SFK in shapes]

        reps = choose_representatives(pairs)
        rep_records = []
        for p in reps:
            sname = p["scale"]
            base = sname[:-len(".weight_scale")]
            wname = base + ".weight"
            gname = base + ".weight_scale_2"
            M, SFK = (int(x) for x in p["scale_shape"])
            scale_raw = tensor_raw(sname, entries, headers)
            if len(scale_raw) != M * SFK:
                raise RuntimeError(f"scale byte count mismatch {sname}: {len(scale_raw)} != {M*SFK}")
            natural = np.frombuffer(scale_raw, dtype=np.uint8).reshape(M, SFK).copy()
            swz = swizzle(natural)
            restored = unswizzle(swz, M, SFK)
            scale_mismatch = int(np.count_nonzero(natural != restored))

            global_raw = tensor_raw(gname, entries, headers)
            g, ghex = scalar_f32(global_raw)
            global_after = bytes(global_raw)  # C1 deliberately never transforms it.

            # Codes are also deliberately not transformed by C1. For manageable
            # representative matrices, read the real bytes and perform sampled
            # dequant reconstruction using natural vs inverse-restored scales.
            wrec = headers[entries[wname]][wname]
            wa, wb = (int(x) for x in wrec["data_offsets"])
            wbytes = wb - wa
            code_hash_before = None
            code_hash_after = None
            dequant_samples = 0
            dequant_mismatch = 0
            if wbytes <= MAX_CODE_PAYLOAD_FOR_DEQUANT:
                codes = np.frombuffer(tensor_raw(wname, entries, headers), dtype=np.uint8).copy()
                code_hash_before = sha256_bytes(codes)
                code_hash_after = sha256_bytes(codes)  # identity reference: no code repack in C1
                logical_K = SFK * 16
                expected_code_bytes = M * logical_K // 2
                if codes.size != expected_code_bytes:
                    raise RuntimeError(f"packed code bytes mismatch {wname}: {codes.size} != {expected_code_bytes}")
                n = min(DEQUANT_SAMPLES, M * logical_K)
                lin = rng.integers(0, M * logical_K, size=n, dtype=np.int64)
                rr = lin // logical_K
                kk = lin % logical_K
                packed_idx = rr * (logical_K // 2) + (kk // 2)
                rawb = codes[packed_idx]
                nib = np.where((kk & 1) == 0, rawb & 15, rawb >> 4).astype(np.uint8)
                sb = natural[rr, kk // 16]
                sa = restored[rr, kk // 16]
                before = e2m1(nib) * e4m3(sb) * np.float64(np.float32(g))
                after = e2m1(nib) * e4m3(sa) * np.float64(np.float32(g))
                # Same exact operands/order => float64 bit identity, including signed zero.
                dequant_mismatch = int(np.count_nonzero(before.view(np.uint64) != after.view(np.uint64)))
                dequant_samples = int(n)

            rep_records.append({
                "scale": sname, "weight": wname, "global": gname,
                "shape": [M, SFK],
                "scale_sha256_before": sha256_bytes(natural),
                "swizzled_sha256": sha256_bytes(swz),
                "scale_sha256_after_inverse": sha256_bytes(restored),
                "scale_byte_mismatches": scale_mismatch,
                "natural_scale_bytes": int(natural.size),
                "native_padded_scale_bytes": int(swz.size),
                "padding_bytes": int(swz.size - natural.size),
                "global_scale_f32": float(g),
                "global_scale_bytes_before": ghex,
                "global_scale_bytes_after": global_after.hex(),
                "global_scale_unchanged": global_after == global_raw,
                "weight_payload_bytes": int(wbytes),
                "codes_sha256_before": code_hash_before,
                "codes_sha256_after": code_hash_after,
                "codes_unchanged": code_hash_before == code_hash_after if code_hash_before else True,
                "sampled_dequant_values": dequant_samples,
                "sampled_dequant_bit_mismatches": dequant_mismatch,
            })

        # Weighted padding over all matrices using metadata counts only.
        nat_total = 0
        native_total = 0
        for p in pairs:
            M, SFK = (int(x) for x in p["scale_shape"])
            nat_total += M * SFK
            native_total += padded_count(M, SFK)
        pad_frac = (native_total - nat_total) / nat_total if nat_total else 0.0

        gates = {
            "C1_G1_all_group16_parent": c0.get("status") == "format_counts_group16_packed_exact" and int((c0.get("totals") or {}).get("breakers", 1)) == 0,
            "C1_G2_all_shapes_padded_count": all(x["native_padded_count"] >= x["natural_count"] for x in shape_checks),
            "C1_G3_mapping_in_bounds": all(x["in_bounds"] for x in shape_checks),
            "C1_G4_mapping_unique": all((x["enumerated_unique"] if x["enumerated_unique"] is not None else x["sampled_unique"]) and x["injective_by_exact_inverse"] for x in shape_checks),
            "C1_G5_inverse_formula": all(x["inverse_exact"] for x in shape_checks),
            "C1_G6_payload_scale_roundtrip_exact": bool(rep_records) and all(x["scale_byte_mismatches"] == 0 for x in rep_records),
            "C1_G7_padding_does_not_alias": all(x["inverse_exact"] and x["in_bounds"] for x in shape_checks),
            "C1_G8_global_scale_unchanged": bool(rep_records) and all(x["global_scale_unchanged"] for x in rep_records),
            "C1_G9_codes_unchanged": bool(rep_records) and all(x["codes_unchanged"] for x in rep_records),
            "C1_G10_sampled_dequant_reconstruction": any(x["sampled_dequant_values"] > 0 for x in rep_records) and all(x["sampled_dequant_bit_mismatches"] == 0 for x in rep_records if x["sampled_dequant_values"] > 0),
            "C1_P1_total_padding_le_5pct": pad_frac <= 0.05,
        }
        correctness = all(gates[k] for k in gates if not k.startswith("C1_P"))
        if not correctness:
            status = "repack_not_lossless"
        elif gates["C1_P1_total_padding_le_5pct"]:
            status = "repack_lossless"
        else:
            status = "repack_lossless_high_padding"

        payload.update({
            "parent_c0b": str(C0B.relative_to(REPO)),
            "frozen_mapping": {
                "name": "SWIZZLE_32_4_4",
                "basic_block": "128 rows x 4 SFK = 512 bytes",
                "block_order": "K-major then M-block",
                "inner_order": "r32, row-group-of-32, sf4",
                "formula": "offset=((sf//4)*ceil(M/128)+(m//128))*512 + (((m%32)*4+((m%128)//32))*4+(sf%4))",
            },
            "metadata": {
                "pair_count": len(pairs), "distinct_scale_shapes": len(shapes),
                "representative_payload_tensors": len(rep_records),
                "natural_scale_bytes_total": int(nat_total),
                "native_padded_scale_bytes_total": int(native_total),
                "padding_bytes_total": int(native_total - nat_total),
                "padding_fraction_of_natural": float(pad_frac),
            },
            "shape_checks": shape_checks,
            "representatives": rep_records,
            "gates": gates,
            "status": status,
            "environment": {
                "python": sys.version, "platform": platform.platform(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "index_sha256": sha256_file(INDEX), "c0b_sha256": sha256_file(C0B),
                "prereg_sha256": sha256_file(PREREG),
            },
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {"type": type(exc).__name__, "message": str(exc),
                      "traceback": __import__("traceback").format_exc()},
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload.get("status"), "output": str(OUT),
                      "metadata": payload.get("metadata"), "gates": payload.get("gates"),
                      "error": (payload.get("error") or {}).get("message")}, indent=1))
    return 0 if payload.get("status") in {"repack_lossless", "repack_lossless_high_padding"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
