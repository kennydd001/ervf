from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


RESIDENT_PATTERNS = (
    re.compile(r"^backbone\.layers\.\d+\.mixer\.(in_proj|out_proj)\.weight$"),
    re.compile(r"^backbone\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.weight$"),
    re.compile(r"^backbone\.layers\.\d+\.mixer\.gate\.weight$"),
    re.compile(r"^backbone\.layers\.\d+\.mixer\.shared_experts\.(up_proj|down_proj)\.weight$"),
    re.compile(r"^lm_head\.weight$"),
)
EXPERT_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"
)
TILE_SIZES = (128, 256, 512, 1024)
PALETTE_BITS = (4, 5, 6)


def entropy_from_counts(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def summarize_stream(raw: np.ndarray, tile_size: int) -> dict:
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    counts = np.bincount(raw, minlength=256).astype(np.int64)
    tile_unique = []
    tile_entropy = []
    palette_bits = {str(k): 0.0 for k in PALETTE_BITS}
    palette_symbols = {str(k): 0 for k in PALETTE_BITS}
    total = int(raw.size)
    for start in range(0, total, tile_size):
        tile = raw[start : start + tile_size]
        tc = np.bincount(tile, minlength=256).astype(np.int64)
        n = int(tile.size)
        unique = int(np.count_nonzero(tc))
        tile_unique.append(unique)
        tile_entropy.append(entropy_from_counts(tc))
        for k in PALETTE_BITS:
            slots = 1 << k
            top = np.sort(tc)[-slots:]
            represented = int(top.sum())
            escapes = n - represented
            symbols = min(unique, slots)
            palette_symbols[str(k)] += symbols
            palette_bits[str(k)] += float(n * k + escapes * 8 + symbols * 8)
    delta = (raw[1:].astype(np.int16) - raw[:-1].astype(np.int16)) % 256
    delta_counts = np.bincount(delta.astype(np.uint8), minlength=256)
    boundaries = np.flatnonzero(raw[1:] != raw[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, total]
    runs = (ends - starts).astype(np.int32)
    run_counts = np.bincount(np.minimum(runs, 255), minlength=256)
    return {
        "bytes": total,
        "byte_entropy_bits": entropy_from_counts(counts),
        "unique_symbols": int(np.count_nonzero(counts)),
        "delta_entropy_bits": entropy_from_counts(delta_counts),
        "run_entropy_bits": entropy_from_counts(run_counts),
        "run_mean_bytes": float(runs.mean()) if runs.size else 0.0,
        "tiles": {
            "count": len(tile_unique),
            "size_bytes": tile_size,
            "mean_unique_symbols": float(np.mean(tile_unique)) if tile_unique else 0.0,
            "p95_unique_symbols": float(np.percentile(tile_unique, 95)) if tile_unique else 0.0,
            "weighted_entropy_bits": float(
                sum(e * min(tile_size, total - i * tile_size) for i, e in enumerate(tile_entropy))
                / total
            ) if total else 0.0,
            "palette_encoded_bits": palette_bits,
            "palette_symbols": palette_symbols,
        },
    }


def split_entropy(raw: np.ndarray, dtype_kind: str) -> dict:
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if dtype_kind == "fp8":
        fields = {
            "sign": (raw >> 7) & 1,
            "exponent": (raw >> 3) & 0xF,
            "mantissa": raw & 0x7,
        }
    elif dtype_kind == "bf16":
        if raw.size % 2:
            raise ValueError("BF16 stream has odd byte count")
        words = raw.view("<u2")
        fields = {
            "sign": (words >> 15) & 1,
            "exponent": (words >> 7) & 0xFF,
            "mantissa": words & 0x7F,
        }
    else:
        return {}
    return {
        name: entropy_from_counts(np.bincount(values.astype(np.int64)))
        for name, values in fields.items()
    }


def tensor_kind(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.bfloat16:
        return "bf16"
    if tensor.dtype == torch.uint8:
        return "fp8"
    if tensor.dtype == torch.float32:
        return "f32"
    raise ValueError(f"unsupported selected dtype: {tensor.dtype}")


def raw_bytes(tensor: torch.Tensor, kind: str) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous().reshape(-1)
    if kind == "bf16":
        return tensor.view(torch.uint8).numpy()
    if kind == "f32":
        return tensor.view(torch.uint8).numpy()
    return tensor.numpy().astype(np.uint8, copy=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(name: str) -> str:
    if name == "lm_head.weight":
        return "lm_head"
    if ".shared_experts." in name:
        return "shared_expert"
    if ".mixer.gate." in name:
        return "router"
    if ".mixer.in_proj." in name or ".mixer.out_proj." in name:
        return "mamba"
    if any(f".mixer.{x}." in name for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention"
    raise ValueError(f"cannot classify {name}")


def selected_names(weight_map: dict[str, str]) -> list[str]:
    return sorted(
        name
        for name in weight_map
        if name.endswith(".weight") and any(pattern.match(name) for pattern in RESIDENT_PATTERNS)
    )


def expert_sample_names(weight_map: dict[str, str]) -> list[str]:
    chosen = []
    for name in sorted(weight_map):
        match = EXPERT_RE.match(name)
        if not match:
            continue
        layer, expert, _ = map(int, (match.group(1), match.group(2), "0"))
        if layer in {1, 27, 51} and expert in {0, 64, 127}:
            chosen.append(name)
    return chosen


def load_one(
    root: Path, weight_map: dict[str, str], name: str
) -> tuple[np.ndarray, str, tuple[int, ...], int]:
    path = root / weight_map[name]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if name == "lm_head.weight":
            view = handle.get_slice(name)
            shape = tuple(int(x) for x in view.get_shape())
            tensor = view[::16, :]
            sample_stride = 16
        else:
            tensor = handle.get_tensor(name)
            shape = tuple(int(x) for x in tensor.shape)
            sample_stride = 1
    kind = tensor_kind(tensor)
    return raw_bytes(tensor, kind), kind, shape, sample_stride


def aggregate(rows: list[dict]) -> dict:
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    out = {}
    for family, family_rows in sorted(by_family.items()):
        weight_count = sum(r["elements"] for r in family_rows)
        raw_bytes_total = sum(r["raw_bytes"] for r in family_rows)
        best = {}
        for k in PALETTE_BITS:
            bits = sum(r["palette_bits_per_stream"][str(k)] for r in family_rows)
            best[str(k)] = {
                "encoded_bits": bits,
                "encoded_bits_per_weight": bits / weight_count,
                "raw_byte_fraction": bits / (raw_bytes_total * 8),
            }
        best_key = min(best, key=lambda key: best[key]["encoded_bits_per_weight"])
        out[family] = {
            "matrix_count": len(family_rows),
            "weight_count": weight_count,
            "raw_bytes": raw_bytes_total,
            "raw_bits_per_weight": (raw_bytes_total * 8) / weight_count,
            "weighted_byte_entropy_bits": sum(
                r["byte_entropy_bits"] * r["raw_bytes"] for r in family_rows
            ) / raw_bytes_total,
            "weighted_delta_entropy_bits": sum(
                r["delta_entropy_bits"] * r["raw_bytes"] for r in family_rows
            ) / raw_bytes_total,
            "palette": best,
            "best_palette_bits_per_weight": best[best_key]["encoded_bits_per_weight"],
            "best_palette_bits": int(best_key),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="S100 Phase 13A lossless entropy census")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/nemotron_3_5_lightning"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pro_research/results/s100_phase13a/S100_PHASE13A_ENTROPY.json"),
    )
    args = parser.parse_args()
    root = args.model_dir.resolve()
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    resident = selected_names(weight_map)
    experts = expert_sample_names(weight_map)
    if len(resident) != 140:
        raise RuntimeError(f"expected Phase-12C resident matrix count 140, got {len(resident)}")
    if len(experts) != 18:
        raise RuntimeError(f"expected 18 deterministic expert samples, got {len(experts)}")

    rows = []
    for name in resident + experts:
        raw, kind, shape, sample_stride = load_one(root, weight_map, name)
        is_expert = EXPERT_RE.match(name) is not None
        summary_1024 = summarize_stream(raw, 1024)
        analysis_scale = float(sample_stride)
        row = {
            "name": name,
            "family": "routed_expert_sample" if is_expert else classify(name),
            "kind": kind,
            "shape": list(shape),
            "elements": int(np.prod(shape)),
            "raw_bytes": int(raw.size * sample_stride),
            "analysis_bytes": int(raw.size),
            "analysis_sampling": (
                "full stream" if sample_stride == 1 else f"every {sample_stride}th row"
            ),
            "element_bytes": {"bf16": 2, "fp8": 1, "f32": 4}[kind],
            "byte_entropy_bits": float(summary_1024["byte_entropy_bits"]),
            "delta_entropy_bits": float(summary_1024["delta_entropy_bits"]),
            "run_entropy_bits": float(summary_1024["run_entropy_bits"]),
            "run_mean_bytes": float(summary_1024["run_mean_bytes"]),
            "split_entropy_bits": split_entropy(raw, kind),
            "palette_bits_per_stream": {
                "raw": float(raw.size * sample_stride * 8),
                **{
                    key: value * analysis_scale
                    for key, value in summary_1024["tiles"]["palette_encoded_bits"].items()
                },
            },
            "tiles": {},
        }
        for tile_size in TILE_SIZES:
            summary = summary_1024 if tile_size == 1024 else summarize_stream(raw, tile_size)
            row["tiles"][str(tile_size)] = summary["tiles"]
            if tile_size == 1024:
                row["palette_bits_per_stream"] = {
                    "raw": float(raw.size * sample_stride * 8),
                    **{
                        key: value * analysis_scale
                        for key, value in summary["tiles"]["palette_encoded_bits"].items()
                    },
                }
        rows.append(row)
        print(f"{len(rows):3d}/{len(resident) + len(experts)} {name} {kind} {raw.size / 2**20:.1f} MiB", flush=True)

    resident_rows = rows[: len(resident)]
    expert_rows = rows[len(resident) :]
    mamba_fp8 = [r for r in resident_rows if r["family"] == "mamba" and r["kind"] == "fp8"]
    all_rows = resident_rows
    all_raw = sum(r["raw_bytes"] for r in all_rows)
    all_best_bits = sum(min(r["palette_bits_per_stream"].values()) for r in all_rows)
    mamba_raw = sum(r["elements"] for r in mamba_fp8)
    mamba_best_bits = sum(min(r["palette_bits_per_stream"].values()) for r in mamba_fp8)
    result = {
        "kind": "s100_phase13a_entropy_census",
        "status": "measured",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(root),
        "model_identity": {
            "config_sha256": sha256_file(root / "config.json"),
            "index_sha256": sha256_file(index_path),
            "claim": "same checkpoint as Phase 12C: models/nemotron_3_5_lightning",
        },
        "method": {
            "raw_stream": "uint8 FP8 code bytes or little-endian BF16 bytes from safetensors",
            "tile_sizes_bytes": list(TILE_SIZES),
            "palette_bits": list(PALETTE_BITS),
            "palette_estimate": "top-2^k byte symbols plus 8-bit escape values and palette bytes per tile",
            "expert_sample": "layers {1,27,51} x experts {0,64,127} x up/down",
            "not_measured": "decoder throughput, GPU kernel overhead, scale-plane metadata, model quality",
            "large_tensor_sampling": "lm_head analyzed on every 16th row; raw byte/weight totals retain full size",
        },
        "counts": {
            "resident_matrices": len(resident_rows),
            "routed_expert_sample_matrices": len(expert_rows),
            "resident_raw_bytes": all_raw,
            "resident_raw_weights": sum(r["elements"] for r in all_rows),
        },
        "family_aggregates": aggregate(resident_rows),
        "gates": {
            "mamba_fp8_weight_count": mamba_raw,
            "mamba_fp8_best_palette_bits_per_weight": mamba_best_bits / mamba_raw,
            "mamba_fp8_le_6_bits_per_weight": mamba_best_bits / mamba_raw <= 6.0,
            "resident_best_palette_fraction": all_best_bits / (all_raw * 8),
            "resident_le_70_percent_raw_bytes": all_best_bits / (all_raw * 8) <= 0.70,
            "promotion_open": (
                mamba_best_bits / mamba_raw <= 6.0
                or all_best_bits / (all_raw * 8) <= 0.70
            ),
        },
        "matrices": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
