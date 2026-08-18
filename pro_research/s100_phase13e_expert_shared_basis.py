from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open


EXPERT_RE = re.compile(
    r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight$"
)
LAYERS = (1, 27, 51)
EXPERTS = 128
RANKS = (4, 8, 16, 32, 64)
RESIDUAL_FRACTIONS = (0.10, 0.25, 0.50)


def tensor_nbytes(root: Path, weight_map: dict[str, str], name: str) -> int:
    with safe_open(str(root / weight_map[name]), framework="pt", device="cpu") as handle:
        return int(handle.get_tensor(name).numel() * handle.get_tensor(name).element_size())


def load_codes(root: Path, weight_map: dict[str, str], name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    with safe_open(str(root / weight_map[name]), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
        shape = tuple(int(x) for x in tensor.shape)
        codes = tensor.detach().cpu().contiguous().numpy().reshape(-1)
    if codes.dtype != np.uint8:
        raise RuntimeError(f"expected FP8 code bytes for {name}, got {codes.dtype}")
    return codes, shape


def sample_indices(size: int, count: int) -> np.ndarray:
    count = min(size, count)
    return np.linspace(0, size - 1, count, dtype=np.int64)


def census_matrix(
    root: Path,
    weight_map: dict[str, str],
    layer: int,
    projection: str,
    sample_features: int,
) -> dict:
    names = [
        f"backbone.layers.{layer}.mixer.experts.{expert}.{projection}.weight"
        for expert in range(EXPERTS)
    ]
    if not all(name in weight_map for name in names):
        missing = [name for name in names if name not in weight_map]
        raise RuntimeError(f"missing expert tensors for layer {layer} {projection}: {missing[:2]}")

    first_codes, shape = load_codes(root, weight_map, names[0])
    features = int(first_codes.size)
    indices = sample_indices(features, sample_features)
    matrix = np.empty((EXPERTS, indices.size), dtype=np.float32)
    matrix[0] = first_codes[indices]
    for expert, name in enumerate(names[1:], start=1):
        codes, other_shape = load_codes(root, weight_map, name)
        if other_shape != shape or codes.size != features:
            raise RuntimeError(f"inconsistent shape at {name}: {other_shape} vs {shape}")
        matrix[expert] = codes[indices]

    # Each expert has its own weight-scale plane. Keep it in the byte budget,
    # while deliberately applying the shared basis only to the FP8 code plane.
    scale_bytes = 0
    scale_names = []
    for expert in range(EXPERTS):
        prefix = f"backbone.layers.{layer}.mixer.experts.{expert}.{projection}"
        for suffix in ("weight_scale", "weight_scale_2"):
            name = f"{prefix}.{suffix}"
            if name in weight_map:
                scale_bytes += tensor_nbytes(root, weight_map, name)
                scale_names.append(name)
    original_code_bytes = EXPERTS * features
    original_total_bytes = original_code_bytes + scale_bytes
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    total_energy = float(np.square(matrix).sum())
    rank_metrics = []
    for rank in RANKS:
        rank = min(rank, min(matrix.shape))
        reconstruction = (u[:, :rank] * singular[:rank]) @ vt[:rank, :]
        residual = matrix - reconstruction
        residual_energy = float(np.square(residual).sum())
        residual_nrmse = float(np.sqrt(residual_energy / max(total_energy, 1e-12)))
        basis_bytes = rank * features * 2
        coefficient_bytes = EXPERTS * rank * 2
        encoded_total_bytes = basis_bytes + coefficient_bytes + scale_bytes
        top_capture = {}
        flat_abs = np.abs(residual).reshape(-1)
        residual_total = float(np.square(residual).sum())
        for fraction in RESIDUAL_FRACTIONS:
            keep = max(1, int(flat_abs.size * fraction))
            selected = np.argpartition(flat_abs, -keep)[-keep:]
            captured = float(np.square(residual.reshape(-1)[selected]).sum())
            top_capture[str(fraction)] = {
                "kept_sample_fraction": float(keep / flat_abs.size),
                "residual_energy_captured": captured / max(residual_total, 1e-12),
            }
        rank_metrics.append({
            "rank": rank,
            "residual_energy_fraction": residual_energy / max(total_energy, 1e-12),
            "reconstruction_nrmse": residual_nrmse,
            "basis_bytes_bf16": basis_bytes,
            "coefficient_bytes_bf16": coefficient_bytes,
            "scale_bytes_untouched": scale_bytes,
            "ideal_dense_shared_total_bytes": encoded_total_bytes,
            "ideal_dense_shared_byte_reduction": 1.0 - encoded_total_bytes / original_total_bytes,
            "top_residual_capture_on_sample": top_capture,
        })

    return {
        "layer": layer,
        "projection": projection,
        "shape": list(shape),
        "experts": EXPERTS,
        "full_features_per_expert": features,
        "sample_features": int(indices.size),
        "sample_strategy": "deterministic linspace over flattened FP8 code plane",
        "original_code_bytes": original_code_bytes,
        "scale_bytes": scale_bytes,
        "original_total_weight_bytes": original_total_bytes,
        "scale_planes": len(scale_names),
        "ranks": rank_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="S100 Phase 13E expert shared-basis census")
    parser.add_argument("--model-dir", type=Path, default=Path("models/nemotron_3_5_lightning"))
    parser.add_argument("--sample-features", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13e/S100_PHASE13E_SHARED_BASIS.json"))
    args = parser.parse_args()
    root = args.model_dir.resolve()
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    records = []
    for layer in LAYERS:
        for projection in ("up_proj", "down_proj"):
            record = census_matrix(root, weight_map, layer, projection, args.sample_features)
            records.append(record)
            print(
                f"layer={layer} projection={projection} shape={record['shape']} "
                f"scale_bytes={record['scale_bytes']} sample={record['sample_features']}",
                flush=True,
            )

    best = []
    for record in records:
        best.append(min(record["ranks"], key=lambda row: row["reconstruction_nrmse"]))
    result = {
        "kind": "s100_phase13e_expert_shared_basis_census",
        "status": "measured",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(root),
        "model_identity": {
            "config_sha256": __import__("hashlib").sha256((root / "config.json").read_bytes()).hexdigest(),
            "index_sha256": __import__("hashlib").sha256(index_path.read_bytes()).hexdigest(),
            "claim": "same checkpoint as Phase 12C: models/nemotron_3_5_lightning",
        },
        "method": {
            "expert_layers": list(LAYERS),
            "expert_count": EXPERTS,
            "projections": ["up_proj", "down_proj"],
            "ranks": list(RANKS),
            "basis_storage_assumption": "BF16 basis and BF16 expert coefficients; FP8 code plane only",
            "scale_policy": "all per-expert weight_scale and weight_scale_2 bytes retained untouched",
            "not_measured": "decoded FP8 numerical quality, activation/output error, sparse residual indexing, kernel throughput, official validation",
        },
        "counts": {
            "matrix_records": len(records),
            "expert_matrices": len(records) * EXPERTS,
            "routed_weight_bytes": sum(r["original_total_weight_bytes"] for r in records),
            "routed_code_bytes": sum(r["original_code_bytes"] for r in records),
            "routed_scale_bytes": sum(r["scale_bytes"] for r in records),
        },
        "best_reconstruction_rows": best,
        "records": records,
        "gates": {
            "any_ideal_dense_shared_byte_reduction_ge_30": any(
                row["ideal_dense_shared_byte_reduction"] >= 0.30
                for record in records for row in record["ranks"]
            ),
            "decoded_quality_green": False,
            "official_validation_green": False,
            "promotion_open": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
