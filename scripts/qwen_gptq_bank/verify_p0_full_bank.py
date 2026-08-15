from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
PREREG = ROOT / "reports/qwen_gptq_bank/P0_FULL_BANK_PREREGISTRATION.md"
ERRATUM = ROOT / "reports/qwen_gptq_bank/P0_GPTQ_SEMANTICS_ERRATUM.md"
EQUIVALENCE = ROOT / "reports/qwen_gptq_bank/p0_fastgrid_equivalence_result_h.json"
EQUIVALENCE_F = ROOT / "reports/qwen_gptq_bank/p0_noany_equivalence_result_f.json"
CALIBRATION_RESULT = ROOT / "reports/qwen_gptq_bank/p0_calibration_capture_result.json"
RUN_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_bank"
LAYER_DIR = ROOT / "reports/qwen_gptq_bank/p0_bank_layers"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_full_bank_verification.json"
REPORT = ROOT / "reports/qwen_gptq_bank/P0_FULL_BANK_VERIFICATION.md"
LAYERS, EXPERTS, HIDDEN, INTERMEDIATE, GROUP = 48, 128, 2_048, 768, 128
CODES_PER_MATRIX = HIDDEN * INTERMEDIATE
CODES_PER_LAYER = EXPERTS * 3 * CODES_PER_MATRIX
PACKED_BYTES_PER_LAYER = CODES_PER_LAYER // 4
SCALE_ELEMENTS_PER_LAYER = EXPERTS * (
    2 * INTERMEDIATE * (HIDDEN // GROUP) + HIDDEN * (INTERMEDIATE // GROUP)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def independent_unpack(packed: torch.Tensor) -> torch.Tensor:
    values = torch.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
    return (values.reshape(*packed.shape[:-1], -1).to(torch.int8) - 2).contiguous()


def independent_repack(codes: torch.Tensor) -> torch.Tensor:
    values = (codes.to(torch.int16) + 2).to(torch.uint8).reshape(*codes.shape[:-1], -1, 4)
    return (values[..., 0] | values[..., 1] << 2 | values[..., 2] << 4 | values[..., 3] << 6).contiguous()


def source_hashes(layer: int, weight_map: dict[str, str]) -> dict[str, str]:
    mappings = {kind: [] for kind in ("gate", "up", "down")}
    for expert in range(EXPERTS):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        for kind in mappings:
            mappings[kind].append(f"{base}.{kind}_proj.weight")
    loaded = load_checkpoint_tensors(
        MODEL, [name for names in mappings.values() for name in names], weight_map
    )
    hashes = {}
    for kind, names in mappings.items():
        stacked = torch.stack([loaded[name] for name in names]).contiguous()
        hashes[kind] = sha256_tensor(stacked)
        del stacked
    return hashes


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite full-bank verification")
    equivalence = json.loads(EQUIVALENCE.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    checks = {
        "equivalence_pass": equivalence["status"] == "equivalence_pass",
        "calibration_pass": calibration["status"] == "capture_pass",
        "all_48_artifacts_present": True,
        "all_48_reports_present": True,
        "all_artifact_hashes": True,
        "all_metadata": True,
        "all_tensor_sets": True,
        "all_shapes": True,
        "all_dtypes": True,
        "all_codes_in_alphabet": True,
        "all_packed_roundtrips_exact": True,
        "all_code_histograms": True,
        "all_scales_finite_nonzero": True,
        "all_scale_counts": True,
        "all_source_weight_hashes": True,
        "all_calibration_hashes": True,
        "all_layer_controls": True,
    }
    expected_shapes = {
        "gate_codes_packed": (EXPERTS, INTERMEDIATE, HIDDEN // 4),
        "gate_scales": (EXPERTS, INTERMEDIATE, HIDDEN // GROUP),
        "up_codes_packed": (EXPERTS, INTERMEDIATE, HIDDEN // 4),
        "up_scales": (EXPERTS, INTERMEDIATE, HIDDEN // GROUP),
        "down_codes_packed": (EXPERTS, HIDDEN, INTERMEDIATE // 4),
        "down_scales": (EXPERTS, HIDDEN, INTERMEDIATE // GROUP),
    }
    weight_map = checkpoint_weight_map(MODEL)
    manifest = {}
    aggregate_histograms = {kind: torch.zeros(4, dtype=torch.int64) for kind in ("gate", "up", "down")}
    total_codes = total_packed_bytes = total_scale_elements = total_artifact_bytes = 0

    for layer in range(LAYERS):
        artifact = RUN_DIR / f"layer_{layer:02d}.safetensors"
        report = LAYER_DIR / f"layer_{layer:02d}.json"
        checks["all_48_artifacts_present"] &= artifact.is_file()
        checks["all_48_reports_present"] &= report.is_file()
        producer = json.loads(report.read_text(encoding="utf-8"))
        artifact_hash = sha256_file(artifact)
        checks["all_artifact_hashes"] &= artifact_hash == producer["artifact_sha256"]
        layer_histograms = {}
        layer_packed = layer_scales = 0
        with safe_open(artifact, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            expected_equivalence = sha256_file(EQUIVALENCE_F) if layer == 0 else sha256_file(EQUIVALENCE)
            checks["all_metadata"] &= (
                metadata.get("layer") == str(layer)
                and metadata.get("equivalence_result_sha256") == expected_equivalence
                and metadata.get("calibration_sha256") == producer["calibration_sha256"]
            )
            checks["all_tensor_sets"] &= set(handle.keys()) == set(expected_shapes)
            for key, shape in expected_shapes.items():
                value = handle.get_tensor(key)
                checks["all_shapes"] &= tuple(value.shape) == shape
                expected_dtype = torch.uint8 if key.endswith("_packed") else torch.bfloat16
                checks["all_dtypes"] &= value.dtype == expected_dtype
                if key.endswith("_packed"):
                    kind = key.split("_")[0]
                    decoded = independent_unpack(value)
                    checks["all_codes_in_alphabet"] &= bool(((decoded >= -2) & (decoded <= 1)).all())
                    checks["all_packed_roundtrips_exact"] &= torch.equal(
                        independent_repack(decoded), value
                    )
                    hist = torch.bincount((decoded.long() + 2).reshape(-1), minlength=4)
                    layer_histograms[kind] = hist
                    aggregate_histograms[kind] += hist
                    total_codes += decoded.numel()
                    total_packed_bytes += value.numel()
                    layer_packed += value.numel()
                    del decoded, hist
                else:
                    checks["all_scales_finite_nonzero"] &= bool(
                        torch.isfinite(value).all() and (value != 0).all()
                    )
                    total_scale_elements += value.numel()
                    layer_scales += value.numel()
        for kind, hist in layer_histograms.items():
            expected = [producer["histograms"][kind][str(code)] for code in (-2, -1, 0, 1)]
            checks["all_code_histograms"] &= hist.tolist() == expected
        checks["all_scale_counts"] &= (
            layer_packed == PACKED_BYTES_PER_LAYER
            and layer_scales == SCALE_ELEMENTS_PER_LAYER
        )
        checks["all_source_weight_hashes"] &= source_hashes(layer, weight_map) == producer["source_weight_sha256"]
        calibration_path = ROOT / producer["calibration"]
        checks["all_calibration_hashes"] &= (
            sha256_file(calibration_path) == producer["calibration_sha256"]
            == calibration["layers"][str(layer)]["artifact_sha256"]
        )
        checks["all_layer_controls"] &= all(producer["controls"].values())
        total_artifact_bytes += artifact.stat().st_size
        manifest[str(layer)] = {
            "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
            "artifact_sha256": artifact_hash, "report_sha256": sha256_file(report),
            "bytes": artifact.stat().st_size,
        }
        print(json.dumps({"layer": layer, "verified": all(checks.values())}), flush=True)

    expected_total_codes = LAYERS * CODES_PER_LAYER
    checks["total_codes_exact"] = total_codes == expected_total_codes
    checks["total_packed_bytes_exact"] = total_packed_bytes == LAYERS * PACKED_BYTES_PER_LAYER
    checks["total_scale_elements_exact"] = total_scale_elements == LAYERS * SCALE_ELEMENTS_PER_LAYER
    checks["histogram_total_exact"] = sum(int(hist.sum()) for hist in aggregate_histograms.values()) == total_codes
    passed = all(checks.values())
    payload = {
        "kind": "qwen_gptq_bank_p0_full_bank_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_bank_pass" if passed else "full_bank_fail",
        "checks": checks, "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks), "manifest": manifest,
        "bank": {
            "layers": LAYERS, "experts": LAYERS * EXPERTS, "matrices": LAYERS * EXPERTS * 3,
            "codes": total_codes, "packed_code_bytes": total_packed_bytes,
            "scale_elements": total_scale_elements, "scale_bytes": total_scale_elements * 2,
            "artifact_bytes": total_artifact_bytes,
            "effective_bits_per_weight_including_bf16_scales": (
                total_packed_bytes * 8 + total_scale_elements * 16
            ) / total_codes,
            "histograms": {
                kind: {str(code - 2): int(value) for code, value in enumerate(hist.tolist())}
                for kind, hist in aggregate_histograms.items()
            },
        },
        "inputs": {
            "preregistration_sha256": sha256_file(PREREG),
            "semantics_erratum_sha256": sha256_file(ERRATUM),
            "equivalence_result_sha256": sha256_file(EQUIVALENCE),
            "layer_0_equivalence_result_f_sha256": sha256_file(EQUIVALENCE_F),
            "calibration_result_sha256": sha256_file(CALIBRATION_RESULT),
            "model_index_sha256": sha256_file(MODEL / "model.safetensors.index.json"),
        },
        "claim_boundary": "Complete pure-GPTQ source-bank verification; CORETAIL physical P0 and runtime remain separate.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Qwen GPTQ Bank — full-bank verification\n\n"
        f"Uitkomst: **{payload['status']}** ({payload['passed_checks']}/{payload['total_checks']}).\n\n"
        f"Fysiek geverifieerd: {payload['bank']['experts']:,} experts, {payload['bank']['matrices']:,} matrices, "
        f"{total_codes:,} codes en {total_scale_elements:,} BF16-scales. Effectieve bronrate: "
        f"{payload['bank']['effective_bits_per_weight_including_bf16_scales']:.6f} bpp.\n\n"
        "De verifier decodeerde en herpakte iedere code, herberekende alle histogrammen en vergeleek "
        "alle brongewicht-, calibratie-, artifact- en rapporthashes.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}",
        "experts": payload["bank"]["experts"], "codes": total_codes,
        "artifact_bytes": total_artifact_bytes,
    }, indent=2))
