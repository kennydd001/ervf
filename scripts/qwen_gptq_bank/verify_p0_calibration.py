from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from moe_lab.reporting import ROOT


LAYERS, EXPERTS, ROWS, HIDDEN, TOP_K = 48, 128, 128, 2_048, 8
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
SOURCES = (
    ("hera", ROOT / "reports/runs/hera_moe/p0_routes", DOMAINS),
    ("dhera", ROOT / "reports/runs/dhera_moe/p0_routes", DOMAINS),
    ("supplement_a", ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_routes", DOMAINS),
    ("supplement_b", ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_b_routes", ("math", "instruction")),
)
SELECTION_LOCK = ROOT / "reports/qwen_gptq_bank/p0_calibration_selection_lock.json"
SELECTION_ARTIFACT = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration_selection.safetensors"
CAPTURE_RESULT = ROOT / "reports/qwen_gptq_bank/p0_calibration_capture_result.json"
CALIBRATION_DIR = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration"
LAYER_DIR = ROOT / "reports/qwen_gptq_bank/p0_calibration_layers"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_calibration_verification.json"
REPORT = ROOT / "reports/qwen_gptq_bank/P0_CALIBRATION_VERIFICATION.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite calibration verification")
    lock = json.loads(SELECTION_LOCK.read_text(encoding="utf-8"))
    capture = json.loads(CAPTURE_RESULT.read_text(encoding="utf-8"))
    selected = load_file(SELECTION_ARTIFACT)
    checks = {
        "capture_status_pass": capture["status"] == "capture_pass",
        "selection_artifact_hash": sha256_file(SELECTION_ARTIFACT) == lock["artifact_sha256"],
        "selection_shapes": all(
            tuple(value.shape) == (LAYERS, EXPERTS, ROWS) for value in selected.values()
        ),
        "selection_coordinates_nonnegative": all(bool((value >= 0).all()) for value in selected.values()),
        "all_route_artifact_hashes": True,
        "all_selected_routes_match_expert": True,
        "all_selected_rows_unique_within_expert": True,
        "all_calibration_artifact_hashes": True,
        "all_calibration_report_hashes": True,
        "all_calibration_metadata": True,
        "all_calibration_shapes": True,
        "all_calibration_dtypes": True,
        "all_calibration_finite": True,
        "all_calibration_tensor_hashes": True,
        "all_coordinate_copies_exact": True,
        "all_producer_route_controls": True,
    }
    route_handles = {}
    total_rows = 0
    layer_hashes = {}
    for layer in range(LAYERS):
        layer_source = selected["source_index"][layer].long()
        layer_domain = selected["domain_index"][layer].long()
        layer_token = selected["token_index"][layer].long()
        layer_slot = selected["slot_index"][layer].long()
        identities = set()
        for expert in range(EXPERTS):
            rows = [
                (int(layer_source[expert, row]), int(layer_domain[expert, row]),
                 int(layer_token[expert, row]), int(layer_slot[expert, row]))
                for row in range(ROWS)
            ]
            checks["all_selected_rows_unique_within_expert"] &= len(set(rows)) == ROWS
            identities.update((expert, *row) for row in rows)
        total_rows += len(identities)
        for source_id, (source_name, route_dir, source_domains) in enumerate(SOURCES):
            route_path = route_dir / f"layer_{layer:02d}.safetensors"
            checks["all_route_artifact_hashes"] &= (
                sha256_file(route_path) == lock["route_artifact_sha256"][source_name][str(layer)]
            )
            with safe_open(route_path, framework="pt", device="cpu") as handle:
                for domain in source_domains:
                    domain_id = DOMAINS.index(domain)
                    mask = (layer_source == source_id) & (layer_domain == domain_id)
                    locations = mask.nonzero(as_tuple=False)
                    if not locations.numel():
                        continue
                    tokens = layer_token[mask]
                    slots = layer_slot[mask]
                    ids = handle.get_tensor(f"{domain}_router_ids")
                    checks["all_selected_routes_match_expert"] &= bool(
                        ids[tokens, slots].long().eq(locations[:, 0]).all()
                    )

        artifact = CALIBRATION_DIR / f"layer_{layer:02d}.safetensors"
        report = LAYER_DIR / f"layer_{layer:02d}.json"
        producer = json.loads(report.read_text(encoding="utf-8"))
        artifact_hash = sha256_file(artifact)
        checks["all_calibration_artifact_hashes"] &= artifact_hash == producer["artifact_sha256"]
        checks["all_calibration_report_hashes"] &= (
            sha256_file(report) == capture["layers"][str(layer)]["report_sha256"]
        )
        with safe_open(artifact, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            x = handle.get_tensor("moe_input")
            checks["all_calibration_metadata"] &= (
                metadata.get("layer") == str(layer)
                and metadata.get("selection_sha256") == sha256_file(SELECTION_ARTIFACT)
                and metadata.get("moe_input_sha256") == sha256_tensor(x)
            )
            checks["all_calibration_shapes"] &= tuple(x.shape) == (EXPERTS, ROWS, HIDDEN)
            checks["all_calibration_dtypes"] &= x.dtype == torch.bfloat16
            checks["all_calibration_finite"] &= bool(torch.isfinite(x).all())
            checks["all_calibration_tensor_hashes"] &= (
                sha256_tensor(x) == producer["moe_input_sha256"]
                == capture["layers"][str(layer)]["moe_input_sha256"]
            )
            checks["all_coordinate_copies_exact"] &= all((
                torch.equal(handle.get_tensor("source_index").long(), layer_source),
                torch.equal(handle.get_tensor("domain_index").long(), layer_domain),
                torch.equal(handle.get_tensor("token_index").long(), layer_token),
                torch.equal(handle.get_tensor("slot_index").long(), layer_slot),
            ))
        checks["all_producer_route_controls"] &= producer["official_routes_exact"] is True
        layer_hashes[str(layer)] = artifact_hash
        print(json.dumps({"layer": layer, "verified": all(checks.values())}), flush=True)

    checks["total_rows_exact"] = total_rows == LAYERS * EXPERTS * ROWS
    passed = all(checks.values())
    payload = {
        "kind": "qwen_gptq_bank_p0_calibration_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "total_checks": len(checks), "total_rows": total_rows,
        "layer_artifact_sha256": layer_hashes,
        "claim_boundary": "Independent route-provenance, coordinate, tensor, and integrity verification; no GPTQ claim.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Qwen GPTQ Bank — calibration verification\n\n"
        f"Uitkomst: **{payload['status']}** ({payload['passed_checks']}/{payload['total_checks']}).\n\n"
        f"De verifier herleidde alle {total_rows:,} selecties opnieuw tot hun officiële route-ID, "
        "controleerde uniciteit, alle coördinatenkopieën, BF16-vormen, eindigheid en alle artifact-, "
        "rapport- en tensorhashes.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "checks": f"{payload['passed_checks']}/{payload['total_checks']}", "rows": total_rows}, indent=2))
