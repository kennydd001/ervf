from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from moe_lab.reporting import ROOT


LAYERS = 48
EXPERTS = 128
ROWS = 128
TOP_K = 8
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
SOURCES = (
    ("hera", ROOT / "reports/runs/hera_moe/p0_routes", DOMAINS),
    ("dhera", ROOT / "reports/runs/dhera_moe/p0_routes", DOMAINS),
    ("supplement_a", ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_routes", DOMAINS),
    (
        "supplement_b",
        ROOT / "reports/runs/qwen_gptq_bank/p0_supplement_b_routes",
        ("math", "instruction"),
    ),
)
COVERAGE = ROOT / "reports/qwen_gptq_bank/p0_coverage_result_b.json"
PREREG = ROOT / "reports/qwen_gptq_bank/P0_FULL_BANK_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/qwen_gptq_bank/p0_calibration_selection_lock.json"
ARTIFACT = ROOT / "reports/runs/qwen_gptq_bank/p0_calibration_selection.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite the calibration selection lock")
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    if coverage["status"] != "coverage_pass" or not coverage["coverage"]["all_pairs_pass"]:
        raise RuntimeError("coverage gate did not pass")

    source_index = torch.full((LAYERS, EXPERTS, ROWS), -1, dtype=torch.int8)
    domain_index = torch.full_like(source_index, -1)
    token_index = torch.full((LAYERS, EXPERTS, ROWS), -1, dtype=torch.int32)
    slot_index = torch.full_like(source_index, -1)
    route_hashes: dict[str, dict[str, str]] = {
        name: {} for name, _directory, _domains in SOURCES
    }
    selected_by_source = torch.zeros((LAYERS, len(SOURCES)), dtype=torch.int64)
    selected_by_domain = torch.zeros((LAYERS, len(DOMAINS)), dtype=torch.int64)

    for layer in range(LAYERS):
        filled = torch.zeros(EXPERTS, dtype=torch.int64)
        for source_id, (source_name, directory, source_domains) in enumerate(SOURCES):
            path = directory / f"layer_{layer:02d}.safetensors"
            route_hashes[source_name][str(layer)] = sha256(path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                for domain in source_domains:
                    domain_id = DOMAINS.index(domain)
                    flat = handle.get_tensor(f"{domain}_router_ids").long().reshape(-1)
                    counts = torch.bincount(flat, minlength=EXPERTS)
                    order = torch.argsort(flat, stable=True)
                    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
                    for expert in torch.where(filled < ROWS)[0].tolist():
                        take = min(ROWS - int(filled[expert]), int(counts[expert]))
                        if take == 0:
                            continue
                        positions = order[offsets[expert] : offsets[expert] + take]
                        begin = int(filled[expert])
                        end = begin + take
                        source_index[layer, expert, begin:end] = source_id
                        domain_index[layer, expert, begin:end] = domain_id
                        token_index[layer, expert, begin:end] = positions.div(TOP_K, rounding_mode="floor").to(torch.int32)
                        slot_index[layer, expert, begin:end] = positions.remainder(TOP_K).to(torch.int8)
                        filled[expert] = end
                        selected_by_source[layer, source_id] += take
                        selected_by_domain[layer, domain_id] += take
        if not bool((filled == ROWS).all()):
            missing = {expert: int(filled[expert]) for expert in torch.where(filled < ROWS)[0].tolist()}
            raise RuntimeError(f"selection incomplete at layer {layer}: {missing}")
        print(json.dumps({"layer": layer, "selected_rows": int(filled.sum())}), flush=True)

    if bool((source_index < 0).any() or (domain_index < 0).any() or (token_index < 0).any() or (slot_index < 0).any()):
        raise RuntimeError("negative selection coordinate remained")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "source_index": source_index,
        "domain_index": domain_index,
        "token_index": token_index,
        "slot_index": slot_index,
    }
    save_file(tensors, ARTIFACT, metadata={
        "kind": "qwen_gptq_bank_p0_calibration_selection",
        "rows_per_expert": str(ROWS),
        "ordering": "source_domain_token_slot",
    })
    payload = {
        "kind": "qwen_gptq_bank_p0_calibration_selection_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "coverage_result_sha256": sha256(COVERAGE),
        "artifact": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_sha256": sha256(ARTIFACT),
        "shape": [LAYERS, EXPERTS, ROWS],
        "sources": [name for name, _directory, _domains in SOURCES],
        "domains": list(DOMAINS),
        "ordering": "source-major, domain-major, token-major, top-k-slot-major",
        "selected_by_source_per_layer": selected_by_source.tolist(),
        "selected_by_domain_per_layer": selected_by_domain.tolist(),
        "route_artifact_sha256": route_hashes,
        "all_pairs_exactly_128": True,
        "routes_opened_before_rule": False,
        "note": "The selection rule was preregistered; this file materializes its deterministic result.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": payload["artifact"],
        "artifact_sha256": payload["artifact_sha256"],
        "selected_rows": int(source_index.numel()),
    }, indent=2))
