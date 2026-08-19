from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

OFFICIAL_LIGHTNING_ID = (
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
)
OFFICIAL_NANO_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"

SMALL_TEXT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "README.md",
    "configuration_nemotron_h.py",
    "modeling_nemotron_h.py",
    "model.safetensors.index.json",
    "ACQUISITION_PROVENANCE.json",
)

MTP_RE = re.compile(
    r"(?:^|[._/\-])(?:mtp|multi[_\- ]?token(?:[_\- ]?prediction)?|"
    r"nextn|next[_\- ]?token|prediction[_\- ]?head|speculator)(?:$|[._/\-])",
    re.I,
)
LATENT_RE = re.compile(
    r"(?:latent[_\- ]?moe|latentmoe|latent[_\- ]?expert|"
    r"expert[_\- ]?latent|latent[_\- ]?proj|latent_dim)",
    re.I,
)
NANO_RE = re.compile(r"Nemotron[-_ ]?3[-_ ]?Nano", re.I)
LIGHTNING_RE = re.compile(r"Nemotron[-_ ]?3(?:\.5|_5)[-_ ]?Lightning", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path, limit: int = 8 << 20) -> str:
    try:
        if path.stat().st_size > limit:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(item, name)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def scalar(config: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in config:
            return config[name]
    return None


def classify(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    config = read_json(model_dir / "config.json")
    index = read_json(model_dir / "model.safetensors.index.json")
    acquisition = read_json(model_dir / "ACQUISITION_PROVENANCE.json")

    texts: dict[str, str] = {}
    for name in SMALL_TEXT_FILES:
        path = model_dir / name
        if path.exists():
            texts[name] = read_text(path)

    tensor_names: list[str] = []
    if isinstance(index, dict) and isinstance(index.get("weight_map"), dict):
        tensor_names = sorted(str(x) for x in index["weight_map"])

    # Human-readable README text may compare several model families and is
    # never strong identity evidence. Structural identity uses config/index
    # and the acquisition manifest; text remains supporting-only evidence.
    structural_lines: list[str] = []
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "configuration_nemotron_h.py",
        "modeling_nemotron_h.py",
        "ACQUISITION_PROVENANCE.json",
    ):
        structural_lines.append(texts.get(name, ""))
    structural_lines.extend(tensor_names)
    structural_combined = "\n".join(structural_lines)
    supporting_combined = "\n".join(
        [str(model_dir), *texts.values(), *tensor_names]
    )

    nano_hits = sorted(set(NANO_RE.findall(supporting_combined)))
    lightning_hits = sorted(set(LIGHTNING_RE.findall(supporting_combined)))
    structural_nano_hits = sorted(
        set(NANO_RE.findall(structural_combined))
    )
    structural_lightning_hits = sorted(
        set(LIGHTNING_RE.findall(structural_combined))
    )
    mtp_hits = sorted(
        x for x in tensor_names if MTP_RE.search(x)
    )
    latent_hits = sorted(
        x for x in tensor_names if LATENT_RE.search(x)
    )

    text_mtp_hits: list[str] = []
    text_latent_hits: list[str] = []
    for name, text in texts.items():
        if MTP_RE.search(text):
            text_mtp_hits.append(name)
        if LATENT_RE.search(text):
            text_latent_hits.append(name)

    config_flat = list(flatten(config or {}))
    config_mtp_hits = [
        key for key, value in config_flat
        if MTP_RE.search(f"{key}={value}")
    ]
    config_latent_hits = [
        key for key, value in config_flat
        if LATENT_RE.search(f"{key}={value}")
    ]

    repo_id = None
    revision = None
    if isinstance(acquisition, dict):
        repo_id = acquisition.get("repo_id")
        revision = (
            acquisition.get("resolved_revision")
            or acquisition.get("revision")
        )

    # Macro-shape agreement is necessary but deliberately not sufficient:
    # Nano and Lightning are both 30B-A3B Nemotron-H hybrids.
    macro = {
        "architectures": (config or {}).get("architectures"),
        "model_type": (config or {}).get("model_type"),
        "hidden_size": scalar(config or {}, "hidden_size"),
        "num_hidden_layers": scalar(config or {}, "num_hidden_layers"),
        "n_routed_experts": scalar(
            config or {}, "n_routed_experts", "num_local_experts"
        ),
        "num_experts_per_tok": scalar(
            config or {}, "num_experts_per_tok", "num_experts_per_token"
        ),
        "max_position_embeddings": scalar(
            config or {}, "max_position_embeddings"
        ),
    }
    macro_green = (
        macro["hidden_size"] == 2688
        and macro["num_hidden_layers"] == 52
        and macro["n_routed_experts"] == 128
        and macro["num_experts_per_tok"] == 6
    )

    nano_strong = (
        repo_id == OFFICIAL_NANO_ID
        or OFFICIAL_NANO_ID.lower() in structural_combined.lower()
        or bool(structural_nano_hits)
    )
    exact_lightning_manifest = repo_id == OFFICIAL_LIGHTNING_ID
    revision_green = bool(
        isinstance(revision, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", revision)
    )
    # README/model-card wording is supporting evidence only. Structural
    # confirmation must come from config fields or tensor names.
    mtp_structural_green = bool(mtp_hits or config_mtp_hits)
    latent_structural_green = bool(latent_hits or config_latent_hits)
    mtp_green = mtp_structural_green
    latent_green = latent_structural_green

    conflict = nano_strong and (
        exact_lightning_manifest or bool(lightning_hits)
    )

    if conflict:
        classification = "identity_conflict"
    elif nano_strong:
        classification = "nano_v3_confirmed"
    elif (
        exact_lightning_manifest
        and revision_green
        and macro_green
        and mtp_green
        and latent_green
    ):
        classification = "lightning35_confirmed"
    elif (
        exact_lightning_manifest
        or bool(lightning_hits)
        or "lightning" in model_dir.name.lower()
    ):
        classification = "lightning_claim_unverified"
    else:
        classification = "unknown"

    metadata_hashes = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "generation_config.json",
        "ACQUISITION_PROVENANCE.json",
    ):
        path = model_dir / name
        metadata_hashes[name] = sha256(path) if path.exists() else None

    return {
        "kind": "s100_lightning_model_guard",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "classification": classification,
        "official_lightning_id": OFFICIAL_LIGHTNING_ID,
        "official_nano_id": OFFICIAL_NANO_ID,
        "acquisition_repo_id": repo_id,
        "acquisition_revision": revision,
        "immutable_revision_green": revision_green,
        "macro_contract": macro,
        "macro_contract_green": macro_green,
        "mtp_marker_green": mtp_green,
        "latentmoe_marker_green": latent_green,
        "nano_marker_green": nano_strong,
        "identity_conflict": conflict,
        "marker_evidence": {
            "mtp_tensor_hits": mtp_hits[:100],
            "latent_tensor_hits": latent_hits[:100],
            "mtp_text_files_supporting_only": text_mtp_hits,
            "latent_text_files_supporting_only": text_latent_hits,
            "mtp_config_keys": config_mtp_hits[:100],
            "latent_config_keys": config_latent_hits[:100],
            "nano_regex_hits_supporting_only": nano_hits,
            "lightning_regex_hits_supporting_only": lightning_hits,
            "nano_structural_hits": structural_nano_hits,
            "lightning_structural_hits": structural_lightning_hits,
        },
        "metadata_sha256": metadata_hashes,
        "gate": classification == "lightning35_confirmed",
        "policy": (
            "A directory name or shared 30B-A3B macro shape is never enough. "
            "Confirmed Lightning requires the official acquisition manifest, "
            "an immutable resolved revision, the macro contract, and structural "
            "MTP and LatentMoE markers from config or tensor names. README words "
            "are supporting evidence only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <model_dir>/MODEL_PROVENANCE.json",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Return zero for lightning_claim_unverified; audit use only.",
    )
    args = parser.parse_args()

    result = classify(args.model_dir)
    output = args.output or args.model_dir / "MODEL_PROVENANCE.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))

    if result["classification"] == "lightning35_confirmed":
        return 0
    if (
        args.allow_unverified
        and result["classification"] == "lightning_claim_unverified"
    ):
        return 0
    if result["classification"] == "nano_v3_confirmed":
        return 3
    if result["classification"] == "identity_conflict":
        return 5
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
