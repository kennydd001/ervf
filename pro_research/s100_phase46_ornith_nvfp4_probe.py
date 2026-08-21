"""Phase46: real Ornith-1.5 checkpoint layout and SM120 NVFP4 transfer probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from common import REPO, environment_snapshot, utc_now, write_json_atomic


RESULTS = REPO / "pro_research" / "results" / "s100_phase46"
PREREG = REPO / "pro_research" / "S100_PHASE46_ORNITH_NVFP4_PREREGISTRATION.md"
NRMSE_MAX = 0.020
COSINE_MIN = 0.9990
NMAX_MAX = 0.050
M8_ROW_NMAX_MAX = 0.005


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def _tensor_bytes(record: dict[str, Any]) -> int:
    begin, end = (int(v) for v in record["data_offsets"])
    return end - begin


def _layer_root(entries: dict[str, str], num_layers: int) -> str:
    suffix = ".0.input_layernorm.weight"
    matches = sorted(name[:-len(suffix)] for name in entries if name.endswith(suffix))
    complete = [
        root for root in matches
        if all(f"{root}.{layer}.input_layernorm.weight" in entries for layer in range(num_layers))
    ]
    if len(complete) != 1:
        raise RuntimeError(
            f"unable to identify one complete {num_layers}-layer language root: "
            f"candidates={matches} complete={complete}"
        )
    return complete[0]


def _triple_category(base: str) -> str:
    if base == "lm_head":
        return "lm_head"
    if ".mlp.experts." in base:
        return "routed_" + base.rsplit(".", 1)[-1]
    if ".mlp.shared_expert." in base:
        return "shared_" + base.rsplit(".", 1)[-1]
    return "other"


def _select(triples: list[dict[str, Any]], label: str, suffix: str) -> dict[str, Any]:
    matches = [row for row in triples if row["base"].endswith(suffix)]
    if not matches:
        raise RuntimeError(f"no NVFP4 triple for {label}: *{suffix}")
    row = dict(sorted(matches, key=lambda value: value["base"])[0])
    row["label"] = label
    return row


def _layout_audit(
    source: str,
    config: dict[str, Any],
    entries: dict[str, str],
    headers: dict[str, dict[str, Any]],
    triples: list[dict[str, Any]],
) -> dict[str, Any]:
    text = _text_config(config)
    expected = {
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "full_attention_interval": 4,
    }
    observed = {key: text.get(key) for key in expected}
    root = _layer_root(entries, int(text.get("num_hidden_layers", -1)))
    layers: list[dict[str, Any]] = []
    incomplete_experts: list[dict[str, int]] = []
    incomplete_shared: list[int] = []
    for layer in range(int(text.get("num_hidden_layers", -1))):
        prefix = f"{root}.{layer}"
        linear = any(name.startswith(prefix + ".linear_attn.") for name in entries)
        full = any(name.startswith(prefix + ".self_attn.") for name in entries)
        router = prefix + ".mlp.gate.weight" in entries
        complete = 0
        for expert in range(int(text.get("num_experts", -1))):
            ep = f"{prefix}.mlp.experts.{expert}"
            if all(
                ep + f".{proj}.{tail}" in entries
                for proj in ("gate_proj", "up_proj", "down_proj")
                for tail in ("weight", "weight_scale", "weight_scale_2")
            ):
                complete += 1
        if complete != int(text.get("num_experts", -1)):
            incomplete_experts.append({"layer": layer, "complete": complete})
        shared = f"{prefix}.mlp.shared_expert"
        shared_complete = all(
            shared + f".{proj}.{tail}" in entries
            for proj in ("gate_proj", "up_proj", "down_proj")
            for tail in ("weight", "weight_scale", "weight_scale_2")
        )
        if not shared_complete:
            incomplete_shared.append(layer)
        layers.append({
            "layer": layer,
            "kind": "linear" if linear and not full else "full" if full and not linear else "invalid",
            "router": router,
            "complete_routed_experts": complete,
            "shared_complete": shared_complete,
        })

    category_counts = Counter(_triple_category(row["base"]) for row in triples)
    category_bytes: Counter[str] = Counter()
    payload_categories: Counter[str] = Counter()
    tensor_payload = 0
    language_parent = root.rsplit(".layers", 1)[0]
    for shard, header in headers.items():
        del shard
        for name, record in header.items():
            if name == "__metadata__":
                continue
            size = _tensor_bytes(record)
            tensor_payload += size
            if name.startswith("mtp."):
                payload_categories["mtp"] += size
            elif ".mlp.experts." in name:
                payload_categories["routed_experts"] += size
            elif ".mlp.shared_expert." in name:
                payload_categories["shared_experts"] += size
            elif ".visual." in name or name.startswith("visual."):
                payload_categories["vision"] += size
            elif "embed_tokens" in name:
                payload_categories["token_embedding"] += size
            elif name == "lm_head.weight" or name.startswith("lm_head."):
                payload_categories["lm_head"] += size
            elif name.startswith(language_parent + "."):
                payload_categories["language_trunk"] += size
            else:
                payload_categories["other"] += size
    for row in triples:
        size = sum(
            _tensor_bytes(headers[entries[row[key]]][row[key]])
            for key in ("weight", "scale", "global")
        )
        category_bytes[_triple_category(row["base"])] += size

    layer0_expert0 = [
        row for row in triples
        if f"{root}.0.mlp.experts.0." in row["base"]
    ]
    one_expert = sum(
        sum(
            _tensor_bytes(headers[entries[row[key]]][row[key]])
            for key in ("weight", "scale", "global")
        )
        for row in layer0_expert0
    )
    one_expert_front = sum(
        sum(
            _tensor_bytes(headers[entries[row[key]]][row[key]])
            for key in ("weight", "scale", "global")
        )
        for row in layer0_expert0
        if row["base"].endswith((".gate_proj", ".up_proj"))
    )
    one_expert_down = one_expert - one_expert_front
    top_k = int(text["num_experts_per_tok"])
    num_layers = int(text["num_hidden_layers"])
    cache_capacity = 72
    text_shell = sum(
        payload_categories[key]
        for key in ("shared_experts", "token_embedding", "lm_head", "language_trunk")
    )
    dflash_bytes = 771_812_352
    usable_vram = int(8 * 2**30 * 0.90)
    runtime_reserve = 512 * 2**20
    cache_budget = max(0, usable_vram - runtime_reserve - dflash_bytes - text_shell)
    cache_capacity_with_dflash = cache_budget // max(one_expert * num_layers, 1)
    front_cache_capacity_with_dflash = cache_budget // max(one_expert_front * num_layers, 1)
    plan = {
        "one_complete_swiglu_expert_bytes": one_expert,
        "one_complete_swiglu_expert_MiB": one_expert / 2**20,
        "one_gate_plus_up_expert_bytes": one_expert_front,
        "one_gate_plus_up_expert_MiB": one_expert_front / 2**20,
        "one_down_expert_bytes": one_expert_down,
        "one_down_expert_MiB": one_expert_down / 2**20,
        "top8_one_layer_stream_bytes": one_expert * top_k,
        "top8_one_layer_stream_MiB": one_expert * top_k / 2**20,
        "top8_all_layers_without_reuse_GiB": one_expert * top_k * num_layers / 2**30,
        "cache72_all_layers_complete_expert_GiB": one_expert * cache_capacity * num_layers / 2**30,
        "all_routed_nvfp4_payload_GiB": sum(
            value for key, value in category_bytes.items() if key.startswith("routed_")
        ) / 2**30,
        "all_shared_nvfp4_payload_MiB": sum(
            value for key, value in category_bytes.items() if key.startswith("shared_")
        ) / 2**20,
        "non_routed_tensor_payload_GiB": (
            tensor_payload - sum(
                value for key, value in category_bytes.items() if key.startswith("routed_")
            )
        ) / 2**30,
        "payload_categories_GiB": {
            key: value / 2**30 for key, value in sorted(payload_categories.items())
        },
        "text_only_resident_shell_GiB": text_shell / 2**30,
        "dflash_resident_GiB": dflash_bytes / 2**30,
        "planning_usable_vram_GiB": usable_vram / 2**30,
        "planning_runtime_reserve_GiB": runtime_reserve / 2**30,
        "complete_expert_cache_capacity_with_dflash": int(cache_capacity_with_dflash),
        "complete_expert_cache_GiB_with_dflash": (
            cache_capacity_with_dflash * one_expert * num_layers / 2**30
        ),
        "gate_plus_up_cache_capacity_with_dflash": int(front_cache_capacity_with_dflash),
        "gate_plus_up_cache_GiB_with_dflash": (
            front_cache_capacity_with_dflash * one_expert_front * num_layers / 2**30
        ),
        "sparse_down_warning": (
            "Ornith uses SwiGLU (SiLU(gate)*up), so the Nemotron ReLU2 exact-zero "
            "masked-down shortcut is not expected to transfer. Full-down traffic or "
            "a separate down cache is required."
        ),
    }
    full_layers = [row["layer"] for row in layers if row["kind"] == "full"]
    linear_layers = [row["layer"] for row in layers if row["kind"] == "linear"]
    gates = {
        "P46_G1_qwen35_geometry": observed == expected,
        "P46_G2_attention_pattern_30_linear_10_full": (
            len(linear_layers) == 30
            and full_layers == list(range(3, 40, 4))
        ),
        "P46_G3_all_routers_present": all(row["router"] for row in layers),
        "P46_G4_all_routed_swiglu_experts_complete": not incomplete_experts,
        "P46_G5_all_shared_swiglu_experts_complete": not incomplete_shared,
        "P46_G6_expected_nvfp4_triple_counts": category_counts == Counter({
            "routed_gate_proj": 40 * 256,
            "routed_up_proj": 40 * 256,
            "routed_down_proj": 40 * 256,
            "shared_gate_proj": 40,
            "shared_up_proj": 40,
            "shared_down_proj": 40,
            "lm_head": 1,
        }),
    }
    return {
        "source": source,
        "model_type": config.get("model_type"),
        "architecture": config.get("architectures"),
        "observed_geometry": observed,
        "expected_geometry": expected,
        "layer_root": root,
        "linear_layers": linear_layers,
        "full_attention_layers": full_layers,
        "incomplete_experts": incomplete_experts,
        "incomplete_shared": incomplete_shared,
        "nvfp4_triple_count": len(triples),
        "nvfp4_category_counts": dict(sorted(category_counts.items())),
        "nvfp4_category_bytes": dict(sorted(category_bytes.items())),
        "tensor_payload_bytes": tensor_payload,
        "offload_plan": plan,
        "gates": gates,
    }


def _remote_headers(repo_id: str):
    """Read config/index plus safetensors JSON headers without weight payloads."""
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    encoded_repo = quote(repo_id, safe="/")
    def retry(call, label: str):
        last = None
        for attempt in range(4):
            try:
                return call()
            except (ConnectionError, OSError, TimeoutError) as exc:
                last = exc
                if attempt == 3:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"{label} failed after 4 attempts: {last}") from last

    def read_revision() -> str:
        with urlopen(f"https://huggingface.co/api/models/{encoded_repo}", timeout=90) as response:
            return json.loads(response.read())["sha"]

    revision = retry(read_revision, "model metadata")
    base = f"https://huggingface.co/{encoded_repo}/resolve/{revision}/"

    def get_small(name: str) -> bytes:
        def read() -> bytes:
            with urlopen(base + quote(name, safe="/"), timeout=90) as response:
                return response.read()

        return retry(read, name)

    def get_range(name: str, begin: int, end: int, timeout: int):
        def read():
            request = Request(
                base + quote(name, safe="/"),
                headers={"Range": f"bytes={begin}-{end}"},
            )
            with urlopen(request, timeout=timeout) as response:
                return response.status, dict(response.headers), response.read()

        return retry(read, f"{name}[{begin}:{end}]")

    config_raw = get_small("config.json")
    index_raw = get_small("model.safetensors.index.json")
    config = json.loads(config_raw)
    index = json.loads(index_raw)
    entries = index["weight_map"]
    headers = {}
    header_hashes = {}
    shard_sizes = {}
    for shard in sorted(set(entries.values())):
        first_status, first_headers, first_content = get_range(shard, 0, 7, 90)
        if first_status != 206 or len(first_content) != 8:
            raise RuntimeError(
                f"{shard}: range preflight failed status={first_status} "
                f"bytes={len(first_content)}"
            )
        header_length = int.from_bytes(first_content, "little")
        content_range = next(
            (value for key, value in first_headers.items() if key.lower() == "content-range"),
            "",
        )
        if "/" not in content_range:
            raise RuntimeError(f"{shard}: missing total size in Content-Range")
        shard_sizes[shard] = int(content_range.rsplit("/", 1)[1])
        last = 8 + header_length - 1
        status, _headers, content = get_range(shard, 8, last, 180)
        if status != 206 or len(content) != header_length:
            raise RuntimeError(
                f"{shard}: header range failed status={status} "
                f"bytes={len(content)} expected={header_length}"
            )
        headers[shard] = json.loads(content)
        header_hashes[shard] = hashlib.sha256(content).hexdigest()
    provenance = {
        "repo_id": repo_id,
        "revision": revision,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "header_sha256": header_hashes,
        "shard_bytes": shard_sizes,
        "payload_bytes_not_downloaded": sum(shard_sizes.values()),
    }
    return config, entries, headers, provenance


def _native_probe(
    model_dir: Path,
    entries: dict[str, str],
    headers: dict[str, dict[str, Any]],
    triples: list[dict[str, Any]],
    probe_layer: int,
) -> dict[str, Any]:
    # Importing Torch creates the CUDA context, so the WDDM idle check must run
    # first. The downloader itself does not use CUDA and is not a blocker.
    from diag_native_nvfp4_c3a_real_weight_v2 import require_gpu_idle_wddm

    idle = require_gpu_idle_wddm()
    import torch
    import torch.nn.functional as F
    import native_nvfp4_c3a_layout_v2 as layout_v2
    import native_nvfp4_c3a_lib as c3lib

    c3lib.MODEL_DIR = model_dir
    c3lib.INDEX = model_dir / "model.safetensors.index.json"
    layout_v2.install(c3lib)
    st, sw = F.ScalingType, F.SwizzleType
    capability = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (-1, -1)
    api_ok = bool(
        torch.cuda.is_available()
        and capability >= (12, 0)
        and hasattr(F, "scaled_mm")
        and hasattr(torch, "float4_e2m1fn_x2")
        and hasattr(st, "BlockWise1x16")
        and hasattr(st, "TensorWise")
        and hasattr(sw, "SWIZZLE_32_4_4")
        and hasattr(sw, "NO_SWIZZLE")
    )
    selections = [
        _select(triples, "routed_gate", f".layers.{probe_layer}.mlp.experts.0.gate_proj"),
        _select(triples, "routed_down", f".layers.{probe_layer}.mlp.experts.0.down_proj"),
        _select(triples, "shared_gate", f".layers.{probe_layer}.mlp.shared_expert.gate_proj"),
        _select(triples, "lm_head", "lm_head"),
    ]
    contract = all(
        row["weight_shape"] == [row["N"], row["K"] // 2]
        and row["scale_shape"] == [row["N"], row["K"] // 16]
        and row["weight_dtype"] == "U8"
        and row["scale_dtype"] == "F8_E4M3"
        and row["global_dtype"] == "F32"
        for row in selections
    )
    smoke = c3lib.two_level_smoke(torch, F, st, sw) if api_ok else {}
    l2 = 0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        l2 = int(getattr(props, "L2_cache_size", 0) or getattr(props, "l2_cache_size", 0) or 0)
    families = (
        [c3lib.run_family(torch, F, st, sw, row, entries, headers, l2) for row in selections]
        if api_ok and contract and smoke.get("all_equal_expected") and l2 > 0
        else []
    )
    finite = bool(families) and all(
        family["native"][mode]["finite"]
        for family in families
        for mode in ("M1", "M8")
    )
    reference = bool(families) and all(
        family["native"]["M1"]["reference_metrics_first_row"]["normalized_rmse"] <= NRMSE_MAX
        and family["native"]["M1"]["reference_metrics_first_row"]["cosine"] >= COSINE_MIN
        and family["native"]["M1"]["reference_metrics_first_row"]["normalized_max_abs_error"] <= NMAX_MAX
        for family in families
    )
    m8_rows = bool(families) and all(
        family["M8_identical_rows_normalized_max_diff"] <= M8_ROW_NMAX_MAX
        for family in families
    )
    gates = {
        "P46_G7_sm120_native_api": api_ok and l2 > 0,
        "P46_G8_selected_real_triples_match_contract": contract,
        "P46_G9_two_level_known_value_exact": bool(smoke.get("all_equal_expected")),
        "P46_G10_real_M1_M8_execute_finite": finite,
        "P46_G11_independent_reference_thresholds": reference,
        "P46_G12_identical_M8_rows_agree": m8_rows,
    }
    return {
        "gpu_idle_preflight": idle,
        "probe_layer": int(probe_layer),
        "api": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(capability),
            "l2_bytes": l2,
        },
        "selection": selections,
        "two_level_smoke": smoke,
        "families": families,
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ornith-1.5 real NVFP4 transfer probe")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--probe-layer", type=int, default=0)
    args = parser.parse_args()
    if not args.model_dir and not args.repo_id:
        parser.error("one of --model-dir or --repo-id is required")
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else None
    if not args.layout_only and model_dir is None:
        parser.error("native execution requires --model-dir (optionally with --repo-id headers)")
    output = RESULTS / f"S100_PHASE46_{args.tag.upper()}.json"
    payload: dict[str, Any] = {
        "kind": "s100_phase46_ornith_nvfp4_transfer_probe",
        "status": "started",
        "tag": args.tag,
        "model_dir": str(model_dir) if model_dir else None,
        "repo_id": args.repo_id,
        "layout_only": bool(args.layout_only),
        "probe_layer": int(args.probe_layer),
        "started_utc": utc_now(),
        "preregistration": str(PREREG.relative_to(REPO)),
        "claim_boundary": (
            "real checkpoint layout, offload byte plan and native matrix kernels only; "
            "not a full Qwen3.5 token decoder or DFlash acceptance claim"
        ),
    }
    try:
        import native_nvfp4_c3a_lib as c3lib

        extra_environment_files = [Path(__file__), PREREG]
        if args.repo_id:
            config, entries, headers, provenance = _remote_headers(args.repo_id)
            payload["remote_provenance"] = provenance
            source_name = f"hf://{args.repo_id}@{provenance['revision']}"
        else:
            assert model_dir is not None
            config_path = model_dir / "config.json"
            index_path = model_dir / "model.safetensors.index.json"
            if not config_path.is_file() or not index_path.is_file():
                raise FileNotFoundError(f"incomplete model snapshot: {model_dir}")
            c3lib.MODEL_DIR = model_dir
            c3lib.INDEX = index_path
            config = json.loads(config_path.read_text(encoding="utf-8"))
            entries, headers = c3lib.load_index_headers()
            source_name = str(model_dir)
            extra_environment_files.extend((config_path, index_path))
        triples = c3lib.all_nvfp4_triples(entries, headers)
        layout = _layout_audit(source_name, config, entries, headers, triples)
        payload["layout"] = layout
        if not args.layout_only:
            assert model_dir is not None
            payload["native"] = _native_probe(
                model_dir, entries, headers, triples, int(args.probe_layer)
            )
        all_gates = dict(layout["gates"])
        if not args.layout_only:
            all_gates.update(payload["native"]["gates"])
        payload["gates"] = all_gates
        payload["status"] = "measured_pass" if all(all_gates.values()) else "measured_fail"
        payload["environment"] = environment_snapshot(extra_environment_files)
        payload["completed_utc"] = utc_now()
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    write_json_atomic(output, payload, archive=True)
    native = payload.get("native") or {}
    print(json.dumps({
        "status": payload.get("status"),
        "tag": args.tag,
        "gates": payload.get("gates"),
        "geometry": (payload.get("layout") or {}).get("observed_geometry"),
        "offload_plan": (payload.get("layout") or {}).get("offload_plan"),
        "native_api": native.get("api"),
        "family_M8_over_M1": {
            row["label"]: row["cold_timing"].get("M8_over_M1")
            for row in native.get("families", [])
        },
        "error": (payload.get("error") or {}).get("message"),
        "output": str(output),
    }, indent=2))
    return 0 if payload.get("status") == "measured_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
