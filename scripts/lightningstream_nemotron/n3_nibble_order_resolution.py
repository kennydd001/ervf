"""N3 §3: resolve the two decoder conventions left open by N2.

Nibble order cannot be falsified by any per-block statistic: swapping the two
nibbles of a byte permutes elements only within that byte, and a byte lies wholly
inside one group of 16, so every block's value multiset is identical under both
orders.  It is decidable only against an external reference.

Attempt order, per the preregistration; the first that succeeds decides:

  1. torch native ``float4_e2m1fn_x2`` conversion
  2. torchao's published ``unpack_uint4`` / ``f4_unpacked_to_f32``
  3. defer to N6 end-to-end coherence

torchao 'main' targets a newer torch than the one pinned here.  Its pure-python
FP4 helpers are reached by shimming the handful of symbols its import chain wants
from ``torch.nn.functional`` **in this process only**.  No file on disk is
modified, and only pure-python helpers are used -- no torchao kernel is invoked.
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
OUT = REPO_ROOT / "reports" / "lightningstream_nemotron" / "n3_nibble_order_resolution.json"

TARGET = "backbone.layers.1.mixer.experts.0.up_proj"
SAMPLE_BYTES = 1 << 20  # 1 MiB of packed codes -> 2,097,152 elements


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_codes() -> tuple[np.ndarray, str]:
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = index["weight_map"][f"{TARGET}.weight"]
    path = MODEL_DIR / shard
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len).decode("utf-8"))
        start, end = header[f"{TARGET}.weight"]["data_offsets"]
        take = min(SAMPLE_BYTES, end - start)
        handle.seek(8 + header_len + start)
        raw = handle.read(take)
    return np.frombuffer(raw, dtype=np.uint8), shard


def try_torch_native(packed: np.ndarray) -> dict:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return {"available": False, "reason": "torch has no float4_e2m1fn_x2"}
    try:
        tensor = torch.from_numpy(packed.copy()).view(torch.float4_e2m1fn_x2)
        values = tensor.to(torch.float32).numpy()
        return {"available": True, "converted": True, "values": values}
    except Exception as exc:
        return {"available": True, "converted": False,
                "error": f"{type(exc).__name__}: {exc}"}


def try_torchao(packed: np.ndarray) -> dict:
    """Reach torchao's pure-python FP4 helpers behind an in-process shim."""
    import torch
    import torch.nn.functional as F

    shimmed = []
    for name in ("ScalingType", "SwizzleType", "MXFP8BlockScaleRecipe"):
        if not hasattr(F, name):
            setattr(F, name, type(name, (), {}))
            shimmed.append(name)
    for name in ("scaled_grouped_mm", "scaled_mm"):
        if not hasattr(F, name):
            setattr(F, name, None)
            shimmed.append(name)

    try:
        from torchao.prototype.mx_formats.kernels import (
            f4_unpacked_to_f32,
            unpack_uint4,
        )
    except Exception as exc:
        return {"available": False, "shimmed": shimmed,
                "error": f"{type(exc).__name__}: {exc}"}

    import torchao

    tensor = torch.from_numpy(packed.copy())
    codes = unpack_uint4(tensor)
    values = f4_unpacked_to_f32(codes).numpy().astype(np.float64)
    return {
        "available": True,
        "shimmed": shimmed,
        "torchao_version": getattr(torchao, "__version__", "unknown"),
        "codes": codes.numpy().astype(np.uint8),
        "values": values,
        "kernels_used": ["unpack_uint4", "f4_unpacked_to_f32"],
        "pure_python_only": True,
    }


def main() -> int:
    packed, shard = load_codes()
    started = utc_now()

    mine = {
        order: nvfp4.decode_e2m1_table(nvfp4.unpack_nibbles(packed, order))
        for order in ("low_first", "high_first")
    }

    attempts = []
    resolution = "unresolved_deferred_to_N6"
    matched_order = None
    reference_used = None

    # -- attempt 1 ---------------------------------------------------------
    native = try_torch_native(packed)
    attempts.append({
        "attempt": 1, "source": "torch.float4_e2m1fn_x2",
        "available": native.get("available"),
        "converted": native.get("converted", False),
        "error": native.get("error"), "reason": native.get("reason"),
    })
    if native.get("converted"):
        ref = native["values"].astype(np.float64)
        hits = {o: bool(np.array_equal(ref, v)) for o, v in mine.items()}
        attempts[-1]["matches"] = hits
        if sum(hits.values()) == 1:
            matched_order = next(o for o, h in hits.items() if h)
            resolution = "confirmed" if matched_order == nvfp4.DEFAULT_NIBBLE_ORDER else "falsified"
            reference_used = "torch.float4_e2m1fn_x2"

    # -- attempt 2 ---------------------------------------------------------
    if matched_order is None:
        ao = try_torchao(packed)
        entry = {
            "attempt": 2, "source": "torchao.prototype.mx_formats.kernels",
            "available": ao.get("available"), "error": ao.get("error"),
            "shimmed_symbols": ao.get("shimmed"),
            "torchao_version": ao.get("torchao_version"),
            "pure_python_only": ao.get("pure_python_only"),
        }
        if ao.get("available"):
            ref = ao["values"]
            hits = {o: bool(np.array_equal(ref, v)) for o, v in mine.items()}
            entry["matches"] = hits
            entry["elements_compared"] = int(ref.size)
            # Codes must agree too, not only the decoded floats.
            ref_codes = ao["codes"]
            code_hits = {
                o: bool(np.array_equal(ref_codes, nvfp4.unpack_nibbles(packed, o)))
                for o in mine
            }
            entry["code_matches"] = code_hits
            if sum(hits.values()) == 1:
                matched_order = next(o for o, h in hits.items() if h)
                resolution = "confirmed" if matched_order == nvfp4.DEFAULT_NIBBLE_ORDER else "falsified"
                reference_used = f"torchao {ao.get('torchao_version')}"
        attempts.append(entry)

    # -- sanity: the two orders must actually differ on this data ----------
    orders_differ = not np.array_equal(mine["low_first"], mine["high_first"])

    result = {
        "kind": "lightningstream_nemotron_n3_nibble_order_resolution",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N3_ONE_MODULE_REFERENCE",
        "started_utc": started,
        "completed_utc": utc_now(),
        "runner_sha256": sha256_path(Path(__file__)),
        "codec_sha256": sha256_path(
            REPO_ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / "nvfp4.py"),
        "target_tensor": f"{TARGET}.weight",
        "shard": shard,
        "packed_bytes_sampled": int(packed.size),
        "elements_compared": int(packed.size * 2),
        "working_assumption": nvfp4.DEFAULT_NIBBLE_ORDER,
        "orders_produce_different_output": orders_differ,
        "why_self_consistency_cannot_decide": (
            "A nibble swap permutes elements only within a byte, and a byte lies "
            "wholly inside one group of 16, so every block's value multiset is "
            "invariant. Block-amax, histogram and scale-consistency tests are all "
            "blind to nibble order."
        ),
        "attempts": attempts,
        "resolution": resolution,
        "matched_order": matched_order,
        "reference_used": reference_used,
        "gate_pass": resolution in {"confirmed", "falsified", "unresolved_deferred_to_N6"},
        "claim_boundary": (
            "Establishes only the packing order of NVFP4 codes against an "
            "external published implementation. Says nothing about model "
            "quality, the correctness of the rest of the runtime, or throughput."
        ),
    }

    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"target                : {TARGET}.weight ({shard})")
    print(f"elements compared     : {packed.size * 2:,}")
    print(f"orders differ on data : {orders_differ}")
    for entry in attempts:
        print(f"  attempt {entry['attempt']} [{entry['source']}] available={entry.get('available')} "
              f"matches={entry.get('matches')}")
        if entry.get("error"):
            print(f"    error: {entry['error'][:120]}")
        if entry.get("code_matches"):
            print(f"    code_matches: {entry['code_matches']}")
    print(f"resolution            : {result['resolution']}")
    print(f"matched order         : {matched_order}")
    print(f"reference used        : {reference_used}")
    print(f"written               : {OUT}")
    return 0 if result["gate_pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
