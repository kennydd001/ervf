"""Range-read weight loader for the public Nemotron 3 Nano NVFP4 checkpoint.

Reads exactly the byte ranges a tensor occupies rather than mapping whole
shards, which is the access pattern the later streaming phases need anyway.
Nothing is written; the model directory is treated as immutable after N2.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import nvfp4

# safetensors dtype -> (numpy dtype for the raw bytes, element size)
RAW_DTYPE = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "BOOL": np.dtype("?"),
}


def bf16_bytes_to_float32(raw: np.ndarray) -> np.ndarray:
    """BF16 is the top 16 bits of a float32, so widening is a shift."""
    as_u16 = raw.view(np.uint16).astype(np.uint32)
    return (as_u16 << 16).view(np.float32)


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    header_len: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class ShardIndex:
    """Header index over the five shards, with byte-range tensor reads."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.entries: dict[str, TensorEntry] = {}
        self._header_len: dict[str, int] = {}

        index_path = self.model_dir / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

        for shard in sorted(set(weight_map.values())):
            path = self.model_dir / shard
            with path.open("rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_len).decode("utf-8"))
            self._header_len[shard] = header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                start, end = meta["data_offsets"]
                self.entries[name] = TensorEntry(
                    name=name, shard=shard, dtype=meta["dtype"],
                    shape=tuple(meta["shape"]), start=start, end=end,
                    header_len=header_len,
                )

        self.config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def read_raw(self, name: str) -> np.ndarray:
        entry = self.entries[name]
        path = self.model_dir / entry.shard
        with path.open("rb") as handle:
            handle.seek(8 + entry.header_len + entry.start)
            raw = handle.read(entry.nbytes)
        if len(raw) != entry.nbytes:
            raise IOError(f"short read for {name}: {len(raw)} != {entry.nbytes}")
        return np.frombuffer(raw, dtype=np.uint8)

    def get_float32(self, name: str) -> np.ndarray:
        """Read a non-quantized tensor and widen it to float32."""
        entry = self.entries[name]
        raw = self.read_raw(name)
        if entry.dtype == "BF16":
            values = bf16_bytes_to_float32(raw)
        elif entry.dtype in RAW_DTYPE:
            values = raw.view(RAW_DTYPE[entry.dtype]).astype(np.float32)
        else:
            raise ValueError(f"{name}: unsupported dtype {entry.dtype}")
        return values.reshape(entry.shape) if entry.shape else values.reshape(())

    def get_scalar(self, name: str) -> float:
        return float(np.asarray(self.get_float32(name)).reshape(-1)[0])

    def dequantize_linear(self, prefix: str) -> np.ndarray:
        """Dequantize one NVFP4 linear weight to float32.

        ``prefix`` is the module path without a field suffix, e.g.
        ``backbone.layers.1.mixer.experts.0.up_proj``.  Returns ``[out, in]``,
        matching the ``nn.Linear`` weight convention.
        """
        codes = self.read_raw(f"{prefix}.weight")
        scales = self.read_raw(f"{prefix}.weight_scale")
        weight_scale_2 = self.get_scalar(f"{prefix}.weight_scale_2")

        entry = self.entries[f"{prefix}.weight"]
        rows, packed_cols = entry.shape
        cols = packed_cols * 2

        scale_entry = self.entries[f"{prefix}.weight_scale"]
        if scale_entry.shape != (rows, cols // nvfp4.GROUP_SIZE):
            raise ValueError(
                f"{prefix}: scale shape {scale_entry.shape} inconsistent with "
                f"weight shape {entry.shape}"
            )

        values = nvfp4.dequantize(codes, scales, weight_scale_2)
        return values.astype(np.float32).reshape(rows, cols)

    def input_scale(self, prefix: str) -> float:
        return self.get_scalar(f"{prefix}.input_scale")

    def quant_kind(self, prefix: str) -> str:
        """Which of the three 3.5 Lightning weight formats this module uses.

        nvfp4      : U8 codes + F8_E4M3 group-16 scales + F32 weight_scale_2
        fp8_tensor : F8_E4M3 weights, one byte each, + F32 scalar weight_scale
        bf16       : plain BF16

        Nemotron 3 Nano only had nvfp4 and bf16; 3.5 Lightning quantises the
        Mamba projections FP8 per tensor and, notably, moves lm_head to NVFP4.
        """
        if f"{prefix}.weight_scale_2" in self.entries:
            return "nvfp4"
        if f"{prefix}.weight_scale" in self.entries:
            return "fp8_tensor"
        return "bf16"

    def is_quantized(self, prefix: str) -> bool:
        return self.quant_kind(prefix) == "nvfp4"

    def load_linear(self, prefix: str) -> np.ndarray:
        """Quantized or not, return a float32 ``[out, in]`` weight."""
        if self.is_quantized(prefix):
            return self.dequantize_linear(prefix)
        return self.get_float32(f"{prefix}.weight")

    @lru_cache(maxsize=None)
    def layer_types(self) -> tuple[str, ...]:
        """Layer roles, from either config spelling.

        Nemotron 3 Nano encodes the hybrid schedule as the string
        ``hybrid_override_pattern``; 3.5 Lightning uses an explicit
        ``layers_block_type`` list. Same schedule, different spelling.
        """
        explicit = self.config.get("layers_block_type")
        if explicit:
            return tuple(explicit)
        pattern = self.config["hybrid_override_pattern"]
        mapping = {"M": "mamba", "*": "attention", "-": "mlp"}
        return tuple(mapping.get(ch, "moe") for ch in pattern)

    @lru_cache(maxsize=None)
    def pattern_string(self) -> str:
        """Compact MEM* form, synthesised when only the list spelling exists."""
        rev = {"mamba": "M", "attention": "*", "mlp": "-", "moe": "E"}
        return "".join(rev[t] for t in self.layer_types())


