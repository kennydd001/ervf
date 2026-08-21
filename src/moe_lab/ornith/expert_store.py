"""Read-only mmap access to Ornith NVFP4 expert records.

The checkpoint stores every linear projection as three safetensors tensors:
packed NVFP4 codes, FP8-E4M3 block scales, and one FP32 ``weight_scale_2``.
This module parses the safetensors index and shard headers once.  Expert
lookups then only create NumPy views over already-mapped bytes; they do not
read or copy tensor payloads.

The six segment views in an :class:`ExpertRecord` are ordered as gate codes,
gate block scales, up codes, up block scales, down codes, and down block
scales.  The three global scales are zero-dimensional read-only NumPy views.
Callers that need mutable GPU/H2D staging can use :meth:`OrnithExpertStore.copy_expert`.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_HEADER_PREFIX_BYTES = 8
_RAW_DTYPES: dict[str, np.dtype[Any]] = {
    "U8": np.dtype(np.uint8),
    # Safetensors exposes FP8 as raw bytes.  Decoding is deliberately left to
    # the consumer; this reader must not transform or copy the payload.
    "F8_E4M3": np.dtype(np.uint8),
    "F32": np.dtype("<f4"),
}

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_EXPECTED_LOGICAL_SHAPES = {
    "gate_proj": (512, 2048),
    "up_proj": (512, 2048),
    "down_proj": (2048, 512),
}
_STAGING_KEYS = (
    "gate_codes",
    "gate_scales",
    "up_codes",
    "up_scales",
    "down_codes",
    "down_scales",
    "global_scales",
)


@dataclass(frozen=True)
class _TensorEntry:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int
    header_bytes: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class TensorSegmentView:
    """One immutable tensor view into a safetensors mmap."""

    name: str
    safetensors_dtype: str
    shape: tuple[int, ...]
    data: np.ndarray

    @property
    def view(self) -> np.ndarray:
        """Return the underlying read-only NumPy view."""

        return self.data


@dataclass(frozen=True)
class ProjectionSegments:
    """Packed codes, block scales, and global scale for one projection."""

    projection: str
    logical_shape: tuple[int, int]
    codes: TensorSegmentView
    scales: TensorSegmentView
    global_scale: np.ndarray

    @property
    def segment_views(self) -> tuple[np.ndarray, np.ndarray]:
        return self.codes.data, self.scales.data


@dataclass(frozen=True)
class ExpertRecord:
    """The six payload views and three scalar scale views for one expert."""

    layer: int
    expert: int
    gate: ProjectionSegments
    up: ProjectionSegments
    down: ProjectionSegments

    @property
    def segment_views(self) -> tuple[np.ndarray, ...]:
        """Return exactly six read-only views in stable projection order."""

        return (
            self.gate.codes.data,
            self.gate.scales.data,
            self.up.codes.data,
            self.up.scales.data,
            self.down.codes.data,
            self.down.scales.data,
        )

    @property
    def global_scales(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return gate, up, and down zero-dimensional scale views."""

        return (
            self.gate.global_scale,
            self.up.global_scale,
            self.down.global_scale,
        )


@dataclass
class ExpertStaging:
    """Preallocated writable NumPy destinations for one expert record."""

    gate_codes: np.ndarray
    gate_scales: np.ndarray
    up_codes: np.ndarray
    up_scales: np.ndarray
    down_codes: np.ndarray
    down_scales: np.ndarray
    global_scales: np.ndarray


@dataclass
class _MappedShard:
    path: Path
    handle: Any
    mapping: mmap.mmap


class OrnithExpertStore:
    """Read-only mmap store for Ornith NVFP4 expert tensors.

    ``checkpoint`` may be either a single ``.safetensors`` shard or a model
    directory containing ``model.safetensors.index.json``.  For a directory,
    all shards named by the index are opened and mapped once during
    construction.  ``prefix_template`` can be supplied for synthetic
    fixtures or alternate checkpoint naming; it must contain ``{layer}`` and
    ``{expert}``.

    Views remain valid only while the store is open.  As with all NumPy views
    backed by an mmap, callers must drop those views before closing the store.
    ``close`` reports a live-view ``BufferError`` instead of silently leaving
    a partially closed store.
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        prefix_template: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self._entries: dict[str, _TensorEntry] = {}
        self._mapped: dict[str, _MappedShard] = {}
        self._live_records: dict[int, weakref.ReferenceType[ExpertRecord]] = {}
        self._prefix_template = prefix_template
        self._closed = False
        self._parse_index_and_headers()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def tensor_names(self) -> tuple[str, ...]:
        """Names from the parsed headers, useful for diagnostics/tests."""

        return tuple(self._entries)

    def _parse_index_and_headers(self) -> None:
        if self.checkpoint.is_dir():
            index_path = self.checkpoint / "model.safetensors.index.json"
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index["weight_map"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid safetensors index: {index_path}") from exc
            shard_names = tuple(dict.fromkeys(weight_map.values()))
            root = self.checkpoint
        elif self.checkpoint.is_file():
            root = self.checkpoint.parent
            shard_names = (self.checkpoint.name,)
        else:
            raise FileNotFoundError(self.checkpoint)

        for shard_name in shard_names:
            path = root / shard_name
            if not path.is_file():
                raise FileNotFoundError(path)
            entries, header_bytes = self._read_header(path, shard_name)
            handle = path.open("rb")
            try:
                mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                handle.close()
                raise
            self._mapped[shard_name] = _MappedShard(path, handle, mapping)
            for entry in entries:
                if entry.name in self._entries:
                    raise ValueError(f"duplicate tensor name: {entry.name}")
                self._entries[entry.name] = entry

    @staticmethod
    def _read_header(path: Path, shard: str) -> tuple[list[_TensorEntry], int]:
        try:
            with path.open("rb") as handle:
                prefix = handle.read(_HEADER_PREFIX_BYTES)
                if len(prefix) != _HEADER_PREFIX_BYTES:
                    raise ValueError("truncated safetensors header length")
                (header_len,) = struct.unpack("<Q", prefix)
                file_size = path.stat().st_size
                if header_len > file_size - _HEADER_PREFIX_BYTES:
                    raise ValueError("safetensors header exceeds file size")
                raw_header = handle.read(header_len)
                if len(raw_header) != header_len:
                    raise ValueError("truncated safetensors header")
            header = json.loads(raw_header.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as exc:
            raise ValueError(f"invalid safetensors header: {path}") from exc
        if not isinstance(header, dict):
            raise ValueError(f"safetensors header is not an object: {path}")

        data_bytes = file_size - _HEADER_PREFIX_BYTES - header_len
        result: list[_TensorEntry] = []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"{name}: tensor metadata is not an object")
            try:
                dtype = str(meta["dtype"])
                shape = tuple(int(value) for value in meta["shape"])
                start, end = (int(value) for value in meta["data_offsets"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{name}: malformed tensor metadata") from exc
            if any(value < 0 for value in shape):
                raise ValueError(f"{name}: negative shape")
            if start < 0 or end < start or end > data_bytes:
                raise ValueError(f"{name}: data offsets outside shard payload")
            result.append(_TensorEntry(name, shard, dtype, shape, start, end, header_len))
        return result, header_len

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OrnithExpertStore is closed")

    def _resolve_prefix(self, layer: int, expert: int) -> str:
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not isinstance(expert, int) or isinstance(expert, bool) or expert < 0:
            raise ValueError("expert must be a non-negative integer")
        if self._prefix_template is not None:
            try:
                prefix = self._prefix_template.format(layer=layer, expert=expert)
            except (KeyError, ValueError) as exc:
                raise ValueError("prefix_template must contain {layer} and {expert}") from exc
            if not any(f"{prefix}.{p}.weight" in self._entries for p in _PROJECTIONS):
                raise KeyError(f"expert prefix not found: {prefix}")
            return prefix

        candidates = (
            f"model.language_model.layers.{layer}.mlp.experts.{expert}",
            f"model.layers.{layer}.mlp.experts.{expert}",
            f"backbone.layers.{layer}.mixer.experts.{expert}",
        )
        for prefix in candidates:
            if all(f"{prefix}.{p}.weight" in self._entries for p in _PROJECTIONS):
                return prefix
        raise KeyError(f"Ornith expert not found: layer={layer}, expert={expert}")

    def _view(self, entry: _TensorEntry) -> np.ndarray:
        self._require_open()
        try:
            dtype = _RAW_DTYPES[entry.dtype]
        except KeyError as exc:
            raise ValueError(f"{entry.name}: unsupported dtype {entry.dtype}") from exc
        expected_bytes = int(np.prod(entry.shape, dtype=np.int64)) * dtype.itemsize
        if expected_bytes != entry.nbytes:
            raise ValueError(
                f"{entry.name}: byte size {entry.nbytes} does not match "
                f"shape {entry.shape} and dtype {entry.dtype}"
            )
        mapped = self._mapped[entry.shard].mapping
        offset = _HEADER_PREFIX_BYTES + entry.header_bytes + entry.start
        view = np.ndarray(entry.shape, dtype=dtype, buffer=mapped, offset=offset)
        view.setflags(write=False)
        return view

    def _projection(self, prefix: str, projection: str) -> ProjectionSegments:
        logical_shape = _EXPECTED_LOGICAL_SHAPES[projection]
        rows, cols = logical_shape
        codes_entry = self._entries[f"{prefix}.{projection}.weight"]
        scales_entry = self._entries[f"{prefix}.{projection}.weight_scale"]
        scalar_entry = self._entries[f"{prefix}.{projection}.weight_scale_2"]

        if codes_entry.dtype != "U8":
            raise ValueError(f"{codes_entry.name}: expected U8, got {codes_entry.dtype}")
        if codes_entry.shape != (rows, cols // 2):
            raise ValueError(
                f"{codes_entry.name}: expected packed shape {(rows, cols // 2)}, "
                f"got {codes_entry.shape}; logical shape is {logical_shape}"
            )
        if scales_entry.dtype != "F8_E4M3":
            raise ValueError(
                f"{scales_entry.name}: expected F8_E4M3, got {scales_entry.dtype}"
            )
        expected_scale_shape = (rows, cols // 16)
        if scales_entry.shape != expected_scale_shape:
            raise ValueError(
                f"{scales_entry.name}: expected shape {expected_scale_shape}, "
                f"got {scales_entry.shape}"
            )
        if scalar_entry.dtype != "F32" or scalar_entry.shape != ():
            raise ValueError(
                f"{scalar_entry.name}: expected scalar F32, "
                f"got dtype={scalar_entry.dtype}, shape={scalar_entry.shape}"
            )

        return ProjectionSegments(
            projection=projection,
            logical_shape=logical_shape,
            codes=TensorSegmentView(
                codes_entry.name,
                codes_entry.dtype,
                codes_entry.shape,
                self._view(codes_entry),
            ),
            scales=TensorSegmentView(
                scales_entry.name,
                scales_entry.dtype,
                scales_entry.shape,
                self._view(scales_entry),
            ),
            global_scale=self._view(scalar_entry),
        )

    def expert(self, layer: int, expert: int) -> ExpertRecord:
        """Return one expert's six segment views and three scale views."""

        self._require_open()
        prefix = self._resolve_prefix(layer, expert)
        gate = self._projection(prefix, "gate_proj")
        up = self._projection(prefix, "up_proj")
        down = self._projection(prefix, "down_proj")
        record = ExpertRecord(layer, expert, gate, up, down)
        record_id = id(record)
        self._live_records[record_id] = weakref.ref(
            record, lambda reference, key=record_id: self._live_records.pop(key, None)
        )
        return record

    @staticmethod
    def _destination(staging: ExpertStaging, key: str) -> np.ndarray:
        value = getattr(staging, key)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"staging.{key} must be a NumPy array")
        if not value.flags.writeable:
            raise ValueError(f"staging.{key} is not writable")
        return value

    @staticmethod
    def _copy_segment(destination: np.ndarray, source: np.ndarray, key: str) -> None:
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError(
                f"staging.{key}: expected shape/dtype {source.shape}/{source.dtype}, "
                f"got {destination.shape}/{destination.dtype}"
            )
        np.copyto(destination, source, casting="no")

    def copy_expert(self, layer: int, expert: int, staging: ExpertStaging) -> None:
        """Copy one record into caller-owned, preallocated staging arrays."""

        if not isinstance(staging, ExpertStaging):
            raise TypeError("staging must be an ExpertStaging instance")
        record = self.expert(layer, expert)
        for key, source in zip(
            _STAGING_KEYS[:6], record.segment_views, strict=True
        ):
            self._copy_segment(self._destination(staging, key), source, key)
        global_destination = self._destination(staging, "global_scales")
        if global_destination.shape != (3,) or global_destination.dtype != np.dtype("<f4"):
            raise ValueError(
                "staging.global_scales: expected shape/dtype (3,)/float32, "
                f"got {global_destination.shape}/{global_destination.dtype}"
            )
        for index, source in enumerate(record.global_scales):
            global_destination[index] = source[()]

    def close(self) -> None:
        """Close all mmaps and file handles, refusing unsafe partial closure."""

        if self._closed:
            return
        self._live_records = {
            key: reference
            for key, reference in self._live_records.items()
            if reference() is not None
        }
        if self._live_records:
            raise BufferError(
                "cannot close OrnithExpertStore while mmap-backed views are alive; "
                "drop ExpertRecord/segment views first"
            )
        try:
            for shard in self._mapped.values():
                if not shard.mapping.closed:
                    shard.mapping.close()
        except BufferError as exc:
            raise BufferError(
                "cannot close OrnithExpertStore while mmap-backed views are alive; "
                "drop ExpertRecord/segment views first"
            ) from exc
        finally:
            # If mmap.close succeeded, all handles can be released.  If it
            # failed, retain the mapping and handle so a later close is safe.
            if all(shard.mapping.closed for shard in self._mapped.values()):
                for shard in self._mapped.values():
                    shard.handle.close()
                self._closed = True

    def __enter__(self) -> "OrnithExpertStore":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Destructors must never mask the original exception.  Explicit
        # close/context-manager use remains the deterministic lifecycle API.
        try:
            self.close()
        except (BufferError, OSError):
            pass


__all__ = [
    "ExpertRecord",
    "ExpertStaging",
    "OrnithExpertStore",
    "ProjectionSegments",
    "TensorSegmentView",
]
