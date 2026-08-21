import json
import struct
from pathlib import Path

import numpy as np
import pytest

from moe_lab.ornith.expert_store import ExpertStaging, OrnithExpertStore


PREFIX = "model.layers.3.mlp.experts.7"


def _fixture_tensors(*, bad_gate_shape: bool = False, bad_scalar_dtype: bool = False):
    specs = {
        f"{PREFIX}.gate_proj.weight": ("U8", (512, 1024)),
        f"{PREFIX}.gate_proj.weight_scale": ("F8_E4M3", (512, 128)),
        f"{PREFIX}.gate_proj.weight_scale_2": (
            "F16" if bad_scalar_dtype else "F32",
            (),
        ),
        f"{PREFIX}.up_proj.weight": ("U8", (512, 1024)),
        f"{PREFIX}.up_proj.weight_scale": ("F8_E4M3", (512, 128)),
        f"{PREFIX}.up_proj.weight_scale_2": ("F32", ()),
        f"{PREFIX}.down_proj.weight": ("U8", (2048, 256)),
        f"{PREFIX}.down_proj.weight_scale": ("F8_E4M3", (2048, 32)),
        f"{PREFIX}.down_proj.weight_scale_2": ("F32", ()),
    }
    if bad_gate_shape:
        specs[f"{PREFIX}.gate_proj.weight"] = ("U8", (512, 1023))
    return specs


def _payload(dtype: str, shape: tuple[int, ...], seed: int) -> bytes:
    count = max(1, int(np.prod(shape, dtype=np.int64)))
    if dtype in ("U8", "F8_E4M3"):
        return np.arange(count, dtype=np.uint8).tobytes()
    if dtype == "F32":
        return np.asarray(seed + 0.25, dtype="<f4").tobytes()
    if dtype == "F16":
        return np.asarray(seed + 0.5, dtype="<f2").tobytes()
    raise AssertionError(dtype)


def _write_safetensors(path: Path, specs: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    header = {}
    payload = bytearray()
    for seed, (name, (dtype, shape)) in enumerate(specs.items()):
        raw = _payload(dtype, shape, seed)
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    header["__metadata__"] = {"fixture": "ornith-expert-store"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _write_indexed_fixture(tmp_path: Path, specs=None) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    _write_safetensors(model_dir / shard_name, specs or _fixture_tensors())
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": (model_dir / shard_name).stat().st_size},
                "weight_map": {name: shard_name for name in (specs or _fixture_tensors())},
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def test_index_and_headers_are_mapped_once_and_record_has_exact_views(tmp_path: Path):
    model_dir = _write_indexed_fixture(tmp_path)
    with OrnithExpertStore(model_dir) as store:
        record = store.expert(3, 7)
        assert len(record.segment_views) == 6
        assert len(record.global_scales) == 3
        assert [view.shape for view in record.segment_views] == [
            (512, 1024),
            (512, 128),
            (512, 1024),
            (512, 128),
            (2048, 256),
            (2048, 32),
        ]
        assert [view.dtype for view in record.segment_views] == [
            np.dtype("u1"),
            np.dtype("u1"),
            np.dtype("u1"),
            np.dtype("u1"),
            np.dtype("u1"),
            np.dtype("u1"),
        ]
        assert [part.logical_shape for part in (record.gate, record.up, record.down)] == [
            (512, 2048),
            (512, 2048),
            (2048, 512),
        ]
        assert [float(scale[()]) for scale in record.global_scales] == pytest.approx(
            [2.25, 5.25, 8.25]
        )
        assert all(not view.flags.owndata for view in record.segment_views)
        assert all(not view.flags.writeable for view in record.segment_views)
        assert all(not view.flags.writeable for view in record.global_scales)

        repeated = store.expert(3, 7)
        assert np.shares_memory(record.gate.codes.data, repeated.gate.codes.data)

        # The source views point into the mmap; no payload read API is involved.
        assert type(record.gate.codes.data.base).__name__ in {"mmap", "memoryview"}
        del repeated
        del record

    assert store.closed
    store.close()


def test_copy_expert_writes_only_into_preallocated_staging(tmp_path: Path):
    model_dir = _write_indexed_fixture(tmp_path)
    shapes = {
        "gate_codes": (512, 1024),
        "gate_scales": (512, 128),
        "up_codes": (512, 1024),
        "up_scales": (512, 128),
        "down_codes": (2048, 256),
        "down_scales": (2048, 32),
    }
    staging = ExpertStaging(
        **{key: np.full(shape, 255, dtype=np.uint8) for key, shape in shapes.items()},
        global_scales=np.full(3, -1, dtype="<f4"),
    )
    with OrnithExpertStore(model_dir) as store:
        store.copy_expert(3, 7, staging)
        assert staging.gate_codes[0, 0] == 0
        assert staging.up_codes[0, 0] == 0
        assert staging.down_codes[0, 0] == 0
        assert staging.global_scales.tolist() == pytest.approx([2.25, 5.25, 8.25])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bad_gate_shape": True}, "packed shape"),
        ({"bad_scalar_dtype": True}, "expected scalar F32"),
    ],
)
def test_shape_and_dtype_contract_is_enforced(tmp_path: Path, kwargs, message):
    model_dir = _write_indexed_fixture(tmp_path, _fixture_tensors(**kwargs))
    with OrnithExpertStore(model_dir) as store:
        with pytest.raises(ValueError, match=message):
            store.expert(3, 7)


def test_close_requires_views_to_be_released(tmp_path: Path):
    model_dir = _write_indexed_fixture(tmp_path)
    store = OrnithExpertStore(model_dir)
    record = store.expert(3, 7)
    with pytest.raises(BufferError, match="mmap-backed views"):
        store.close()
    assert not store.closed
    del record
    store.close()
    assert store.closed
