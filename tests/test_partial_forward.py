from pathlib import Path

import pytest

from moe_lab.partial_forward import checkpoint_state_for_prefix


def test_checkpoint_state_rejects_unknown_prefix(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"model.known.weight": "one.safetensors"}}',
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="no tensors"):
        checkpoint_state_for_prefix(tmp_path, "model.unknown")
