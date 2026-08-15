from __future__ import annotations

from io import BytesIO

import numpy as np

from scripts.coretail_moe.run_p0_full_bank_format import core_record, tail_record
from scripts.coretail_moe.verify_p0_full_bank_format import apply_tail, read_core


def test_coretail_record_roundtrip_and_metadata() -> None:
    codes = np.asarray(
        [
            [-2, -1, 0, 1, 0, -2, 1, -1],
            [0, 0, 0, 0, -1, 1, -2, 0],
            [1, -2, -1, 0, 1, 0, -2, -1],
        ],
        dtype=np.int8,
    )
    scale_bytes = np.arange(6, dtype="<u2").tobytes()
    core_bytes, core_written = core_record(7, 19, "gate", codes, scale_bytes)
    tail_bytes, tail_written = tail_record(7, 19, "gate", codes)

    core_codes, decoded_scales, core_read = read_core(BytesIO(core_bytes), 0)
    decoded, tail_read = apply_tail(BytesIO(tail_bytes), 0, core_codes)

    assert np.array_equal(decoded, codes)
    assert decoded_scales == scale_bytes
    assert core_read["crc_ok"] and core_read["offsets_valid"]
    assert tail_read["record_crc_ok"] and tail_read["block_crc_ok"] and tail_read["layout_ok"]
    assert core_written["record_bytes"] == len(core_bytes)
    assert tail_written["record_bytes"] == len(tail_bytes)
    assert core_written["crc32"] == core_read["crc32"]
    assert tail_written["crc32"] == tail_read["crc32"]
    assert tail_written["raw_fallback_bytes"] == tail_read["raw_fallback_bytes"]
