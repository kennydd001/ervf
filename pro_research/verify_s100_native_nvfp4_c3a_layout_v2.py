"""Independent stdlib verifier for the additive C3A-v2 layout preflight."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULT = REPO / "pro_research" / "results" / "native_nvfp4" / "C3A_V2_LAYOUT_PREFLIGHT.json"
REVISION = "c3a_v2_torchao_to_blocked_row_block_major"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blocked_reference(raw: bytes, rows: int, cols: int, legacy: bool) -> bytes:
    nrb = (rows + 127) // 128
    ncb = (cols + 3) // 4
    out = bytearray(nrb * ncb * 512)
    for r in range(rows):
        rb, rr = r // 128, r % 128
        r32, g32 = rr % 32, rr // 32
        for c in range(cols):
            cb, cc = c // 4, c % 4
            outer = cb * nrb + rb if legacy else rb * ncb + cb
            out[outer * 512 + ((r32 * 4 + g32) * 4 + cc)] = raw[r * cols + c]
    return bytes(out)


def main() -> int:
    if not RESULT.exists():
        print(f"FAIL: missing {RESULT}")
        return 2
    d = json.loads(RESULT.read_text(encoding="utf-8"))
    failures: list[str] = []
    if d.get("status") != "layout_v2_preflight_pass":
        failures.append(f"preflight status={d.get('status')}")
    if d.get("revision") != REVISION:
        failures.append(f"revision={d.get('revision')}")

    rows, cols = 256, 8
    raw = bytes(((r * 13 + c * 29 + 17) % 251) for r in range(rows) for c in range(cols))
    expected = blocked_reference(raw, rows, cols, False)
    legacy = blocked_reference(raw, rows, cols, True)
    legacy_mismatch = sum(a != b for a, b in zip(legacy, expected))
    w = d.get("layout_witness") or {}
    witness_ok = bool(
        w.get("revision") == REVISION
        and w.get("rows") == rows and w.get("cols") == cols
        and w.get("input_sha256") == sha(raw)
        and w.get("expected_row_major_sha256") == sha(expected)
        and w.get("actual_sha256") == sha(expected)
        and int(w.get("byte_mismatches", -1)) == 0
        and int(w.get("legacy_k_major_byte_mismatches", -1)) == legacy_mismatch
        and legacy_mismatch > 0
        and w.get("passes") is True
    )
    if not witness_ok:
        failures.append("independent row-block-major byte witness failed")

    s = d.get("nonuniform_native_smoke") or {}
    smoke_ok = bool(
        s.get("revision") == REVISION and s.get("finite") is True
        and s.get("all_equal_expected") is True and s.get("passes") is True
        and float(s.get("expected_first_128", -1)) == 48.0
        and float(s.get("expected_second_128", -1)) == 192.0
        and float(s.get("max_abs_error", -1)) == 0.0
    )
    if not smoke_ok:
        failures.append("nonuniform native 2x2-block smoke failed")

    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "revision": d.get("revision"),
        "independent_row_major_witness": witness_ok,
        "legacy_k_major_byte_mismatches": legacy_mismatch,
        "nonuniform_native_smoke_exact": smoke_ok,
        "failures": failures,
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
