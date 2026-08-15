from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PTX = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8_direct_nvrtc_noftz.ptx"
R8 = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8_direct_nvrtc_noftz_diagnostic.json"
OUT = ROOT / "reports/streamq5_moe/het_next_l0_ph0x_r8r1_noftz_ptx_parser_correction.json"
EXPECTED_PTX_SHA = "ec4789735f548123be0df3c2ff20c3e05c7b3741d9ed5f00b7b51eaeaa8ca7ae"
EXPECTED_R8_SHA = "c5df7a09ea13e4c29caa0d9acf40120131ae6a45033e73126f8563180f005ff2"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    ptx = PTX.read_bytes()
    r8b = R8.read_bytes()
    text = ptx.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if "shfl.sync.down.b32" in line]
    parsed = []
    pattern = re.compile(r"shfl\.sync\.down\.b32\s+[^,]+,\s*[^,]+,\s*(\d+),\s*(\d+),\s*(%r\d+);")
    for line in lines:
        match = pattern.search(line)
        parsed.append({"line": line, "offset": int(match.group(1)) if match else None, "clamp_segment": int(match.group(2)) if match else None, "membermask": match.group(3) if match else None})
    result = {
        "kind": "ph0x_r8r1_noftz_ptx_parser_correction",
        "ptx_sha256": sha(ptx),
        "r8_json_sha256": sha(r8b),
        "ptx_bytes": len(ptx),
        "ftz_modifier_count": text.count(".ftz"),
        "mul_f32_count": text.count("mul.f32"),
        "fma_rn_f32_count": text.count("fma.rn.f32"),
        "add_rn_f32_count": text.count("add.rn.f32"),
        "shuffles": parsed,
    }
    result["pass"] = bool(
        result["ptx_sha256"] == EXPECTED_PTX_SHA
        and result["r8_json_sha256"] == EXPECTED_R8_SHA
        and result["ftz_modifier_count"] == 0
        and result["mul_f32_count"] == 256
        and result["fma_rn_f32_count"] == 256
        and result["add_rn_f32_count"] == 34
        and [row["offset"] for row in parsed] == [4, 2, 1]
        and all(row["clamp_segment"] == 6175 and row["membermask"] is not None for row in parsed)
    )
    OUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pass": result["pass"], "shuffles": parsed}, indent=2))
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
