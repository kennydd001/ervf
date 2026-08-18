from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


RESIDENT = (
    re.compile(r"^backbone\.layers\.\d+\.mixer\.(in_proj|out_proj)\.weight$"),
    re.compile(r"^backbone\.layers\.\d+\.mixer\.(q_proj|k_proj|v_proj|o_proj)\.weight$"),
    re.compile(r"^backbone\.layers\.\d+\.mixer\.gate\.weight$"),
)
TILE = 1024
BITS = (4, 5, 6)


def raw_bytes(root, weight_map, name):
    with safe_open(str(root / weight_map[name]), framework="pt", device="cpu") as h:
        t = h.get_tensor(name).detach().cpu().contiguous()
        return t.view(torch.uint8).numpy().reshape(-1).copy(), str(t.dtype), tuple(int(x) for x in t.shape)


def encode_tile(tile: np.ndarray, bits: int) -> tuple[bytes, dict]:
    tile = np.asarray(tile, dtype=np.uint8).reshape(-1)
    slots = 1 << bits
    counts = np.bincount(tile, minlength=256)
    palette = np.argsort(counts)[-slots:][::-1].astype(np.uint8)
    lookup = {int(value): idx for idx, value in enumerate(palette)}
    codes = np.array([lookup.get(int(value), slots) for value in tile], dtype=np.uint16)
    width = bits + 1
    packed = bytearray((len(codes) * width + 7) // 8)
    for i, code in enumerate(codes):
        bit = i * width
        for j in range(width):
            if code & (1 << j):
                packed[(bit + j) // 8] |= 1 << ((bit + j) % 8)
    escapes = tile[codes == slots].tobytes()
    header = len(tile).to_bytes(2, "little") + bytes([bits]) + palette.tobytes()
    encoded = header + bytes(packed) + escapes
    return encoded, {"raw_bytes": int(tile.size), "encoded_bytes": len(encoded), "escapes": int(len(escapes))}


def encode(raw: np.ndarray, bits: int) -> tuple[bytes, dict]:
    pieces = []
    tiles = []
    for start in range(0, raw.size, TILE):
        piece, stats = encode_tile(raw[start:start + TILE], bits)
        pieces.append(piece)
        tiles.append(stats)
    return b"".join(pieces), {"tiles": len(tiles), "raw_bytes": int(raw.size), "encoded_bytes": sum(x["encoded_bytes"] for x in tiles), "escapes": sum(x["escapes"] for x in tiles)}


def decode(blob: bytes, raw_size: int, bits: int) -> np.ndarray:
    out = np.empty(raw_size, dtype=np.uint8)
    cursor = 0
    out_cursor = 0
    slots = 1 << bits
    width = bits + 1
    while out_cursor < raw_size:
        n = int.from_bytes(blob[cursor:cursor + 2], "little"); cursor += 2
        actual_bits = blob[cursor]; cursor += 1
        if actual_bits != bits:
            raise ValueError("codec bit width mismatch")
        palette = np.frombuffer(blob[cursor:cursor + slots], dtype=np.uint8); cursor += slots
        packed_size = (n * width + 7) // 8
        packed = blob[cursor:cursor + packed_size]; cursor += packed_size
        codes = np.zeros(n, dtype=np.uint16)
        for i in range(n):
            bit = i * width
            code = 0
            for j in range(width):
                code |= ((packed[(bit + j) // 8] >> ((bit + j) % 8)) & 1) << j
            codes[i] = code
        escape_count = int(np.count_nonzero(codes == slots))
        escapes = np.frombuffer(blob[cursor:cursor + escape_count], dtype=np.uint8); cursor += escape_count
        decoded = np.empty(n, dtype=np.uint8)
        escape_cursor = 0
        for i, code in enumerate(codes):
            if code == slots:
                decoded[i] = escapes[escape_cursor]; escape_cursor += 1
            else:
                decoded[i] = palette[code]
        out[out_cursor:out_cursor + n] = decoded
        out_cursor += n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("models/nemotron_3_5_lightning"))
    ap.add_argument("--output", type=Path, default=Path("pro_research/results/s100_phase13g/S100_PHASE13G_ENTROPY_CODEC.json"))
    args = ap.parse_args()
    root = args.model_dir.resolve()
    index_path = root / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    names = sorted(name for name in weight_map if name.endswith(".weight") and any(p.match(name) for p in RESIDENT))
    # Full census already covered every stream; this codec test uses six
    # deterministic representatives to keep encode/decode timing bounded.
    selected = [n for n in names if ".mixer.in_proj." in n][:2] + [n for n in names if ".mixer.out_proj." in n][:2] + [n for n in names if ".mixer.q_proj." in n][:2]
    records = []
    for name in selected:
        raw, dtype, shape = raw_bytes(root, weight_map, name)
        raw = raw[: min(raw.size, 1 << 20)]
        for bits in BITS:
            t0 = time.perf_counter(); blob, stats = encode(raw, bits); encode_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter(); decoded = decode(blob, raw.size, bits); decode_ms = (time.perf_counter() - t0) * 1000
            if not np.array_equal(raw, decoded):
                raise RuntimeError(f"roundtrip mismatch for {name} bits={bits}")
            records.append({"name": name, "dtype": dtype, "shape": list(shape), "bits": bits, **stats, "encoded_fraction": stats["encoded_bytes"] / stats["raw_bytes"], "roundtrip_exact": True, "encode_ms": encode_ms, "decode_ms": decode_ms, "decode_mib_s": stats["raw_bytes"] / 2**20 / max(decode_ms / 1000, 1e-12)})
        print(f"codec tested {name} {dtype} {shape}", flush=True)
    result = {"kind": "s100_phase13g_lossless_entropy_codec", "status": "measured", "created_utc": datetime.now(timezone.utc).isoformat(), "model_dir": str(root), "claim_boundary": "CPU tile codec roundtrip and overhead screen; no GPU decoder or end-to-end inference claim", "method": {"tile_bytes": TILE, "bits": list(BITS), "encoding": "local top-symbol palette, fixed-width palette/escape codes, raw escape bytes", "selected_streams": selected, "missing": ["GPU register/shared-memory decode", "kernel integration", "full model latency", "quality risk under imperfect decode"]}, "records": records, "gates": {"all_roundtrips_exact": all(r["roundtrip_exact"] for r in records), "gpu_runtime_green": False, "promotion_open": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "records": len(records), "promotion_open": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
