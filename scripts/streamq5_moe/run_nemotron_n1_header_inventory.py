from __future__ import annotations

import hashlib
import json
import re
import struct
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from moe_lab.reporting import ROOT

R = ROOT / "reports/streamq5_moe"
PREREG = R / "NEMOTRON_N1_HEADER_INVENTORY_PREREGISTRATION.md"
N0 = R / "nemotron_n0_metadata_gate.json"
OUTPUT = R / "nemotron_n1_header_inventory.json"
REPORT = R / "NEMOTRON_N1_HEADER_INVENTORY_REPORT_2026-08-12.md"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
COMMIT = "ce1b118ae66ec705d02c241525192832eb045fd3"
BASE = f"https://huggingface.co/{MODEL}/resolve/{COMMIT}/"
ROUTED = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.")
SHARED = re.compile(r"^backbone\.layers\.(\d+)\.mixer\.shared_experts\.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_prefix(name: str, size: int) -> bytes:
    request = urllib.request.Request(BASE + name, headers={"Range": f"bytes=0-{size - 1}", "User-Agent": "STREAMQ5-header-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        value = response.read(size)
    if len(value) != size:
        raise RuntimeError(f"{name}: requested {size} header bytes, received {len(value)}")
    return value


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite NEM1 result")
    n0 = json.loads(N0.read_text(encoding="utf-8"))
    shards = n0["identity"]["shards"]
    expected_index_keys = int(n0["identity"]["weight_map_tensors"])
    expected_total = int(n0["identity"]["index_total_size_bytes"])
    tensors = {}
    shard_evidence = []
    layout_failures = []
    for shard in shards:
        first = fetch_prefix(shard, 8)
        header_bytes = struct.unpack("<Q", first)[0]
        raw = fetch_prefix(shard, 8 + header_bytes)
        header = json.loads(raw[8:].decode("utf-8"))
        rows = {name: value for name, value in header.items() if name != "__metadata__"}
        intervals = []
        for name, value in rows.items():
            begin, end = map(int, value["data_offsets"])
            if begin < 0 or end < begin:
                layout_failures.append(f"invalid:{shard}:{name}")
            intervals.append((begin, end, name))
            tensors[name] = {"shard": shard, "dtype": value["dtype"], "shape": value["shape"], "bytes": end - begin}
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if left[1] > right[0]:
                layout_failures.append(f"overlap:{shard}:{left[2]}:{right[2]}")
        payload_extent = max((end for _, end, _ in intervals), default=0)
        shard_evidence.append({"shard": shard, "header_bytes": header_bytes, "header_sha256": hashlib.sha256(raw).hexdigest(), "tensors": len(rows), "payload_extent_bytes": payload_extent})

    routed_records: dict[tuple[int, int], int] = defaultdict(int)
    buckets = Counter()
    dtypes = Counter()
    moe_layers = set()
    expert_ids = defaultdict(set)
    for name, value in tensors.items():
        size = int(value["bytes"])
        match = ROUTED.match(name)
        if match:
            layer, expert = map(int, match.groups())
            routed_records[(layer, expert)] += size
            moe_layers.add(layer)
            expert_ids[layer].add(expert)
            bucket = "routed_experts"
        elif SHARED.match(name):
            bucket = "shared_experts"
        else:
            bucket = "trunk_other"
        buckets[bucket] += size
        dtypes[str(value["dtype"])] += size

    record_sizes = Counter(routed_records.values())
    record_bytes = next(iter(record_sizes)) if len(record_sizes) == 1 else None
    active_records = len(moe_layers) * 6
    all_cold_bytes = active_records * record_bytes if record_bytes is not None else None
    floor_ms = all_cold_bytes / 26.158915e9 * 1000 if all_cold_bytes is not None else None
    gates = {
        "five_headers": len(shard_evidence) == 5,
        "index_key_count": len(tensors) == expected_index_keys,
        "no_offset_failures": not layout_failures,
        "tensor_bytes_equal_index_total": sum(value["bytes"] for value in tensors.values()) == expected_total,
        "moe_layers_23": len(moe_layers) == 23,
        "each_layer_128_experts": len(expert_ids) == 23 and all(values == set(range(128)) for values in expert_ids.values()),
        "routed_records_uniform": len(routed_records) == 23 * 128 and len(record_sizes) == 1,
    }
    result = {
        "kind": "nemotron_n1_header_inventory",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "header_inventory_pass" if all(gates.values()) else "header_inventory_fail",
        "overall_pass": all(gates.values()),
        "inputs": {"preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)), "n0_sha256": sha256(N0), "model": MODEL, "commit": COMMIT},
        "shards": shard_evidence,
        "inventory": {
            "tensor_count": len(tensors),
            "tensor_bytes": sum(value["bytes"] for value in tensors.values()),
            "bucket_bytes": dict(buckets),
            "dtype_bytes": dict(dtypes),
            "moe_layers": sorted(moe_layers),
            "routed_records": len(routed_records),
            "routed_record_size_distribution": {str(key): value for key, value in record_sizes.items()},
            "routed_record_bytes": record_bytes,
            "all_cold_top6_records_per_token": active_records,
            "all_cold_top6_bytes_per_token": all_cold_bytes,
            "all_cold_floor_ms_at_26_158915_gb_s": floor_ms,
        },
        "layout_failures": layout_failures,
        "gates": gates,
        "claim_boundary": "Pinned official safetensors-header/index inventory only; no tensor payload downloaded, decoded or executed and no quality/performance result.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# Nemotron N1 — pinned safetensors-header inventory\n\n"
        f"Verdict: **{result['status']}**. {len(tensors):,} tensors; total {result['inventory']['tensor_bytes'] / 2**30:.3f} GiB. "
        f"Routed bank: {buckets['routed_experts'] / 2**30:.3f} GiB; shared: {buckets['shared_experts'] / 2**30:.3f} GiB; trunk/other: {buckets['trunk_other'] / 2**30:.3f} GiB.\n\n"
        f"There are {len(moe_layers)} MoE layers × 128 uniform routed experts. One stored routed expert is {record_bytes:,} bytes. "
        f"Top-6 all-cold routed traffic is {all_cold_bytes / 1e6:.3f} MB/token, with a {floor_ms:.3f}-ms floor at 26.158915 GB/s.\n\n"
        "Only safetensors headers were read; no model payload was downloaded or executed.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
