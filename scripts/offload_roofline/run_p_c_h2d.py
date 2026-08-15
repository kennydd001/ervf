from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/offload_roofline/P_C_H2D_ROOFLINE_PREREGISTRATION.md"
OUT_JSON = ROOT / "reports/offload_roofline/p_c_h2d_result.json"
OUT_MD = ROOT / "reports/offload_roofline/P_C_H2D_REPORT.md"
SIZES_MIB = (64, 256, 512)
WARMUPS = 10
TRIALS = 50
K3_EXTERNAL_TRUNK_GB = 27.28


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def pcie_info():
    query = "name,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current"
    try:
        process = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return process.stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def measure(size_mib: int):
    size_bytes = size_mib * 2**20
    source = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    source.fill_(17)
    destination = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
    for _ in range(WARMUPS):
        destination.copy_(source, non_blocking=True)
    torch.cuda.synchronize()
    milliseconds = []
    for _ in range(TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        destination.copy_(source, non_blocking=True)
        end.record()
        end.synchronize()
        milliseconds.append(float(start.elapsed_time(end)))
    bandwidth = [size_bytes / (value / 1000.0) / 1e9 for value in milliseconds]
    result = {
        "size_mib": size_mib, "size_bytes": size_bytes,
        "warmups": WARMUPS, "trials": TRIALS,
        "milliseconds": milliseconds, "bandwidth_gb_s": bandwidth,
        "latency_ms": {
            "p05": percentile(milliseconds, 0.05), "median": statistics.median(milliseconds),
            "mean": statistics.mean(milliseconds), "p95": percentile(milliseconds, 0.95),
        },
        "effective_bandwidth_gb_s": {
            "p05": percentile(bandwidth, 0.05), "median": statistics.median(bandwidth),
            "mean": statistics.mean(bandwidth), "p95": percentile(bandwidth, 0.95),
        },
    }
    del destination, source
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    if OUT_JSON.exists() or OUT_MD.exists():
        raise FileExistsError("refusing to overwrite P-C outputs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; P-C hardware leg cannot run")
    free_before, total = torch.cuda.mem_get_info()
    rows, failures = [], []
    for size_mib in SIZES_MIB:
        try:
            rows.append(measure(size_mib))
        except Exception as exc:
            failures.append({"size_mib": size_mib, "exception": type(exc).__name__, "message": str(exc)})
            torch.cuda.empty_cache()
    if rows:
        primary = max(rows, key=lambda row: row["size_mib"])
        primary_bandwidth = primary["effective_bandwidth_gb_s"]["median"]
        conditional_ceiling = primary_bandwidth / K3_EXTERNAL_TRUNK_GB
        hardware_leg_supports_le_1 = conditional_ceiling <= 1.0
        verdict = "hardware_leg_supports_conditional_k3_le_1" if hardware_leg_supports_le_1 else "hardware_leg_falsifies_conditional_k3_le_1"
    else:
        primary = None
        primary_bandwidth = conditional_ceiling = None
        hardware_leg_supports_le_1 = False
        verdict = "hardware_leg_blocked"
    payload = {
        "kind": "offload_roofline_p_c_pinned_h2d",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "preregistration_sha256": sha256(PREREG),
        "hardware": {
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "total_vram_bytes": total,
            "free_vram_before_bytes": free_before, "nvidia_smi_pcie": pcie_info(),
        },
        "measurements": rows, "failures": failures,
        "primary": {
            "largest_successful_size_mib": primary["size_mib"] if primary else None,
            "median_bandwidth_gb_s": primary_bandwidth,
            "external_k3_trunk_gb_per_token": K3_EXTERNAL_TRUNK_GB,
            "conditional_trunk_ceiling_tokens_per_second": conditional_ceiling,
        },
        "gates": {
            "at_least_one_size_measured": bool(rows),
            "primary_bandwidth_le_27_28_gb_s": bool(rows) and primary_bandwidth <= K3_EXTERNAL_TRUNK_GB,
            "hardware_leg_supports_conditional_le_1_tps": hardware_leg_supports_le_1,
            "actual_k3_trunk_bytes_measured": False,
            "actual_64_token_k3_decode_measured": False,
            "full_p_c_proven": False,
        },
        "claim_boundary": "New local pinned-H2D measurement. The 27.28 GB K3 trunk is an unverified external input; no local K3 checkpoint/runtime or 64-token decode is present.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# P-C pinned-H2D-roofline — resultaat", "", f"**Uitkomst hardwareleg: {verdict}.**", ""]
    for row in rows:
        bw = row["effective_bandwidth_gb_s"]
        lines.append(f"- {row['size_mib']} MiB: mediaan {bw['median']:.3f} GB/s (p05 {bw['p05']:.3f}, p95 {bw['p95']:.3f}).")
    if primary:
        lines.extend(["", f"Conditioneel plafond bij T=27,28 GB: **{conditional_ceiling:.4f} tok/s**."])
    lines.extend(["", "Dit is geen volledige K3-meting: actieve trunkbytes en de 64-token decode ontbreken lokaal. De uitkomst valideert of falsifieert alleen de busleg onder de externe T-aanname.", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "device": payload["hardware"]["device"], "primary": payload["primary"], "failures": failures}, indent=2))
