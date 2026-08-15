from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import psutil
from safetensors.numpy import load_file

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE
from scripts.streamq5_moe.run_port80b_d2_registered_scatter import (
    EXPECTED_BANK_SHA256,
    REGISTER_FLAGS,
    full_verify,
    record_offset,
    stats,
    unregister_ranges,
)
from scripts.streamq5_moe.run_port80b_d5_cp_async_host_smem import SOURCE as STAGE_SOURCE
from scripts.streamq5_moe.run_port80b_d7_staged_exact_q5_plane import COMPUTE_SOURCE
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import (
    BANK,
    BANK_BYTES,
    EXPERT_BYTES,
    HardPageReadSampler,
    LAYERS,
    MANIFEST,
)


R = ROOT / "reports" / "streamq5_moe"
RUNS = ROOT / "reports" / "runs" / "streamq5_moe"
PREREG = R / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_PREREGISTRATION.md"
COMPILE_OUT = R / "port80b_d10a_next_component_composition_compile.json"
COMPILE_REPORT = R / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_COMPILE_REPORT_2026-08-13.md"
COMPONENT_OUT = R / "port80b_d10a_next_component_composition.json"
ENDURANCE_OUT = R / "port80b_d10a_next_component_endurance_10k.json"
REPORT = R / "PORT80B_D10A_NEXT_COMPONENT_COMPOSITION_REPORT_2026-08-13.md"
D9 = R / "port80b_d9_capacity_aware_bank_bridge.json"
D9_VERIFY = R / "port80b_d9_capacity_aware_bank_bridge_independent_verification.json"
N4A = R / "n4a_synthetic_80b_shape_capacity.json"
CAPTURE = R / "p4d_route_capture_result.json"
ROUTE_DIR = RUNS / "p4d_routes"

DOMAINS = ("general", "code", "math", "multilingual", "instruction")
PREFIX = 499
TOP_K = 10
HIDDEN = 2048
INTER = 512
TOKEN_BYTES = LAYERS * TOP_K * EXPERT_BYTES
TILES_PER_RECORD = EXPERT_BYTES // 4096
STAGE_BLOCKS = 1024
STAGE_THREADS = 256
CORRECTNESS = (0, 8)
VALIDATION = (512, 576)
ENDURANCE_SOURCE = (768, 1024)
ENDURANCE_STEPS = 10_000
ACK = "D10A_10000_AFTER_COMPONENT_PASS"
MIN_RAM_BEFORE = 50 * 2**30
MIN_RAM_AFTER_TOUCH = 2 * 2**30
EMERGENCY_RAM = int(1.5 * 2**30)
VRAM_RESERVE = 512 * 2**20
DENSE_BYTES = 1_933_921_280
KV_BYTES = 12 * 2 * 2 * 4096 * 256 * 2
RECURRENT_BYTES = 36 * 32 * 128 * 128 * 4
CONV_BYTES = 36 * (16 * 128 * 2 + 32 * 128) * 4 * 2
SHARED_BYTES = LAYERS * EXPERT_BYTES
RUNTIME_BYTES = 256 * 2**20
COLD_SLOTS = 32
COLD_BYTES = COLD_SLOTS * EXPERT_BYTES
OUTPUT_BYTES = LAYERS * TOP_K * (INTER + INTER + HIDDEN) * 4
REFERENCE_BYTES = TOP_K * EXPERT_BYTES
DEVICE_REQUEST = (
    DENSE_BYTES + KV_BYTES + RECURRENT_BYTES + CONV_BYTES + SHARED_BYTES
    + RUNTIME_BYTES + COLD_BYTES + TOKEN_BYTES * 2 + OUTPUT_BYTES * 2 + REFERENCE_BYTES
)
MASK64 = (1 << 64) - 1
LIFT_SEED = 0xD10A_499D_1308_2026


COMPONENT_SOURCE = r'''
extern "C" __global__ void differentiate_q5_from_header(unsigned char* records) {
  const unsigned long long record_bytes = 2027520ULL;
  const unsigned long long matrix_bytes = 675840ULL;
  const unsigned long long scale_offset = 64ULL + 655360ULL;
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total = 480ULL * 3ULL;
  for (; index < total; index += (unsigned long long)blockDim.x * gridDim.x) {
    unsigned long long row = index % 3ULL;
    unsigned long long record = index / 3ULL;
    unsigned char* header = records + record * record_bytes;
    unsigned short layer = *(unsigned short*)(header + 6);
    unsigned short expert = *(unsigned short*)(header + 8);
    unsigned int identifier = (unsigned int)layer * 512U + (unsigned int)expert;
    unsigned int digit = (identifier >> (5U * (unsigned int)row)) & 31U;
    unsigned char* up = records + record * record_bytes + matrix_bytes;
    ((unsigned short*)(up + scale_offset))[row * 16ULL] = (unsigned short)(0x3e80U + 4U * digit);
  }
}

extern "C" __global__ void differentiate_q5_expected(
    unsigned char* records, const short* route) {
  const unsigned long long record_bytes = 2027520ULL;
  const unsigned long long matrix_bytes = 675840ULL;
  const unsigned long long scale_offset = 64ULL + 655360ULL;
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total = 480ULL * 3ULL;
  for (; index < total; index += (unsigned long long)blockDim.x * gridDim.x) {
    unsigned long long row = index % 3ULL;
    unsigned long long record = index / 3ULL;
    unsigned int layer = (unsigned int)(record / 10ULL);
    unsigned int expert = (unsigned short)route[record];
    unsigned int identifier = layer * 512U + expert;
    unsigned int digit = (identifier >> (5U * (unsigned int)row)) & 31U;
    unsigned char* up = records + record * record_bytes + matrix_bytes + scale_offset;
    ((unsigned short*)up)[row * 16ULL] = (unsigned short)(0x3e80U + 4U * digit);
  }
}

extern "C" __global__ void verify_q5_fingerprints(
    const unsigned char* records, const short* route, unsigned long long* errors) {
  const unsigned long long record_bytes = 2027520ULL;
  const unsigned long long matrix_bytes = 675840ULL;
  const unsigned long long scale_offset = 64ULL + 655360ULL;
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total = 480ULL * 3ULL;
  unsigned long long local = 0;
  for (; index < total; index += (unsigned long long)blockDim.x * gridDim.x) {
    unsigned long long row = index % 3ULL;
    unsigned long long record = index / 3ULL;
    unsigned int layer = (unsigned int)(record / 10ULL);
    unsigned int expert = (unsigned short)route[record];
    unsigned int identifier = layer * 512U + expert;
    unsigned int digit = (identifier >> (5U * (unsigned int)row)) & 31U;
    const unsigned char* up = records + record * record_bytes + matrix_bytes + scale_offset;
    local += (unsigned long long)(((const unsigned short*)up)[row * 16ULL] != (unsigned short)(0x3e80U + 4U * digit));
  }
  if (local) atomicAdd(errors, local);
}

extern "C" __global__ void extract_q5_canaries(
    const unsigned char* records, unsigned short* actual_ids, unsigned short* words) {
  const unsigned long long record_bytes = 2027520ULL;
  const unsigned long long matrix_bytes = 675840ULL;
  const unsigned long long scale_offset = 64ULL + 655360ULL;
  unsigned long long record = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (record < 480ULL) {
    const unsigned char* header = records + record * record_bytes;
    unsigned short layer = *(const unsigned short*)(header + 6);
    unsigned short expert = *(const unsigned short*)(header + 8);
    actual_ids[record] = (unsigned short)((unsigned int)layer * 512U + (unsigned int)expert);
    const unsigned short* up = (const unsigned short*)(header + matrix_bytes + scale_offset);
    words[record * 3ULL + 0ULL] = up[0];
    words[record * 3ULL + 1ULL] = up[16];
    words[record * 3ULL + 2ULL] = up[32];
  }
}

extern "C" __global__ void dense_shell_work(
    const unsigned char* dense, unsigned long long bytes, unsigned long long step,
    unsigned long long* checksum) {
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  unsigned long long local = 0;
  for (; index < bytes; index += stride) local += ((unsigned long long)dense[index]) * (index + 1ULL + step);
  if (local) atomicAdd(checksum, local);
}

extern "C" __global__ void next_attention_kv_qgate(
    unsigned short* kv, float* output, int position, int step) {
  int index = (int)blockIdx.x * blockDim.x + threadIdx.x;
  int total = 12 * 16 * 256;
  if (index >= total) return;
  int layer = index / (16 * 256);
  int local = index - layer * 16 * 256;
  int head = local / 256;
  int dim = local - head * 256;
  int kv_head = head >> 3;
  unsigned long long base_k = (((((unsigned long long)layer * 2ULL) * 2ULL + kv_head) * 4096ULL + position) * 256ULL + dim);
  unsigned long long base_v = (((((unsigned long long)layer * 2ULL + 1ULL) * 2ULL + kv_head) * 4096ULL + position) * 256ULL + dim);
  float q = ((float)(((step + layer * 17 + head * 5 + dim) & 255) - 127)) / 128.0f;
  float k = ((float)(((step * 3 + layer * 11 + kv_head * 7 + dim) & 255) - 127)) / 128.0f;
  float v = ((float)(((step * 5 + layer * 13 + kv_head * 3 + dim) & 255) - 127)) / 128.0f;
  kv[base_k] = float_to_bf16(k);
  kv[base_v] = float_to_bf16(v);
  float gate = 1.0f / (1.0f + expf(-q));
  output[index] = round_bf16(gate * round_bf16(q + k * v));
}

extern "C" __global__ void gated_deltanet_step(
    float* recurrent, unsigned short* conv, float* output, int step) {
  unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long total = 36ULL * 32ULL * 128ULL * 128ULL;
  if (index < total) {
    unsigned long long cell = index % (128ULL * 128ULL);
    int i = (int)(cell / 128ULL), j = (int)(cell % 128ULL);
    int head = (int)((index / (128ULL * 128ULL)) % 32ULL);
    int layer = (int)(index / (32ULL * 128ULL * 128ULL));
    float k = ((float)(((step + layer * 7 + head * 3 + i) & 63) - 31)) / 64.0f;
    float v = ((float)(((step * 3 + layer * 5 + head + j) & 63) - 31)) / 64.0f;
    float next = recurrent[index] * 0.99609375f + k * v * 0.00390625f;
    recurrent[index] = next;
    if (i == 0 && j == 0) output[layer * 32 + head] = next;
  }
  unsigned long long conv_total = 36ULL * 8192ULL * 4ULL;
  if (index < conv_total) {
    unsigned long long channel = (index / 4ULL) % 8192ULL;
    unsigned long long layer = index / (8192ULL * 4ULL);
    unsigned long long slot = index & 3ULL;
    float value = ((float)(((step + (int)layer * 19 + (int)channel) & 127) - 63)) / 64.0f;
    if (slot == ((unsigned long long)step & 3ULL)) conv[index] = float_to_bf16(value);
  }
}

extern "C" __global__ void shared_q5_gate_up(
    const float* x, const unsigned char* bank, float* gate, float* up) {
  int group = (int)threadIdx.x >> 3;
  int lane = (int)threadIdx.x & 7;
  int global_row = (int)blockIdx.x * 32 + group;
  int layer = global_row >> 10;
  int local = global_row - layer * 1024;
  int projection = local >= 512;
  int row = local - projection * 512;
  long long base = (long long)layer * 2027520LL + (long long)projection * 675840LL;
  const unsigned char* packed = bank + base + 64;
  const unsigned short* scales = (const unsigned short*)(bank + base + 64 + 655360);
  float value = q5_ervf_row<8>(x, packed, scales, row, 2048, lane);
  if (lane == 0) { if (projection) up[layer * 512 + row] = round_bf16(value); else gate[layer * 512 + row] = round_bf16(value); }
}

extern "C" __global__ void shared_swiglu(float* gate, const float* up) {
  int index = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 48 * 512) {
    float g = round_bf16(gate[index]), u = round_bf16(up[index]);
    gate[index] = round_bf16(round_bf16(g / (1.0f + expf(-g))) * u);
  }
}

extern "C" __global__ void shared_q5_down(
    const float* activation, const unsigned char* bank, float* down) {
  int group = (int)threadIdx.x >> 3;
  int lane = (int)threadIdx.x & 7;
  int global_row = (int)blockIdx.x * 32 + group;
  int layer = global_row >> 11;
  int row = global_row - layer * 2048;
  long long base = (long long)layer * 2027520LL + 2LL * 675840LL;
  const unsigned char* packed = bank + base + 64;
  const unsigned short* scales = (const unsigned short*)(bank + base + 64 + 655360);
  float value = q5_ervf_row<8>(activation + layer * 512, packed, scales, row, 512, lane);
  if (lane == 0) down[layer * 2048 + row] = round_bf16(value);
}

extern "C" __global__ void compose_routed_shared(
    const float* routed, const float* shared, const float* attention,
    const float* delta, float* state) {
  int index = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 48 * 2048) {
    int layer = index / 2048, dim = index - layer * 2048;
    float route = routed[layer * 10 * 2048 + dim];
    float share = shared[index];
    float shell = ((layer + 1) % 4 == 0) ? attention[(layer / 4) * 16 * 256 + dim] : delta[(layer - ((layer + 1) / 4)) * 32 + (dim & 31)];
    float gate = 1.0f / (1.0f + expf(-((float)((layer + dim) & 31) - 15.0f) / 8.0f));
    state[index] = round_bf16(route + gate * share + shell);
  }
}
'''


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            value.update(block)
    return value.hexdigest()


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def load_routes() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    routes = {domain: np.empty((1024, LAYERS, 8), dtype=np.int16) for domain in DOMAINS}
    hashes: dict[str, str] = {}
    for layer in range(LAYERS):
        path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        hashes[str(layer)] = sha256(path)
        if hashes[str(layer)] != capture["manifests"][str(layer)]["artifact_sha256"]:
            raise RuntimeError(f"route hash mismatch at layer {layer}")
        tensors = load_file(path)
        for domain in DOMAINS:
            routes[domain][:, layer] = tensors[f"{domain}_router_ids"].astype(np.int16)
    return routes, hashes


def lift(source: np.ndarray, domain_index: int, token: int, epoch: int) -> np.ndarray:
    output = np.empty((LAYERS, TOP_K), dtype=np.int16)
    for layer in range(LAYERS):
        used: set[int] = set()
        for rank, raw in enumerate(source[layer]):
            state = LIFT_SEED ^ (domain_index << 56) ^ (token << 24) ^ (epoch << 16) ^ (layer << 8) ^ rank
            expert = int(raw) * 4 + int(splitmix64(state) & 3)
            output[layer, rank] = expert
            used.add(expert)
        state = LIFT_SEED ^ (domain_index << 52) ^ (token << 20) ^ (epoch << 12) ^ layer
        rank = 8
        while rank < TOP_K:
            state = splitmix64(state)
            expert = int(state % 512)
            if expert not in used:
                output[layer, rank] = expert
                used.add(expert)
                rank += 1
    return output


def route_inventory(routes: dict[str, np.ndarray]) -> dict[str, Any]:
    specs = {"correctness": CORRECTNESS, "validation": VALIDATION, "endurance_source": ENDURANCE_SOURCE}
    result: dict[str, Any] = {}
    for name, bounds in specs.items():
        digest = hashlib.sha256()
        cold: list[int] = []
        coverage: set[tuple[int, int]] = set()
        epochs = 8 if name == "endurance_source" else 1
        emitted = 0
        for epoch in range(epochs):
            for domain_index, domain in enumerate(DOMAINS):
                for token in range(bounds[0], bounds[1]):
                    if name == "endurance_source" and emitted >= ENDURANCE_STEPS:
                        break
                    lifted = lift(routes[domain][token], domain_index, token, epoch)
                    digest.update(lifted.tobytes())
                    cold.append(int(np.count_nonzero(lifted >= PREFIX)))
                    coverage.update((layer, int(expert)) for layer, row in enumerate(lifted) for expert in row)
                    emitted += 1
                if name == "endurance_source" and emitted >= ENDURANCE_STEPS:
                    break
            if name == "endurance_source" and emitted >= ENDURANCE_STEPS:
                break
        values = np.asarray(cold, dtype=np.float64)
        result[name] = {
            "label": "p4d_shaped_synthetic_proxy",
            "source_partition": list(bounds),
            "cases": emitted,
            "route_sha256": digest.hexdigest(),
            "cold_records": int(values.sum()),
            "cold_rate": float(values.sum() / (emitted * LAYERS * TOP_K)),
            "cold_per_step_p50": float(np.percentile(values, 50)),
            "cold_per_step_p95": float(np.percentile(values, 95)),
            "cold_per_step_p99": float(np.percentile(values, 99)),
            "cold_per_step_max": int(values.max()),
            "layer_expert_coverage": len(coverage),
        }
    return result


def build_module() -> tuple[cp.RawModule, list[str]]:
    from scripts.streamq5_moe.run_port80b_d2_registered_scatter import VERIFY_SOURCE

    names = [
        "host_to_smem_pipeline", "verify_record_bytes", "staged_q5_gate_up",
        "staged_q5_down", "canonical_swiglu_d7", "differentiate_q5_from_header",
        "differentiate_q5_expected", "verify_q5_fingerprints", "dense_shell_work",
        "extract_q5_canaries",
        "next_attention_kv_qgate", "gated_deltanet_step", "shared_q5_gate_up",
        "shared_swiglu", "shared_q5_down", "compose_routed_shared",
    ]
    cuda_include = ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "include"
    module = cp.RawModule(
        code=CUDA_SOURCE + ERVF_SOURCE + STAGE_SOURCE + COMPUTE_SOURCE + VERIFY_SOURCE + COMPONENT_SOURCE,
        options=("--std=c++14", f"--include-path={cuda_include}"), name_expressions=names,
    )
    for name in names:
        module.get_function(name)
    return module, names


def audit() -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    d9 = json.loads(D9.read_text(encoding="utf-8"))
    d9v = json.loads(D9_VERIFY.read_text(encoding="utf-8"))
    n4a = json.loads(N4A.read_text(encoding="utf-8"))
    routes, route_hashes = load_routes()
    mapped_readonly = False
    if BANK.is_file() and BANK.stat().st_size == BANK_BYTES:
        mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
        mapped_readonly = not mapped.flags.writeable
        del mapped
    checks = {
        "only_required_bulk_bank_exists": BANK.is_file() and BANK.stat().st_size == BANK_BYTES,
        "bank_manifest_sha": manifest.get("bank_sha256") == EXPECTED_BANK_SHA256,
        "bank_mapping_readonly": mapped_readonly,
        "d9_strong_and_clean": d9.get("strong_pass") is True and d9.get("unregister_failures") == [],
        "d9_independently_verified": d9v.get("all_checks_pass") is True,
        "n4a_shape_pass": n4a.get("overall_pass") is True,
        "next_geometry": n4a["architecture"]["layers"] == 48
        and n4a["architecture"]["full_attention_layers"] == 12
        and n4a["architecture"]["linear_attention_layers"] == 36
        and n4a["architecture"]["top_k"] == 10,
        "device_byte_math": DEVICE_REQUEST <= int(4.25 * 2**30),
        "partitions_disjoint": CORRECTNESS[1] <= VALIDATION[0] and VALIDATION[1] <= ENDURANCE_SOURCE[0],
        "all_route_hashes": len(route_hashes) == LAYERS,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "required_bulk_files": [{"path": str(BANK), "bytes": BANK_BYTES}],
        "explicitly_not_required": ["P1D bank", "P6A Q8 bank", "checkpoint shards", "generated dense sidecar"],
        "device_request_bytes": DEVICE_REQUEST,
        "device_request_gib": DEVICE_REQUEST / 2**30,
        "device_reserve_bytes": VRAM_RESERVE,
        "registered_bytes": LAYERS * PREFIX * EXPERT_BYTES,
        "registered_gib": LAYERS * PREFIX * EXPERT_BYTES / 2**30,
        "available_ram_bytes": int(psutil.virtual_memory().available),
        "free_disk_bytes": int(psutil.disk_usage(str(ROOT)).free),
        "architecture_limit": "No local official Next payload/hidden-state oracle; component exactness is against D10A synthetic references only.",
    }, routes, route_hashes


def canary_audit() -> dict[str, Any]:
    triples: list[tuple[int, int, int]] = []
    roundtrip_failures: list[int] = []
    boundary_failures: list[int] = []
    for layer in range(LAYERS):
        for expert in range(512):
            identifier = layer * 512 + expert
            digits = tuple((identifier >> (5 * place)) & 31 for place in range(3))
            words = tuple(0x3E80 + 4 * digit for digit in digits)
            triples.append(words)
            decoded_digits = tuple((word - 0x3E80) // 4 for word in words)
            decoded = decoded_digits[0] + 32 * decoded_digits[1] + 1024 * decoded_digits[2]
            if decoded != identifier:
                roundtrip_failures.append(identifier)
        hot_id = layer * 512 + 498
        cold_id = layer * 512 + 499
        hot = tuple(0x3E80 + 4 * ((hot_id >> (5 * place)) & 31) for place in range(3))
        cold = tuple(0x3E80 + 4 * ((cold_id >> (5 * place)) & 31) for place in range(3))
        if hot == cold or hot_id >= layer * 512 + PREFIX or cold_id < layer * 512 + PREFIX:
            boundary_failures.append(layer)
    return {
        "pairs": len(triples),
        "unique_canary_triples": len(set(triples)),
        "injective": len(set(triples)) == LAYERS * 512,
        "roundtrip_failures": roundtrip_failures,
        "roundtrip_pass": not roundtrip_failures,
        "boundary_498_499_failures": boundary_failures,
        "boundary_498_499_pass": not boundary_failures,
        "scale_word_min": min(min(row) for row in triples),
        "scale_word_max": max(max(row) for row in triples),
        "procedure": "id=512*layer+expert; three radix-32 digits; word=0x3e80+4*digit",
    }


def compile_phase() -> None:
    if COMPILE_OUT.exists() or COMPILE_REPORT.exists():
        raise FileExistsError("refusing to overwrite D10A compile/preflight evidence")
    started = time.perf_counter()
    error = None
    evidence: dict[str, Any] = {}
    names: list[str] = []
    inventory: dict[str, Any] = {}
    route_hashes: dict[str, str] = {}
    canaries: dict[str, Any] = {}
    try:
        evidence, routes, route_hashes = audit()
        if not evidence["pass"]:
            raise RuntimeError("preflight evidence failed")
        inventory = route_inventory(routes)
        canaries = canary_audit()
        if not canaries["injective"] or not canaries["roundtrip_pass"] or not canaries["boundary_498_499_pass"]:
            raise RuntimeError("exhaustive numerical-canary audit failed")
        if inventory["endurance_source"]["cases"] != ENDURANCE_STEPS:
            raise RuntimeError("endurance inventory is not exactly 10,000")
        if inventory["endurance_source"]["cold_per_step_max"] > COLD_SLOTS:
            raise RuntimeError("frozen cold escape exceeds allocated slots")
        compile(Path(__file__).read_text(encoding="utf-8"), str(Path(__file__)), "exec")
        _, names = build_module()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "kind": "port80b_d10a_next_component_composition_compile",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "compile_preflight_pass_endurance_closed" if error is None else "compile_preflight_fail",
        "pass": error is None and evidence.get("pass") is True,
        "inputs": {
            "preregistration_sha256": sha256(PREREG), "runner_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST), "d9_sha256": sha256(D9),
            "d9_verification_sha256": sha256(D9_VERIFY), "n4a_sha256": sha256(N4A),
            "capture_sha256": sha256(CAPTURE), "route_hashes": route_hashes,
        },
        "audit": evidence,
        "route_inventory": inventory,
        "canary_audit": canaries,
        "cuda_compile": {
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8"),
            "free_total_bytes": list(map(int, cp.cuda.runtime.memGetInfo())),
            "resolved_symbols": names,
        },
        "physical_actions": {"host_registration": False, "large_device_allocation": False, "kernel_launch": False, "bank_scan": False},
        "component_opened": False,
        "endurance_opened": False,
        "error": error,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Compile/read-only preflight only; no component execution or endurance authorization.",
    }
    COMPILE_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    COMPILE_REPORT.write_text(
        "# PORT80B-D10A1 component/composition compile report\n\n"
        f"Verdict: **{result['status']}**.\n\n"
        f"Resolved CUDA symbols: **{len(names)}**. Device request for the separately "
        f"authorized component run: **{DEVICE_REQUEST / 2**30:.3f} GiB**, plus a "
        f"{VRAM_RESERVE / 2**20:.0f} MiB reserve. Registered host prefix in that run: "
        f"**{LAYERS * PREFIX * EXPERT_BYTES / 2**30:.3f} GiB**.\n\n"
        "This phase compiled only. It launched no kernel, registered no host range, "
        "allocated no large device buffer, and did not scan the bank. The executable "
        "D10A1 component gate is implemented; the 10,000-step executor remains closed "
        "until a clean component result and separate authorization.\n\n"
        f"Claim boundary: {result['claim_boundary']}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


def locked_compile() -> dict[str, Any]:
    if not COMPILE_OUT.is_file():
        raise RuntimeError("compile evidence missing")
    result = json.loads(COMPILE_OUT.read_text(encoding="utf-8"))
    if not result.get("pass"):
        raise RuntimeError("compile evidence negative")
    if result["inputs"]["preregistration_sha256"] != sha256(PREREG) or result["inputs"]["runner_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("preregistration or runner changed after compile")
    return result


def process_memory() -> dict[str, int]:
    value = psutil.Process().memory_info()
    return {
        "rss": int(value.rss),
        "peak_wset": int(getattr(value, "peak_wset", value.rss)),
        "private": int(getattr(value, "private", 0)),
        "pagefile": int(getattr(value, "pagefile", 0)),
        "num_page_faults": int(getattr(value, "num_page_faults", 0)),
    }


def bf16_words(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_round(values: np.ndarray) -> np.ndarray:
    return (bf16_words(values).astype(np.uint32) << 16).view(np.float32)


def selected_cases(routes: dict[str, np.ndarray], bounds: tuple[int, int], limit: int | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(DOMAINS):
        for token in range(bounds[0], bounds[1]):
            lifted = lift(routes[domain][token], domain_index, token, 0)
            result.append({"domain": domain, "domain_index": domain_index, "token": token, "epoch": 0, "route": lifted})
            if limit is not None and len(result) >= limit:
                return result
    return result


def register_prefix(mapped: np.memmap) -> tuple[list[int], list[int]]:
    hosts: list[int] = []
    aliases: list[int] = []
    size = PREFIX * EXPERT_BYTES
    try:
        for layer in range(LAYERS):
            host = int(mapped.ctypes.data) + record_offset(layer, 0)
            cp.cuda.runtime.hostRegister(host, size, REGISTER_FLAGS)
            hosts.append(host)
            alias = int(cp.cuda.runtime.pointerGetAttributes(host).devicePointer)
            if not alias:
                raise RuntimeError(f"layer {layer}: null mapped alias")
            aliases.append(alias)
    except Exception:
        unregister_ranges(hosts)
        raise
    return hosts, aliases


def expected_canaries(route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = route.reshape(-1).astype(np.uint16)
    layers = np.repeat(np.arange(LAYERS, dtype=np.uint16), TOP_K)
    identifiers = (layers.astype(np.uint32) * 512 + flat.astype(np.uint32)).astype(np.uint16)
    words = np.empty((identifiers.size, 3), dtype=np.uint16)
    ids32 = identifiers.astype(np.uint32)
    for place in range(3):
        words[:, place] = (0x3E80 + 4 * ((ids32 >> (5 * place)) & 31)).astype(np.uint16)
    return identifiers, words


def compare_bits(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(left, dtype=np.float32)
    b = np.ascontiguousarray(right, dtype=np.float32)
    return {
        "elements": int(a.size),
        "different_bits": int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))),
        "bitwise_equal": bool(np.array_equal(a.view(np.uint32), b.view(np.uint32))),
        "max_abs": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0)),
        "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
        "left_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "right_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
    }


def component_phase() -> None:
    locked_compile()
    if COMPONENT_OUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D10A component evidence")
    ram_before = int(psutil.virtual_memory().available)
    if ram_before < MIN_RAM_BEFORE:
        raise RuntimeError("hard stop: less than 50 GiB system RAM available")
    free_vram, _ = cp.cuda.runtime.memGetInfo()
    if int(free_vram) < DEVICE_REQUEST + VRAM_RESERVE:
        raise RuntimeError("hard stop: exact device request plus 512 MiB reserve unavailable")
    audit_result, routes, _ = audit()
    if not audit_result["pass"]:
        raise RuntimeError("immutable audit failed")
    correctness_cases = selected_cases(routes, CORRECTNESS)
    validation_cases = selected_cases(routes, VALIDATION, 32)
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    module, names = build_module()
    kernels = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    started = time.perf_counter()
    hosts: list[int] = []
    aliases: list[int] = []
    unregister_failures: list[str] = []
    error = None
    payload: dict[str, Any] = {}
    page_sampler = HardPageReadSampler()
    available_after_registration: int | None = None
    available_after_first_touch: int | None = None
    free_after_allocations: int | None = None
    try:
        # All fixed shell/state/runtime buffers physically exist before host registration.
        dense = cp.full(DENSE_BYTES, np.uint8(0x5A), dtype=cp.uint8)
        kv = cp.zeros(KV_BYTES // 2, dtype=cp.uint16)
        recurrent = cp.zeros(RECURRENT_BYTES // 4, dtype=cp.float32)
        conv = cp.zeros(CONV_BYTES // 2, dtype=cp.uint16)
        shared = cp.empty(SHARED_BYTES, dtype=cp.uint8)
        runtime = cp.full(RUNTIME_BYTES, np.uint8(0xA5), dtype=cp.uint8)
        cold = cp.empty(COLD_BYTES, dtype=cp.uint8)
        staging = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
        oracle_stage = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
        reference10 = cp.empty(REFERENCE_BYTES, dtype=cp.uint8)
        free_after_allocations = int(cp.cuda.runtime.memGetInfo()[0])
        if free_after_allocations < VRAM_RESERVE:
            raise RuntimeError("hard stop: post-allocation VRAM reserve below 512 MiB")

        hosts, aliases = register_prefix(mapped)
        available_after_registration = int(psutil.virtual_memory().available)
        if len(hosts) != LAYERS or available_after_registration < MIN_RAM_AFTER_TOUCH:
            raise RuntimeError("hard stop after registration")

        for layer in range(LAYERS):
            cp.cuda.runtime.memcpyAsync(
                int(shared.data.ptr) + layer * EXPERT_BYTES,
                int(mapped.ctypes.data) + record_offset(layer, 512),
                EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
            )
        cp.cuda.runtime.memcpyAsync(reference10.data.ptr, int(mapped.ctypes.data), REFERENCE_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
        stream.synchronize()

        rng = np.random.default_rng(10_130_826)
        x_host = rng.standard_normal(HIDDEN, dtype=np.float32)
        x = cp.asarray(x_host)
        gate = cp.empty(TOP_K * INTER, dtype=cp.float32)
        up = cp.empty_like(gate)
        down = cp.empty(TOP_K * HIDDEN, dtype=cp.float32)
        routed_capture = cp.empty(OUTPUT_BYTES // 4, dtype=cp.float32)
        oracle_capture = cp.empty_like(routed_capture)
        routed_down = cp.empty(LAYERS * TOP_K * HIDDEN, dtype=cp.float32)
        shared_gate = cp.empty(LAYERS * INTER, dtype=cp.float32)
        shared_up = cp.empty_like(shared_gate)
        shared_down = cp.empty(LAYERS * HIDDEN, dtype=cp.float32)
        attention = cp.empty(12 * 16 * 256, dtype=cp.float32)
        delta = cp.empty(36 * 32, dtype=cp.float32)
        state = cp.empty(LAYERS * HIDDEN, dtype=cp.float32)
        dense_checksum = cp.zeros(1, dtype=cp.uint64)
        canary_errors = cp.zeros(1, dtype=cp.uint64)
        actual_ids_device = cp.empty(LAYERS * TOP_K, dtype=cp.uint16)
        canary_words_device = cp.empty(LAYERS * TOP_K * 3, dtype=cp.uint16)

        def pairs(route: np.ndarray) -> list[tuple[int, int]]:
            return [(layer, int(expert)) for layer, row in enumerate(route) for expert in row]

        def pointer_table(route: np.ndarray, override: tuple[int, int, int] | None = None) -> cp.ndarray:
            values: list[int] = []
            cold_slot = 0
            for record_index, (layer, expert) in enumerate(pairs(route)):
                source_layer, source_expert = layer, expert
                if override is not None and record_index == override[0]:
                    source_layer, source_expert = override[1], override[2]
                if source_expert < PREFIX:
                    values.append(aliases[source_layer] + source_expert * EXPERT_BYTES)
                else:
                    if cold_slot >= COLD_SLOTS:
                        raise RuntimeError("cold escape slot overflow")
                    target = int(cold.data.ptr) + cold_slot * EXPERT_BYTES
                    cp.cuda.runtime.memcpyAsync(
                        target, int(mapped.ctypes.data) + record_offset(source_layer, source_expert),
                        EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
                    )
                    values.append(target)
                    cold_slot += 1
            return cp.asarray(np.asarray(values, dtype=np.uint64))

        def stage(route: np.ndarray, override: tuple[int, int, int] | None = None) -> cp.ndarray:
            table = pointer_table(route, override)
            kernels["host_to_smem_pipeline"](
                (STAGE_BLOCKS,), (STAGE_THREADS,), (table, staging, np.uint64(480 * TILES_PER_RECORD)),
                shared_mem=4096, stream=stream,
            )
            # Keep the temporary device pointer table alive through every cold copy
            # and staging read before CuPy can recycle it. Inclusive wall timing still
            # begins before stage(), so the synchronization is deliberately measured.
            stream.synchronize()
            return table

        def assemble_oracle(route: np.ndarray) -> cp.ndarray:
            selected = pairs(route)
            for record_index, (layer, expert) in enumerate(selected):
                cp.cuda.runtime.memcpyAsync(
                    int(oracle_stage.data.ptr) + record_index * EXPERT_BYTES,
                    int(mapped.ctypes.data) + record_offset(layer, expert), EXPERT_BYTES,
                    cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
                )
            route_device = cp.asarray(route.reshape(-1).astype(np.int16))
            kernels["differentiate_q5_expected"]((32,), (256,), (oracle_stage, route_device), stream=stream)
            return route_device

        def q5_capture(bank: cp.ndarray, target: cp.ndarray, save_down: bool) -> None:
            stride = TOP_K * (INTER + INTER + HIDDEN)
            for layer in range(LAYERS):
                kernels["staged_q5_gate_up"]((TOP_K * 1024 // 32,), (256,), (x, bank, np.int32(layer), gate, up), stream=stream)
                kernels["canonical_swiglu_d7"](((TOP_K * INTER + 255) // 256,), (256,), (gate, up), stream=stream)
                kernels["staged_q5_down"]((TOP_K * HIDDEN // 32,), (256,), (gate, bank, np.int32(layer), down), stream=stream)
                base = int(target.data.ptr) + layer * stride * 4
                cp.cuda.runtime.memcpyAsync(base, gate.data.ptr, gate.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                cp.cuda.runtime.memcpyAsync(base + gate.nbytes, up.data.ptr, up.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                cp.cuda.runtime.memcpyAsync(base + gate.nbytes + up.nbytes, down.data.ptr, down.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                if save_down:
                    cp.cuda.runtime.memcpyAsync(
                        int(routed_down.data.ptr) + layer * TOP_K * HIDDEN * 4, down.data.ptr, down.nbytes,
                        cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr,
                    )

        def extract_canaries() -> tuple[np.ndarray, np.ndarray]:
            kernels["extract_q5_canaries"]((2,), (256,), (staging, actual_ids_device, canary_words_device), stream=stream)
            stream.synchronize()
            return cp.asnumpy(actual_ids_device), cp.asnumpy(canary_words_device).reshape(-1, 3)

        raw_canaries: list[dict[str, Any]] = []
        correctness_rows: list[dict[str, Any]] = []
        output_digests: list[str] = []
        for case_index, case in enumerate(correctness_cases):
            route = case["route"]
            stage(route)
            stream.synchronize()
            header_mismatches = full_verify(kernels["verify_record_bytes"], staging, pairs(route), stream)
            kernels["differentiate_q5_from_header"]((32,), (256,), (staging,), stream=stream)
            route_device = assemble_oracle(route)
            canary_errors.fill(0)
            kernels["verify_q5_fingerprints"]((32,), (256,), (staging, route_device, canary_errors), stream=stream)
            q5_capture(staging, routed_capture, True)
            q5_capture(oracle_stage, oracle_capture, False)
            stream.synchronize()
            candidate_host = cp.asnumpy(routed_capture)
            oracle_host = cp.asnumpy(oracle_capture)
            comparison = compare_bits(candidate_host, oracle_host)
            actual_ids, observed_words = extract_canaries()
            intended_ids, expected_words = expected_canaries(route)
            raw_canaries.append({
                "case": case_index, "domain": case["domain"], "token": case["token"],
                "intended_ids": intended_ids.astype(int).tolist(),
                "actual_header_ids": actual_ids.astype(int).tolist(),
                "expected_words": expected_words.astype(int).tolist(),
                "observed_words": observed_words.astype(int).tolist(),
            })
            digest = hashlib.sha256(candidate_host.tobytes()).hexdigest()
            output_digests.append(digest)
            correctness_rows.append({
                "case": case_index, "header_mismatches": header_mismatches,
                "canary_mismatches": int(canary_errors.get()[0]), "comparison": comparison,
                "raw_canary_exact": bool(np.array_equal(intended_ids, actual_ids) and np.array_equal(expected_words, observed_words)),
                "output_sha256": digest,
            })

        first_route = correctness_cases[0]["route"]
        injection_index = next(index for index, expert in enumerate(first_route.reshape(-1)) if int(expert) < PREFIX - 1)
        injection_layer = injection_index // TOP_K
        intended_expert = int(first_route.reshape(-1)[injection_index])
        injected_rows: dict[str, Any] = {}
        assemble_oracle(first_route)
        q5_capture(oracle_stage, oracle_capture, False)
        stream.synchronize()
        intended_output = cp.asnumpy(oracle_capture)
        for name, override in {
            "wrong_expert": (injection_index, injection_layer, intended_expert + 1),
            "wrong_layer": (injection_index, (injection_layer + 1) % LAYERS, intended_expert),
        }.items():
            stage(first_route, override)
            stream.synchronize()
            mismatches = full_verify(kernels["verify_record_bytes"], staging, pairs(first_route), stream)
            kernels["differentiate_q5_from_header"]((32,), (256,), (staging,), stream=stream)
            q5_capture(staging, routed_capture, False)
            stream.synchronize()
            wrong_output = cp.asnumpy(routed_capture)
            actual_ids, observed_words = extract_canaries()
            intended_ids, expected_words = expected_canaries(first_route)
            injected_rows[name] = {
                "override": list(override), "header_mismatches": mismatches,
                "numerical_comparison": compare_bits(wrong_output, intended_output),
                "intended_ids": intended_ids.astype(int).tolist(),
                "actual_header_ids": actual_ids.astype(int).tolist(),
                "expected_words": expected_words.astype(int).tolist(),
                "observed_words": observed_words.astype(int).tolist(),
            }

        # Independent physical component checks before composed validation.
        attention.fill(0); recurrent.fill(0); conv.fill(0); shared_gate.fill(0); shared_up.fill(0); shared_down.fill(0)
        kernels["next_attention_kv_qgate"](((12 * 16 * 256 + 255) // 256,), (256,), (kv, attention, np.int32(0), np.int32(0)), stream=stream)
        kernels["gated_deltanet_step"](((RECURRENT_BYTES // 4 + 255) // 256,), (256,), (recurrent, conv, delta, np.int32(0)), stream=stream)
        kernels["shared_q5_gate_up"]((48 * 1024 // 32,), (256,), (x, shared, shared_gate, shared_up), stream=stream)
        kernels["shared_swiglu"](((48 * INTER + 255) // 256,), (256,), (shared_gate, shared_up), stream=stream)
        kernels["shared_q5_down"]((48 * HIDDEN // 32,), (256,), (shared_gate, shared, shared_down), stream=stream)
        dense_checksum.fill(0)
        kernels["dense_shell_work"]((4096,), (256,), (dense, np.uint64(DENSE_BYTES), np.uint64(0), dense_checksum), stream=stream)
        # The retained bank has a uniform numerical payload. Run the ordinary D7
        # kernels on layer-0/expert-0 only, then compare that independent resident
        # record against each layer's shared expert. Treating this ten-record buffer
        # as a 48-layer bank would be an out-of-bounds read.
        kernels["staged_q5_gate_up"]((TOP_K * 1024 // 32,), (256,), (x, reference10, np.int32(0), gate, up), stream=stream)
        kernels["canonical_swiglu_d7"](((TOP_K * INTER + 255) // 256,), (256,), (gate, up), stream=stream)
        kernels["staged_q5_down"]((TOP_K * HIDDEN // 32,), (256,), (gate, reference10, np.int32(0), down), stream=stream)
        stream.synchronize()
        attention_host = cp.asnumpy(attention)
        recurrent_host = cp.asnumpy(recurrent)
        conv_host = cp.asnumpy(conv)
        shared_host = cp.asnumpy(shared_down).reshape(LAYERS, HIDDEN)
        resident_one = cp.asnumpy(down).reshape(TOP_K, HIDDEN)[0]
        resident = np.broadcast_to(resident_one, (LAYERS, HIDDEN)).copy()

        idx = np.arange(12 * 16 * 256, dtype=np.int64)
        layer = idx // (16 * 256); local = idx - layer * 16 * 256; head = local // 256; dim = local - head * 256; kv_head = head // 8
        q = (((layer * 17 + head * 5 + dim) & 255) - 127).astype(np.float32) / np.float32(128)
        k = (((layer * 11 + kv_head * 7 + dim) & 255) - 127).astype(np.float32) / np.float32(128)
        v = (((layer * 13 + kv_head * 3 + dim) & 255) - 127).astype(np.float32) / np.float32(128)
        attention_expected = bf16_round((1.0 / (1.0 + np.exp(-q))) * bf16_round(q + k * v))
        sample_idx = np.arange(0, recurrent_host.size, max(1, recurrent_host.size // 4096), dtype=np.int64)[:4096]
        cell = sample_idx % (128 * 128); ri = cell // 128; rj = cell % 128; rh = (sample_idx // (128 * 128)) % 32; rl = sample_idx // (32 * 128 * 128)
        rk = (((rl * 7 + rh * 3 + ri) & 63) - 31).astype(np.float32) / np.float32(64)
        rv = (((rl * 5 + rh + rj) & 63) - 31).astype(np.float32) / np.float32(64)
        recurrent_expected = rk * rv * np.float32(0.00390625)
        dense_expected = (0x5A * (DENSE_BYTES * (DENSE_BYTES + 1) // 2)) & MASK64
        components = {
            "attention_max_abs": float(np.max(np.abs(attention_host - attention_expected))),
            "attention_finite": bool(np.isfinite(attention_host).all()),
            "recurrent_sample_max_abs": float(np.max(np.abs(recurrent_host[sample_idx] - recurrent_expected))),
            "conv_nonzero": int(np.count_nonzero(conv_host)),
            "shared_vs_resident": compare_bits(shared_host, resident),
            "dense_checksum_observed": int(dense_checksum.get()[0]),
            "dense_checksum_expected": int(dense_expected),
            "runtime_touch_sentinels": [int(runtime[0].get()), int(runtime[-1].get())],
        }

        recurrent.fill(0); conv.fill(0); kv.fill(0); dense_checksum.fill(0)
        page_sampler.start()
        validation_wall: list[float] = []
        validation_event: list[float] = []
        telemetry: list[dict[str, Any]] = []
        for warmup in range(8):
            case = validation_cases[warmup % len(validation_cases)]
            stage(case["route"]); kernels["differentiate_q5_from_header"]((32,), (256,), (staging,), stream=stream); q5_capture(staging, routed_capture, True)
            kernels["shared_q5_gate_up"]((48 * 1024 // 32,), (256,), (x, shared, shared_gate, shared_up), stream=stream)
            kernels["shared_swiglu"](((48 * INTER + 255) // 256,), (256,), (shared_gate, shared_up), stream=stream)
            kernels["shared_q5_down"]((48 * HIDDEN // 32,), (256,), (shared_gate, shared, shared_down), stream=stream)
            kernels["next_attention_kv_qgate"](((12 * 16 * 256 + 255) // 256,), (256,), (kv, attention, np.int32(warmup), np.int32(warmup)), stream=stream)
            kernels["gated_deltanet_step"](((RECURRENT_BYTES // 4 + 255) // 256,), (256,), (recurrent, conv, delta, np.int32(warmup)), stream=stream)
            kernels["dense_shell_work"]((4096,), (256,), (dense, np.uint64(DENSE_BYTES), np.uint64(warmup), dense_checksum), stream=stream)
            kernels["compose_routed_shared"](((LAYERS * HIDDEN + 255) // 256,), (256,), (routed_down, shared_down, attention, delta, state), stream=stream)
        stream.synchronize()
        available_after_first_touch = int(psutil.virtual_memory().available)
        if available_after_first_touch < MIN_RAM_AFTER_TOUCH:
            raise RuntimeError("hard stop: less than 2 GiB after first touch")
        for step, case in enumerate(validation_cases):
            if int(psutil.virtual_memory().available) < EMERGENCY_RAM:
                raise RuntimeError("emergency RAM stop during validation")
            begin_event, end_event = cp.cuda.Event(), cp.cuda.Event()
            begin_wall = time.perf_counter(); begin_event.record(stream)
            stage(case["route"]); kernels["differentiate_q5_from_header"]((32,), (256,), (staging,), stream=stream); q5_capture(staging, routed_capture, True)
            kernels["shared_q5_gate_up"]((48 * 1024 // 32,), (256,), (x, shared, shared_gate, shared_up), stream=stream)
            kernels["shared_swiglu"](((48 * INTER + 255) // 256,), (256,), (shared_gate, shared_up), stream=stream)
            kernels["shared_q5_down"]((48 * HIDDEN // 32,), (256,), (shared_gate, shared, shared_down), stream=stream)
            position = step & 4095
            kernels["next_attention_kv_qgate"](((12 * 16 * 256 + 255) // 256,), (256,), (kv, attention, np.int32(position), np.int32(step)), stream=stream)
            kernels["gated_deltanet_step"](((RECURRENT_BYTES // 4 + 255) // 256,), (256,), (recurrent, conv, delta, np.int32(step)), stream=stream)
            kernels["dense_shell_work"]((4096,), (256,), (dense, np.uint64(DENSE_BYTES), np.uint64(step), dense_checksum), stream=stream)
            kernels["compose_routed_shared"](((LAYERS * HIDDEN + 255) // 256,), (256,), (routed_down, shared_down, attention, delta, state), stream=stream)
            end_event.record(stream); end_event.synchronize()
            validation_wall.append((time.perf_counter() - begin_wall) * 1000.0)
            validation_event.append(float(cp.cuda.get_elapsed_time(begin_event, end_event)))
            telemetry.append({
                "step": step, "available_ram": int(psutil.virtual_memory().available),
                "free_vram": int(cp.cuda.runtime.memGetInfo()[0]), "process": process_memory(),
            })
        page_sampler.stop()
        wall_stats = stats(validation_wall)
        event_stats = stats(validation_event)
        page_rates = [float(row.get("page_reads_per_sec", 0.0)) for row in page_sampler.samples]
        memory_loss = (telemetry[0]["available_ram"] - telemetry[-1]["available_ram"]) if telemetry else math.inf
        component_gates = {
            "canary_exhaustive_injective_roundtrip_boundary": all(canary_audit()[key] for key in ("injective", "roundtrip_pass", "boundary_498_499_pass")),
            "all_correctness_headers_zero_mismatch": all(row["header_mismatches"] == 0 for row in correctness_rows),
            "all_canaries_raw_exact": all(row["raw_canary_exact"] and row["canary_mismatches"] == 0 for row in correctness_rows),
            "all_routed_q5_bitexact": all(row["comparison"]["bitwise_equal"] for row in correctness_rows),
            "output_digest_uniqueness_ge_95pct": len(set(output_digests)) / len(output_digests) >= 0.95,
            "wrong_expert_header_and_numerical_detected": injected_rows["wrong_expert"]["header_mismatches"] > 0 and not injected_rows["wrong_expert"]["numerical_comparison"]["bitwise_equal"],
            "wrong_layer_header_and_numerical_detected": injected_rows["wrong_layer"]["header_mismatches"] > 0 and not injected_rows["wrong_layer"]["numerical_comparison"]["bitwise_equal"],
            "attention_reference_abs_rel_le_2e_5": components["attention_max_abs"] <= 2e-5,
            "gdn_reference_abs_rel_le_2e_5": components["recurrent_sample_max_abs"] <= 2e-5 and components["conv_nonzero"] > 0,
            "shared_q5_bitexact": components["shared_vs_resident"]["bitwise_equal"],
            "dense_and_runtime_touched": components["dense_checksum_observed"] == components["dense_checksum_expected"] and components["runtime_touch_sentinels"] == [0xA5, 0xA5],
            "validation_32_finite": len(validation_wall) == 32 and bool(np.isfinite(validation_wall).all()),
            "validation_wall_p95_le_150ms": wall_stats["p95"] <= 150.0,
            "validation_wall_p99_le_200ms": wall_stats["p99"] <= 200.0,
            "post_warmup_page_reads_no_sample_gt_2048": bool(page_rates) and max(page_rates) <= 2048.0,
            "validation_memory_loss_le_1gib": memory_loss <= 2**30,
            "ram_after_first_touch_ge_2gib": available_after_first_touch >= MIN_RAM_AFTER_TOUCH,
            "vram_reserve_ge_512mib": min(row["free_vram"] for row in telemetry) >= VRAM_RESERVE,
            "registration_48_ranges": len(hosts) == LAYERS,
            "no_cuda_or_runner_error": True,
        }
        payload = {
            "raw_canary_arrays": raw_canaries,
            "correctness": correctness_rows,
            "output_digests": output_digests,
            "negative_controls": injected_rows,
            "components": components,
            "validation": {"wall_ms": validation_wall, "cuda_event_ms": validation_event, "wall_stats": wall_stats, "cuda_event_stats": event_stats},
            "telemetry": telemetry,
            "page_reads": {"samples": page_sampler.samples, "error": page_sampler.error},
            "gates": component_gates,
        }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            page_sampler.stop()
        except Exception:
            pass
    finally:
        try:
            stream.synchronize()
        except Exception:
            pass
        unregister_failures = unregister_ranges(hosts)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    gates = payload.setdefault("gates", {})
    gates["clean_unregister_48_ranges"] = len(hosts) == LAYERS and not unregister_failures
    gates["no_cuda_or_runner_error"] = error is None
    overall = bool(gates) and all(gates.values())
    result = {
        "kind": "port80b_d10a_next_component_composition",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "component_composition_pass_endurance_authorizable" if overall else "component_composition_negative_endurance_closed",
        "overall_pass": overall,
        "endurance_authorized_by_evidence": overall,
        "inputs": {
            "preregistration_sha256": sha256(PREREG), "runner_sha256": sha256(Path(__file__)),
            "compile_sha256": sha256(COMPILE_OUT), "bank_sha256_from_manifest": EXPECTED_BANK_SHA256,
            "route_label": "p4d_shaped_synthetic_proxy",
        },
        "physical": {
            "device_request_bytes": DEVICE_REQUEST, "registered_bytes": LAYERS * PREFIX * EXPERT_BYTES,
            "available_ram_before": ram_before, "available_ram_after_registration": available_after_registration,
            "available_ram_after_first_touch": available_after_first_touch, "available_ram_after_cleanup": int(psutil.virtual_memory().available),
            "free_vram_after_allocations": free_after_allocations,
        },
        **payload,
        "error": error,
        "unregister_failures": unregister_failures,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Synthetic shape-informed physical shell stress/composition only; not an exact Qwen3-Next shell, real checkpoint, natural routing, quality or endurance result.",
    }
    COMPONENT_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# PORT80B-D10A — Next component/composition report\n\n"
        f"Verdict: **{result['status']}**. Endurance evidence-authorized: **{overall}**.\n\n"
        f"Validation wall p50/p95/p99: {payload.get('validation', {}).get('wall_stats', {}).get('p50', '—')} / "
        f"{payload.get('validation', {}).get('wall_stats', {}).get('p95', '—')} / {payload.get('validation', {}).get('wall_stats', {}).get('p99', '—')} ms.\n\n"
        f"Claim boundary: {result['claim_boundary']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "overall_pass": overall, "error": error, "unregister_failures": unregister_failures}, indent=2))


def endurance_phase(acknowledgement: str | None) -> None:
    locked_compile()
    if acknowledgement != ACK:
        raise RuntimeError("endurance acknowledgement missing")
    if not COMPONENT_OUT.is_file():
        raise RuntimeError("passing component evidence missing")
    component = json.loads(COMPONENT_OUT.read_text(encoding="utf-8"))
    if not component.get("overall_pass") or component.get("unregister_failures") != []:
        raise RuntimeError("component gate is not a clean pass")
    if ENDURANCE_OUT.exists():
        raise FileExistsError(ENDURANCE_OUT)
    raise RuntimeError("10k executor remains fail-closed until a post-component authorization turn; no GPU action was taken")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("compile", "component", "endurance"), required=True)
    parser.add_argument("--acknowledge-endurance")
    args = parser.parse_args()
    if args.phase == "compile":
        compile_phase()
    elif args.phase == "component":
        component_phase()
    else:
        endurance_phase(args.acknowledge_endurance)


if __name__ == "__main__":
    main()
