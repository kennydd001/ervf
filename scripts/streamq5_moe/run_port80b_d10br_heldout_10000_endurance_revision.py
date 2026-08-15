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
    header_reference,
    record_offset,
    stats,
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
PREREG = R / "PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_PREREGISTRATION.md"
COMPILE_OUT = R / "port80b_d10br_heldout_10000_endurance_revision_preflight.json"
COMPILE_REPORT = R / "PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_PREFLIGHT_REPORT_2026-08-13.md"
COMPONENT_OUT = R / "port80b_d10a2r2_gdn36_oracle_repair.json"
ENDURANCE_OUT = R / "port80b_d10br_heldout_10000_endurance_revision.json"
REPORT = R / "PORT80B_D10BR_HELDOUT_10000_ENDURANCE_REVISION_REPORT_2026-08-13.md"
D10B_FAILED_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10b_heldout_10000_endurance.py"
D10B_FAILED_PREREG = R / "PORT80B_D10B_HELDOUT_10000_ENDURANCE_PREREGISTRATION.md"
D10B_FAILED_PREFLIGHT = R / "port80b_d10b_heldout_10000_endurance_preflight.json"
D10B_FAILED_PREFLIGHT_REPORT = R / "PORT80B_D10B_HELDOUT_10000_ENDURANCE_PREFLIGHT_REPORT_2026-08-13.md"
EXPECTED_D10B_FAILED_LOCKS = {
    "runner": "4f8226d82d7d804195a9728bc9852cc9b75fa33ec6d8481e86d94ae90ff3cb68",
    "prereg": "8d171ac876d03681d35de9155b100ec01e3345588ca31eb926e9acddaa59b977",
    "preflight": "91bb855940f0d39f241c29159ae39c011c46e4e9a3297d50bdb696e90fde985e",
    "preflight_report": "c0e9fcbcc9e1010307bffe23334b9a829cf7d5f7310a55092a0bfb1eb2c4dd21",
}
CONV_UNIT_TEST = ROOT / "scripts" / "streamq5_moe" / "test_port80b_d10a2r2_conv_oracle.py"
CONV_UNIT_RESULT = R / "port80b_d10a2r2_conv_oracle_unit.json"
D10A2R_RESULT = R / "port80b_d10a2r_single_stream_repair_revision.json"
D10A2R_REPORT = R / "PORT80B_D10A2R_SINGLE_STREAM_REPAIR_REVISION_REPORT_2026-08-13.md"
D10A2R_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a2r_single_stream_repair_revision.py"
D10A2R_PREREG = R / "PORT80B_D10A2R_SINGLE_STREAM_REPAIR_REVISION_PREREGISTRATION.md"
EXPECTED_D10A2R_LOCKS = {
    "result": "e328d555eefa0140c6b3075d30c2ae76db4fdce0e141794ad71eab5a61ef3a7f",
    "report": "a2397b163ada4d134b47a2107d5ae7c25ff9c4abb9fb26f40ef9dbfd7617edf9",
    "runner": "ea85a6e9d27627c883b6db3b63a1cdfb12040009fb89ef77be83b16eba51c275",
    "prereg": "3598c9d2da024cfe4d2ad749e7b1811982f444a3e46fd0cdcb21fa8f60da3bc0",
}
INDEPENDENT_AUDIT = R / "port80b_d10a2r2_component_independent_verification.json"
EXPECTED_INDEPENDENT_AUDIT_SHA256 = "409c379600b733bc466b21c981f75342d6087612f3c60f6e7c4889f31828ab6d"
INDEPENDENT_AUDIT_REPORT = R / "PORT80B_D10A2R2_COMPONENT_INDEPENDENT_VERIFICATION_REPORT_2026-08-13.md"
EXPECTED_INDEPENDENT_AUDIT_REPORT_SHA256 = "a02e1caa978e27a6cf717629128ec6d57292232d36a351934c48e6dba81efbaa"
D10A2R2_ERRATUM = R / "PORT80B_D10A2R2_REPORT_ERRATUM_2026-08-13.md"
EXPECTED_D10A2R2_ERRATUM_SHA256 = "1335889ac3c6668339bf50b6a8d707dc2714f21e07207683b206da074ff1393f"
D10A2R2_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a2r2_gdn36_oracle_repair.py"
D10A2R2_PREREG = R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_PREREGISTRATION.md"
D10A2R2_UNIT_TEST = ROOT / "scripts" / "streamq5_moe" / "test_port80b_d10a2r2_conv_oracle.py"
D10A2R2_UNIT = R / "port80b_d10a2r2_conv_oracle_unit.json"
D10A2R2_PREFLIGHT = R / "port80b_d10a2r2_gdn36_oracle_repair_preflight.json"
D10A2R2_PREFLIGHT_REPORT = R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_PREFLIGHT_REPORT_2026-08-13.md"
D10A2R2_RESULT = COMPONENT_OUT
D10A2R2_REPORT = R / "PORT80B_D10A2R2_GDN36_ORACLE_REPAIR_REPORT_2026-08-13.md"
EXPECTED_D10A2R2_LOCKS = {
    "runner": "8bce44334d9d416ad53e7f8499b676133e21248964dee79bccd16cc04f65cf8c",
    "prereg": "46cffb32de3228b30ff6b45003bdc288c39d11d02857a00409334e29a980022a",
    "unit_test": "02dfa87cee8ac58b54a8b71656d109a55d96298db659228c2726f447214f3650",
    "unit": "ba7c398facaaa88b46ad95ec020bd031fed324755ed0bf7550af0c63ba9941c1",
    "preflight": "eba995a770b8671f05fdea9c4fd593c9eacf04f05f26cb82ba343b5a0afb160c",
    "preflight_report": "f7b2ceb9edc4ae06f2741fffc4010211348945c0fb5fc6bb0f7510e447d6349a",
    "result": "cd4486221dae9073a14a7e0d617c803120f7f3e094580559c81d9035111063b1",
    "report": "a8388d54b4e0b5e76e2eec6a28f9d7cd3ef7e9e858d2051a8cdcd2cf104e601b",
}
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
ACK = "D10B_HELDOUT_10000_AFTER_AUDIT_AND_PREFLIGHT"
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
EXPECTED_D10A1R_SHA256 = "c92e5dda380c8f9ed0669fc8961056bef58fabbf758946776b426fa7feb888ae"
D10A1R = R / "port80b_d10a1r_conservative_resource_retry.json"
D10A1R_RUNNER = ROOT / "scripts" / "streamq5_moe" / "run_port80b_d10a1r_conservative_resource_retry.py"
D10A1R_PREREG = R / "PORT80B_D10A1R_CONSERVATIVE_RESOURCE_RETRY_PREREGISTRATION.md"
COUNTER_AUDIT = R / "D10A1R_FAILURE_COUNTER_AUDIT_2026-08-13.md"
EXPECTED_PRIOR_LOCKS = {
    "d10a1r": EXPECTED_D10A1R_SHA256,
    "d10a1r_runner": "a9bb549b6f7a21dfedaf28a44b8e249b28a8b75747502e2113d31fe02f5c189d",
    "d10a1r_prereg": "d606b438595a3aba7f4a1fb11aa93c97bf42a1a63f544138a5477c7c94fc62c7",
    "counter_audit": "75f5ac247a2c270f5f2d0480cbd63833ad3d1755b509a7929946c948b95fc5e1",
}
START_RAM_GATE = 52_652_163_072
MIN_RAM_BEFORE = START_RAM_GATE
CONV_EXPECTED_NONZERO_STEP0 = 292_608
GDN_LAYERS = 36
CONV_EXPECTED_WORDS = 1_179_648
CONV_EXPECTED_SHA256 = "cedf5736557919b023d6f7cce73d0064df07236ff1e18b5d8b3fec49d658fa1e"
POISON = np.float32(12345.25)
DIGEST_STEPS = tuple([0] + list(range(99, ENDURANCE_STEPS, 100)))
EXPECTED_ENDURANCE_ROUTE_SHA256 = "85f12fb0020bb8568dfc3683662e8251b29bf83684beb296dbb6d8734f5ffd20"
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
    conv_unit = json.loads(CONV_UNIT_RESULT.read_text(encoding="utf-8"))
    routes, route_hashes = load_routes()
    prior_locks = {
        "d10a1r": sha256(D10A1R),
        "d10a1r_runner": sha256(D10A1R_RUNNER),
        "d10a1r_prereg": sha256(D10A1R_PREREG),
        "counter_audit": sha256(COUNTER_AUDIT),
    }
    d10a2r_locks = {
        "result": sha256(D10A2R_RESULT),
        "report": sha256(D10A2R_REPORT),
        "runner": sha256(D10A2R_RUNNER),
        "prereg": sha256(D10A2R_PREREG),
    }
    d10a2r2_locks = {
        "runner": sha256(D10A2R2_RUNNER),
        "prereg": sha256(D10A2R2_PREREG),
        "unit_test": sha256(D10A2R2_UNIT_TEST),
        "unit": sha256(D10A2R2_UNIT),
        "preflight": sha256(D10A2R2_PREFLIGHT),
        "preflight_report": sha256(D10A2R2_PREFLIGHT_REPORT),
        "result": sha256(D10A2R2_RESULT),
        "report": sha256(D10A2R2_REPORT),
    }
    d10b_failed_locks = {
        "runner": sha256(D10B_FAILED_RUNNER),
        "prereg": sha256(D10B_FAILED_PREREG),
        "preflight": sha256(D10B_FAILED_PREFLIGHT),
        "preflight_report": sha256(D10B_FAILED_PREFLIGHT_REPORT),
    }
    d10a2r2_result = json.loads(D10A2R2_RESULT.read_text(encoding="utf-8"))
    independent = json.loads(INDEPENDENT_AUDIT.read_text(encoding="utf-8"))
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
        "immutable_d10a1r_and_counter_audit": prior_locks == EXPECTED_PRIOR_LOCKS,
        "immutable_d10a2r_blocked_execution": d10a2r_locks == EXPECTED_D10A2R_LOCKS,
        "immutable_d10a2r2_clean_component_pass": (
            d10a2r2_locks == EXPECTED_D10A2R2_LOCKS
            and d10a2r2_result.get("overall_pass") is True
            and all(d10a2r2_result.get("gates", {}).values())
            and d10a2r2_result.get("error") is None
            and d10a2r2_result.get("unregister_failures") == []
            and d10a2r2_result.get("endurance_authorized_by_evidence") is False
        ),
        "independent_d10a2r2_26_of_26_and_erratum_locked": (
            sha256(INDEPENDENT_AUDIT) == EXPECTED_INDEPENDENT_AUDIT_SHA256
            and sha256(INDEPENDENT_AUDIT_REPORT) == EXPECTED_INDEPENDENT_AUDIT_REPORT_SHA256
            and sha256(D10A2R2_ERRATUM) == EXPECTED_D10A2R2_ERRATUM_SHA256
            and independent.get("verification_pass") is True
            and len(independent.get("independent_checks", {})) == 26
            and all(independent.get("independent_checks", {}).values())
            and independent["endurance_decision"]["component_evidence_supports_new_preregistered_arm"] is True
            and independent["endurance_decision"]["d10a2r2_endurance_actually_open"] is False
        ),
        "immutable_d10b_failed_preflight": d10b_failed_locks == EXPECTED_D10B_FAILED_LOCKS,
        "conservative_start_ram_gate": MIN_RAM_BEFORE == START_RAM_GATE,
        "conv_oracle_cpu_unit_locked_and_passed": (
            conv_unit.get("pass") is True
            and conv_unit["inputs"]["runner_sha256"] == EXPECTED_D10A2R2_LOCKS["runner"]
            and conv_unit["inputs"]["preregistration_sha256"] == EXPECTED_D10A2R2_LOCKS["prereg"]
            and conv_unit["inputs"]["unit_test_sha256"] == EXPECTED_D10A2R2_LOCKS["unit_test"]
            and conv_unit["audit"]["sha256"] == CONV_EXPECTED_SHA256
            and conv_unit["audit"]["nonzero_words"] == CONV_EXPECTED_NONZERO_STEP0
            and conv_unit["audit"]["shape"] == [CONV_EXPECTED_WORDS]
        ),
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
        "immutable_prior_locks": prior_locks,
        "immutable_d10a2r_locks": d10a2r_locks,
        "immutable_d10a2r2_locks": d10a2r2_locks,
        "immutable_d10b_failed_locks": d10b_failed_locks,
        "independent_audit": {
            "json_sha256": sha256(INDEPENDENT_AUDIT),
            "report_sha256": sha256(INDEPENDENT_AUDIT_REPORT),
            "erratum_sha256": sha256(D10A2R2_ERRATUM),
            "verification_pass": independent.get("verification_pass"),
            "check_count": len(independent.get("independent_checks", {})),
        },
        "conv_oracle_cpu_unit": conv_unit,
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
    if INDEPENDENT_AUDIT is None or EXPECTED_INDEPENDENT_AUDIT_SHA256 is None:
        raise RuntimeError("D10B preflight prohibited until independent new_pack_audit is hash-locked")
    if sha256(INDEPENDENT_AUDIT) != EXPECTED_INDEPENDENT_AUDIT_SHA256:
        raise RuntimeError("D10B independent audit hash mismatch")
    if COMPILE_OUT.exists() or COMPILE_REPORT.exists():
        raise FileExistsError("refusing to overwrite D10A compile/preflight evidence")
    started = time.perf_counter()
    error = None
    evidence: dict[str, Any] = {}
    names: list[str] = []
    inventory: dict[str, Any] = {}
    route_hashes: dict[str, str] = {}
    canaries: dict[str, Any] = {}
    required_source_contract: dict[str, bool] = {}
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
        source = Path(__file__).read_text(encoding="utf-8")
        compile(source, str(Path(__file__)), "exec")
        required_source_contract = {
            "explicit_nonblocking_stream": "cp.cuda.Stream(non_blocking=True)" in source,
            "single_stream_context": "with stream:" in source,
            "same_stream_verifier": "def same_stream_full_verify" in source,
            "exact_conv_oracle": all(value in source for value in ("GDN_LAYERS = 36", "CONV_EXPECTED_NONZERO_STEP0", "CONV_EXPECTED_SHA256")),
            "shared_payload_oracle": "shared_payload_comparison" in source,
            "full_state_digests": "validation_output_evidence" in source,
            "registration_attempt_rows": "registration_attempts" in source,
            "unregister_attempt_rows": "unregister_attempts" in source,
            "heldout_10000_executor": "def heldout_endurance_cases" in source and "len(cases) == ENDURANCE_STEPS" in source,
            "exact_digest_cadence": "DIGEST_STEPS" in source and "len(checkpoint_evidence) == len(DIGEST_STEPS)" in source,
        }
        if not all(required_source_contract.values()):
            raise RuntimeError("D10A2 static mutation contract failed")
        names = [
            "host_to_smem_pipeline", "verify_record_bytes", "staged_q5_gate_up",
            "staged_q5_down", "canonical_swiglu_d7", "differentiate_q5_from_header",
            "differentiate_q5_expected", "verify_q5_fingerprints", "dense_shell_work",
            "extract_q5_canaries", "next_attention_kv_qgate", "gated_deltanet_step",
            "shared_q5_gate_up", "shared_swiglu", "shared_q5_down", "compose_routed_shared",
        ]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "kind": "port80b_d10br_heldout_10000_endurance_revision_cpu_preflight",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "compile_preflight_pass_endurance_closed" if error is None else "compile_preflight_fail",
        "pass": error is None and evidence.get("pass") is True,
        "inputs": {
            "preregistration_sha256": sha256(PREREG), "runner_sha256": sha256(Path(__file__)),
            "manifest_sha256": sha256(MANIFEST), "d9_sha256": sha256(D9),
            "d9_verification_sha256": sha256(D9_VERIFY), "n4a_sha256": sha256(N4A),
            "capture_sha256": sha256(CAPTURE), "route_hashes": route_hashes,
            "conv_unit_test_sha256": sha256(CONV_UNIT_TEST),
            "conv_unit_result_sha256": sha256(CONV_UNIT_RESULT),
            "independent_audit_sha256": sha256(INDEPENDENT_AUDIT),
            "independent_audit_report_sha256": sha256(INDEPENDENT_AUDIT_REPORT),
            "d10a2r2_erratum_sha256": sha256(D10A2R2_ERRATUM),
        },
        "audit": evidence,
        "route_inventory": inventory,
        "canary_audit": canaries,
        "static_cuda_contract": {"expected_symbols": names, "nvrtc_compile_deferred": True},
        "source_mutation_contract": required_source_contract,
        "physical_actions": {"cuda_initialized": False, "nvrtc_compile": False, "host_registration": False, "large_device_allocation": False, "kernel_launch": False, "bank_scan": False},
        "component_opened": False,
        "endurance_opened": False,
        "error": error,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Compile/read-only preflight only; no component execution or endurance authorization.",
    }
    COMPILE_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    COMPILE_REPORT.write_text(
        "# PORT80B-D10B-R held-out 10,000-step CPU-only preflight report\n\n"
        f"Verdict: **{result['status']}**.\n\n"
        f"Statically expected CUDA symbols: **{len(names)}**. Device request for the separately "
        f"authorized component run: **{DEVICE_REQUEST / 2**30:.3f} GiB**, plus a "
        f"{VRAM_RESERVE / 2**20:.0f} MiB reserve. Registered host prefix in that run: "
        f"**{LAYERS * PREFIX * EXPERT_BYTES / 2**30:.3f} GiB**.\n\n"
        "This CPU-only phase initialized no CUDA context, invoked no NVRTC compiler, "
        "launched no kernel, registered no host range, allocated no large device buffer, "
        "and did not scan the bank. The exact 10,000-step executor is implemented but "
        "remains closed until a separate explicit GPU authorization.\n\n"
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


def heldout_endurance_cases(routes: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for epoch in range(8):
        for domain_index, domain in enumerate(DOMAINS):
            for token in range(ENDURANCE_SOURCE[0], ENDURANCE_SOURCE[1]):
                route = lift(routes[domain][token], domain_index, token, epoch)
                digest.update(route.tobytes())
                result.append({
                    "domain": domain,
                    "domain_index": domain_index,
                    "token": token,
                    "epoch": epoch,
                    "route": route,
                })
                if len(result) == ENDURANCE_STEPS:
                    if digest.hexdigest() != EXPECTED_ENDURANCE_ROUTE_SHA256:
                        raise RuntimeError("held-out endurance route SHA-256 mismatch")
                    return result
    raise RuntimeError(f"held-out route generator emitted only {len(result)} cases")


def attempt_rows(action: str) -> list[dict[str, Any]]:
    return [
        {
            "action": action,
            "layer": layer,
            "bytes": PREFIX * EXPERT_BYTES,
            "attempted": False,
            "success": False,
            "host_pointer": None,
            "device_alias": None,
            "error": None,
        }
        for layer in range(LAYERS)
    ]


def register_prefix(
    mapped: np.memmap,
    attempts: list[dict[str, Any]],
    hosts: list[int],
    aliases: list[int],
) -> None:
    size = PREFIX * EXPERT_BYTES
    for layer in range(LAYERS):
        row = attempts[layer]
        host = int(mapped.ctypes.data) + record_offset(layer, 0)
        row["host_pointer"] = host
        row["attempted"] = True
        try:
            cp.cuda.runtime.hostRegister(host, size, REGISTER_FLAGS)
            hosts.append(host)
            alias = int(cp.cuda.runtime.pointerGetAttributes(host).devicePointer)
            if not alias:
                raise RuntimeError(f"layer {layer}: null mapped alias")
            aliases.append(alias)
            row["device_alias"] = alias
            row["success"] = True
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise


def unregister_prefix(hosts: list[int], attempts: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    registered = {layer: host for layer, host in enumerate(hosts)}
    for layer in range(LAYERS):
        row = attempts[layer]
        host = registered.get(layer)
        if host is None:
            row["error"] = "not_registered"
            continue
        row["host_pointer"] = host
        row["attempted"] = True
        try:
            cp.cuda.runtime.hostUnregister(host)
            row["success"] = True
        except Exception as exc:
            message = f"layer {layer}: {type(exc).__name__}: {exc}"
            row["error"] = message
            failures.append(message)
    return failures


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


def compare_words(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    a = np.ascontiguousarray(left, dtype=np.uint16)
    b = np.ascontiguousarray(right, dtype=np.uint16)
    return {
        "elements": int(a.size),
        "different_words": int(np.count_nonzero(a != b)),
        "bitwise_equal": bool(np.array_equal(a, b)),
        "left_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "right_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
    }


def conv_step0_oracle() -> np.ndarray:
    layers = np.arange(GDN_LAYERS, dtype=np.int32)[:, None]
    channels = np.arange(8192, dtype=np.int32)[None, :]
    values = (((layers * 19 + channels) & 127) - 63).astype(np.float32) / np.float32(64)
    expected = np.zeros((GDN_LAYERS, 8192, 4), dtype=np.uint16)
    expected[:, :, 0] = bf16_words(values)
    assert int(np.count_nonzero(expected)) == CONV_EXPECTED_NONZERO_STEP0
    return expected.reshape(-1)


def conv_oracle_unit_audit() -> dict[str, Any]:
    values = conv_step0_oracle()
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    checks = {
        "dtype_uint16": values.dtype == np.uint16,
        "flattened_shape_exact": values.shape == (CONV_EXPECTED_WORDS,),
        "nonzero_exact_292608": int(np.count_nonzero(values)) == CONV_EXPECTED_NONZERO_STEP0,
        "sha256_exact": digest == CONV_EXPECTED_SHA256,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "nonzero_words": int(np.count_nonzero(values)),
        "sha256": digest,
    }


def shared_payload_comparison(mapped: np.memmap) -> dict[str, Any]:
    matrix_bytes = EXPERT_BYTES // 3
    header_bytes = 64
    reference_hashes: list[str] = []
    for projection in range(3):
        start = record_offset(0, 0) + projection * matrix_bytes + header_bytes
        reference_hashes.append(hashlib.sha256(memoryview(mapped)[start:start + matrix_bytes - header_bytes]).hexdigest())
    rows: list[dict[str, Any]] = []
    for layer in range(LAYERS):
        hashes: list[str] = []
        for projection in range(3):
            start = record_offset(layer, 512) + projection * matrix_bytes + header_bytes
            hashes.append(hashlib.sha256(memoryview(mapped)[start:start + matrix_bytes - header_bytes]).hexdigest())
        rows.append({"layer": layer, "projection_sha256": hashes, "matches_reference": hashes == reference_hashes})
    return {
        "header_bytes_excluded_per_projection": header_bytes,
        "payload_bytes_per_projection": matrix_bytes - header_bytes,
        "reference_projection_sha256": reference_hashes,
        "layers": rows,
        "all_48_match_reference": all(row["matches_reference"] for row in rows),
    }


def array_evidence(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "poison_count": int(np.count_nonzero(array == POISON)),
    }


def same_stream_full_verify(
    kernel: cp.RawKernel,
    destination: cp.ndarray,
    selected: list[tuple[int, int]],
    stream: cp.cuda.Stream,
) -> int:
    with stream:
        headers = cp.asarray(header_reference(selected))
        mismatches = cp.zeros(1, dtype=cp.uint64)
        kernel(
            (4096,), (256,),
            (destination, headers, np.uint64(TOKEN_BYTES), mismatches),
            stream=stream,
        )
    stream.synchronize()
    with stream:
        observed = cp.asnumpy(mismatches)
    stream.synchronize()
    return int(observed[0])


def component_phase() -> None:
    locked_compile()
    if COMPONENT_OUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D10A2-R2 component evidence")
    ram_before = int(psutil.virtual_memory().available)
    if ram_before < MIN_RAM_BEFORE:
        raise RuntimeError(f"hard stop: available RAM below D10A2-R2 frozen {START_RAM_GATE}-byte start gate")
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
    # D10A2's governing repair: make this explicit stream the CuPy current
    # stream as well as the raw-kernel/runtime stream for the entire phase.
    stream.use()
    started = time.perf_counter()
    hosts: list[int] = []
    aliases: list[int] = []
    registration_attempts = attempt_rows("host_register")
    unregister_attempts = attempt_rows("host_unregister")
    unregister_failures: list[str] = []
    error = None
    payload: dict[str, Any] = {}
    page_sampler = HardPageReadSampler()
    available_after_registration: int | None = None
    available_after_first_touch: int | None = None
    free_after_allocations: int | None = None
    shared_payload_check: dict[str, Any] = {}
    try:
        # All fixed shell/state/runtime buffers physically exist before host registration.
        with stream:
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
        stream.synchronize()
        free_after_allocations = int(cp.cuda.runtime.memGetInfo()[0])
        if free_after_allocations < VRAM_RESERVE:
            raise RuntimeError("hard stop: post-allocation VRAM reserve below 512 MiB")

        register_prefix(mapped, registration_attempts, hosts, aliases)
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
        shared_payload_check = shared_payload_comparison(mapped)

        rng = np.random.default_rng(10_130_826)
        x_host = rng.standard_normal(HIDDEN, dtype=np.float32)
        with stream:
            x = cp.asarray(x_host)
            gate = cp.full(TOP_K * INTER, POISON, dtype=cp.float32)
            up = cp.full_like(gate, POISON)
            down = cp.full(TOP_K * HIDDEN, POISON, dtype=cp.float32)
            routed_capture = cp.full(OUTPUT_BYTES // 4, POISON, dtype=cp.float32)
            oracle_capture = cp.full_like(routed_capture, POISON)
            routed_down = cp.full(LAYERS * TOP_K * HIDDEN, POISON, dtype=cp.float32)
            shared_gate = cp.full(LAYERS * INTER, POISON, dtype=cp.float32)
            shared_up = cp.full_like(shared_gate, POISON)
            shared_down = cp.full(LAYERS * HIDDEN, POISON, dtype=cp.float32)
            attention = cp.full(12 * 16 * 256, POISON, dtype=cp.float32)
            delta = cp.full(36 * 32, POISON, dtype=cp.float32)
            state = cp.full(LAYERS * HIDDEN, POISON, dtype=cp.float32)
            dense_checksum = cp.zeros(1, dtype=cp.uint64)
            canary_errors = cp.zeros(1, dtype=cp.uint64)
            actual_ids_device = cp.empty(LAYERS * TOP_K, dtype=cp.uint16)
            canary_words_device = cp.empty(LAYERS * TOP_K * 3, dtype=cp.uint16)
        stream.synchronize()

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
            with stream:
                table = cp.asarray(np.asarray(values, dtype=np.uint64))
            return table

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
            with stream:
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
            header_mismatches = same_stream_full_verify(kernels["verify_record_bytes"], staging, pairs(route), stream)
            kernels["differentiate_q5_from_header"]((32,), (256,), (staging,), stream=stream)
            route_device = assemble_oracle(route)
            with stream:
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
                "candidate_poison_count": int(np.count_nonzero(candidate_host == POISON)),
                "oracle_poison_count": int(np.count_nonzero(oracle_host == POISON)),
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
            mismatches = same_stream_full_verify(kernels["verify_record_bytes"], staging, pairs(first_route), stream)
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
        with stream:
            attention.fill(POISON)
            recurrent.fill(0)
            conv.fill(0)
            delta.fill(POISON)
            shared_gate.fill(POISON)
            shared_up.fill(POISON)
            shared_down.fill(POISON)
            dense_checksum.fill(0)
        kernels["next_attention_kv_qgate"](((12 * 16 * 256 + 255) // 256,), (256,), (kv, attention, np.int32(0), np.int32(0)), stream=stream)
        kernels["gated_deltanet_step"](((RECURRENT_BYTES // 4 + 255) // 256,), (256,), (recurrent, conv, delta, np.int32(0)), stream=stream)
        kernels["shared_q5_gate_up"]((48 * 1024 // 32,), (256,), (x, shared, shared_gate, shared_up), stream=stream)
        kernels["shared_swiglu"](((48 * INTER + 255) // 256,), (256,), (shared_gate, shared_up), stream=stream)
        kernels["shared_q5_down"]((48 * HIDDEN // 32,), (256,), (shared_gate, shared, shared_down), stream=stream)
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
        delta_host = cp.asnumpy(delta)
        shared_gate_host = cp.asnumpy(shared_gate)
        shared_up_host = cp.asnumpy(shared_up)
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
        conv_expected = conv_step0_oracle()
        conv_comparison = compare_words(conv_host, conv_expected)
        dense_expected = (0x5A * (DENSE_BYTES * (DENSE_BYTES + 1) // 2)) & MASK64
        components = {
            "attention_max_abs": float(np.max(np.abs(attention_host - attention_expected))),
            "attention_finite": bool(np.isfinite(attention_host).all()),
            "recurrent_sample_max_abs": float(np.max(np.abs(recurrent_host[sample_idx] - recurrent_expected))),
            "conv_nonzero": int(np.count_nonzero(conv_host)),
            "conv_expected_nonzero": CONV_EXPECTED_NONZERO_STEP0,
            "conv_full_bf16_comparison": conv_comparison,
            "shared_vs_resident": compare_bits(shared_host, resident),
            "shared_payload_comparison": shared_payload_check,
            "component_output_sentinels": {
                "attention": int(np.count_nonzero(attention_host == POISON)),
                "delta": int(np.count_nonzero(delta_host == POISON)),
                "shared_gate": int(np.count_nonzero(shared_gate_host == POISON)),
                "shared_up": int(np.count_nonzero(shared_up_host == POISON)),
                "shared_down": int(np.count_nonzero(shared_host == POISON)),
            },
            "dense_checksum_observed": int(dense_checksum.get()[0]),
            "dense_checksum_expected": int(dense_expected),
            "runtime_touch_sentinels": [int(runtime[0].get()), int(runtime[-1].get())],
        }

        with stream:
            recurrent.fill(0)
            conv.fill(0)
            kv.fill(0)
            dense_checksum.fill(0)
            state.fill(POISON)
        page_sampler.start()
        validation_wall: list[float] = []
        validation_event: list[float] = []
        validation_output_evidence: list[dict[str, Any]] = []
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
            stream.synchronize()
            with stream:
                output_arrays = {
                    "routed_capture": cp.asnumpy(routed_capture),
                    "routed_down": cp.asnumpy(routed_down),
                    "shared_down": cp.asnumpy(shared_down),
                    "attention": cp.asnumpy(attention),
                    "delta": cp.asnumpy(delta),
                    "kv_state": cp.asnumpy(kv),
                    "recurrent_state": cp.asnumpy(recurrent),
                    "conv_state": cp.asnumpy(conv),
                    "composed_state": cp.asnumpy(state),
                }
            stream.synchronize()
            validation_output_evidence.append({
                "step": step,
                "domain": case["domain"],
                "token": case["token"],
                "arrays": {name: array_evidence(values) for name, values in output_arrays.items()},
            })
            telemetry.append({
                "step": step, "available_ram": int(psutil.virtual_memory().available),
                "free_vram": int(cp.cuda.runtime.memGetInfo()[0]), "process": process_memory(),
            })
        page_sampler.stop()
        wall_stats = stats(validation_wall)
        event_stats = stats(validation_event)
        page_rates = [float(row.get("page_reads_per_sec", 0.0)) for row in page_sampler.samples]
        memory_loss = (telemetry[0]["available_ram"] - telemetry[-1]["available_ram"]) if telemetry else math.inf
        outputs_exactly_32_finite_digested = (
            len(validation_output_evidence) == 32
            and all(
                row["arrays"]
                and all(value["finite"] and value["poison_count"] == 0 and len(value["sha256"]) == 64 for value in row["arrays"].values())
                for row in validation_output_evidence
            )
        )
        component_gates = {
            "canary_exhaustive_injective_roundtrip_boundary": all(canary_audit()[key] for key in ("injective", "roundtrip_pass", "boundary_498_499_pass")),
            "all_correctness_headers_zero_mismatch": all(row["header_mismatches"] == 0 for row in correctness_rows),
            "all_canaries_raw_exact": all(row["raw_canary_exact"] and row["canary_mismatches"] == 0 for row in correctness_rows),
            "all_routed_q5_bitexact": all(row["comparison"]["bitwise_equal"] for row in correctness_rows),
            "all_routed_q5_outputs_finite": all(row["comparison"]["finite"] for row in correctness_rows),
            "all_routed_q5_outputs_fully_written": all(
                row["candidate_poison_count"] == 0 and row["oracle_poison_count"] == 0
                for row in correctness_rows
            ),
            "output_digest_uniqueness_ge_95pct": len(set(output_digests)) / len(output_digests) >= 0.95,
            "wrong_expert_header_and_numerical_detected": injected_rows["wrong_expert"]["header_mismatches"] > 0 and not injected_rows["wrong_expert"]["numerical_comparison"]["bitwise_equal"],
            "wrong_layer_header_and_numerical_detected": injected_rows["wrong_layer"]["header_mismatches"] > 0 and not injected_rows["wrong_layer"]["numerical_comparison"]["bitwise_equal"],
            "attention_reference_abs_rel_le_2e_5": components["attention_max_abs"] <= 2e-5,
            "gdn_reference_abs_rel_le_2e_5": components["recurrent_sample_max_abs"] <= 2e-5 and components["conv_nonzero"] > 0,
            "conv_step0_full_bf16_bitexact_and_292608_nonzero": (
                components["conv_nonzero"] == CONV_EXPECTED_NONZERO_STEP0
                and components["conv_full_bf16_comparison"]["bitwise_equal"]
            ),
            "shared_q5_bitexact": components["shared_vs_resident"]["bitwise_equal"],
            "shared_48_payloads_match_reference_excluding_headers": components["shared_payload_comparison"]["all_48_match_reference"],
            "component_outputs_no_poison_remaining": all(value == 0 for value in components["component_output_sentinels"].values()),
            "dense_and_runtime_touched": components["dense_checksum_observed"] == components["dense_checksum_expected"] and components["runtime_touch_sentinels"] == [0xA5, 0xA5],
            "validation_32_finite": len(validation_wall) == 32 and bool(np.isfinite(validation_wall).all()),
            "validation_full_outputs_finite_digested_no_poison": outputs_exactly_32_finite_digested,
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
            "validation_output_evidence": validation_output_evidence,
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
        unregister_failures = unregister_prefix(hosts, unregister_attempts)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    gates = payload.setdefault("gates", {})
    gates["registration_attempt_rows_48_all_success"] = (
        len(registration_attempts) == LAYERS
        and all(row["attempted"] and row["success"] and row["error"] is None for row in registration_attempts)
    )
    gates["clean_unregister_48_ranges"] = (
        len(hosts) == LAYERS
        and len(unregister_attempts) == LAYERS
        and all(row["attempted"] and row["success"] and row["error"] is None for row in unregister_attempts)
        and not unregister_failures
    )
    gates["no_cuda_or_runner_error"] = error is None
    overall = bool(gates) and all(gates.values())
    result = {
        "kind": "port80b_d10a2r2_gdn36_oracle_repair",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "component_composition_pass_endurance_closed_pending_new_authorization" if overall else "component_composition_negative_endurance_closed",
        "overall_pass": overall,
        "endurance_authorized_by_evidence": False,
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
        "registration_attempts": registration_attempts,
        "unregister_attempts": unregister_attempts,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Synthetic shape-informed physical shell stress/composition only; not an exact Qwen3-Next shell, real checkpoint, natural routing, quality or endurance result.",
    }
    COMPONENT_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# PORT80B-D10A2-R2 — GDN-36 oracle repair report\n\n"
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
        raise RuntimeError("exact D10B acknowledgement missing")
    if ENDURANCE_OUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D10B endurance evidence")

    component = json.loads(COMPONENT_OUT.read_text(encoding="utf-8"))
    if (
        sha256(COMPONENT_OUT) != EXPECTED_D10A2R2_LOCKS["result"]
        or component.get("overall_pass") is not True
        or not all(component.get("gates", {}).values())
        or component.get("error") is not None
        or component.get("unregister_failures") != []
    ):
        raise RuntimeError("immutable D10A2-R2 clean component pass missing")

    ram_before = int(psutil.virtual_memory().available)
    if ram_before < START_RAM_GATE:
        raise RuntimeError(f"hard stop: available RAM below frozen {START_RAM_GATE}-byte start gate")
    free_vram, _ = cp.cuda.runtime.memGetInfo()
    if int(free_vram) < DEVICE_REQUEST + VRAM_RESERVE:
        raise RuntimeError("hard stop: exact device request plus 512 MiB reserve unavailable")

    audit_result, routes, _ = audit()
    if not audit_result["pass"]:
        raise RuntimeError("immutable D10B input audit failed")
    cases = heldout_endurance_cases(routes)
    if len(cases) != ENDURANCE_STEPS:
        raise RuntimeError("held-out stream is not exactly 10,000 cases")

    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    module, names = build_module()
    kernels = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    stream.use()
    started = time.perf_counter()
    hosts: list[int] = []
    aliases: list[int] = []
    registration_attempts = attempt_rows("host_register")
    unregister_attempts = attempt_rows("host_unregister")
    unregister_failures: list[str] = []
    page_sampler = HardPageReadSampler()
    error: str | None = None
    payload: dict[str, Any] = {}
    available_after_registration: int | None = None
    available_after_first_touch: int | None = None
    free_after_allocations: int | None = None

    try:
        with stream:
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
        stream.synchronize()
        free_after_allocations = int(cp.cuda.runtime.memGetInfo()[0])
        if free_after_allocations < VRAM_RESERVE:
            raise RuntimeError("hard stop: post-allocation VRAM reserve below 512 MiB")

        register_prefix(mapped, registration_attempts, hosts, aliases)
        available_after_registration = int(psutil.virtual_memory().available)
        if len(hosts) != LAYERS or available_after_registration < MIN_RAM_AFTER_TOUCH:
            raise RuntimeError("hard stop after registration")

        for layer in range(LAYERS):
            cp.cuda.runtime.memcpyAsync(
                int(shared.data.ptr) + layer * EXPERT_BYTES,
                int(mapped.ctypes.data) + record_offset(layer, 512),
                EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
            )
        cp.cuda.runtime.memcpyAsync(
            reference10.data.ptr, int(mapped.ctypes.data), REFERENCE_BYTES,
            cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
        )
        stream.synchronize()

        rng = np.random.default_rng(10_130_826)
        x_host = rng.standard_normal(HIDDEN, dtype=np.float32)
        with stream:
            x = cp.asarray(x_host)
            gate = cp.full(TOP_K * INTER, POISON, dtype=cp.float32)
            up = cp.full_like(gate, POISON)
            down = cp.full(TOP_K * HIDDEN, POISON, dtype=cp.float32)
            routed_capture = cp.full(OUTPUT_BYTES // 4, POISON, dtype=cp.float32)
            oracle_capture = cp.full_like(routed_capture, POISON)
            routed_down = cp.full(LAYERS * TOP_K * HIDDEN, POISON, dtype=cp.float32)
            shared_gate = cp.full(LAYERS * INTER, POISON, dtype=cp.float32)
            shared_up = cp.full_like(shared_gate, POISON)
            shared_down = cp.full(LAYERS * HIDDEN, POISON, dtype=cp.float32)
            attention = cp.full(12 * 16 * 256, POISON, dtype=cp.float32)
            delta = cp.full(36 * 32, POISON, dtype=cp.float32)
            state = cp.full(LAYERS * HIDDEN, POISON, dtype=cp.float32)
            dense_checksum = cp.zeros(1, dtype=cp.uint64)
        stream.synchronize()

        def pairs(route: np.ndarray) -> list[tuple[int, int]]:
            return [(layer, int(expert)) for layer, row in enumerate(route) for expert in row]

        def pointer_table(route: np.ndarray) -> cp.ndarray:
            values: list[int] = []
            cold_slot = 0
            for layer, expert in pairs(route):
                if expert < PREFIX:
                    values.append(aliases[layer] + expert * EXPERT_BYTES)
                else:
                    if cold_slot >= COLD_SLOTS:
                        raise RuntimeError("cold escape slot overflow")
                    target = int(cold.data.ptr) + cold_slot * EXPERT_BYTES
                    cp.cuda.runtime.memcpyAsync(
                        target, int(mapped.ctypes.data) + record_offset(layer, expert),
                        EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr,
                    )
                    values.append(target)
                    cold_slot += 1
            with stream:
                table = cp.asarray(np.asarray(values, dtype=np.uint64))
            return table

        def stage(route: np.ndarray) -> None:
            table = pointer_table(route)
            kernels["host_to_smem_pipeline"](
                (STAGE_BLOCKS,), (STAGE_THREADS,),
                (table, staging, np.uint64(LAYERS * TOP_K * TILES_PER_RECORD)),
                shared_mem=4096, stream=stream,
            )
            stream.synchronize()

        def q5_capture() -> None:
            stride = TOP_K * (INTER + INTER + HIDDEN)
            for layer in range(LAYERS):
                kernels["staged_q5_gate_up"](
                    (TOP_K * 1024 // 32,), (256,),
                    (x, staging, np.int32(layer), gate, up), stream=stream,
                )
                kernels["canonical_swiglu_d7"](
                    ((TOP_K * INTER + 255) // 256,), (256,), (gate, up), stream=stream,
                )
                kernels["staged_q5_down"](
                    (TOP_K * HIDDEN // 32,), (256,),
                    (gate, staging, np.int32(layer), down), stream=stream,
                )
                base = int(routed_capture.data.ptr) + layer * stride * 4
                cp.cuda.runtime.memcpyAsync(base, gate.data.ptr, gate.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                cp.cuda.runtime.memcpyAsync(base + gate.nbytes, up.data.ptr, up.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                cp.cuda.runtime.memcpyAsync(base + gate.nbytes + up.nbytes, down.data.ptr, down.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr)
                cp.cuda.runtime.memcpyAsync(
                    int(routed_down.data.ptr) + layer * TOP_K * HIDDEN * 4,
                    down.data.ptr, down.nbytes, cp.cuda.runtime.memcpyDeviceToDevice, stream.ptr,
                )

        def execute(case: dict[str, Any], step: int) -> None:
            stage(case["route"])
            kernels["differentiate_q5_from_header"]((32,), (256,), (staging,), stream=stream)
            q5_capture()
            kernels["shared_q5_gate_up"]((LAYERS * 1024 // 32,), (256,), (x, shared, shared_gate, shared_up), stream=stream)
            kernels["shared_swiglu"](((LAYERS * INTER + 255) // 256,), (256,), (shared_gate, shared_up), stream=stream)
            kernels["shared_q5_down"]((LAYERS * HIDDEN // 32,), (256,), (shared_gate, shared, shared_down), stream=stream)
            kernels["next_attention_kv_qgate"](((12 * 16 * 256 + 255) // 256,), (256,), (kv, attention, np.int32(step & 4095), np.int32(step)), stream=stream)
            kernels["gated_deltanet_step"](((RECURRENT_BYTES // 4 + 255) // 256,), (256,), (recurrent, conv, delta, np.int32(step)), stream=stream)
            kernels["dense_shell_work"]((4096,), (256,), (dense, np.uint64(DENSE_BYTES), np.uint64(step), dense_checksum), stream=stream)
            kernels["compose_routed_shared"](((LAYERS * HIDDEN + 255) // 256,), (256,), (routed_down, shared_down, attention, delta, state), stream=stream)

        def device_evidence(array: cp.ndarray) -> dict[str, Any]:
            stream.synchronize()
            with stream:
                host = cp.asnumpy(array)
            stream.synchronize()
            return array_evidence(host)

        with stream:
            recurrent.fill(0)
            conv.fill(0)
            kv.fill(0)
            dense_checksum.fill(0)
            state.fill(POISON)
        for warmup in range(8):
            execute(cases[warmup], warmup)
        stream.synchronize()
        # Precompile and execute the per-step state guard outside measured timing.
        with stream:
            warm_finite = cp.isfinite(state).all()
            warm_written = (state != POISON).all()
        stream.synchronize()
        if not bool(cp.asnumpy(warm_finite)) or not bool(cp.asnumpy(warm_written)):
            raise RuntimeError("warm-up composed state is not finite and fully written")

        available_after_first_touch = int(psutil.virtual_memory().available)
        if available_after_first_touch < MIN_RAM_AFTER_TOUCH:
            raise RuntimeError("hard stop: less than 2 GiB after first touch")

        page_sampler.start()
        wall_ms: list[float] = []
        event_ms: list[float] = []
        telemetry: list[dict[str, Any]] = []
        state_checks: list[bool] = []
        checkpoint_evidence: list[dict[str, Any]] = []
        checkpoint_set = set(DIGEST_STEPS)

        for step, case in enumerate(cases):
            available_now = int(psutil.virtual_memory().available)
            if available_now < EMERGENCY_RAM:
                raise RuntimeError(f"emergency RAM stop before endurance step {step}")
            begin_event, end_event = cp.cuda.Event(), cp.cuda.Event()
            begin_wall = time.perf_counter()
            begin_event.record(stream)
            execute(case, step)
            end_event.record(stream)
            end_event.synchronize()
            wall_ms.append((time.perf_counter() - begin_wall) * 1000.0)
            event_ms.append(float(cp.cuda.get_elapsed_time(begin_event, end_event)))

            with stream:
                state_finite = cp.isfinite(state).all()
                state_written = (state != POISON).all()
            stream.synchronize()
            state_ok = bool(cp.asnumpy(state_finite)) and bool(cp.asnumpy(state_written))
            state_checks.append(state_ok)
            telemetry.append({
                "step": step,
                "available_ram": int(psutil.virtual_memory().available),
                "free_vram": int(cp.cuda.runtime.memGetInfo()[0]),
                "process": process_memory(),
                "state_finite_and_written": state_ok,
            })

            if step in checkpoint_set:
                arrays = {
                    "routed_capture": routed_capture,
                    "routed_down": routed_down,
                    "shared_down": shared_down,
                    "attention": attention,
                    "delta": delta,
                    "kv_state": kv,
                    "recurrent_state": recurrent,
                    "conv_state": conv,
                    "composed_state": state,
                }
                checkpoint_evidence.append({
                    "step": step,
                    "domain": case["domain"],
                    "token": case["token"],
                    "epoch": case["epoch"],
                    "arrays": {name: device_evidence(array) for name, array in arrays.items()},
                })

        page_sampler.stop()
        wall_stats = stats(wall_ms)
        event_stats = stats(event_ms)
        first_1000 = stats(wall_ms[:1000])
        last_1000 = stats(wall_ms[-1000:])
        drift_ratio = last_1000["p95"] / first_1000["p95"]
        page_rates = [float(row.get("page_reads_per_sec", 0.0)) for row in page_sampler.samples]
        memory_loss = telemetry[0]["available_ram"] - telemetry[-1]["available_ram"]
        array_rows = [value for row in checkpoint_evidence for value in row["arrays"].values()]
        composed_digests = [row["arrays"]["composed_state"]["sha256"] for row in checkpoint_evidence]

        per_call_base = (0x5A * (DENSE_BYTES * (DENSE_BYTES + 1) // 2)) & MASK64
        per_step = (0x5A * DENSE_BYTES) & MASK64
        invocation_count = 8 + ENDURANCE_STEPS
        summed_steps = sum(range(8)) + (ENDURANCE_STEPS * (ENDURANCE_STEPS - 1) // 2)
        dense_expected = (invocation_count * per_call_base + per_step * summed_steps) & MASK64
        dense_observed = int(dense_checksum.get()[0])
        runtime_sentinels = [int(runtime[0].get()), int(runtime[-1].get())]

        gates = {
            "exactly_10000_heldout_cases": len(cases) == ENDURANCE_STEPS,
            "latency_vectors_exactly_10000_finite_positive": (
                len(wall_ms) == ENDURANCE_STEPS
                and len(event_ms) == ENDURANCE_STEPS
                and bool(np.isfinite(wall_ms).all())
                and bool(np.isfinite(event_ms).all())
                and min(wall_ms) > 0.0 and min(event_ms) > 0.0
            ),
            "wall_p95_le_150ms": wall_stats["p95"] <= 150.0,
            "wall_p99_le_200ms": wall_stats["p99"] <= 200.0,
            "last_first_1000_p95_ratio_le_1_20": drift_ratio <= 1.20,
            "all_10000_composed_states_finite_and_written": len(state_checks) == ENDURANCE_STEPS and all(state_checks),
            "exact_101_checkpoint_schedule": [row["step"] for row in checkpoint_evidence] == list(DIGEST_STEPS),
            "checkpoint_arrays_exactly_909_finite_digested_no_poison": (
                len(checkpoint_evidence) == len(DIGEST_STEPS)
                and len(array_rows) == len(DIGEST_STEPS) * 9
                and all(value["finite"] and value["poison_count"] == 0 and len(value["sha256"]) == 64 for value in array_rows)
            ),
            "all_composed_checkpoint_digests_unique": len(set(composed_digests)) == len(DIGEST_STEPS),
            "telemetry_exactly_10000": len(telemetry) == ENDURANCE_STEPS,
            "post_warmup_page_reads_no_sample_gt_2048": bool(page_rates) and max(page_rates) <= 2048.0,
            "endurance_memory_loss_le_1gib": memory_loss <= 2**30,
            "ram_after_first_touch_ge_2gib": available_after_first_touch >= MIN_RAM_AFTER_TOUCH,
            "vram_reserve_ge_512mib": min(row["free_vram"] for row in telemetry) >= VRAM_RESERVE,
            "dense_checksum_exact_and_runtime_touched": dense_observed == dense_expected and runtime_sentinels == [0xA5, 0xA5],
            "registration_48_ranges": len(hosts) == LAYERS,
            "no_cuda_or_runner_error": True,
        }
        payload = {
            "route_contract": {
                "label": "p4d_shaped_synthetic_proxy",
                "partition": list(ENDURANCE_SOURCE),
                "route_sha256": EXPECTED_ENDURANCE_ROUTE_SHA256,
                "steps": len(cases),
                "warmups": 8,
            },
            "latency": {
                "wall_ms": wall_ms,
                "cuda_event_ms": event_ms,
                "wall_stats": wall_stats,
                "cuda_event_stats": event_stats,
                "first_1000_wall_stats": first_1000,
                "last_1000_wall_stats": last_1000,
                "last_first_p95_ratio": drift_ratio,
            },
            "state_checks": state_checks,
            "checkpoint_evidence": checkpoint_evidence,
            "telemetry": telemetry,
            "page_reads": {"samples": page_sampler.samples, "error": page_sampler.error},
            "dense_runtime": {
                "dense_checksum_observed": dense_observed,
                "dense_checksum_expected": dense_expected,
                "runtime_sentinels": runtime_sentinels,
            },
            "gates": gates,
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
        unregister_failures = unregister_prefix(hosts, unregister_attempts)
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    gates = payload.setdefault("gates", {})
    gates["registration_attempt_rows_48_all_success"] = (
        len(registration_attempts) == LAYERS
        and all(row["attempted"] and row["success"] and row["error"] is None for row in registration_attempts)
    )
    gates["clean_unregister_48_ranges"] = (
        len(hosts) == LAYERS
        and len(unregister_attempts) == LAYERS
        and all(row["attempted"] and row["success"] and row["error"] is None for row in unregister_attempts)
        and not unregister_failures
    )
    gates["no_cuda_or_runner_error"] = error is None
    overall = bool(gates) and all(gates.values())
    result = {
        "kind": "port80b_d10br_heldout_10000_endurance_revision",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "heldout_10000_endurance_pass" if overall else "heldout_10000_endurance_negative",
        "overall_pass": overall,
        "inputs": {
            "preregistration_sha256": sha256(PREREG),
            "runner_sha256": sha256(Path(__file__)),
            "preflight_sha256": sha256(COMPILE_OUT),
            "d10a2r2_component_sha256": sha256(COMPONENT_OUT),
            "bank_sha256_from_manifest": EXPECTED_BANK_SHA256,
            "route_label": "p4d_shaped_synthetic_proxy",
        },
        "physical": {
            "device_request_bytes": DEVICE_REQUEST,
            "registered_bytes": LAYERS * PREFIX * EXPERT_BYTES,
            "available_ram_before": ram_before,
            "available_ram_after_registration": available_after_registration,
            "available_ram_after_first_touch": available_after_first_touch,
            "available_ram_after_cleanup": int(psutil.virtual_memory().available),
            "free_vram_after_allocations": free_after_allocations,
        },
        **payload,
        "error": error,
        "registration_attempts": registration_attempts,
        "unregister_attempts": unregister_attempts,
        "unregister_failures": unregister_failures,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Synthetic shape-informed 10,000-step endurance on held-out P4D-shaped proxy routes and uniform Q5 payloads only; not checkpoint, natural routing, quality, production throughput or breakthrough evidence.",
    }
    ENDURANCE_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "# PORT80B-D10B-R held-out 10,000-step endurance report\n\n"
        f"Verdict: **{result['status']}**.\n\n"
        f"Wall p50/p95/p99: {payload.get('latency', {}).get('wall_stats', {}).get('p50', '—')} / "
        f"{payload.get('latency', {}).get('wall_stats', {}).get('p95', '—')} / "
        f"{payload.get('latency', {}).get('wall_stats', {}).get('p99', '—')} ms.\n\n"
        f"Claim boundary: {result['claim_boundary']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "overall_pass": overall,
        "error": error,
        "unregister_failures": unregister_failures,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "endurance"), required=True)
    parser.add_argument("--acknowledge-endurance")
    args = parser.parse_args()
    if args.phase == "preflight":
        compile_phase()
    else:
        endurance_phase(args.acknowledge_endurance)


if __name__ == "__main__":
    main()
