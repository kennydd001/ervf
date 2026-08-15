from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from moe_lab.reporting import ROOT
from scripts.streamq5_moe.run_p6a_end_to_end_decode import CUDA_SOURCE
from scripts.streamq5_moe.run_p7b_ervf_kernel import ERVF_SOURCE
from scripts.streamq5_moe.run_port80b_d2_registered_scatter import EXPECTED_BANK_SHA256, REGISTER_FLAGS, TOKEN_BYTES, record_offset, routes, stats, unregister_ranges
from scripts.streamq5_moe.run_port80b_d5_cp_async_host_smem import SOURCE as STAGE_SOURCE
from scripts.streamq5_moe.run_port80b_p0_physical_host_bank import BANK, BANK_BYTES, EXPERT_BYTES, LAYERS, MANIFEST

R = ROOT / "reports/streamq5_moe"
PREREG = R / "PORT80B_D7_STAGED_EXACT_Q5_PLANE_PREREGISTRATION.md"
OUTPUT = R / "port80b_d7_staged_exact_q5_plane.json"
REPORT = R / "PORT80B_D7_STAGED_EXACT_Q5_PLANE_REPORT_2026-08-12.md"
EXPERTS_REGISTERED = 307
ACTIVE = 10
HIDDEN = 2048
INTER = 512
SEED = 120_827
WARMUPS = 5
VALIDATION_ROUNDS = 24
TEST_ROUNDS = 120
DENSE_SHELL_P95 = 28.077_227

COMPUTE_SOURCE = r'''
extern "C" __global__ void staged_q5_gate_up(
    const float* x, const unsigned char* bank, int layer,
    float* gate, float* up) {
  int group = (int)threadIdx.x >> 3;
  int lane = (int)threadIdx.x & 7;
  int global_row = (int)blockIdx.x * 32 + group;
  int expert = global_row >> 10;
  int local = global_row - expert * 1024;
  int projection = local >= 512;
  int row = local - projection * 512;
  long long base = ((long long)layer * 10LL + expert) * 2027520LL + (long long)projection * 675840LL;
  const unsigned char* packed = bank + base + 64;
  const unsigned short* scales = (const unsigned short*)(bank + base + 64 + 655360);
  float value = q5_ervf_row<8>(x, packed, scales, row, 2048, lane);
  if (lane == 0) {
    if (projection) up[expert * 512 + row] = round_bf16(value);
    else gate[expert * 512 + row] = round_bf16(value);
  }
}

extern "C" __global__ void staged_q5_down(
    const float* activation, const unsigned char* bank, int layer,
    float* down) {
  int group = (int)threadIdx.x >> 3;
  int lane = (int)threadIdx.x & 7;
  int global_row = (int)blockIdx.x * 32 + group;
  int expert = global_row >> 11;
  int row = global_row - expert * 2048;
  long long base = ((long long)layer * 10LL + expert) * 2027520LL + 2LL * 675840LL;
  const unsigned char* packed = bank + base + 64;
  const unsigned short* scales = (const unsigned short*)(bank + base + 64 + 655360);
  float value = q5_ervf_row<8>(activation + expert * 512, packed, scales, row, 512, lane);
  if (lane == 0) down[expert * 2048 + row] = round_bf16(value);
}

extern "C" __global__ void canonical_swiglu_d7(float* gate, const float* up) {
  int index = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 10 * 512) {
    float g = round_bf16(gate[index]);
    float u = round_bf16(up[index]);
    float silu = round_bf16(g / (1.0f + expf(-g)));
    gate[index] = round_bf16(silu * u);
  }
}
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register(mapped: np.memmap) -> tuple[list[int], list[int]]:
    hosts, aliases = [], []
    size = EXPERTS_REGISTERED * EXPERT_BYTES
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


def compare(observed: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    left, right = observed.view(np.uint32), expected.view(np.uint32)
    return {"elements": int(expected.size), "different_bits": int(np.count_nonzero(left != right)), "bitwise_equal": bool(np.array_equal(left, right)), "max_abs": float(np.max(np.abs(observed.astype(np.float64) - expected.astype(np.float64)), initial=0.0)), "finite": bool(np.isfinite(observed).all())}


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite D7 result")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not BANK.is_file() or BANK.stat().st_size != BANK_BYTES or manifest.get("bank_sha256") != EXPECTED_BANK_SHA256:
        raise RuntimeError("immutable bank/manifest contract failed")
    mapped = np.memmap(BANK, dtype=np.uint8, mode="r", shape=(BANK_BYTES,))
    cuda_include = ROOT / ".venv/Lib/site-packages/nvidia/cu13/include"
    names = ("host_to_smem_pipeline", "staged_q5_gate_up", "staged_q5_down", "canonical_swiglu_d7")
    module = cp.RawModule(code=CUDA_SOURCE + ERVF_SOURCE + STAGE_SOURCE + COMPUTE_SOURCE, options=("--std=c++14", f"--include-path={cuda_include}"), name_expressions=names)
    kernels = {name: module.get_function(name) for name in names}
    stream = cp.cuda.Stream(non_blocking=True)
    started = time.perf_counter()
    hosts: list[int] = []
    payload: dict[str, object] = {}
    error = None
    unregister_failures: list[str] = []
    try:
        hosts, aliases = register(mapped)
        tokens = [109_999] + list(range(110_000, 110_000 + VALIDATION_ROUNDS)) + list(range(111_000, 111_000 + TEST_ROUNDS))
        pointer_host = np.stack([np.asarray([aliases[layer] + expert * EXPERT_BYTES for layer, expert in routes(token, EXPERTS_REGISTERED)], dtype=np.uint64) for token in tokens])
        pointer_table = cp.asarray(pointer_host)
        rows = {token: index for index, token in enumerate(tokens)}
        staging = cp.empty(TOKEN_BYTES, dtype=cp.uint8)
        reference = cp.empty(ACTIVE * EXPERT_BYTES, dtype=cp.uint8)
        # All payloads are invariant; stage the first ten records once for the oracle.
        cp.cuda.runtime.memcpyAsync(reference.data.ptr, int(mapped.ctypes.data), ACTIVE * EXPERT_BYTES, cp.cuda.runtime.memcpyHostToDevice, stream.ptr)
        rng = np.random.default_rng(SEED)
        x_host = rng.standard_normal(HIDDEN, dtype=np.float32)
        x = cp.asarray(x_host)
        gate = cp.empty(ACTIVE * INTER, dtype=cp.float32)
        up = cp.empty_like(gate)
        down = cp.empty(ACTIVE * HIDDEN, dtype=cp.float32)

        def compute(bank: cp.ndarray, layer: int, capture: np.ndarray | None, output_layer: int) -> None:
            kernels["staged_q5_gate_up"]((ACTIVE * 1024 // 32,), (256,), (x, bank, np.int32(layer), gate, up), stream=stream)
            kernels["canonical_swiglu_d7"](((ACTIVE * INTER + 255) // 256,), (256,), (gate, up), stream=stream)
            kernels["staged_q5_down"]((ACTIVE * HIDDEN // 32,), (256,), (gate, bank, np.int32(layer), down), stream=stream)
            if capture is not None:
                stream.synchronize()
                capture[output_layer] = np.concatenate((cp.asnumpy(gate), cp.asnumpy(up), cp.asnumpy(down)))

        def oracle_capture() -> np.ndarray:
            output = np.empty((LAYERS, ACTIVE * (INTER + INTER + HIDDEN)), dtype=np.float32)
            for layer in range(LAYERS):
                compute(reference, 0, output, layer)
            return output

        def candidate(token: int, capture: bool = False) -> np.ndarray | None:
            output = np.empty((LAYERS, ACTIVE * (INTER + INTER + HIDDEN)), dtype=np.float32) if capture else None
            kernels["host_to_smem_pipeline"]((1024,), (256,), (pointer_table[rows[token]], staging, np.uint64(480 * 495)), shared_mem=4096, stream=stream)
            for layer in range(LAYERS):
                compute(staging, layer, output, layer)
            return output

        expected = oracle_capture()
        observed = candidate(109_999, True)
        assert observed is not None
        correctness = compare(observed, expected)
        digests = {"resident": hashlib.sha256(expected.tobytes()).hexdigest(), "staged": hashlib.sha256(observed.tobytes()).hexdigest()}
        exact = correctness["bitwise_equal"] and len(set(digests.values())) == 1

        raw_validation: list[float] = []
        raw_test: list[float] = []
        if exact:
            for warmup in range(WARMUPS):
                candidate(110_000 + warmup, False)
            stream.synchronize()
            for token in range(110_000, 110_000 + VALIDATION_ROUNDS):
                begin, end = cp.cuda.Event(), cp.cuda.Event(); begin.record(stream); candidate(token, False); end.record(stream); end.synchronize(); raw_validation.append(float(cp.cuda.get_elapsed_time(begin, end)))
        validation_stats = stats(raw_validation) if raw_validation else None
        validation_open = bool(validation_stats and float(validation_stats["p50"]) <= 65.0)
        if validation_open:
            for token in range(111_000, 111_000 + TEST_ROUNDS):
                begin, end = cp.cuda.Event(), cp.cuda.Event(); begin.record(stream); candidate(token, False); end.record(stream); end.synchronize(); raw_test.append(float(cp.cuda.get_elapsed_time(begin, end)))
        test_stats = stats(raw_test) if raw_test else None
        effective = TOKEN_BYTES / (float(test_stats["p95"]) / 1000.0) / 1e9 if test_stats else None
        projected_total = float(test_stats["p95"]) + DENSE_SHELL_P95 if test_stats else None
        gates = {
            "all_outputs_bit_exact_and_digest_equal": bool(exact),
            "test_120_finite": len(raw_test) == TEST_ROUNDS and bool(np.isfinite(raw_test).all()),
            "test_p95_le_65ms": bool(test_stats and float(test_stats["p95"]) <= 65.0),
            "effective_remote_payload_gb_s_ge_15": bool(effective is not None and effective >= 15.0),
            "projected_total_p95_le_100ms": bool(projected_total is not None and projected_total <= 100.0),
            "strong_test_p95_le_55ms": bool(test_stats and float(test_stats["p95"]) <= 55.0),
            "strong_projected_total_p95_le_90ms": bool(projected_total is not None and projected_total <= 90.0),
            "registration_48_ranges": len(hosts) == LAYERS,
            "no_cuda_or_runner_error": True,
        }
        primary = all(gates[name] for name in ("all_outputs_bit_exact_and_digest_equal", "test_120_finite", "test_p95_le_65ms", "effective_remote_payload_gb_s_ge_15", "projected_total_p95_le_100ms", "registration_48_ranges", "no_cuda_or_runner_error"))
        strong = primary and gates["strong_test_p95_le_55ms"] and gates["strong_projected_total_p95_le_90ms"]
        payload = {"input_sha256": hashlib.sha256(x_host.tobytes()).hexdigest(), "pointer_table_sha256": hashlib.sha256(pointer_host.tobytes()).hexdigest(), "correctness": correctness, "output_digests": digests, "validation": {"raw_ms": raw_validation, "stats": validation_stats, "open": validation_open}, "test": {"raw_ms": raw_test, "stats": test_stats}, "effective_remote_payload_gb_s_at_p95": effective, "dense_projection": {"frozen_dense_shell_p95_ms": DENSE_SHELL_P95, "projected_total_p95_ms": projected_total}, "gates": gates, "primary_pass": primary, "strong_pass": strong}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            stream.synchronize()
        except Exception:
            pass
        unregister_failures = unregister_ranges(hosts)

    primary = bool(payload.get("primary_pass")) and error is None and not unregister_failures
    strong = bool(payload.get("strong_pass")) and primary
    result = {"kind": "port80b_d7_staged_exact_q5_plane", "completed_utc": datetime.now(timezone.utc).isoformat(), "status": "staged_exact_q5_strong_pass" if strong else ("staged_exact_q5_primary_pass" if primary else "staged_exact_q5_negative"), "primary_pass": primary, "strong_pass": strong, "full_bank_pass": False, "inputs": {"preregistration_sha256": sha256(PREREG), "evaluator_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(MANIFEST), "bank_sha256_from_manifest": manifest["bank_sha256"], "seed": SEED}, "physical": {"remote_payload_bytes": TOKEN_BYTES, "hbm_work_buffer_bytes": TOKEN_BYTES, "registered_experts_per_layer": EXPERTS_REGISTERED, "registered_gib": LAYERS * EXPERTS_REGISTERED * EXPERT_BYTES / 2**30, "layers": LAYERS, "active_experts": ACTIVE}, "protocol": {"stage_blocks": 1024, "stage_threads": 256, "q5_width": 8, "warmups": WARMUPS, "validation_rounds": VALIDATION_ROUNDS, "test_rounds": TEST_ROUNDS}, **payload, "error": error, "unregister_failures": unregister_failures, "wall_seconds": time.perf_counter() - started, "claim_boundary": "Exact synthetic staged Q5 active plane on a 60%-registered bank; no full bank, real checkpoint, natural routing, quality, physical dense shell, end-to-end tok/s or endurance claim."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    test_stats = payload.get("test", {}).get("stats") if payload else None
    REPORT.write_text("# PORT80B-D7 — cp.async staged exact Q5 plane report\n\n" f"Verdict: **{result['status']}**. Bit differences: {payload.get('correctness', {}).get('different_bits', '—')}. Test p50/p95: {test_stats.get('p50') if test_stats else '—'} / {test_stats.get('p95') if test_stats else '—'} ms. Effective rate: {payload.get('effective_remote_payload_gb_s_at_p95', '—')} GB/s. Projected total: {payload.get('dense_projection', {}).get('projected_total_p95_ms', '—')} ms.\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "primary_pass": primary, "strong_pass": strong, "correctness": payload.get("correctness"), "digests": payload.get("output_digests"), "validation": payload.get("validation", {}).get("stats"), "test": test_stats, "effective": payload.get("effective_remote_payload_gb_s_at_p95"), "projected_total": payload.get("dense_projection", {}).get("projected_total_p95_ms"), "gates": payload.get("gates"), "error": error, "unregister_failures": unregister_failures}, indent=2))


if __name__ == "__main__":
    main()
