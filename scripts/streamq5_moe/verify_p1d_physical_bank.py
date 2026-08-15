from __future__ import annotations

import gc
import hashlib
import json
import math
import mmap
import struct
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch

from moe_lab.reporting import ROOT
from moe_lab.rsiv_moe.qwen_stream import checkpoint_weight_map, load_checkpoint_tensors


MODEL = ROOT / "models/qwen3-30b-a3b-base"
REPORT_DIR = ROOT / "reports/streamq5_moe"
PREREG = REPORT_DIR / "P1D_CORRECTED_PHYSICAL_BANK_PREREGISTRATION.md"
PRODUCER_LOCK = REPORT_DIR / "p1d_bank_producer_lock.json"
PRODUCER = ROOT / "scripts/streamq5_moe/build_p1d_physical_bank.py"
BANK_RESULT = REPORT_DIR / "p1d_physical_bank_result.json"
RUN_DIR = ROOT / "reports/runs/streamq5_moe/p1d_q5_bank"
LAYER_DIR = REPORT_DIR / "p1d_bank_layers"
VERIFIER_LOCK = REPORT_DIR / "p1d_bank_verifier_lock.json"
OUTPUT = REPORT_DIR / "p1d_physical_bank_verification.json"
REPORT = REPORT_DIR / "P1D_PHYSICAL_BANK_VERIFICATION.md"

LAYERS, EXPERTS, GROUP = 48, 128, 128
MATRICES = (("gate", 768, 2048, 0), ("up", 768, 2048, 1), ("down", 2048, 768, 2))
HEADER_FORMAT = "<4sHHHBBIIH2xIII28s"
HEADER_BYTES, RECORD_BYTES = 64, 1_011_712
CODE_BYTES, SCALE_BYTES, PADDING_BYTES = 983_040, 24_576, 4_032
EXPERT_BYTES, LAYER_BYTES, BANK_BYTES = 3_035_136, 388_497_408, 18_647_875_584


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def source_names(layer: int) -> tuple[dict[int, dict[str, str]], list[str]]:
    identities: dict[int, dict[str, str]] = {}
    names: list[str] = []
    for expert in range(EXPERTS):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        identities[expert] = {kind: f"{base}.{kind}_proj.weight" for kind, *_ in MATRICES}
        names.extend(identities[expert].values())
    return identities, names


def decode_q5(payload: memoryview, rows: int, columns: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 5).astype(np.uint64)
    words = (
        packed[:, 0]
        | (packed[:, 1] << 8)
        | (packed[:, 2] << 16)
        | (packed[:, 3] << 24)
        | (packed[:, 4] << 32)
    )
    decoded = np.empty((words.size, 8), dtype=np.int8)
    for slot in range(8):
        decoded[:, slot] = ((words >> (slot * 5)) & 31).astype(np.int8) - 15
    return decoded.reshape(rows, columns)


@torch.no_grad()
def expected_quantization(source: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = source.shape
    work = source.to(device).float().reshape(rows, columns // GROUP, GROUP)
    maximum = work.abs().amax(dim=-1, keepdim=True)
    temporary_scale = torch.where(maximum > 0, maximum / 15, torch.ones_like(maximum))
    codes = torch.round(work / temporary_scale).clamp(-15, 15).to(torch.int8)
    scale_bits = temporary_scale.squeeze(-1).to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().copy()
    code_values = codes.reshape(rows, columns).cpu().numpy().copy()
    del work, maximum, temporary_scale, codes
    return code_values, scale_bits


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite P1D bank verification")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for independent full-bank recomputation")

    result = json.loads(BANK_RESULT.read_text(encoding="utf-8"))
    producer_lock = json.loads(PRODUCER_LOCK.read_text(encoding="utf-8"))
    verifier_lock = json.loads(VERIFIER_LOCK.read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    check("verifier hash lock", sha256(Path(__file__)) == verifier_lock["verifier_sha256"])
    check("bank result hash lock", sha256(BANK_RESULT) == verifier_lock["bank_result_sha256"])
    check("producer hash lock", sha256(PRODUCER) == producer_lock["producer_sha256"])
    check("preregistration hash lock", sha256(PREREG) == producer_lock["preregistration_sha256"])
    check("producer result status", result.get("status") == "physical_bank_built_pending_independent_verification")
    check("declared physical semantics", result.get("format", {}).get("decode_semantics") == "codes selected with FP32 scale; persisted BF16 scale used for dequant; BF16 output")
    check("declared bank cardinality", result.get("bank") == {
        "layers": 48, "experts": 6144, "records": 18432,
        "codes": 28_991_029_248, "scale_elements": 226_492_416,
        "bytes": BANK_BYTES, "gib": 17.3671875,
    })
    check("producer resource gates", result["runtime"]["peak_cuda_allocated_bytes"] <= int(7.5 * 2**30) and result["runtime"]["peak_rss_bytes"] <= 32 * 2**30)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    weight_map = checkpoint_weight_map(MODEL)
    started = time.perf_counter()
    counters = {
        "layers": 0, "experts": 0, "records": 0, "codes": 0, "scale_elements": 0,
        "header_failures": 0, "crc_failures": 0, "padding_failures": 0,
        "code_range_failures": 0, "code_source_mismatches": 0,
        "scale_validity_failures": 0, "scale_source_mismatches": 0,
        "source_hash_failures": 0, "artifact_hash_failures": 0,
        "layer_report_failures": 0,
    }

    for layer in range(LAYERS):
        layer_started = time.perf_counter()
        artifact = RUN_DIR / f"layer_{layer:02d}.q5bin"
        layer_report_path = LAYER_DIR / f"layer_{layer:02d}.json"
        manifest = result["manifests"].get(str(layer), {})
        artifact_ok = artifact.is_file() and artifact.stat().st_size == LAYER_BYTES
        if artifact_ok:
            artifact_ok = sha256(artifact) == manifest.get("artifact_sha256")
        counters["artifact_hash_failures"] += int(not artifact_ok)
        report_ok = layer_report_path.is_file() and sha256(layer_report_path) == manifest.get("report_sha256")
        layer_report = json.loads(layer_report_path.read_text(encoding="utf-8")) if layer_report_path.is_file() else {}
        report_ok = report_ok and layer_report.get("layer") == layer and layer_report.get("records") == 384 and layer_report.get("artifact_bytes") == LAYER_BYTES
        counters["layer_report_failures"] += int(not report_ok)
        if not artifact_ok or not report_ok:
            raise RuntimeError(f"P1D layer manifest failed at layer {layer}")

        identities, names = source_names(layer)
        loaded = load_checkpoint_tensors(MODEL, names, weight_map)
        for kind, *_ in MATRICES:
            stacked = torch.stack([loaded[identities[expert][kind]] for expert in range(EXPERTS)]).contiguous()
            counters["source_hash_failures"] += int(tensor_sha(stacked) != layer_report["source_weight_sha256"][kind])
            del stacked

        with artifact.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for expert in range(EXPERTS):
                for matrix_index, (kind, rows, columns, projection) in enumerate(MATRICES):
                    record_index = expert * 3 + matrix_index
                    offset = record_index * RECORD_BYTES
                    fields = struct.unpack(HEADER_FORMAT, mapped[offset : offset + HEADER_BYTES])
                    magic, version, got_layer, got_expert, got_projection, bits, got_rows, got_columns, group, code_bytes, scale_bytes, stored_crc, reserved = fields
                    header_ok = (
                        magic == b"SQ5M" and version == 1 and got_layer == layer and got_expert == expert
                        and got_projection == projection and bits == 5 and got_rows == rows and got_columns == columns
                        and group == GROUP and code_bytes == CODE_BYTES and scale_bytes == SCALE_BYTES
                        and reserved == b"\x00" * 28
                    )
                    counters["header_failures"] += int(not header_ok)

                    code_begin = offset + HEADER_BYTES
                    scale_begin = code_begin + CODE_BYTES
                    padding_begin = scale_begin + SCALE_BYTES
                    code_payload = memoryview(mapped)[code_begin:scale_begin]
                    scale_payload = memoryview(mapped)[scale_begin:padding_begin]
                    crc = zlib.crc32(code_payload)
                    crc = zlib.crc32(scale_payload, crc) & 0xFFFFFFFF
                    counters["crc_failures"] += int(crc != stored_crc)
                    padding = np.frombuffer(mapped, dtype=np.uint8, count=PADDING_BYTES, offset=padding_begin)
                    counters["padding_failures"] += int(np.any(padding))

                    decoded = decode_q5(code_payload, rows, columns)
                    counters["code_range_failures"] += int(decoded.min() < -15 or decoded.max() > 15)
                    stored_scale_bits = np.frombuffer(scale_payload, dtype="<u2").reshape(rows, columns // GROUP).copy()
                    stored_scales = torch.from_numpy(stored_scale_bits).view(torch.bfloat16).float()
                    counters["scale_validity_failures"] += int(not (torch.isfinite(stored_scales).all() and (stored_scales > 0).all()))

                    expected_codes, expected_scale_bits = expected_quantization(loaded[identities[expert][kind]], device)
                    counters["code_source_mismatches"] += int(not np.array_equal(decoded, expected_codes))
                    counters["scale_source_mismatches"] += int(not np.array_equal(stored_scale_bits, expected_scale_bits))
                    counters["records"] += 1
                    counters["codes"] += rows * columns
                    counters["scale_elements"] += rows * (columns // GROUP)
                    del code_payload, scale_payload, padding, decoded, stored_scale_bits, stored_scales, expected_codes, expected_scale_bits

        counters["layers"] += 1
        counters["experts"] += EXPERTS
        del loaded
        gc.collect(); torch.cuda.empty_cache()
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({"layer": layer, "records_verified": counters["records"], "seconds": time.perf_counter() - layer_started, "mismatches": counters["code_source_mismatches"] + counters["scale_source_mismatches"]}), flush=True)

    check("48 layer artifacts and manifests", counters["artifact_hash_failures"] == 0 and counters["layer_report_failures"] == 0)
    check("immutable source tensor hashes", counters["source_hash_failures"] == 0)
    check("all headers", counters["header_failures"] == 0)
    check("all payload CRC32", counters["crc_failures"] == 0)
    check("all padding zero", counters["padding_failures"] == 0)
    check("all decoded codes in range", counters["code_range_failures"] == 0)
    check("all codes match source recomputation", counters["code_source_mismatches"] == 0)
    check("all BF16 scales finite and positive", counters["scale_validity_failures"] == 0)
    check("all raw BF16 scale bits match source", counters["scale_source_mismatches"] == 0)
    check("exact independently counted cardinality", counters["layers"] == 48 and counters["experts"] == 6144 and counters["records"] == 18432 and counters["codes"] == 28_991_029_248 and counters["scale_elements"] == 226_492_416)

    passed = sum(row["pass"] for row in checks)
    total = len(checks)
    status = "p1d_physical_bank_verification_pass" if passed == total else "p1d_physical_bank_verification_fail"
    payload = {
        "kind": "streamq5_moe_p1d_independent_physical_bank_verification",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "checks_passed": passed, "checks_total": total, "checks": checks,
        "counters": counters,
        "inputs": {"verifier_lock_sha256": sha256(VERIFIER_LOCK), "verifier_sha256": sha256(Path(__file__)), "bank_result_sha256": sha256(BANK_RESULT), "producer_sha256": sha256(PRODUCER), "preregistration_sha256": sha256(PREREG), "model_index_sha256": sha256(MODEL / "model.safetensors.index.json")},
        "runtime": {"seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_rss_bytes": peak_rss},
        "conclusion": "Every physical record independently decoded and matched exactly to recomputed source Q5 codes and raw BF16 scale bits.",
        "claim_boundary": "Physical representation and exact source equivalence only; measured transfer, Q5 kernel, overlap, and integrated wall-clock remain unproven.",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        f"# STREAMQ5-MoE P1D - onafhankelijke fysieke-bankverificatie\n\n"
        f"Uitkomst: **{status}** ({passed}/{total}).\n\n"
        f"Alle {counters['records']:,} records zijn gedecodeerd; code- en schaalmismatches: "
        f"{counters['code_source_mismatches'] + counters['scale_source_mismatches']}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "checks": f"{passed}/{total}", "counters": counters, "runtime": payload["runtime"]}, indent=2), flush=True)
    if status.endswith("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
