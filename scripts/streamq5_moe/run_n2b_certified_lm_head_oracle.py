from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.streamq5_moe.run_p12_32g_4k_endurance as p12
import scripts.streamq5_moe.run_p13c_evt_pm_32g_endurance as p13c

R = ROOT / "reports/streamq5_moe"
PREREG = R / "N2B_CERTIFIED_LM_HEAD_ORACLE_PREREGISTRATION.md"
MANIFEST = R / "p6a_exact_runtime_bank_result.json"
DATA = ROOT / "reports/runs/streamq5_moe/p0c_fresh_input_ids.safetensors"
DATA_LOCK = R / "p0c_input_lock.json"
OUT_VALIDATION = R / "n2b_certified_lm_head_oracle_validation.json"
OUT_TEST = R / "n2b_certified_lm_head_oracle_test.json"
DOMAINS = ("general", "code", "math", "multilingual", "instruction")
ROWS, COLS, GROUPS, BITS, CLUSTERS = 151936, 2048, 16, 10, 1024
SEED, CHUNK = 120820, 2048


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_head_record():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = next(row for row in manifest["records"] if row["name"] == "head")
    path = ROOT / record["artifact"]
    if sha256(path) != record["artifact_sha256"]: raise ValueError("head artifact hash mismatch")
    return path, record


def dequant_chunk(codes, scale_bits, start, end, device):
    code = torch.from_numpy(np.array(codes[start:end], copy=True)).to(device=device, dtype=torch.float32)
    raw = np.array(scale_bits[start:end], copy=True)
    scale = torch.from_numpy(((raw.astype(np.uint32) << 16).view(np.float32))).to(device)
    return code * scale.repeat_interleave(128, dim=1)


@torch.no_grad()
def build_clusters():
    path, record = physical_head_record()
    codes = np.memmap(path, mode="r", dtype=np.int8, offset=0, shape=(ROWS, COLS))
    scales = np.memmap(path, mode="r", dtype="<u2", offset=record["code_bytes"], shape=(ROWS, GROUPS))
    rng = np.random.default_rng(SEED)
    raw = rng.standard_normal((COLS, BITS), dtype=np.float32)
    directions, _ = np.linalg.qr(raw, mode="reduced"); directions = directions.astype(np.float32)
    device = torch.device("cuda")
    direction = torch.from_numpy(directions).to(device)
    powers = (1 << torch.arange(BITS, device=device, dtype=torch.int64))
    sums = torch.zeros((CLUSTERS, COLS), device=device, dtype=torch.float32)
    counts = torch.zeros(CLUSTERS, device=device, dtype=torch.float32)
    assignments = np.empty(ROWS, dtype=np.uint16)
    for start in range(0, ROWS, CHUNK):
        end = min(ROWS, start + CHUNK); weight = dequant_chunk(codes, scales, start, end, device)
        cluster = ((weight @ direction) >= 0).to(torch.int64).mul(powers).sum(dim=1)
        sums.index_add_(0, cluster, weight); counts.index_add_(0, cluster, torch.ones(end - start, device=device))
        assignments[start:end] = cluster.cpu().numpy().astype(np.uint16)
    centers = sums / counts.clamp_min(1).unsqueeze(1)
    radius = torch.zeros(CLUSTERS, device=device); normmax = torch.zeros(CLUSTERS, device=device)
    for start in range(0, ROWS, CHUNK):
        end = min(ROWS, start + CHUNK); weight = dequant_chunk(codes, scales, start, end, device)
        cluster = torch.from_numpy(assignments[start:end].astype(np.int64)).to(device)
        distance = torch.linalg.vector_norm(weight - centers[cluster], dim=1)
        norms = torch.linalg.vector_norm(weight, dim=1)
        radius.scatter_reduce_(0, cluster, distance, reduce="amax", include_self=True)
        normmax.scatter_reduce_(0, cluster, norms, reduce="amax", include_self=True)
    result = {"assignments": assignments, "counts": counts.cpu().numpy().astype(np.int32),
              "centers": centers.cpu().numpy(), "radius": radius.cpu().numpy() * np.float32(1.001),
              "normmax": normmax.cpu().numpy(), "directions_sha256": hashlib.sha256(directions.tobytes()).hexdigest()}
    del direction, sums, centers, radius, normmax
    gc.collect(); torch.cuda.empty_cache(); cp.get_default_memory_pool().free_all_blocks()
    return result


def certify(hidden: np.ndarray, logits: np.ndarray, clusters) -> dict:
    centers = clusters["centers"].astype(np.float64)
    h = hidden.astype(np.float64); hnorm = float(np.linalg.norm(h))
    center_dot = centers @ h
    gamma = (4096 * 2**-24) / (1.0 - 4096 * 2**-24)
    upper = center_dot + hnorm * clusters["radius"].astype(np.float64)
    upper += gamma * hnorm * (clusters["normmax"].astype(np.float64) + np.linalg.norm(centers, axis=1))
    upper += np.abs(upper) * (2**-8) + 1e-6
    cluster_max = np.full(CLUSTERS, -np.inf, dtype=np.float32)
    np.maximum.at(cluster_max, clusters["assignments"], logits)
    order = np.argsort(-upper, kind="stable")
    best = -np.inf; rows = 0; stop = CLUSTERS
    for index, cluster in enumerate(order):
        best = max(best, float(cluster_max[cluster])); rows += int(clusters["counts"][cluster])
        remaining = -np.inf if index + 1 == CLUSTERS else float(upper[order[index + 1]])
        if best >= remaining:
            stop = index + 1; break
    exact = int(np.argmax(logits))
    certified = best >= (-np.inf if stop == CLUSTERS else float(upper[order[stop]]))
    return {"rows": rows, "skip_fraction": 1.0 - rows / ROWS, "clusters": stop,
            "certified": bool(certified), "exact_argmax": exact, "best_seen_equals_exact": bool(best == float(logits[exact]))}


def evaluate(split: str, clusters) -> list[dict]:
    Runtime = p13c.load_runtime_class_evt_pm(); lock = json.loads(p12.P6_LOCK.read_text(encoding="utf-8"))
    runtime = Runtime(lock); data = load_file(DATA); rows = []
    for domain in DOMAINS:
        runtime.activate_domain(domain)
        ids = data[f"{split}_{domain}"].numpy().astype(np.int64)
        for context_index, context in enumerate(ids):
            runtime.reset_context()
            for position in range(context.size - 1):
                decoded = runtime.decode(int(context[position]), position, int(context[position + 1]))
                hidden = cp.asnumpy(runtime.normed); logits = cp.asnumpy(runtime.logits)
                row = certify(hidden, logits, clusters)
                row.update({"domain": domain, "context": context_index, "position": position,
                            "runtime_prediction": decoded["prediction"],
                            "runtime_matches_exact": decoded["prediction"] == row["exact_argmax"]})
                rows.append(row)
            print(json.dumps({"split": split, "domain": domain, "context": context_index, "tokens": len(rows)}), flush=True)
    return rows


def summarize(rows):
    skip = np.asarray([row["skip_fraction"] for row in rows], dtype=np.float64)
    gates = {"tokens_1270": len(rows) == 1270, "all_certified": all(row["certified"] for row in rows),
             "all_best_seen_exact": all(row["best_seen_equals_exact"] for row in rows),
             "runtime_argmax_exact": all(row["runtime_matches_exact"] for row in rows),
             "median_skip_ge_60pct": float(np.median(skip)) >= 0.60}
    return {"tokens": len(rows), "skip_fraction": {"mean": float(skip.mean()), "p5": float(np.percentile(skip, 5)),
            "p50": float(np.percentile(skip, 50)), "p95": float(np.percentile(skip, 95)), "min": float(skip.min()), "max": float(skip.max())},
            "rows_evaluated": {"mean": float(np.mean([r["rows"] for r in rows])), "p50": float(np.median([r["rows"] for r in rows]))},
            "gates": gates, "pass": all(gates.values())}


def main():
    if OUT_VALIDATION.exists() or OUT_TEST.exists(): raise FileExistsError("N2B output exists")
    data_lock = json.loads(DATA_LOCK.read_text(encoding="utf-8"))
    if sha256(DATA) != data_lock["artifact_sha256"]: raise ValueError("data hash mismatch")
    started = time.perf_counter(); clusters = build_clusters()
    common = {"inputs": {"preregistration_sha256": sha256(PREREG), "manifest_sha256": sha256(MANIFEST), "data_sha256": sha256(DATA)},
              "clustering": {"bits": BITS, "clusters": CLUSTERS, "nonempty": int(np.count_nonzero(clusters["counts"])),
                             "directions_sha256": clusters["directions_sha256"], "seed": SEED}}
    validation_rows = evaluate("validation", clusters); validation = summarize(validation_rows)
    payload = {"kind": "streamq5_moe_n2b_certified_lm_head_oracle", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "split": "validation", **common, "summary": validation, "runtime_seconds": time.perf_counter() - started,
               "claim_boundary": "Exact-search row-count oracle; all logits are still physically computed and no timing gain is claimed."}
    OUT_VALIDATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": validation, "output": str(OUT_VALIDATION)}, indent=2), flush=True)
    if validation["pass"]:
        test_rows = evaluate("test", clusters); test = summarize(test_rows)
        payload = {"kind": "streamq5_moe_n2b_certified_lm_head_oracle", "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "split": "test", **common, "validation_sha256": sha256(OUT_VALIDATION), "summary": test,
                   "runtime_seconds_total": time.perf_counter() - started,
                   "claim_boundary": "Exact-search row-count oracle; all logits are still physically computed and no timing gain is claimed."}
        OUT_TEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"test": test, "output": str(OUT_TEST)}, indent=2), flush=True)


if __name__ == "__main__": main()
