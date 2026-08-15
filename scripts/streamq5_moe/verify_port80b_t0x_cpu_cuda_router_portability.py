from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "streamq5_moe"
RUN_DIR = ROOT / "reports" / "runs" / "streamq5_moe" / "port80b_t0x_router_portability"
LOCK = REPORTS / "port80b_t0x_router_portability_lock.json"
RESULT = RUN_DIR / "t0x_cpu_cuda_router_result.json"
RAW = RUN_DIR / "t0x_cpu_cuda_router_raw.safetensors"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def u16eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint16), b.view(torch.uint16))


def preflight():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    checks = {
        "runner_hash": sha256(ROOT / lock["runner"]) == lock["source_sha256"]["runner"],
        "verifier_hash": sha256(Path(__file__)) == lock["source_sha256"]["verifier"],
        "prereg_hash": sha256(ROOT / lock["prereg"]) == lock["source_sha256"]["prereg"],
        "outputs_closed": not RESULT.exists() and not RAW.exists(),
    }
    print(json.dumps({"kind": "t0x_preflight", "pass": all(checks.values()), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


def verify():
    j = json.loads(RESULT.read_text(encoding="utf-8"))
    with safe_open(str(RAW), framework="pt", device="cpu") as f:
        t = {k: f.get_tensor(k).contiguous() for k in f.keys()}
    rows = []
    for i in range(16):
        ci, gi = t["cpu_ids"][i], t["gpu1_ids"][i]
        rows.append({
            "ordered_ids_equal": torch.equal(ci, gi),
            "id_set_equal": set(map(int, ci.tolist())) == set(map(int, gi.tolist())),
            "weights_equal": u16eq(t["cpu_weights"][i], t["gpu1_weights"][i]),
        })
    repeat = {
        "logits": u16eq(t["gpu1_logits"], t["gpu2_logits"]),
        "probs": torch.equal(t["gpu1_probs"], t["gpu2_probs"]),
        "ids": torch.equal(t["gpu1_ids"], t["gpu2_ids"]),
        "weights": u16eq(t["gpu1_weights"], t["gpu2_weights"]),
    }
    finite = all(torch.isfinite(t[k].float()).all().item() for k in ("cpu_logits", "cpu_probs", "cpu_weights", "gpu1_logits", "gpu1_probs", "gpu1_weights", "gpu2_logits", "gpu2_probs", "gpu2_weights"))
    exact = all(x["ordered_ids_equal"] and x["weights_equal"] for x in rows) and all(repeat.values()) and finite
    expected_verdict = "exact_cross_backend_pass" if exact else "cross_backend_negative"
    checks = {
        "raw_sha": sha256(RAW) == j["raw_sha256"],
        "row_counts": sum(x["ordered_ids_equal"] for x in rows) == j["ordered_id_equal_rows"] and sum(x["id_set_equal"] for x in rows) == j["id_set_equal_rows"] and sum(x["weights_equal"] for x in rows) == j["weight_bit_equal_rows"],
        "cuda_repeat": repeat == j["cuda_repeat"],
        "finite": finite == j["all_finite"],
        "verdict": j["verdict"] == expected_verdict and j["overall_pass"] == exact,
        "diagnostic_boundary": j["bank_built"] is False and j["host_registered"] is False,
    }
    out = {"kind": "t0x_independent_verification", "verification_pass": all(checks.values()), "scientific_pass": exact, "verdict": expected_verdict, "checks": checks, "ordered_id_equal_rows": sum(x["ordered_ids_equal"] for x in rows), "id_set_equal_rows": sum(x["id_set_equal"] for x in rows), "weight_bit_equal_rows": sum(x["weights_equal"] for x in rows), "claim_boundary": "Independent replay of stored CPU/CUDA tensors; no CUDA rerun."}
    out_path = REPORTS / "port80b_t0x_cpu_cuda_router_portability_independent_verification.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--phase", choices=("preflight", "verify"), required=True); a = p.parse_args()
    raise SystemExit(preflight() if a.phase == "preflight" else verify())
