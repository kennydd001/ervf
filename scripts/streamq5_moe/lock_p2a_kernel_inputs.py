from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/streamq5_moe/P2A_Q5_KERNEL_PREREGISTRATION.md"
P1D_VERIFY = ROOT / "reports/streamq5_moe/p1d_physical_bank_verification.json"
ROUTE_DIR = ROOT / "reports/runs/streamq5_moe/p1c_routes"
ARTIFACT = ROOT / "reports/runs/streamq5_moe/p2a_kernel_vectors.npz"
OUTPUT = ROOT / "reports/streamq5_moe/p2a_kernel_input_lock.json"
LAYERS, TOKEN, SEED = (0, 24, 47), 768, 20260812
MATRICES = (("gate", 2048), ("up", 2048), ("down", 768))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUTPUT.exists() or ARTIFACT.exists():
        raise FileExistsError("refusing to overwrite P2A input lock")
    verified = json.loads(P1D_VERIFY.read_text(encoding="utf-8"))
    if verified.get("status") != "p1d_physical_bank_verification_pass":
        raise RuntimeError("independent P1D pass required")
    rng = np.random.default_rng(SEED)
    experts = {}
    route_hashes = {}
    vectors = {}
    for layer in LAYERS:
        route_path = ROUTE_DIR / f"layer_{layer:02d}.safetensors"
        route_hashes[str(layer)] = sha256(route_path)
        with safe_open(route_path, framework="numpy") as handle:
            ids = handle.get_tensor("general_router_ids")[TOKEN].astype(np.int64)
        if ids.shape != (8,) or len(set(ids.tolist())) != 8 or ids.min() < 0 or ids.max() >= 128:
            raise RuntimeError(f"invalid locked experts for layer {layer}")
        experts[str(layer)] = [int(value) for value in ids]
        for expert in ids:
            for name, columns in MATRICES:
                key = f"layer_{layer:02d}_expert_{int(expert):03d}_{name}"
                vectors[key] = rng.standard_normal(columns, dtype=np.float32)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ARTIFACT, **vectors)
    payload = {
        "kind": "streamq5_moe_p2a_kernel_input_lock",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "p1d_verification_sha256": sha256(P1D_VERIFY),
        "route_artifact_sha256": route_hashes,
        "layers": list(LAYERS), "token": TOKEN, "domain": "general",
        "experts": experts, "seed": SEED,
        "input_vectors": str(ARTIFACT.relative_to(ROOT)).replace("\\", "/"),
        "input_vectors_sha256": sha256(ARTIFACT),
        "cases": len(vectors),
        "kernel": {"threads": 256, "warmup": 100, "iterations": 500},
        "tolerances": {"max_abs": 0.02, "relative_l2": 0.0001},
        "gates": {"weights_per_second": 27200000000.0, "full_token_p95_compute_ms_max": 66.615},
        "outputs_opened": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "locked", "experts": experts, "cases": len(vectors), "artifact_sha256": payload["input_vectors_sha256"]}, indent=2))
