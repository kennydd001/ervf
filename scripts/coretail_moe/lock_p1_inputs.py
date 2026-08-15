from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

from moe_lab.reporting import ROOT


PREREG = ROOT / "reports/coretail_moe/P1_FUSED_KERNEL_PREREGISTRATION.md"
P0 = ROOT / "reports/coretail_moe/p0_full_bank_format_verification.json"
OUT = ROOT / "reports/coretail_moe/p1_fused_kernel_input_lock.json"
VECTOR_FILE = ROOT / "reports/runs/coretail_moe/p1_input_vectors.npz"
LAYERS = (0, 24, 47)
SEED = 0xC07E7A11


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if OUT.exists() or VECTOR_FILE.exists():
        raise FileExistsError("refusing to overwrite P1 input lock")
    p0 = json.loads(P0.read_text(encoding="utf-8"))
    if p0.get("status") != "p0_pass" or not p0.get("p1_authorized"):
        raise ValueError("P0 pass is required")
    routes = {}
    route_hashes = {}
    for layer in LAYERS:
        path = ROOT / f"reports/runs/qwen_gptq_bank/p0_supplement_routes/layer_{layer:02d}.safetensors"
        with safe_open(path, framework="pt", device="cpu") as handle:
            routes[str(layer)] = handle.get_tensor("general_router_ids")[0].tolist()
        route_hashes[str(layer)] = sha256(path)
    rng = np.random.Generator(np.random.PCG64(SEED))
    vectors = {}
    vector_manifest = {}
    for layer in LAYERS:
        for expert in routes[str(layer)]:
            for matrix, cols in (("gate", 2048), ("up", 2048), ("down", 768)):
                key = f"layer_{layer:02d}_expert_{expert:03d}_{matrix}"
                value = rng.standard_normal(cols, dtype=np.float32)
                vectors[key] = value
                vector_manifest[key] = {
                    "shape": [cols],
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
    VECTOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(VECTOR_FILE, **vectors)
    payload = {
        "kind": "coretail_moe_p1_fused_kernel_input_lock",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": sha256(PREREG),
        "p0_verification_sha256": sha256(P0),
        "layers": list(LAYERS),
        "route_source": "general_router_ids token 0",
        "experts": routes,
        "route_artifact_sha256": route_hashes,
        "input_seed": SEED,
        "input_vectors": str(VECTOR_FILE.relative_to(ROOT)).replace("\\", "/"),
        "input_vectors_sha256": sha256(VECTOR_FILE),
        "vector_manifest": vector_manifest,
        "kernel": {"threads": 256, "warmup": 100, "iterations": 500},
        "tolerances": {"max_abs": 0.005, "relative_l2": 0.0001},
        "tail_trace": {"domains": ["general", "instruction", "code", "math", "multilingual"], "tokens_per_domain": 1024, "layers": 48},
        "gates": {"weights_per_second": 27_200_000_000, "tail_decode_h2d_p95_ms": 33.3},
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "locked", "layers": payload["layers"], "experts": routes,
        "vectors": len(vectors), "vector_file_sha256": payload["input_vectors_sha256"],
    }, indent=2))
