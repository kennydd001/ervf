from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from common import REPO

RESULTS = REPO / "pro_research" / "results" / "s100_phase14r"
MODEL = Path(__import__("os").environ.get(
    "LS_MODEL_DIR", REPO / "models" / "nemotron_3_5_lightning"
))
PROMPTS = REPO / "pro_research" / "S100_PHASE3_PROMPTS.json"
TRACE = REPO / "pro_research" / "results" / "S100_PHASE3_V18_TRACE_FULL.npz"
TRACE_META = TRACE.with_suffix(".json")
CAPACITY = (
    REPO / "pro_research" / "results" / "s100_phase9"
    / "S100_PHASE9_CAPACITY_PROFILES.json"
)

E2 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS

def current_map() -> dict[int, int]:
    data = json.loads(CAPACITY.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in data["profiles"]["current"].items()}

def bf16_round(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float32)
    u = a.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    b = ((u + bias) >> np.uint32(16)).astype(np.uint16)
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)

def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    r = np.asarray(reference, np.float64)
    c = np.asarray(candidate, np.float64)
    d = c - r
    rn = np.linalg.norm(r, axis=1)
    dn = np.linalg.norm(d, axis=1)
    cn = np.linalg.norm(c, axis=1)
    cosine = np.sum(r * c, axis=1) / np.maximum(rn * cn, 1e-30)
    rel = dn / np.maximum(rn, 1e-30)
    return {
        "nrmse": float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-30)),
        "mean_cosine": float(np.mean(cosine)),
        "min_cosine": float(np.min(cosine)),
        "p95_relative_row_error": float(np.percentile(rel, 95)),
        "max_relative_row_error": float(np.max(rel)),
        "row_argmax_agreement": float(
            np.mean(np.argmax(r, axis=1) == np.argmax(c, axis=1))
        ),
        "finite": bool(np.isfinite(c).all()),
    }

def normalize_eager_moe(rt):
    """runtime.step expects a tuple; graph-oriented wrappers may return None."""
    import types
    original = rt._moe

    def safe(self, layer, out):
        result = original(layer, out)
        return (None, None) if result is None else result

    rt._moe = types.MethodType(safe, rt)
    return original
