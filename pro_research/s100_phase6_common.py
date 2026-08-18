"""Frozen candidate definitions and shared phase-6 utilities."""
from __future__ import annotations

BACKENDS = ("legacy", "ballot_fused", "direct", "direct_opt")
K1 = {40: 5}
K2 = {40: 5, 34: 5}
K3 = {40: 5, 34: 5, 49: 5}
K4 = {40: 5, 34: 5, 49: 5, 47: 5}

CANDIDATES = {
    "thr_0003": {"layer_k": {}, "alpha": 0.0003},
    "thr_0010": {"layer_k": {}, "alpha": 0.0010},
    "thr_0015": {"layer_k": {}, "alpha": 0.0015},
    "thr_0020": {"layer_k": {}, "alpha": 0.0020},
    "thr_0025": {"layer_k": {}, "alpha": 0.0025},
    "k1": {"layer_k": K1, "alpha": 0.0},
    "k2": {"layer_k": K2, "alpha": 0.0},
    "k3": {"layer_k": K3, "alpha": 0.0},
    "k4": {"layer_k": K4, "alpha": 0.0},
    "thr0010_k1": {"layer_k": K1, "alpha": 0.0010},
    "thr0010_k2": {"layer_k": K2, "alpha": 0.0010},
    "thr0015_k1": {"layer_k": K1, "alpha": 0.0015},
}


def public_spec(spec):
    return {
        "layer_k": {str(k): int(v) for k, v in sorted(spec.get("layer_k", {}).items())},
        "alpha": float(spec.get("alpha", 0.0)),
    }
