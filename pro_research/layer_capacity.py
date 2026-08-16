"""Shared per-layer cache capacity reallocation, extracted from
diag_per_layer_capacity.py (2026-08-16, -14.3% misses on real hit-rate
diagnostic, hit rate 85.61% -> 87.66%) for reuse in a causal A/B and V6
integration.

Mirrors enable_cache's own allocation code (runtime.py:324-378) verbatim for
mode="up_only" -- no edit to runtime.py. _moe_dev already reads c["cap"]
per-layer dynamically (via alloc_device_cache(self.n_experts, c["cap"], ...)
inside _moe_dev/batched_moe_dev), so heterogeneous per-layer capacity was
already structurally supported; only enable_cache's convenience API is
uniform.

Capacity changes do not alter routing or expert contributions (top-k
selection and each expert's math are capacity-independent) -- only which
experts are device-resident vs PCIe-fetched. A1's own adoption test already
established that changing cache capacity (72 vs 56) preserves bit-exact
output once D1's deterministic accumulation order is in place (which is now
always the case) -- this is precedent, not a new claim.
"""

from __future__ import annotations

from collections import OrderedDict

UP_CODE = 2_494_464
UP_SCALE = 311_808

# From diag_per_layer_capacity.json (2026-08-16): highest/lowest miss-rate
# layers at uniform capacity=72 over a 256-token real rollout.
REDUCE_LAYERS = [38, 10, 40, 20, 43, 13]
REDUCE_DELTA = -20
BOOST_LAYERS = [1, 3, 51, 6]
BOOST_DELTA = 30
BASELINE_CAP = 72


def reallocate_layer(rt, layer: int, new_cap: int) -> None:
    cp = rt.cp
    entry = {
        "codes": cp.zeros(new_cap * UP_CODE, dtype=cp.uint8),
        "scales": cp.zeros(new_cap * UP_SCALE, dtype=cp.uint8),
        "map": OrderedDict(),
        "cap": new_cap,
    }
    entry["slot_codes"] = [entry["codes"][k * UP_CODE:(k + 1) * UP_CODE] for k in range(new_cap)]
    entry["slot_scales"] = [entry["scales"][k * UP_SCALE:(k + 1) * UP_SCALE] for k in range(new_cap)]
    rt.cache[layer] = entry


def apply_nonuniform_capacity(rt) -> None:
    """Budget-neutral reallocation: -20 on the 6 lowest-miss layers, +30 on
    the 4 highest-miss layers (total slots unchanged: 1656). Call AFTER
    rt.enable_cache(BASELINE_CAP) and BEFORE rt.setup_graph() (capacity
    changes invalidate the device-LRU tables and any captured graph, exactly
    as enable_cache() itself already documents)."""
    for layer in REDUCE_LAYERS:
        reallocate_layer(rt, layer, BASELINE_CAP + REDUCE_DELTA)
    for layer in BOOST_LAYERS:
        reallocate_layer(rt, layer, BASELINE_CAP + BOOST_DELTA)
    rt._dev_cache = {}
    rt._graph = None
