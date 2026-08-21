from __future__ import annotations

import types

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES, UP_CODE, UP_SCALE
from scale_resident_kernels import PLANE_BYTES
from s100_phase23_group_kernels import ROUTES
from s100_phase31_common import make_attention_head_direct_candidate


CUDA_SOURCE = r"""
#define ROUTES 24

extern "C" __global__ void p44_store_routes(
    const int* __restrict__ ids,
    const int* __restrict__ pos,
    int* __restrict__ table,
    const int layer_slot,
    const int layer_count,
    const int block_count)
{
    const int r = (int)threadIdx.x;
    const int p = pos[0];
    const int b = p >> 2;
    if (r >= ROUTES || (p & 3) != 0 || b < 0 || b >= block_count) return;
    table[((size_t)b * layer_count + layer_slot) * ROUTES + r] = ids[r];
}

extern "C" __global__ void p44_load_routes(
    const int* __restrict__ table,
    const int* __restrict__ pos,
    int* __restrict__ ids,
    const int layer_slot,
    const int layer_count,
    const int block_count)
{
    const int r = (int)threadIdx.x;
    const int p = pos[0];
    const int b = p >> 2;
    if (r >= ROUTES) return;
    if ((p & 3) != 0 || b < 0 || b >= block_count) {
        ids[r] = -1;
        return;
    }
    ids[r] = table[((size_t)b * layer_count + layer_slot) * ROUTES + r];
}

extern "C" __global__ void p44_compare_routes(
    const int* __restrict__ actual,
    const int* __restrict__ oracle,
    int* __restrict__ mismatches)
{
    const int r = (int)threadIdx.x;
    if (r < ROUTES && actual[r] != oracle[r]) atomicAdd(mismatches, 1);
}
"""


class Phase44Kernels:
    def __init__(self):
        import cupy as cp

        module = cp.RawModule(
            code=CUDA_SOURCE,
            options=("-std=c++17",),
            name_expressions=(
                "p44_store_routes",
                "p44_load_routes",
                "p44_compare_routes",
            ),
        )
        self.store_k = module.get_function("p44_store_routes")
        self.load_k = module.get_function("p44_load_routes")
        self.compare_k = module.get_function("p44_compare_routes")

    def store(self, ids, pos, table, layer_slot: int, layer_count: int):
        self.store_k(
            (1,),
            (32,),
            (
                ids,
                pos,
                table,
                np.int32(layer_slot),
                np.int32(layer_count),
                np.int32(table.shape[0]),
            ),
        )

    def load(self, table, pos, ids, layer_slot: int, layer_count: int):
        self.load_k(
            (1,),
            (32,),
            (
                table,
                pos,
                ids,
                np.int32(layer_slot),
                np.int32(layer_count),
                np.int32(table.shape[0]),
            ),
        )

    def compare(self, actual, oracle, mismatches):
        self.compare_k((1,), (32,), (actual, oracle, mismatches))


class RouteCaptureProxy:
    """Capture authoritative Phase31 routes into a position-indexed table."""

    def __init__(self, inner, graph, table, kernels: Phase44Kernels):
        self.inner = inner
        self.graph = graph
        self.table = table
        self.kernels = kernels
        self.layers = tuple(int(x) for x in graph.rt.moe_layers)
        self.layer_slot = {layer: slot for slot, layer in enumerate(self.layers)}
        self.current_layer = None
        self._group_owner = inner.base.k
        self._original_group = self._group_owner.group
        self._group_owner.group = self._capture_group

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, layer, normed, out, collect_stats=False):
        self.current_layer = int(layer)
        return self.inner(layer, normed, out, collect_stats)

    def _capture_group(self, *args, **kwargs):
        result = self._original_group(*args, **kwargs)
        if self.current_layer is None:
            raise RuntimeError("Phase44 route capture has no active layer")
        ids = args[0]
        self.kernels.store(
            ids,
            self.graph.pos_dev,
            self.table,
            self.layer_slot[self.current_layer],
            len(self.layers),
        )
        return result

    def restore(self):
        self._group_owner.group = self._original_group


class PerfectBlockPrefetchOracle:
    """Start all exact per-layer expert transfers at H4 graph entry."""

    def __init__(
        self,
        inner,
        graph,
        route_table,
        kernels: Phase44Kernels,
        *,
        mode: str = "layer_now",
    ):
        import cupy as cp

        self.cp = cp
        self.inner = inner
        self.graph = graph
        self.rt = graph.rt
        self.base = inner.base
        if mode not in ("layer_now", "moe_l1", "moe_l2", "block_all"):
            raise ValueError(mode)
        self.mode = mode
        self.route_table = cp.asarray(route_table, dtype=cp.int32)
        self.kernels = kernels
        self.layers = tuple(int(x) for x in self.rt.moe_layers)
        self.layer_slot = {layer: slot for slot, layer in enumerate(self.layers)}
        self.oracle_ids = {
            layer: cp.empty(ROUTES, cp.int32) for layer in self.layers
        }
        self.oracle_slots = {
            layer: cp.empty(ROUTES, cp.int32) for layer in self.layers
        }
        self.oracle_need = {
            layer: cp.empty(ROUTES, cp.int32) for layer in self.layers
        }
        self.prefetch_done = {
            layer: cp.cuda.Event() for layer in self.layers
        }
        self.prefetch_start = cp.cuda.Event()
        self.prefetch_stream = cp.cuda.Stream(non_blocking=True)
        self.normal_copy_stream = cp.cuda.Stream(non_blocking=True)
        self.mismatches = cp.zeros(1, cp.int32)
        self.current_layer = None

        self._cache_owner = self.base.k
        self._original_cache_assign = self._cache_owner.cache_assign
        self._original_fetch = self.rt.fused.cache_fetch_k
        self._plane_owner = self.base.sres
        self._original_plane_fetch = self._plane_owner.fetch_plane_k
        self._original_copy_stream = self.rt.copy_stream
        self._original_body = graph.body
        self._installed = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, layer, normed, out, collect_stats=False):
        self.current_layer = int(layer)
        if self.mode == "layer_now":
            self.begin_layer(self.current_layer)
        elif self.mode in ("moe_l1", "moe_l2"):
            lead = 1 if self.mode == "moe_l1" else 2
            slot = self.layer_slot[self.current_layer] + lead
            if slot < len(self.layers):
                self.begin_layer(self.layers[slot])
        return self.inner(layer, normed, out, collect_stats)

    def install_for_capture(self):
        if self._installed:
            return
        self._cache_owner.cache_assign = self._cache_assign_after_router
        self.rt.fused.cache_fetch_k = self._skip_fetch
        self._plane_owner.fetch_plane_k = self._skip_fetch
        self.rt.copy_stream = self.normal_copy_stream

        if self.mode == "block_all":
            original_body = self._original_body
            oracle = self

            def oracle_body(_graph_self):
                oracle.begin_block()
                return original_body()

            self.graph.body = types.MethodType(oracle_body, self.graph)
        elif self.mode in ("moe_l1", "moe_l2"):
            original_body = self._original_body
            oracle = self
            lead = 1 if self.mode == "moe_l1" else 2

            def oracle_body(_graph_self):
                oracle.begin_layers(oracle.layers[:lead])
                return original_body()

            self.graph.body = types.MethodType(oracle_body, self.graph)
        self._installed = True

    def restore_after_capture(self):
        if not self._installed:
            return
        self._cache_owner.cache_assign = self._original_cache_assign
        self.rt.fused.cache_fetch_k = self._original_fetch
        self._plane_owner.fetch_plane_k = self._original_plane_fetch
        self.rt.copy_stream = self._original_copy_stream
        self.graph.body = self._original_body
        self._installed = False

    @staticmethod
    def _skip_fetch(*_args, **_kwargs):
        return None

    def begin_block(self):
        main = self.cp.cuda.get_current_stream()
        # Make the auxiliary stream part of the same CUDA graph capture.
        # Without this explicit dependency CUDA correctly rejects a later
        # main-stream wait as a dependency on uncaptured external work.
        self.prefetch_start.record(main)
        with self.prefetch_stream:
            self.prefetch_stream.wait_event(self.prefetch_start)
            for layer in self.layers:
                self._enqueue_layer(layer)

    def begin_layer(self, layer: int):
        self.begin_layers((int(layer),))

    def begin_layers(self, layers):
        main = self.cp.cuda.get_current_stream()
        self.prefetch_start.record(main)
        with self.prefetch_stream:
            self.prefetch_stream.wait_event(self.prefetch_start)
            for layer in layers:
                self._enqueue_layer(int(layer))

    def _enqueue_layer(self, layer: int):
        b = self.base
        rt = self.rt
        slot = self.layer_slot[layer]
        d_ids = self.oracle_ids[layer]
        d_slots = self.oracle_slots[layer]
        d_need = self.oracle_need[layer]
        bank = rt.bank[layer]
        cache = rt.cache[layer]
        dev = b._dev(layer)

        self.kernels.load(
            self.route_table,
            self.graph.pos_dev,
            d_ids,
            slot,
            len(self.layers),
        )
        self._original_cache_assign(
            dev, d_ids, d_slots, d_need, int(cache["cap"])
        )
        self._original_fetch(
            (ROUTES, 64),
            (256,),
            (
                np.uint64(bank["up_codes"].ctypes.data),
                np.uint64(bank["up_scales"].ctypes.data),
                cache["codes"],
                cache["scales"],
                d_ids,
                d_slots,
                d_need,
                np.uint64(UP_CODE),
                np.uint64(UP_SCALE),
            ),
        )
        if layer not in b.sres.planes:
            raise RuntimeError(
                f"Phase44 requires H-SCALE plane on layer {layer}"
            )
        self._original_plane_fetch(
            (ROUTES, 64),
            (256,),
            (
                np.uint64(bank["down_base_ptr"]),
                b.sres.planes[layer],
                d_ids,
                d_slots,
                d_need,
                np.uint64(DOWN_PANEL_BYTES),
                np.uint64(PLANE_BYTES),
                np.int32(b.hidden),
                np.int32(b.npanel),
            ),
        )
        self.prefetch_done[layer].record(self.prefetch_stream)

    def _cache_assign_after_router(self, dev, ids, slots, need, cap):
        del dev, cap
        if self.current_layer is None:
            raise RuntimeError("Phase44 oracle has no active MoE layer")
        layer = self.current_layer
        main = self.cp.cuda.get_current_stream()
        main.wait_event(self.prefetch_done[layer])
        self.kernels.compare(ids, self.oracle_ids[layer], self.mismatches)
        self.cp.copyto(slots, self.oracle_slots[layer])
        self.cp.copyto(need, self.oracle_need[layer])

    def mismatch_count(self) -> int:
        return int(self.cp.asnumpy(self.mismatches)[0])


def make_phase31_parent(context: int):
    return make_attention_head_direct_candidate(int(context), head_mode="m4")


def install_route_capture(graph, block_count: int):
    import cupy as cp

    kernels = Phase44Kernels()
    layers = tuple(int(x) for x in graph.rt.moe_layers)
    table = cp.full((int(block_count), len(layers), ROUTES), -1, cp.int32)
    proxy = RouteCaptureProxy(graph.gmoe, graph, table, kernels)
    graph.gmoe = proxy
    graph.v.moeb = proxy
    return proxy, table, kernels


def install_perfect_prefetch(graph, route_table, *, mode: str = "layer_now"):
    kernels = Phase44Kernels()
    oracle = PerfectBlockPrefetchOracle(
        graph.gmoe, graph, route_table, kernels, mode=mode
    )
    graph.gmoe = oracle
    graph.v.moeb = oracle
    oracle.install_for_capture()
    return oracle, kernels
