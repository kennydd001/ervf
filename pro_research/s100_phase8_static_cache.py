
"""Static full-record routed-down cache, populated before graph capture."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from moe_dev_batched import DOWN_PANEL_BYTES
from s100_phase8_common import selection_hash
from s100_phase8_static_kernels import StaticDownKernels


@dataclass
class LayerStaticCache:
    expert_ids: list[int]
    ids_device: object
    expert_to_static: object
    records: object


class StaticDownCache:
    def __init__(
        self,
        rt,
        selection: dict[int, list[int]],
        reserve_bytes: int = 0,
    ):
        import cupy as cp

        self.cp = cp
        self.kernels = StaticDownKernels()
        self.selection = {
            int(layer): [int(x) for x in experts]
            for layer, experts in selection.items()
        }
        self.selection_sha256 = selection_hash(self.selection)
        self.layers: dict[int, LayerStaticCache] = {}
        self.total_records = sum(
            len(x) for x in self.selection.values()
        )
        self.physical_bytes = (
            self.total_records * int(DOWN_PANEL_BYTES)
        )
        self.reserve_bytes = int(reserve_bytes)

        cp.get_default_memory_pool().free_all_blocks()
        free_before = int(cp.cuda.Device(0).mem_info[0])
        margin = 128 * 1024 * 1024
        required = (
            self.physical_bytes
            + self.reserve_bytes
            + margin
        )
        if required > free_before:
            raise MemoryError(
                "static down cache does not fit: "
                f"records={self.physical_bytes}, "
                f"reserve={self.reserve_bytes}, "
                f"free={free_before}, margin={margin}"
            )
        self.free_before_bytes = free_before

        for layer in sorted(int(x) for x in rt.moe_layers):
            experts = self.selection.get(layer, [])
            mapping = np.full(
                int(rt.n_experts), -1, dtype=np.int32
            )
            for slot, expert in enumerate(experts):
                mapping[expert] = slot

            ids_device = cp.asarray(
                np.asarray(experts, dtype=np.int32)
            )
            records = cp.empty(
                max(1, len(experts) * int(DOWN_PANEL_BYTES)),
                dtype=cp.uint8,
            )
            entry = LayerStaticCache(
                expert_ids=experts,
                ids_device=ids_device,
                expert_to_static=cp.asarray(mapping),
                records=records,
            )
            self.layers[layer] = entry

            if experts:
                self.kernels.preload(
                    (len(experts), 64),
                    (256,),
                    (
                        np.uint64(rt.bank[layer]["down_base_ptr"]),
                        ids_device,
                        records,
                        np.uint64(DOWN_PANEL_BYTES),
                    ),
                )

        cp.cuda.Device(0).synchronize()
        self.free_after_bytes = int(
            cp.cuda.Device(0).mem_info[0]
        )

    def public_record(self) -> dict:
        return {
            "total_records": int(self.total_records),
            "physical_bytes": int(self.physical_bytes),
            "physical_mib": self.physical_bytes / (1024**2),
            "reserved_future_bytes": int(self.reserve_bytes),
            "selection_sha256": self.selection_sha256,
            "free_before_bytes": self.free_before_bytes,
            "free_after_bytes": self.free_after_bytes,
            "by_layer": {
                str(layer): list(entry.expert_ids)
                for layer, entry in sorted(self.layers.items())
                if entry.expert_ids
            },
        }
