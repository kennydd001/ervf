from __future__ import annotations

from common import REPO
from s100_phase21_common import make_rt
from s100_phase32_common import Phase32GraphH8, make_parent
from s100_phase36_native_head import NativeFP4HeadH8


RESULTS = REPO / "pro_research" / "results" / "s100_phase36"


class Phase36NativeHeadGraphH8(Phase32GraphH8):
    def __init__(self, rt):
        super().__init__(rt, "dense_m8")
        self.native_head = NativeFP4HeadH8(rt, self.stream)

    def _head(self) -> None:
        self.native_head(self.core.final_normed, self.core.logits)

    def setup_graph(self):
        info = super().setup_graph()
        info["phase36_native_fp4_head_m8"] = True
        info["native_head_tensor_scale"] = self.native_head.tensor_scale_value
        info["native_head_extra_device_bytes"] = self.native_head.extra_device_bytes
        info["native_head_resources"] = self.native_head.resource_audit()
        return info


def make_candidate(context: int):
    rt, keep = make_rt(int(context), "v6_device_rows")
    graph = Phase36NativeHeadGraphH8(rt)
    return rt, graph, list(keep) + [graph.native_head]
