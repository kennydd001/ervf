from __future__ import annotations

from common import REPO
from s100_phase21_common import make_rt
from s100_phase32_common import Phase32GraphH8, Phase32MoEH8
from s100_phase33_nvfp4_m8 import NVFP4M8Warp32


RESULTS = REPO / "pro_research" / "results" / "s100_phase33"
ARMS = ("head_m8", "shared_m8", "shared_head_m8")


class Phase33MoEH8(Phase32MoEH8):
    def __init__(self, base, *, shared_m8: bool):
        super().__init__(base, "m8")
        self.shared_m8_enabled = bool(shared_m8)
        self.nvfp4_m8 = NVFP4M8Warp32() if self.shared_m8_enabled else None

    def _shared_fork(self, i, d, normed, main) -> None:
        if not self.shared_m8_enabled:
            return super()._shared_fork(i, d, normed, main)
        b, rt = self.base, self.rt
        self.fork_events[i].record(main)
        with self.shared_stream:
            self.shared_stream.wait_event(self.fork_events[i])
            self.nvfp4_m8.nvfp4(
                d["sh_up_c"], d["sh_up_s"],
                rt.fused.e2m1, rt.fused.e4m3,
                normed, b.shared_act, d["sh_up_g"],
                b.shared, b.hidden, True,
            )
            self.nvfp4_m8.nvfp4(
                d["sh_dn_c"], d["sh_dn_s"],
                rt.fused.e2m1, rt.fused.e4m3,
                b.shared_act, self.shared_out, d["sh_dn_g"],
                b.hidden, b.shared, False,
            )
            self.done_events[i].record(self.shared_stream)


class Phase33GraphH8(Phase32GraphH8):
    def __init__(self, rt, arm: str):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(rt, "dense_m8")
        self.phase33_arm = arm
        base = self.gmoe.base
        self.gmoe = Phase33MoEH8(
            base,
            shared_m8=arm in ("shared_m8", "shared_head_m8"),
        )
        self.phase33_head_m8 = (
            NVFP4M8Warp32()
            if arm in ("head_m8", "shared_head_m8")
            else None
        )

    def _head(self) -> None:
        if self.phase33_head_m8 is None:
            return super()._head()
        rt, core = self.rt, self.core
        self.phase33_head_m8.nvfp4(
            rt.lm_head_codes, rt.lm_head_scales,
            rt.fused.e2m1, rt.fused.e4m3,
            core.final_normed, core.logits, rt.lm_head_g,
            rt.vocab, rt.hidden, False,
        )

    def setup_graph(self):
        info = super().setup_graph()
        info["phase33_arm"] = self.phase33_arm
        info["shared_nvfp4_m8"] = bool(self.gmoe.shared_m8_enabled)
        info["head_nvfp4_m8"] = self.phase33_head_m8 is not None
        return info


def make_candidate(context: int, arm: str):
    if arm not in ARMS:
        raise ValueError(arm)
    rt, keep = make_rt(int(context), "v6_device_rows")
    graph = Phase33GraphH8(rt, arm)
    return rt, graph, list(keep) + [graph.gmoe, graph.phase33_head_m8]


def compile_audit() -> dict:
    kernel = NVFP4M8Warp32()
    return {"nvfp4_m8_warp32_direct_l2": kernel.resource_audit()}
