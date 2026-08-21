from __future__ import annotations

from s100_phase31_common import make_attention_head_direct_candidate
from s100_phase45_persistent_up_kernels import PersistentSplitUpKernels


def make_candidate(context: int, schedule: str):
    runtime, graph, keep = make_attention_head_direct_candidate(
        int(context), head_mode="m4"
    )
    persistent = PersistentSplitUpKernels(schedule)
    graph.gmoe.group_dispatch = persistent
    return runtime, graph, list(keep) + [persistent]


def compile_audit(schedule: str):
    return PersistentSplitUpKernels(schedule).resource_audit()
