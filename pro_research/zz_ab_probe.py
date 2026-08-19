"""A/B probe: P16 vs P16R native dispatch in ONE process, same runtime.

Builds once, evaluates the same 2 calibration prompts with each dispatch,
restoring the original mv_bf16 in between. Removes all cross-process
variables (GPU state, allocation history, trace, weights).
"""
from __future__ import annotations

import json

from s100_phase10a_runtime import build
from s100_lightning16_common import (
    assert_lightning,
    evaluate_runtime,
    normalize_eager_moe,
)
import s100_lightning16_native as n16
import s100_lightning16r_native as n16r

CASES = {
    "attention_12_k", "attention_19_k", "attention_26_k",
    "attention_33_k", "attention_42_k", "attention_5_k",
}

def main():
    assert_lightning()
    bundle = build()
    rt = bundle.rt
    rt._graph = None
    rt.graph_mode = False
    normalize_eager_moe(rt)

    original = rt.k.mv_bf16
    out = {}

    d16 = n16.PointerDispatch(
        rt, terms=1, handoff="sync_control", enabled_cases=set(CASES)
    ).install()
    r16 = evaluate_runtime(rt, split="calibration", prompt_limit=2)
    out["p16"] = {
        "summary": r16["summary"],
        "native_calls": d16.native_calls,
        "original_calls": d16.original_calls,
    }
    rt.k.mv_bf16 = original

    d16r = n16r.PointerDispatch(
        rt, terms=1, handoff="sync_control", enabled_cases=set(CASES)
    ).install()
    r16r = evaluate_runtime(rt, split="calibration", prompt_limit=2)
    out["p16r"] = {
        "summary": r16r["summary"],
        "native_calls": d16r.native_calls,
        "original_calls": d16r.original_calls,
    }
    rt.k.mv_bf16 = original

    # Control: same P16 module again, in the "cursed" third position.
    d16b = n16.PointerDispatch(
        rt, terms=1, handoff="sync_control", enabled_cases=set(CASES)
    ).install()
    r16b = evaluate_runtime(rt, split="calibration", prompt_limit=2)
    out["p16_second"] = {
        "summary": r16b["summary"],
        "native_calls": d16b.native_calls,
        "original_calls": d16b.original_calls,
    }
    rt.k.mv_bf16 = original

    print("P16  top1:", out["p16"]["summary"]["top1_agreement"],
          "ce:", out["p16"]["summary"]["mean_ce_delta"],
          "native:", out["p16"]["native_calls"])
    print("P16R top1:", out["p16r"]["summary"]["top1_agreement"],
          "ce:", out["p16r"]["summary"]["mean_ce_delta"],
          "native:", out["p16r"]["native_calls"])
    print("P16#2 top1:", out["p16_second"]["summary"]["top1_agreement"],
          "ce:", out["p16_second"]["summary"]["mean_ce_delta"],
          "native:", out["p16_second"]["native_calls"])
    with open("pro_research/results/s100_lightning16r/ab_probe.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)

if __name__ == "__main__":
    main()
