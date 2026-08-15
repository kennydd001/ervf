"""Read-only diagnostic: the decisive unmeasured term from S10-A.

S10A_MTP_ACCEPTANCE_REPORT_2026-08-15.md (reports/lightningstream_nemotron/)
measured MTP draft acceptance (A=2.114, gate G-S10-1 passed) but explicitly
flagged that step 2 (building a speculative loop) hinges entirely on a term
nobody had measured: how many UNIQUE experts per MoE layer are touched across
a window of D+1=5 consecutive verified tokens, versus 6 for a single token.
The report's own recommendation: "meet die unie eerst... Poortvoorstel: als
de gemiddelde unie over 5 tokens groter is dan ~12 van de 128 per laag, is
stap 2 negatief voor er een regel kernel geschreven is."

This script does exactly that, using tooling that already exists
(`LightningRuntime.step(token_id, capture_routes=...)`) and data that already
exists (the greedy-generated token sequences recorded in
s10a_mtp_acceptance.json's gate_prompts, teacher-forced replay). No new
kernel, no runtime modification, no PRO gate -- a pure analysis of routing
decisions the model already made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import environment_snapshot, percentiles, require_gpu_free, require_model_dir, utc_now, write_json_atomic

S10A = REPO / "reports" / "lightningstream_nemotron" / "s10a_mtp_acceptance.json"
WINDOW = 5  # D+1 = 4 drafts + 1 verification token, per S10 preregistration


def sliding_union_sizes(per_layer_routes: list[list[int]], window: int) -> list[int]:
    """routes: one list of top-k expert ids per position. Returns union size
    for every window of `window` consecutive positions."""
    sizes = []
    for start in range(0, len(per_layer_routes) - window + 1):
        s = set()
        for pos in range(start, start + window):
            s.update(per_layer_routes[pos])
        sizes.append(len(s))
    return sizes


def main() -> int:
    require_gpu_free()
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    s10a = json.loads(S10A.read_text(encoding="utf-8"))
    prompts = s10a["gate_prompts"]

    rt = LightningRuntime(require_model_dir(), contexts_max=4096, embed_on_host=True,
                          fp8_kv=True, verbose=False)
    rt.enable_cache(72)
    rt.load_routed_bank()
    rt.deterministic_accum = True
    # device_cache stays False: _moe_dev returns (None, None) and would not
    # populate capture_routes. Routing is a deterministic function of the
    # hidden state trajectory, not of cache/device_cache mode.

    moe_layers = [i for i, ch in enumerate(rt.pattern) if ch not in ("M", "*")]

    per_prompt_result = []
    all_window_sizes: list[int] = []
    for p in prompts:
        seq = [int(x) for x in p["sequence"]]
        rt.reset()
        capture: dict[str, list] = {}
        for t in seq[:-1]:
            rt.step(int(t), capture_routes=capture)

        layer_window_sizes: dict[str, list[int]] = {}
        prompt_all_windows: list[int] = []
        for i in moe_layers:
            routes = capture.get(str(i))
            if not routes or len(routes) < WINDOW:
                continue
            sizes = sliding_union_sizes(routes, WINDOW)
            layer_window_sizes[str(i)] = sizes
            prompt_all_windows.extend(sizes)
            all_window_sizes.extend(sizes)

        per_prompt_result.append({
            "label": p["label"],
            "sequence_len": len(seq),
            "steps_captured": len(seq) - 1,
            "moe_layers_captured": len(layer_window_sizes),
            "windows_per_layer": len(next(iter(layer_window_sizes.values()))) if layer_window_sizes else 0,
            "union_size_stats": percentiles([float(x) for x in prompt_all_windows]),
        })

    overall = percentiles([float(x) for x in all_window_sizes])
    baseline_single_token = 6  # top_k
    gate_threshold = 12  # the report's own proposed "doubling" gate

    payload = {
        "kind": "diag_mtp_route_union",
        "created_utc": utc_now(),
        "note": "read-only diagnostic answering the decisive open term in S10A_MTP_ACCEPTANCE_REPORT_2026-08-15.md; not a gated PRO experiment",
        "source_report": "reports/lightningstream_nemotron/S10A_MTP_ACCEPTANCE_REPORT_2026-08-15.md",
        "source_sequences": str(S10A.relative_to(REPO)),
        "environment": environment_snapshot((REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py",)),
        "window": WINDOW,
        "top_k": int(rt.top_k),
        "n_experts": int(rt.n_experts),
        "moe_layer_count": len(moe_layers),
        "per_prompt": per_prompt_result,
        "overall_union_size_stats": overall,
        "baseline_single_token_experts": baseline_single_token,
        "reports_proposed_gate_experts": gate_threshold,
        "mean_union_ge_gate_threshold": overall["mean"] is not None and overall["mean"] > gate_threshold,
        "amplification_vs_single_token": (overall["mean"] / baseline_single_token) if overall["mean"] else None,
    }
    out = REPO / "pro_research" / "diag_mtp_route_union.json"
    write_json_atomic(out, payload, archive=False)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
