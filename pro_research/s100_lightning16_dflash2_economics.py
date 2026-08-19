from __future__ import annotations

import json
import math
import traceback

from common import write_json_atomic, utc_now
from s100_phase10a_runtime import build
from s100_lightning16_common import (
    RESULTS, assert_lightning, ensure_results,
)

OUT = RESULTS / "S100_LIGHTNING16_DFLASH2_ECONOMICS.json"

DRAFT_LAYERS = (2, 3, 4, 5)
MLP_RATIOS = (2.0, 3.0, 3.75)
FORMATS = {
    "bf16": 2.0,
    "fp8": 1.0,
    "nvfp4": 0.5 + 1.0 / 16.0,
}
CONTEXTS = (4096, 32768, 131072)
RESERVE = 512 * 1024**2
WORKSPACE = 0.12
PARAMETER_FACTOR = 1.08
SELECTOR_RANK = 256
CONV_KERNEL = 2
CONV_GROUP = 16

def load(name):
    path = RESULTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def parameter_estimate(rt, layers, mlp_ratio):
    hidden = int(rt.hidden)
    q_dim = int(rt.n_heads * rt.head_dim)
    kv_dim = int(rt.n_kv * rt.head_dim)
    vocab = int(rt.vocab)
    intermediate = int(
        math.ceil(hidden * mlp_ratio / 128.0) * 128
    )
    attention = (
        hidden * q_dim
        + 2 * hidden * kv_dim
        + q_dim * hidden
    )
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    context_projection = layers * hidden * hidden
    groups = math.ceil(hidden / CONV_GROUP)
    conv_module = (
        2 * CONV_KERNEL * hidden
        + hidden * 2 * CONV_KERNEL * groups
    )
    conv = layers * 2 * conv_module
    selector = (
        2 * vocab * SELECTOR_RANK
        + hidden * SELECTOR_RANK
    )
    raw = (
        layers * (attention + mlp + norms)
        + context_projection + conv + selector + hidden
    )
    return {
        "layers": layers,
        "mlp_ratio": mlp_ratio,
        "intermediate": intermediate,
        "raw_parameters": int(raw),
        "total_parameters": int(math.ceil(
            raw * PARAMETER_FACTOR
        )),
    }

def main():
    ensure_results()
    payload = {
        "kind": "s100_lightning16_dflash2_economics",
        "status": "started",
        "started_utc": utc_now(),
    }
    bundle = None
    try:
        ident = assert_lightning()
        block = load(
            "S100_LIGHTNING16_BLOCK_VERIFIER.json"
        )
        proxy = load(
            "S100_LIGHTNING16_DFLASH2_PROXY.json"
        )
        if not block or block.get("status") != "measured":
            raise RuntimeError(
                "fresh Lightning block verifier unavailable"
            )

        verifier = []
        for block_size, row in block["per_B"].items():
            B = int(block_size)
            cycle = float(row["cycle_ms_median"])
            budgets = []
            for accepted in range(1, B + 1):
                budgets.append({
                    "accepted_tokens_including_anchor": accepted,
                    "s100_total_budget_ms": 10.0 * accepted,
                    "draft_selector_budget_ms": (
                        10.0 * accepted - cycle
                    ),
                    "s100_possible_before_draft_cost": (
                        cycle <= 10.0 * accepted
                    ),
                })
            verifier.append({
                "block": B,
                "verify_cycle_ms": cycle,
                "perfect_draft_tok_s": 1000.0 * B / cycle,
                "perfect_draft_s100_open": (
                    1000.0 * B / cycle >= 100.0
                ),
                "acceptance_budgets": budgets,
            })

        bundle = build()
        rt = bundle.rt
        import cupy as cp
        free_bytes, total_bytes = cp.cuda.Device(0).mem_info
        memory = []

        for layers in DRAFT_LAYERS:
            for ratio in MLP_RATIOS:
                estimate = parameter_estimate(
                    rt, layers, ratio
                )
                for format_name, bytes_per_parameter in FORMATS.items():
                    weights = int(math.ceil(
                        estimate["total_parameters"]
                        * bytes_per_parameter
                    ))
                    workspace = int(math.ceil(
                        weights * WORKSPACE
                    ))
                    contexts = []
                    for context in CONTEXTS:
                        kv_bf16 = (
                            2 * layers * context
                            * int(rt.n_kv * rt.head_dim) * 2
                        )
                        required = (
                            weights + workspace
                            + kv_bf16 + RESERVE
                        )
                        contexts.append({
                            "context": context,
                            "kv_bf16_bytes": kv_bf16,
                            "required_bytes": required,
                            "fits": required <= free_bytes,
                        })
                    memory.append({
                        **estimate,
                        "format": format_name,
                        "weight_bytes": weights,
                        "workspace_bytes": workspace,
                        "reserve_bytes": RESERVE,
                        "free_bytes": int(free_bytes),
                        "total_vram_bytes": int(total_bytes),
                        "contexts": contexts,
                    })

        memory_open = any(
            context["fits"]
            for row in memory
            for context in row["contexts"]
            if context["context"] == 4096
        )
        proxy_signal = bool(
            proxy
            and proxy.get("status") == "measured"
            and (
                proxy.get("gates") or {}
            ).get("DFLASH2_LIGHTNING_SIGNAL_OPEN")
        )

        # Use the actually observed corrected independent acceptance when
        # available, not the oracle, for the current runtime budget.
        observed_acceptance = None
        if proxy and proxy.get("status") == "measured":
            observed_acceptance = (
                proxy.get("validation_candidates", {})
                .get("corrected", {})
                .get(
                    "mean_acceptance_independent_including_anchor"
                )
            )
        current_budget_open = False
        current_budget_rows = []
        if observed_acceptance is not None:
            for row in verifier:
                if row["block"] != 8:
                    continue
                budget = (
                    10.0 * float(observed_acceptance)
                    - row["verify_cycle_ms"]
                )
                current_budget_rows.append({
                    "block": 8,
                    "observed_acceptance": observed_acceptance,
                    "draft_selector_budget_ms": budget,
                })
                current_budget_open = budget > 0

        perfect_open = any(
            row["perfect_draft_s100_open"]
            for row in verifier
        )
        training_open = bool(
            current_budget_open
            and memory_open
            and proxy_signal
        )

        payload.update({
            "status": "measured",
            "identity": ident,
            "verifier": verifier,
            "observed_acceptance_budget": current_budget_rows,
            "memory": memory,
            "gates": {
                "LIGHTNING_PERFECT_DRAFT_S100_OPEN": perfect_open,
                "LIGHTNING_OBSERVED_ACCEPTANCE_S100_BUDGET_OPEN": (
                    current_budget_open
                ),
                "DFLASH2_RESIDENT_MEMORY_OPEN_4K": memory_open,
                "DFLASH2_LIGHTNING_SIGNAL_OPEN": proxy_signal,
                "DFLASH2_TRAINING_BUILD_OPEN": training_open,
            },
            "completed_utc": utc_now(),
        })
    except Exception as exc:
        payload.update({
            "status": "technical_failure",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_utc": utc_now(),
        })
    finally:
        if bundle is not None:
            try:
                bundle.restore_combined()
                bundle.restore_sel()
            except Exception:
                pass

    write_json_atomic(OUT, payload, archive=True)
    print(json.dumps({
        "status": payload.get("status"),
        "gates": payload.get("gates"),
        "observed_acceptance_budget": payload.get(
            "observed_acceptance_budget"
        ),
        "error": (payload.get("error") or {}).get("message"),
        "output": str(OUT),
    }, indent=2))
    return 0 if payload.get("status") == "measured" else 2

if __name__ == "__main__":
    raise SystemExit(main())
