from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import scipy
import torch

from evaluate_crcq_oracle import git_state, sha256_file, write_json_once
from moe_lab.craft_moe.qerc import scale_layout_accounting
from moe_lab.reporting import ROOT


PHASE_A = ROOT / "reports/runs/craft_moe/qerc_covariance_layer26.json"
COMPONENTS = ROOT / "reports/runs/craft_moe/qerc_layer26_components.safetensors"
PREREGISTRATION = ROOT / "reports/craft_moe/H6_QERC_LAYER26_PREREGISTRATION.md"
OUTPUT = ROOT / "reports/craft_moe/qerc.json"
NEAR_ZERO_THRESHOLD = 0.02


def main() -> None:
    for path in (PHASE_A, COMPONENTS, PREREGISTRATION):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite result: {OUTPUT}")
    phase_a = json.loads(PHASE_A.read_text(encoding="utf-8"))
    cancellation = {
        split: float(
            phase_a["results"][split][
                "global_cancellation_fraction_ratio_of_sums"
            ]
        )
        for split in ("validation", "test")
    }
    near_zero = {
        split: abs(value) < NEAR_ZERO_THRESHOLD
        for split, value in cancellation.items()
    }
    hard_stop = all(near_zero.values())
    if not hard_stop:
        raise RuntimeError(
            "Phase A did not trigger the preregistered stop; implement and run "
            "the fixed Phase-B gain optimization before adjudication"
        )
    exact_control = bool(
        phase_a["controls"]["official_teacher_delta_bit_exact"]
        and phase_a["controls"]["route_recomputation"]["slot_order_ids_exact"]
    )
    verdict = (
        "falsified_phase_a_cross_terms_near_zero"
        if exact_control
        else "invalid_exact_control_failure"
    )
    result = {
        "schema_version": 1,
        "kind": "craft_moe_h6_qerc",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "experiment": "H6_QERC",
        "verdict": verdict,
        "preregistration": str(PREREGISTRATION.resolve()),
        "model": phase_a["model"],
        "dataset": phase_a["dataset"],
        "phase_a": {
            "source": str(PHASE_A.resolve()),
            "source_sha256": sha256_file(PHASE_A),
            "component_artifact": str(COMPONENTS.resolve()),
            "component_sha256": sha256_file(COMPONENTS),
            "global_cancellation_fraction": cancellation,
            "absolute_near_zero_threshold": NEAR_ZERO_THRESHOLD,
            "near_zero_by_split": near_zero,
            "near_zero_both_splits": hard_stop,
            "energy_sums": {
                split: {
                    "diagonal": phase_a["results"][split]["diagonal_energy"][
                        "sum"
                    ],
                    "aggregate": phase_a["results"][split]["aggregate_energy"][
                        "sum"
                    ],
                    "cross": phase_a["results"][split]["cross_term"]["sum"],
                }
                for split in ("validation", "test")
            },
        },
        "phase_b": {
            "status": "not_opened_preregistered_phase_a_hard_stop",
            "reason": (
                "absolute cancellation fraction is below 2% on validation and "
                "test; scale/clipping and floor/ceil tuning stopped before data fitting"
            ),
            "fixed_candidate_definition_retained": {
                "gain_bounds": [0.75, 1.25],
                "ridge_alphas": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "validation_fit_positions": [0, 128],
                "validation_selection_positions": [128, 256],
            },
        },
        "gates": {
            "adjudicated": True,
            "cross_terms_near_zero_both_splits_hard_falsification": hard_stop,
            "aggregate_q3_error_reduction_ge_0_20": {
                "evaluated": False,
                "reason": "blocked by preregistered Phase-A hard stop",
            },
            "layer26_kl_reduction_ge_0_20": {
                "evaluated": False,
                "reason": "blocked by preregistered Phase-A hard stop",
            },
            "same_byte_layout": scale_layout_accounting(),
            "exact_controls_pass": exact_control,
        },
        "controls": phase_a["controls"],
        "reproducibility": {
            "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd().resolve()),
            "repository": git_state(),
            "libraries": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "psutil": psutil.__version__,
            },
            "inputs": {
                str(PHASE_A.resolve()): sha256_file(PHASE_A),
                str(COMPONENTS.resolve()): sha256_file(COMPONENTS),
                str(PREREGISTRATION.resolve()): sha256_file(PREREGISTRATION),
            },
        },
        "limitations": [
            "the hypothesis is falsified at the preregistered covariance precondition; no learned gains were fit",
            "only layer 26 WikiText was inspected because the stop rule blocks layer/corpus transfer",
            "same-byte scale accounting is analytical; no physical Q3 kernel or runtime was executed",
            "a near-zero natural cross term does not prove every conceivable joint quantizer impossible, only the registered QERC mechanism",
        ],
    }
    write_json_once(OUTPUT, result)
    print(f"result={OUTPUT}")
    print(f"verdict={verdict}")
    print(f"cancellation={json.dumps(cancellation, sort_keys=True)}")


if __name__ == "__main__":
    main()
