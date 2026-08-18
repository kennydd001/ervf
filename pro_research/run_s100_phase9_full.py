"""S100 phase 9 full runner: every experiment as a fresh subprocess, in the
preregistered order, reusing the existing 8192-token trace and UPMISS NPZ
captures.

Steps:
  1. RTX staged-vs-DirectHost miss probes for the five captured layers;
  2. Intel Arc OpenCL miss probes N=1/2/3 (need the RTX .ref.npz);
  3. capacity A/B per profile in BASE_A -> CAND_A -> CAND_B -> BASE_B order,
     each arm a fresh process;
  4. per-profile compare with the preregistered gates;
  5. miss economics, repaired summary, repair verification.

Exit 0 when every step completed with an expected status. `infeasible_vram`
is an expected, complete verdict (the candidate genuinely does not fit the
target GPU); `technical_failure` is not.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRO = REPO / "pro_research"
RESULTS = PRO / "results" / "s100_phase9"
PY = REPO / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

LAYERS = (43, 38, 10, 29, 22)
PROFILES = ("budget_neutral", "plus_128", "plus_256", "plus_379")
ROLES = ("base_a", "cand_a", "cand_b", "base_b")
ARM_OK = {"measured", "infeasible_vram"}

LOG = REPO / "pro_research" / "results" / "logs" / (
    f"S100_PHASE9_FULL_{time.strftime('%Y%m%d-%H%M%S')}.log"
)


def run(label, argv, ok_statuses=None, out_json=None):
    line = f"[phase9] {label}: {' '.join(str(a) for a in argv)}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as log:
        log.write(line + "\n")
        started = time.time()
        proc = subprocess.run(
            [str(PY), *argv],
            cwd=PRO,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.time() - started
    note = f"[phase9] {label}: exit={proc.returncode} ({elapsed:.0f}s)"
    status = None
    if out_json is not None and Path(out_json).exists():
        try:
            status = json.loads(
                Path(out_json).read_text(encoding="utf-8")
            ).get("status")
            note += f" status={status}"
        except Exception as exc:  # unreadable json is a failure below
            note += f" status unreadable: {exc}"
    print(note, flush=True)
    with LOG.open("a", encoding="utf-8") as log:
        log.write(note + "\n")
    if proc.returncode != 0:
        return False
    if ok_statuses is not None and status not in ok_statuses:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-capacity", action="store_true",
        help="only run the miss probes + economics + summary",
    )
    parser.add_argument(
        "--skip-probes", action="store_true",
        help="only run capacity A/B + compares + summary",
    )
    args = parser.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    failures = []

    if not args.skip_probes:
        for layer in LAYERS:
            sample = RESULTS / f"UPMISS_LAYER_{layer}.npz"
            rtx = RESULTS / f"RTX_UPMISS_LAYER_{layer}.json"
            if not run(
                f"rtx-upmiss L{layer}",
                ["s100_phase9_upmiss_rtx.py", "--sample", str(sample),
                 "--out", str(rtx)],
                ok_statuses={"measured"}, out_json=rtx,
            ):
                failures.append(f"rtx-upmiss L{layer}")
                continue  # Arc arm needs the RTX reference of this layer
            arc = RESULTS / f"ARC_UPMISS_LAYER_{layer}.json"
            if not run(
                f"arc-upmiss L{layer}",
                ["s100_phase9_upmiss_arc.py", "--sample", str(sample),
                 "--rtx-ref", str(rtx.with_suffix(".ref.npz")),
                 "--out", str(arc)],
                ok_statuses={"measured"}, out_json=arc,
            ):
                failures.append(f"arc-upmiss L{layer}")

    if not args.skip_capacity:
        for profile in PROFILES:
            for role in ROLES:
                out = RESULTS / f"CAP_{profile.upper()}_{role.upper()}.json"
                if not run(
                    f"capacity {profile} {role}",
                    ["s100_phase9_capacity_arm.py", "--profile", profile,
                     "--role", role],
                    ok_statuses=ARM_OK, out_json=out,
                ):
                    failures.append(f"capacity {profile} {role}")
            compare = RESULTS / f"CAP_COMPARE_{profile.upper()}.json"
            if not run(
                f"capacity-compare {profile}",
                ["s100_phase9_capacity_compare.py", "--profile", profile],
                ok_statuses={
                    "capacity_promote", "capacity_below_gate",
                    "measurement_failed", "infeasible_vram",
                },
                out_json=compare,
            ):
                failures.append(f"capacity-compare {profile}")

    steps = [
        ("miss-economics", ["s100_phase9_miss_economics.py",
                            "--dir", str(RESULTS)],
         RESULTS / "S100_PHASE9_MISS_ECONOMICS.json",
         {"measured", "incomplete"}),
        ("summary", ["s100_phase9_summary.py"],
         RESULTS / "S100_PHASE9_SUMMARY.json", None),
        ("repair-verify", ["verify_s100_phase9_repair.py",
                           "--dir", str(RESULTS)],
         RESULTS / "S100_PHASE9_REPAIR_VERIFY.json", {"PASS"}),
    ]
    for label, argv, out, ok in steps:
        if not run(label, argv, ok_statuses=ok, out_json=out):
            failures.append(label)

    print(f"[phase9] DONE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
