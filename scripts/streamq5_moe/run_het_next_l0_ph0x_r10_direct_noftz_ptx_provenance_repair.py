from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts/streamq5_moe"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import het_next_l0_ph0r3_common as common
import run_het_next_l0_ph0x_r9_direct_noftz_ptx_nvidia as r9


RUN = ROOT / "reports/runs/streamq5_moe/het_next_l0_ph0x_r10_direct_noftz_ptx_provenance_repair"
RESULT = RUN / "ph0x_r10_result.json"
PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R10_DIRECT_NOFTZ_PTX_PROVENANCE_REPAIR_PREREGISTRATION_2026-08-13.md"
R9_PREREG = ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R9_DIRECT_NOFTZ_PTX_NVIDIA_COMPLETION_PREREGISTRATION_2026-08-13.md"

EXTRA_EXPECTED = {
    Path(r9.__file__).resolve(): "65bc5d13aaa07689dbdc794f1735cf39310fbdf681f0269c8ee26ca6391217ae",
    R9_PREREG: "6b35c59a34da914baaefd8103846851d67ac7e0a26bb5728bdced9aee13ec213",
    Path(r9.r5.__file__).resolve(): "0f2d1894067c65fd40200c45d7b8d14dd72d35987e7aad4afed8c003dead9f63",
    r9.r6.R5_PREREG: "0824989026f32cb692001b4824937ad636f8a72cfd2f263e71b191d4c196aa71",
    Path(r9.r6.__file__).resolve(): "a1369c314a4e1367fa4ce3584555a7dc4db30ed9480cbdff289aa18af8417bdf",
    r9.r7.R6_PREREG: "7e5c0ad01797120c66ce140f32207ed3460821aa3a0f4acbd6aff8f5a8231732",
    ROOT / "reports/streamq5_moe/HET_NEXT_L0_PH0X_R7_NVIDIA_ONLY_LIFECYCLE_REPAIR_PREREGISTRATION_2026-08-13.md": "3fd0c0429eaaebe1291d9ffbbb31d08df38de300687ff3c63def1b28d0b3eb95",
}
ORIGINAL_R9_GATE = r9.predevice_gate


def expanded_predevice_gate():
    observed = {str(path): common.file_digest(path) for path in EXTRA_EXPECTED}
    if any(observed[str(path)] != expected for path, expected in EXTRA_EXPECTED.items()):
        raise RuntimeError("r10_expanded_provenance_hash_drift")
    inherited, prior = ORIGINAL_R9_GATE()
    inherited.update(observed)
    return inherited, prior


def main() -> int:
    if RUN.exists():
        raise FileExistsError(RUN)
    expanded_predevice_gate()
    original_gate = r9.predevice_gate
    original_run = r9.RUN
    original_result = r9.RESULT
    original_prereg = r9.PREREG
    original_module_file = r9.__file__
    original_writer = common.write_atomic_new

    def r10_writer(path, data):
        value = __import__("json").loads(data)
        value["kind"] = "het_next_l0_ph0x_r10_direct_noftz_ptx_provenance_repair"
        value.setdefault("bindings", {})["r9_runner_sha256"] = EXTRA_EXPECTED[Path(original_module_file).resolve()]
        value["bindings"]["r9_prereg_sha256"] = EXTRA_EXPECTED[R9_PREREG]
        original_writer(path, common.canonical(value))

    r9.predevice_gate = expanded_predevice_gate
    r9.RUN = RUN
    r9.RESULT = RESULT
    r9.PREREG = PREREG
    r9.__file__ = __file__
    common.write_atomic_new = r10_writer
    try:
        return r9.main()
    finally:
        common.write_atomic_new = original_writer
        r9.__file__ = original_module_file
        r9.predevice_gate = original_gate
        r9.RUN = original_run
        r9.RESULT = original_result
        r9.PREREG = original_prereg


if __name__ == "__main__":
    raise SystemExit(main())
