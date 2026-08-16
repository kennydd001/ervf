"""GPU-free preflight for the E100 primitive pack.

Only parses/compiles Python source and runs the CPU reduction-tree mapping test.
It intentionally never imports CuPy or constructs a runtime/GPU context.
"""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRO = ROOT / "pro_research"

FILES = [
    "e100_adopted_baseline.py",
    "e100_mrhs_adopted_bench.py",
    "mrhs_exact_kernels.py",
    "e100_mrhs_v3.py",
    "verify_e100_mrhs_v3.py",
    "mrhs256_exact_kernels.py",
    "e100_mrhs256_v3.py",
    "verify_e100_mrhs256_v3.py",
    "up_proj_pair_batch_kernels.py",
    "e100_pairbatch.py",
    "verify_e100_pairbatch.py",
]


def main() -> int:
    report = {"compiled": {}, "cpu_reduction_selftest": None, "passed": False}
    ok = True
    for rel in FILES:
        path = PRO / rel
        try:
            py_compile.compile(str(path), doraise=True)
            report["compiled"][rel] = "pass"
        except Exception as exc:
            ok = False
            report["compiled"][rel] = f"FAIL: {type(exc).__name__}: {exc}"

    # Import only after syntax compilation. mrhs_exact_kernels itself imports
    # numpy only; its CPU selftest imports the pure-Python reduction helpers in
    # ervf_dense. Neither creates a CUDA context.
    if ok:
        sys.path.insert(0, str(PRO))
        try:
            from mrhs_exact_kernels import cpu_mapping_selftest
            report["cpu_reduction_selftest"] = cpu_mapping_selftest(500)
            ok = ok and bool(report["cpu_reduction_selftest"].get("passed"))
        except Exception as exc:
            ok = False
            report["cpu_reduction_selftest"] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    report["passed"] = bool(ok)
    print(json.dumps(report, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
