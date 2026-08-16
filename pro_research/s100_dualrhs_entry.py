"""Execution shim for the frozen DualRHS microbenchmark.

Only replaces its unnecessarily heavy runtime constructor: no routed expert
bank/cache is needed to benchmark the ten resident projection families.
See S100_DUALRHS_EXECUTION_ADDENDUM.md.
"""
from __future__ import annotations

import sys

import s100_dualrhs_ervf as bench
from common import REPO, require_model_dir


def _lean_runtime(_capacity: int):
    sys.path.insert(0, str(REPO / "src"))
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    return LightningRuntime(
        require_model_dir(), contexts_max=4096, embed_on_host=True,
        fp8_kv=True, verbose=False,
    )


def main() -> int:
    bench._new_runtime = _lean_runtime
    return int(bench.main())


if __name__ == "__main__":
    raise SystemExit(main())
