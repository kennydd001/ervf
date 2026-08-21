from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from common import utc_now, write_json_atomic
from s100_phase21_common import load_trace
from s100_phase44_measure import capture_routes, run_oracle, run_parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated Phase44 arm")
    parser.add_argument("--arm", choices=("routes", "parent", "oracle"), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--routes-file", default="")
    parser.add_argument(
        "--oracle-mode",
        choices=("layer_now", "moe_l1", "moe_l2", "block_all"),
        default="layer_now",
    )
    args = parser.parse_args()

    payload = {
        "kind": "s100_phase44_isolated_arm",
        "arm": args.arm,
        "status": "started",
        "started_utc": utc_now(),
    }
    try:
        tokens = load_trace()["tokens"]
        if args.arm == "routes":
            if not args.routes_file:
                raise ValueError("--routes-file is required for routes")
            table, positions, capture_info = capture_routes(
                tokens, args.context, args.blocks, args.warmup
            )
            np.savez_compressed(
                args.routes_file,
                route_table=np.asarray(table, np.int32),
                positions=np.asarray(positions, np.int32),
            )
            payload.update(
                {
                    "status": "measured",
                    "route_capture": {
                        "positions": positions,
                        "layer_count": int(table.shape[1]),
                        "routes_per_layer": int(table.shape[2]),
                        "capture_info": capture_info,
                        "routes_file": args.routes_file,
                    },
                }
            )
        elif args.arm == "parent":
            payload.update(
                {
                    "status": "measured",
                    "result": run_parent(
                        tokens, args.context, args.blocks, args.warmup
                    ),
                }
            )
        else:
            if not args.routes_file:
                raise ValueError("--routes-file is required for oracle")
            with np.load(args.routes_file, allow_pickle=False) as data:
                table = np.asarray(data["route_table"], np.int32)
            payload.update(
                {
                    "status": "measured",
                    "result": run_oracle(
                        tokens,
                        table,
                        args.context,
                        args.blocks,
                        args.warmup,
                        args.oracle_mode,
                    ),
                }
            )
        payload["completed_utc"] = utc_now()
    except Exception as exc:
        payload.update(
            {
                "status": "technical_failure",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                "completed_utc": utc_now(),
            }
        )

    output_path = Path(args.output)
    write_json_atomic(output_path, payload, archive=True)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "status": payload.get("status"),
                "summary": (payload.get("result") or {}).get("summary"),
                "error": (payload.get("error") or {}).get("message"),
                "output": args.output,
            },
            indent=2,
        )
    )
    return 0 if payload.get("status") == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
