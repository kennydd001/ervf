from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.streamq5_moe.run_n1a2_q5_staging_end_to_end import (
    N1A, P7_TEST, R, TOKENS, execute, runtime_class, sha256,
)

PREREG = R / "N1A2R_Q5_STAGING_REVERSE_REPLICATION_PREREGISTRATION.md"
N1A2 = R / "n1a2_q5_staging_end_to_end.json"
OUTPUT = R / "n1a2r_q5_staging_reverse.json"


def main():
    if OUTPUT.exists(): raise FileExistsError(OUTPUT)
    original = json.loads(N1A2.read_text(encoding="utf-8"))
    if not original["overall_pass"]: raise RuntimeError("N1A2 pass required")
    p7 = json.loads(P7_TEST.read_text(encoding="utf-8"))["rollout"]
    inputs = [int(x) for x in p7["prompt_ids"]] + [int(x) for x in p7["feedback_ids"][:TOKENS]]
    candidate = execute(runtime_class(True), inputs)
    baseline = execute(runtime_class(False), inputs)
    exact_predictions = [x["prediction"] for x in baseline["tokens"]] == [x["prediction"] for x in candidate["tokens"]]
    exact_misses = [x["misses"] for x in baseline["tokens"]] == [x["misses"] for x in candidate["tokens"]]
    exact_kv = baseline["kv_digest"] == candidate["kv_digest"]
    ratios = {"mean": candidate["timing"]["mean"] / baseline["timing"]["mean"],
              "p95": candidate["timing"]["p95"] / baseline["timing"]["p95"]}
    gates = {"candidate_executed_first": True, "tokens_256": len(candidate["tokens"]) - len(p7["prompt_ids"]) == TOKENS,
             "finite": baseline["finite"] and candidate["finite"], "predictions_exact": exact_predictions,
             "misses_exact": exact_misses, "kv_exact": exact_kv,
             "mean_ratio_le_0_95": ratios["mean"] <= 0.95, "p95_ratio_le_0_95": ratios["p95"] <= 0.95}
    result = {"kind": "streamq5_moe_n1a2r_q5_staging_reverse_replication",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"preregistration_sha256": sha256(PREREG), "n1a_sha256": sha256(N1A),
                         "n1a2_sha256": sha256(N1A2), "p7_test_sha256": sha256(P7_TEST)},
              "execution_order": ["candidate", "baseline"], "workload": original["workload"],
              "baseline": baseline, "candidate": candidate, "ratios": ratios,
              "gates": gates, "overall_pass": all(gates.values()),
              "claim_boundary": "Reverse-order 256-token exact-input replication; 10K endurance remains unopened."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate": candidate["timing"], "baseline": baseline["timing"], "ratios": ratios,
                      "gates": gates, "overall_pass": result["overall_pass"]}, indent=2), flush=True)


if __name__ == "__main__": main()
