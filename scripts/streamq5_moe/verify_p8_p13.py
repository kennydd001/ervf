from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
OUTPUT = R / "p8_p13_independent_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


checks = []


def check(name, condition, detail=None):
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


p8a = json.loads((R / "p8a_projection_adaptive_ervf.json").read_text(encoding="utf-8"))
p8b = json.loads((R / "p8b_scale_broadcast_ervf.json").read_text(encoding="utf-8"))
p8d = json.loads((R / "p8d_q5_code_audit.json").read_text(encoding="utf-8"))
p8a2 = json.loads((R / "p8a2_q5_width8_end_to_end_validation.json").read_text(encoding="utf-8"))
p7v = json.loads((R / "p7c_ervf_end_to_end_validation.json").read_text(encoding="utf-8"))
p10 = json.loads((R / "p10_cache_family.json").read_text(encoding="utf-8"))
p10b = json.loads((R / "p10b_domain_cache_physical.json").read_text(encoding="utf-8"))
p11a = json.loads((R / "p11a_cpu_q5_miss_compute.json").read_text(encoding="utf-8"))
p11b = json.loads((R / "p11b_npu_expert_microbench.json").read_text(encoding="utf-8"))
p11c = json.loads((R / "p11c_nvme_tier.json").read_text(encoding="utf-8"))
p12r2 = json.loads((R / "p12r2_pinned_window_streaming.json").read_text(encoding="utf-8"))
p13a = json.loads((R / "p13a_exact_virtual_attention.json").read_text(encoding="utf-8"))
p13b = json.loads((R / "p13b_explicit_add_attention.json").read_text(encoding="utf-8"))
p13c = json.loads((R / "p13c_evt_pm_32g_endurance.json").read_text(encoding="utf-8"))

for path, payload in (
    (R / "P8A_PROJECTION_ADAPTIVE_ERVF_PREREGISTRATION.md", p8a),
    (R / "P8B_SCALE_BROADCAST_ERVF_PREREGISTRATION.md", p8b),
    (R / "P8D_Q5_CODE_AUDIT_PREREGISTRATION.md", p8d),
    (R / "P11A_CPU_Q5_MISS_COMPUTE_PREREGISTRATION.md", p11a),
    (R / "P11B_NPU_EXPERT_MICROBENCH_PREREGISTRATION.md", p11b),
    (R / "P11C_NVME_TIER_PREREGISTRATION.md", p11c),
    (R / "P13A_EXACT_VIRTUAL_ATTENTION_PREREGISTRATION.md", p13a),
    (R / "P13B_EXPLICIT_ADD_ATTENTION_PREREGISTRATION.md", p13b),
    (R / "P13C_EVT_PM_32G_ENDURANCE_PREREGISTRATION.md", p13c),
):
    check(f"prereg hash {path.name}", sha256(path) == payload["preregistration_sha256"])

check("P8A Q5 full exact", p8a["correctness"]["q5"]["bitwise_equal"] and p8a["correctness"]["q5"]["elements"] == 1_376_256)
check("P8A combined negative", not p8a["overall_pass"] and not p8a["test"]["q8"]["pass"])
check("P8B exact but negative", p8b["correctness"]["bitwise_equal"] and not p8b["overall_pass"])
check("P8A2 CE exact", p8a2["quality"]["aggregate"]["next_token_cross_entropy"] == p7v["quality"]["aggregate"]["next_token_cross_entropy"])
check("P8A2 predictions exact", p8a2["quality"]["aggregate"]["predictions_sha256"] == p7v["quality"]["aggregate"]["predictions_sha256"])
check("P8D all codes", p8d["codes"] == 28_991_029_248 and sum(p8d["histogram_unsigned_0_31"]) == p8d["codes"])
check("P8D per projection", all(sum(values) == 9_663_676_416 for values in p8d["projection_histograms"].values()))
check("P8D overflow recompute", sum(p8d["histogram_unsigned_0_31"][:8]) + sum(p8d["histogram_unsigned_0_31"][23:]) == p8d["overflow_abs_gt_7"])
check("P10 no policy selected", p10["selected"] is None and not p10["cache_family_pass"])
check("P10B copy counts exact", all(p10b["exact_counts"].values()))
check("P10B negative p95", not p10b["gates"]["auto_p95_le_95pct_universal"] and not p10b["overall_pass"])
check("P11A bit exact negative", p11a["correctness"]["bitwise_equal"] and not p11a["overall_pass"])
check("P11B NPU actually ran", p11b["gates"]["npu_compiled"] and not p11b["overall_pass"])
check("P11C direct IO integrity", p11c["direct_io"] and p11c["integrity_failures"] == 0)
check("P11C tail negative", not p11c["gates"]["projected_p95_le_150ms"] and not p11c["overall_pass"])
check("P12R2 tokens", p12r2["tokens"] == 10_000 and p12r2["kv_4k"]["context"] == 4096)
check("P12R2 capacity pass", p12r2["gates"]["peak_commit_le_32g"] and p12r2["gates"]["kv_4k_complete"])
check("P12R2 speed fail", not p12r2["gates"]["tps_ge_10"] and not p12r2["overall_pass"])
check("P13A near exact rejected", not p13a["eligible"] and not p13a["overall_pass"])
for context in ("128", "512", "1024", "4096"):
    item = p13b["correctness"][context]
    check(f"P13B exact {context}", item["scores_bitwise_equal_all_layers"] and item["score_differences"] == 0 and item["values"]["bitwise_equal"])
check("P13B isolated pass", p13b["overall_pass"])
check("P13B 4K p50 gate", p13b["test"]["4096"]["p50_ratio"] <= 0.50)
check("P13B 4K p95 gate", p13b["test"]["4096"]["p95_ratio"] <= 0.50)
check("P13C tokens and 4K", p13c["tokens"] == 10_000 and p13c["kv_4k"]["context"] == 4096)
check("P13C exact predictions", p13c["exactness_vs_p12r2"]["predictions_sha256_equal"])
check("P13C exact misses", p13c["exactness_vs_p12r2"]["misses_equal"])
check("P13C exact KV", p13c["exactness_vs_p12r2"]["kv_4k_sha256_equal"])
check("P13C commit", p13c["memory_final"]["peak_pagefile"] <= 32 * 2**30)
check("P13C latency", p13c["timing"]["mean"] <= 100 and p13c["timing"]["p95"] <= 150 and p13c["timing"]["p99"] < 110)
check("P13C throughput", p13c["tokens_per_second"] >= 10)
check("P13C paired thermal", p13c["context_paired_thermal"]["mean_ratio"] <= 1.10 and p13c["context_paired_thermal"]["p95_ratio"] <= 1.10)
check("P13C all gates", p13c["overall_pass"] and all(p13c["gates"].values()))

result = {
    "kind": "streamq5_moe_p8_p13_independent_verification",
    "checks": checks, "passed": sum(item["pass"] for item in checks),
    "total": len(checks), "overall_pass": all(item["pass"] for item in checks),
}
OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["overall_pass"]: raise SystemExit(1)
