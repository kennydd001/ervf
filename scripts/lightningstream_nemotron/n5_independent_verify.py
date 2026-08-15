"""Independent verifier for N5_PHYSICAL_RESIDENT_SHELL.

Separate program from the runner.  Re-derives the tensor partition directly from
the checkpoint headers, checks it against the frozen N2 inventory, re-derives the
device accounting from the recorded free-memory samples, tests the variant deltas
against the actual embedding and lm_head byte counts, and re-evaluates every gate.

Imports nothing from the runner.  Opens no GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.loader import ShardIndex  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
RESULT = OUT_DIR / "n5_resident_shell.json"
OUT = OUT_DIR / "n5_independent_verification.json"

GIB = 1024 ** 3
MIB = 1024 ** 2

# Frozen N2 inventory values.
N2_TOTAL = 19_339_781_632
N2_ROUTED = 16_523_376_640
N2_SHARED = 258_177_392
N2_TRUNK_OTHER = 2_558_227_600
KV_BYTES_PER_TOKEN = 3_072

ROUTED_RE = re.compile(r"^backbone\.layers\.\d+\.mixer\.experts\.\d+\.")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {}

    # -- provenance ---------------------------------------------------------
    checks["phase_correct"] = result.get("phase") == "N5_PHYSICAL_RESIDENT_SHELL"
    checks["claim_boundary_present"] = bool(result.get("claim_boundary"))
    checks["no_tok_s_promotion"] = "never promoted to tok/s" in result["claim_boundary"]
    runner = REPO_ROOT / "scripts/lightningstream_nemotron/n5_resident_shell.py"
    checks["runner_hash_matches_recorded"] = sha256_path(runner) == result.get("runner_sha256")
    checks["allocations_were_touched"] = result.get("allocations_touched") is True
    checks["measured_from_driver_not_allocator"] = "cuMemGetInfo" in result.get(
        "measurement_source", "")

    # -- independent partition of the checkpoint ----------------------------
    index = ShardIndex(MODEL_DIR)
    routed_bytes = shell_bytes = 0
    routed_n = shell_n = 0
    for name, entry in index.entries.items():
        if ROUTED_RE.match(name):
            routed_bytes += entry.nbytes
            routed_n += 1
        else:
            shell_bytes += entry.nbytes
            shell_n += 1

    inv = result["inventory"]
    checks["routed_bytes_independently_reproduced"] = routed_bytes == inv["routed_bytes"]
    checks["shell_bytes_independently_reproduced"] = shell_bytes == inv["shell_bytes"]
    checks["routed_matches_n2"] = routed_bytes == N2_ROUTED
    checks["shell_equals_trunk_plus_shared"] = shell_bytes == N2_TRUNK_OTHER + N2_SHARED
    checks["total_matches_n2"] = routed_bytes + shell_bytes == N2_TOTAL
    checks["routed_tensor_count_reproduced"] = routed_n == inv["routed_tensor_count"]
    checks["shell_tensor_count_reproduced"] = shell_n == inv["shell_tensor_count"]
    detail["routed_bytes"] = routed_bytes
    detail["shell_bytes"] = shell_bytes

    embed = index.entries["backbone.embeddings.weight"].nbytes
    head = index.entries["lm_head.weight"].nbytes
    checks["embed_bytes_reproduced"] = embed == inv["embeddings_bytes"]
    checks["lm_head_bytes_reproduced"] = head == inv["lm_head_bytes"]

    # -- variant accounting -------------------------------------------------
    variants = result["variants"]
    fitting = {k: v for k, v in variants.items() if v.get("fits")}
    checks["all_three_variants_fit"] = len(fitting) == 3

    device_total = result["device_total_bytes"]
    acct_ok = True
    for name, v in fitting.items():
        recomputed_used = v["free_before"] - v["free_after_all"]
        if recomputed_used != v["device_used_bytes"]:
            acct_ok = False
        if device_total - v["free_after_all"] != v["peak_total_device_bytes"]:
            acct_ok = False
        detail[f"{name}_peak_gib"] = round(v["peak_total_device_bytes"] / GIB, 4)
    checks["device_accounting_recomputed"] = acct_ok

    # Moving the embedding to host must save exactly its byte count, and moving
    # the head must save exactly its own. This is the sharpest consistency test.
    a = variants["A_embed_device_head_device"]
    b = variants["B_embed_host_head_device"]
    c = variants["C_embed_host_head_host"]
    checks["A_minus_B_equals_embedding_bytes"] = (
        a["uploaded_bytes"] - b["uploaded_bytes"] == embed)
    checks["B_minus_C_equals_lm_head_bytes"] = (
        b["uploaded_bytes"] - c["uploaded_bytes"] == head)
    detail["A_minus_B_uploaded"] = a["uploaded_bytes"] - b["uploaded_bytes"]
    detail["B_minus_C_uploaded"] = b["uploaded_bytes"] - c["uploaded_bytes"]

    checks["variant_A_uploaded_equals_full_shell"] = a["uploaded_bytes"] == shell_bytes
    checks["host_resident_shell_accounted"] = (
        c["host_resident_shell_bytes"] == embed + head)

    # -- KV accounting ------------------------------------------------------
    kv_ok = True
    for ctx in result["constants"]["contexts_tested"]:
        expected = ctx * KV_BYTES_PER_TOKEN
        for v in fitting.values():
            if v["kv_by_context"][str(ctx)]["kv_bytes"] != expected:
                kv_ok = False
    checks["kv_bytes_per_context_correct"] = kv_ok
    checks["kv_held_is_largest_context"] = all(
        v["kv_held_bytes"] == max(result["constants"]["contexts_tested"]) * KV_BYTES_PER_TOKEN
        for v in fitting.values())

    # -- headroom, teardown, process ----------------------------------------
    best_peak = min(v["peak_total_device_bytes"] for v in fitting.values())
    worst_peak = max(v["peak_total_device_bytes"] for v in fitting.values())
    checks["worst_variant_still_under_8gib"] = worst_peak <= 8 * GIB
    checks["every_variant_has_256mib_free"] = all(
        v["free_headroom_bytes"] >= 256 * MIB for v in fitting.values())
    detail["worst_peak_gib"] = round(worst_peak / GIB, 4)
    detail["best_peak_gib"] = round(best_peak / GIB, 4)
    detail["variant_A_free_mib"] = round(a["free_headroom_bytes"] / MIB, 3)

    td = result["teardown"]
    checks["teardown_leak_is_zero"] = td["leak_bytes"] == 0
    checks["teardown_flag_consistent"] = td["clean"] is (abs(td["leak_bytes"]) <= 64 * MIB)

    proc = result["process_memory_after"]
    checks["process_memory_actually_measured"] = (
        "error" not in proc and proc.get("peak_commit_bytes", 0) > 0)
    checks["process_commit_under_32gib"] = proc.get("peak_commit_bytes", 1 << 62) <= 32 * GIB
    detail["peak_commit_gib"] = round(proc.get("peak_commit_bytes", 0) / GIB, 4)

    # -- gates --------------------------------------------------------------
    gates = result["gates"]
    checks["gates_all_pass_consistent"] = result["gates_all_pass"] == all(gates.values())
    checks["terminal_state_consistent"] = result["terminal_state"] == (
        "n5_resident_shell_fits" if result["gates_all_pass"] else "n5_resident_shell_fail")
    checks["no_precision_reduction_declared"] = gates["S6_no_precision_reduction"] is True

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    payload = {
        "kind": "lightningstream_nemotron_n5_independent_verification",
        "registry": "LIGHTNINGSTREAM_NEMOTRON",
        "phase": "N5_PHYSICAL_RESIDENT_SHELL",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256_path(Path(__file__)),
        "verified_result_sha256": sha256_path(RESULT),
        "gpu_opened": False,
        "runner_imported": False,
        "checks": checks,
        "passed": passed,
        "total": total,
        "pass": passed == total,
        "detail": detail,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for name, value in checks.items():
        print(f"  {'OK  ' if value else 'FAIL'} {name}")
    print()
    print(f"variant A peak / free : {detail['A_embed_device_head_device_peak_gib']} GiB / "
          f"{detail['variant_A_free_mib']} MiB")
    print(f"worst peak            : {detail['worst_peak_gib']} GiB")
    print(f"peak process commit   : {detail['peak_commit_gib']} GiB")
    print(f"\n{passed}/{total} verification checks passed")
    return 0 if payload["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
