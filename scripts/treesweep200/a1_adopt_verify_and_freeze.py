"""A1b -- verify the adopted defaults, then freeze the deterministic anchor.

The point of this run: construct LightningRuntime with NO flags set at all and
check that what it produces is exactly what A1 measured for the adoption stack.
That is what "adopted" has to mean -- the default path IS the validated path.

  G-A1B-DEFAULT  a default-constructed runtime reproduces, bitwise, the token
                 sequences A1 measured under the explicit adoption stack
  G-A1B-FLAGS    the flags really are on by default (asserted, not assumed)

Only if both hold is the new anchor written. The old V35 anchor is NOT touched:
it records the hit-then-miss accumulation order and remains the reference for
every measurement taken before adoption.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "treesweep200"
OLD_ANCHOR = OUT_DIR / "V35_GENERATION_ANCHOR.json"
NEW_ANCHOR = OUT_DIR / "V36_DETERMINISTIC_ANCHOR.json"
A1 = OUT_DIR / "A1_ADOPTION_PRECONDITION.json"


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    import cupy as cp

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    old = json.loads(OLD_ANCHOR.read_text(encoding="utf-8"))
    a1 = json.loads(A1.read_text(encoding="utf-8"))
    expected = a1["gates"]["G_A2_ANCHOR_informative"]["produced_ids"]

    rt = LightningRuntime(MODEL_DIR, contexts_max=4096, embed_on_host=True, fp8_kv=True)
    rt.enable_cache(int(old["capacity"]))
    rt.load_routed_bank()

    # G-A1B-FLAGS: assert the defaults, do not assume them
    flags = {"use_ervf": bool(rt.fused.use_ervf),
             "ervf_width": int(rt.fused.ervf_width),
             "deterministic_accum": bool(rt.deterministic_accum),
             "gatherless_down": bool(rt.fused.gatherless_down),
             # bound methods compare unequal by identity, so compare the
             # underlying function objects
             "attn_is_v4": (getattr(rt.attn, "__func__", rt.attn)
                            is rt.k.attention_fp8_gqa4.__func__),
             "attn_is_not_v1": (getattr(rt.attn, "__func__", rt.attn)
                                is not rt.k.attention_fp8_gqa.__func__)}
    flags_ok = (flags["use_ervf"] and flags["deterministic_accum"]
                and flags["attn_is_v4"] and flags["attn_is_not_v1"]
                and not flags["gatherless_down"]
                and flags["ervf_width"] == 16)
    print("  defaults: " + json.dumps(flags))

    n = int(old["gen_tokens"])
    rows, matches = [], {}
    for p in old["prompts"]:
        rt.reset()
        nxt = None
        for t in p["prompt_ids"]:
            nxt = rt.step(int(t))
        gen = [int(nxt)]
        for _ in range(n - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        exp = expected.get(p["prompt"])
        matches[p["prompt"]] = bool(exp is not None and exp == gen)
        rows.append({"prompt": p["prompt"], "prompt_ids": p["prompt_ids"],
                     "generated_ids": gen,
                     "digest": hashlib.sha256(json.dumps(gen).encode()).hexdigest()})
        print("    %-38s matches A1: %s" % (p["prompt"][:38], matches[p["prompt"]]))

    default_ok = all(matches.values())
    passed = bool(default_ok and flags_ok)

    payload = {
        "kind": "treesweep200_a1b_adoption_verify", "registry": "TREESWEEP200",
        "phase": "A1B", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "fused_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"),
        "observed_defaults": flags,
        "gates": {
            "G_A1B_DEFAULT": {"per_prompt": matches, "passed": bool(default_ok)},
            "G_A1B_FLAGS": {"passed": bool(flags_ok)}},
        "anchor_written": bool(passed),
        "claim_boundary": (
            "A default-constructed runtime, no flags set by the caller, on the "
            "old anchor's own prompts, capacity and token count. It checks that "
            "the default path is the path A1 validated. It is not a latency "
            "measurement and makes no throughput claim."),
    }
    (OUT_DIR / "A1B_ADOPTION_VERIFY.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if passed:
        NEW_ANCHOR.write_text(json.dumps({
            "kind": "nemotron_v36_deterministic_generation_anchor",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "why": ("Frozen after the A1 adoption. The default runtime is now "
                    "ERVF w16 + attention_fp8_gqa4 + route-order accumulation. "
                    "The previous anchor V35_GENERATION_ANCHOR.json records the "
                    "hit-then-miss accumulation order and is NOT bit-comparable "
                    "with this one: it is the reference for every measurement "
                    "taken before adoption, and is kept for exactly that reason."),
            "supersedes_for_new_work": "V35_GENERATION_ANCHOR.json",
            "model_dir": MODEL_DIR.name,
            "runtime_sha256": payload["runtime_sha256"],
            "fused_sha256": payload["fused_sha256"],
            "stack": "ervf_w16 + attention_fp8_gqa4 + deterministic_accum",
            "capacity": old["capacity"], "gen_tokens": n,
            "prompts": rows,
        }, indent=2) + "\n", encoding="utf-8")

    print("\n  G-A1B-FLAGS   defaults are the adopted stack : %s" % flags_ok)
    print("  G-A1B-DEFAULT default run == A1 adoption run : %s" % default_ok)
    print("  new anchor written: %s" % passed)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
