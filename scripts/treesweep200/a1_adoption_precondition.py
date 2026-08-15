"""A1 -- adoption preconditions: is the output independent of cache capacity?

Preregistered in A1_ADOPTION_PREREGISTRATION_2026-08-15.md. Gates frozen there:

  G-A1-CAP  (hard) with deterministic_accum=True the generated token sequence at
            capacity 72 is identical to capacity 56, 2 prompts x 256 tokens
  G-A1-CTL  (control, MUST FAIL) the same comparison with deterministic_accum
            =False. If this also passes, the test is blind and G-A1-CAP proves
            nothing -> no adoption.
  G-A2-ANCHOR (informative) does the adoption stack reproduce the frozen
            V35 anchor bitwise?

Changing the cache capacity changes which experts hit and which miss on nearly
every layer of every token, so it changes the hit-then-miss accumulation order
radically -- without touching a single weight, route or kernel. The
mathematically correct output cannot depend on the cache size.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "treesweep200"
ANCHOR = OUT_DIR / "V35_GENERATION_ANCHOR.json"
CORPUS = REPO_ROOT / "reports/lightningstream_nemotron/s10a_corpus.json"
TOKENS = 256
CAPS = (72, 56)


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def rollout(rt, cp, tokenizer, prompts, n):
    out = []
    for p in prompts:
        ids = tokenizer.encode(p["text"], add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        cp.cuda.Device(0).synchronize()
        gen = [int(nxt)]
        for _ in range(n - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        out.append({"id": p["id"], "generated_ids": gen,
                    "digest": hashlib.sha256(json.dumps(gen).encode()).hexdigest()})
    return out


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    if [l for l in o.stdout.strip().splitlines()
            if l.strip() and int(l.split(",")[0]) != os.getpid()]:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    started = datetime.now(timezone.utc).isoformat()
    prompts = json.loads(CORPUS.read_text(encoding="utf-8"))["gate_prompts"][:2]

    rt = LightningRuntime(MODEL_DIR, contexts_max=4096, embed_on_host=True, fp8_kv=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    v1_attn = rt.k.attention_fp8_gqa
    v4_attn = rt.k.attention_fp8_gqa4

    # Both variants run the SAME adoption stack (ERVF + v4); the only variable
    # inside each variant is the capacity, and between variants only D1.
    runs = {}
    for det in (True, False):
        key = "det" if det else "control"
        runs[key] = {}
        for cap in CAPS:
            rt.enable_cache(cap)
            rt.load_routed_bank()
            rt.fused.use_ervf = True
            rt.k.attention_fp8_gqa = v4_attn
            rt.deterministic_accum = det
            assert rt.fused.use_ervf is True
            assert rt.deterministic_accum is det
            assert rt.k.attention_fp8_gqa is v4_attn
            t0 = time.perf_counter()
            r = rollout(rt, cp, tokenizer, prompts, TOKENS)
            runs[key][cap] = r
            print("  %-7s cap %d : %s  (%.1fs)"
                  % (key, cap, " ".join(x["digest"][:12] for x in r),
                     time.perf_counter() - t0), flush=True)
            cp.get_default_memory_pool().free_all_blocks()

    def same(pair):
        a, b = pair[CAPS[0]], pair[CAPS[1]]
        return all(x["generated_ids"] == y["generated_ids"] for x, y in zip(a, b))

    def first_div(pair):
        a, b = pair[CAPS[0]], pair[CAPS[1]]
        d = {}
        for x, y in zip(a, b):
            i = next((j for j, (u, v) in enumerate(zip(x["generated_ids"],
                                                       y["generated_ids"])) if u != v), None)
            d[x["id"]] = i
        return d

    cap_ok = same(runs["det"])
    ctl_ok = same(runs["control"])

    # G-A2: does the adoption stack reproduce the frozen anchor, on the anchor's
    # OWN terms -- its prompts, its capacity, its token count. The anchor was
    # frozen with the v1 attention kernel and hit-then-miss accumulation, so this
    # is informative only; the preregistration fixed what either outcome means.
    anc = json.loads(ANCHOR.read_text(encoding="utf-8"))
    rt.enable_cache(int(anc["capacity"]))
    rt.load_routed_bank()
    rt.fused.use_ervf = True
    rt.k.attention_fp8_gqa = v4_attn
    rt.deterministic_accum = True
    n_anc = int(anc["gen_tokens"])
    matches, got = {}, {}
    for p in anc["prompts"]:
        rt.reset()
        nxt = None
        for t in p["prompt_ids"]:
            nxt = rt.step(int(t))
        gen = [int(nxt)]
        for _ in range(n_anc - 1):
            gen.append(int(rt.step(gen[-1])))
        cp.cuda.Device(0).synchronize()
        got[p["prompt"]] = gen
        ref = p["generated_ids"]
        k = min(len(ref), len(gen))
        i = next((j for j in range(k) if ref[j] != gen[j]), None)
        matches[p["prompt"]] = {"identical": bool(ref[:k] == gen[:k]),
                                "first_divergence_index": i}
    anchor_cmp = {
        "anchor_kernel": anc["kernel"], "anchor_capacity": anc["capacity"],
        "stack_under_test": "ervf_w16 + attention_fp8_gqa4 + deterministic_accum",
        "per_prompt": matches,
        "all_match": bool(all(v["identical"] for v in matches.values())),
        "produced_ids": got}

    verdict = ("adopt" if (cap_ok and not ctl_ok)
               else "no_adopt_test_blind" if (cap_ok and ctl_ok)
               else "no_adopt_d1_insufficient")

    payload = {
        "kind": "treesweep200_a1_adoption_precondition", "registry": "TREESWEEP200",
        "phase": "A1", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "runtime_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
        "config": {"capacities": list(CAPS), "tokens": TOKENS,
                   "prompts": [p["id"] for p in prompts], "max_ctx": 4096,
                   "stack": "ervf_w16 + attention_fp8_gqa4"},
        "digests": {k: {str(c): [x["digest"] for x in v]
                        for c, v in d.items()} for k, d in runs.items()},
        "gates": {
            "G_A1_CAP_hard": {"required": "identical across capacity 72 vs 56 with D1",
                              "identical": bool(cap_ok),
                              "first_divergence_index": first_div(runs["det"]),
                              "passed": bool(cap_ok)},
            "G_A1_CTL_control": {"required": "MUST differ without D1, else test is blind",
                                 "identical": bool(ctl_ok),
                                 "first_divergence_index": first_div(runs["control"]),
                                 "passed": bool(not ctl_ok)},
            "G_A2_ANCHOR_informative": anchor_cmp,
        },
        "verdict": verdict,
        "claim_boundary": (
            "2 prompts x 256 causal tokens per arm, capacity 72 against 56, one "
            "cache rebuild per capacity, contexts_max 4096. This tests ORDER "
            "INDEPENDENCE of the output, not speed; no latency claim is made "
            "here. It demonstrates determinism across the two capacities tested, "
            "not across all possible cache states."),
    }
    (OUT_DIR / "A1_ADOPTION_PRECONDITION.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n  G-A1-CAP  (hard, D1 on) identical across capacity : %s" % cap_ok)
    print("  G-A1-CTL  (control, must differ) identical        : %s -> gate %s"
          % (ctl_ok, not ctl_ok))
    if anchor_cmp is not None:
        print("  G-A2      adoption stack reproduces V35 anchor   : %s"
              % anchor_cmp["all_match"])
    print("  VERDICT: " + verdict)
    print("\nwritten A1_ADOPTION_PRECONDITION.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
