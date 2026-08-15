"""NERVF-5: full-model validation of ERVF over 512-token causal rollouts.

Gates frozen here, before the run:
  G-NERVF-5C  every generated token identical between ERVF on and off, over
              3 prompt domains x 512 causal tokens
  G-NERVF-5P  p50 token time improves in every domain and the gain exceeds the
              drift between the two baseline arms
  G-NERVF-5M  no VRAM regression
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

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
OUT_DIR = REPO_ROOT / "reports" / "nervf_nemotron"
CORPUS = REPO_ROOT / "reports/lightningstream_nemotron/s10a_corpus.json"
ROLLOUT = 512


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def pct(v):
    a = np.asarray(v, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def rollout(rt, cp, tokenizer, prompts, n):
    out = []
    for p in prompts:
        ids = tokenizer.encode(p["text"], add_special_tokens=False)
        rt.reset()
        nxt = None
        for t in ids:
            nxt = rt.step(t)
        cp.cuda.Device(0).synchronize()
        gen, ms = [int(nxt)], []
        for _ in range(n - 1):
            t0 = time.perf_counter_ns()
            nx = int(rt.step(gen[-1]))
            cp.cuda.Device(0).synchronize()
            ms.append((time.perf_counter_ns() - t0) / 1e6)
            gen.append(nx)
        lat = pct(ms)
        out.append({"id": p["id"], "generated_ids": gen, "latency": lat,
                    "digest": hashlib.sha256(json.dumps(gen).encode()).hexdigest()})
        print("    " + p["id"].ljust(11)
              + " p50 %7.3f p95 %7.3f p99 %7.3f ms" % (lat["p50"], lat["p95"],
                                                       lat["p99"]), flush=True)
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
    prompts = json.loads(CORPUS.read_text(encoding="utf-8"))["gate_prompts"]

    rt = LightningRuntime(MODEL_DIR, contexts_max=4096, embed_on_host=True, fp8_kv=True)
    rt.enable_cache(72)
    rt.load_routed_bank()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    arms, ref = {}, None
    DET = os.environ.get("D1", "0") == "1"
    for name, use in (("base_a", False), ("ervf", True), ("base_b", False)):
        rt.fused.use_ervf = use
        rt.deterministic_accum = DET
        assert rt.fused.use_ervf is use
        assert rt.deterministic_accum is DET
        print("\narm " + name + ": use_ervf=" + str(use), flush=True)
        r = rollout(rt, cp, tokenizer, prompts, ROLLOUT)
        if ref is None:
            ref = r
        same = all(a["generated_ids"] == b["generated_ids"] for a, b in zip(r, ref))
        free_now, _ = cp.cuda.runtime.memGetInfo()
        arms[name] = {"arm": name, "use_ervf": use, "rollout": r,
                      "identical_to_base_a": bool(same),
                      "device_free_bytes": int(free_now)}
        print("  identical to base_a: " + str(same), flush=True)
    rt.fused.use_ervf = False

    per = {}
    for k, p in enumerate(prompts):
        a1 = arms["base_a"]["rollout"][k]["latency"]
        a2 = arms["base_b"]["rollout"][k]["latency"]
        b = arms["ervf"]["rollout"][k]["latency"]
        base = 0.5 * (a1["p50"] + a2["p50"])
        drift = abs(a2["p50"] - a1["p50"])
        per[p["id"]] = {
            "base_p50": base, "ervf_p50": b["p50"], "gain_ms": base - b["p50"],
            "drift_ms": drift,
            "base_p95": 0.5 * (a1["p95"] + a2["p95"]), "ervf_p95": b["p95"],
            "base_p99": 0.5 * (a1["p99"] + a2["p99"]), "ervf_p99": b["p99"],
            "conclusive": bool(abs(base - b["p50"]) > drift)}
        v = per[p["id"]]
        print("  " + p["id"].ljust(11)
              + " p50 %7.3f -> %7.3f (%+.3f, drift %.3f, concl=%s)"
              % (base, b["p50"], v["gain_ms"], drift, v["conclusive"]), flush=True)

    exact = all(a["identical_to_base_a"] for a in arms.values())
    vram_ok = arms["ervf"]["device_free_bytes"] >= arms["base_a"]["device_free_bytes"]
    allc = all(v["conclusive"] and v["gain_ms"] > 0 for v in per.values())

    payload = {
        "kind": "nervf_nemotron_full_model", "namespace": "NERVF_NEMOTRON",
        "phase": "D1_DETERMINISM" if DET else "NERVF_5", "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": MODEL_DIR.name,
        "runner_sha256": sha256_path(Path(__file__)),
        "fused_sha256": sha256_path(
            REPO_ROOT / "src/moe_lab/lightningstream_nemotron/fused_nvfp4.py"),
        "config": {"rollout_tokens": ROLLOUT,
                   "domains": [p["id"] for p in prompts],
                   "capacity": 72, "max_ctx": 4096, "deterministic_accum": DET},
        "arms": arms, "per_domain": per,
        "gates": {
            "G_NERVF_5C_exact": {"required": "identical tokens over 3 x 512",
                                 "passed": bool(exact)},
            "G_NERVF_5P_latency": {
                "per_domain_gain_ms": {k: v["gain_ms"] for k, v in per.items()},
                "all_conclusive_and_positive": bool(allc),
                "passed": bool(exact and allc)},
            "G_NERVF_5M_vram": {"base_free": arms["base_a"]["device_free_bytes"],
                                "ervf_free": arms["ervf"]["device_free_bytes"],
                                "passed": bool(vram_ok)}},
        "claim_boundary": (
            "512-token causal rollouts on this GPU at capacity 72, three prompt "
            "domains, three arms base/ervf/base so the repeat bounds drift. "
            "Exactness is a hard gate over every generated token. Latency is "
            "end-to-end per-token wall time including the synchronisation; "
            "p95/p99 are over 511 timed steps per domain. Context grows during "
            "the rollout, so these numbers are not comparable to n7b's frozen "
            "fixed-depth figures, and they may not be added to other component "
            "gains."),
    }
    (OUT_DIR / ("d1_determinism.json" if DET else "nervf5_full_model.json")).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\n  G-NERVF-5C exact   : " + str(exact))
    print("  G-NERVF-5P latency : " + str(exact and allc))
    print("  G-NERVF-5M vram    : " + str(vram_ok))
    print("\nwritten nervf5_full_model.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
