"""S5 baseline: freeze greedy generation of the UNMODIFIED runtime.

Must run before any S5 code lands. The frozen token ids are the G-S5-C1
correctness anchor for the masked column-selective path.
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

MODEL_DIR = REPO_ROOT / "models" / "nemotron_3_5_lightning"
OUT_DIR = REPO_ROOT / "reports" / "lightningstream_nemotron"
PROMPTS = ["The capital of France is", "The history of computing began when"]
GEN = 32


def sha256_path(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def main() -> int:
    import cupy as cp
    from transformers import AutoTokenizer

    try:
        o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        foreign = [l for l in o.stdout.strip().splitlines()
                   if l.strip() and int(l.split(",")[0]) != os.getpid()]
    except Exception:
        foreign = ["query failed"]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    rt = LightningRuntime(MODEL_DIR, contexts_max=4096, verbose=False)
    rt.enable_cache(31)
    rt.load_routed_bank()
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    out = {"kind": "lightningstream_nemotron_s5_baseline_generation",
           "started_utc": datetime.now(timezone.utc).isoformat(),
           "runtime_sha256": sha256_path(
               REPO_ROOT / "src/moe_lab/lightningstream_nemotron/runtime.py"),
           "capacity": 31, "gen_tokens": GEN, "prompts": []}
    for prompt in PROMPTS:
        ids = tok.encode(prompt, add_special_tokens=False)
        rt.reset()
        cur, gen = ids[0], []
        for s in range(len(ids) + GEN):
            nxt = rt.step(cur)
            if s >= len(ids) - 1:
                gen.append(nxt)
                cur = nxt
            else:
                cur = ids[s + 1]
        cp.cuda.Device(0).synchronize()
        out["prompts"].append({"prompt": prompt, "prompt_ids": ids,
                               "generated_ids": gen,
                               "generated_text": tok.decode(gen)})
        print(f"{prompt!r} -> {tok.decode(gen)[:70]!r}", flush=True)

    (OUT_DIR / "s5_baseline_generation.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("written s5_baseline_generation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
