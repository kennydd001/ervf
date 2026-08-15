"""S10-A smoke: does the MTP block load, fit, and produce finite logits?

No routed bank, so the backbone cannot decode here -- this only checks shapes,
kernels, memory footprint and finiteness before the expensive full run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from moe_lab.lightningstream_nemotron.runtime import LightningRuntime  # noqa: E402
from moe_lab.lightningstream_nemotron.mtp import MTPBlock  # noqa: E402

MODEL_DIR = REPO_ROOT / "models" / os.environ.get("LS_MODEL_DIR", "nemotron_3_5_lightning")
GIB = 1024 ** 3


def main() -> int:
    import cupy as cp

    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    foreign = [l for l in o.stdout.strip().splitlines()
               if l.strip() and int(l.split(",")[0]) != os.getpid()]
    if foreign:
        print("BLOCKED: another PID holds a CUDA context.")
        return 4

    free0, total = cp.cuda.runtime.memGetInfo()
    rt = LightningRuntime(MODEL_DIR, contexts_max=8192, embed_on_host=True, fp8_kv=True)
    free_shell, _ = cp.cuda.runtime.memGetInfo()
    print(f"shell {(free0 - free_shell) / GIB:.3f} GiB")

    mtp = MTPBlock(rt, max_ctx=8192, concat_order="eh")
    free_mtp, _ = cp.cuda.runtime.memGetInfo()
    print(f"mtp   {(free_shell - free_mtp) / GIB:.3f} GiB "
          f"(experts {(mtp.exp_up.nbytes + mtp.exp_dn.nbytes) / GIB:.3f} GiB)")
    print(f"free  {free_mtp / GIB:.3f} GiB")

    print("\nshapes from the checkpoint:")
    for name in ("mtp.layers.0.eh_proj.weight", "mtp.layers.0.mixer.q_proj.weight",
                 "mtp.layers.1.mixer.gate.weight",
                 "mtp.layers.1.mixer.experts.0.up_proj.weight",
                 "mtp.layers.1.mixer.shared_experts.up_proj.weight"):
        e = rt.index.entries[name]
        print(f"  {name:<52} {e.dtype:<8} {e.shape}")

    h = cp.asarray((cp.random.random(rt.hidden, dtype=cp.float32) - 0.5) * 0.1)
    for order in ("eh", "he"):
        mtp.concat_order = order
        mtp.reset()
        tok, x, y = mtp.forward(1000, h, 0)
        lg = mtp.logits
        finite = bool(cp.all(cp.isfinite(lg)))
        print(f"\norder={order}: draft={tok} finite={finite} "
              f"logit[min,max]=[{float(lg.min()):.3f},{float(lg.max()):.3f}] "
              f"rms(x)={float(cp.sqrt(cp.mean(x * x))):.4f}")
        tok2, _, _ = mtp.forward(tok, x, 1)
        print(f"          chained draft={tok2}")

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
