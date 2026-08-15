"""Remove the duplicate GQA kernel I added before noticing Kimi's S7 build.

Kimi already implemented attn_decode_warp_fp8_gqa / attention_fp8_gqa and wired
runtime._attention to it. My attn_decode_gqa_fp8 / attention_gqa_fp8 were never
registered, so they are inert -- but duplicate dead code in a research repo is
worse than no code. This deletes exactly my additions and leaves Kimi's build
untouched.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / "gpu_kernels.py"
R = ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py"

src = K.read_text(encoding="utf-8")

start = src.find("// ---- S7: GQA-grouped attention ---")
end = src.find("// Warp-per-position flash decoding over an FP8 KV cache.")
if start != -1 and end != -1 and start < end:
    src = src[:start] + src[end:]
    print("removed duplicate CUDA kernel block")
else:
    print("kernel block not found (already clean)")

m = re.search(r"    def attention_gqa_fp8\(.*?\n\n", src, flags=re.S)
if m:
    src = src[:m.start()] + src[m.end():]
    print("removed duplicate attention_gqa_fp8 wrapper")
else:
    print("wrapper not found (already clean)")

K.write_text(src, encoding="utf-8")

rt = R.read_text(encoding="utf-8")
before = rt
rt = rt.replace(", gqa_attn: bool = True):", "):")
rt = rt.replace("        self.gqa_attn = gqa_attn\n", "")
if rt != before:
    print("removed unused gqa_attn flag from runtime")
else:
    print("runtime already clean")
R.write_text(rt, encoding="utf-8")

# sanity: Kimi's names must still be present
for needle in ("attn_decode_warp_fp8_gqa", "def attention_fp8_gqa"):
    print(f"  kept: {needle}: {needle in K.read_text(encoding='utf-8')}")
print(f"  runtime calls attention_fp8_gqa: "
      f"{'attention_fp8_gqa' in R.read_text(encoding='utf-8')}")
