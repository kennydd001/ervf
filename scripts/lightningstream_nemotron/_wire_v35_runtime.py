"""Wire the runtime for 3.5 Lightning's three weight formats.

Nano had nvfp4 + bf16. 3.5 Lightning adds fp8_tensor for the Mamba projections
and moves lm_head from BF16 to NVFP4 (704 MB -> 198 MB). Both are handled by
dispatching on loader.quant_kind() instead of a boolean.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py"

src = R.read_text(encoding="utf-8")
changed = []

# --- shell load: three-way for mamba in_proj/out_proj ---------------------
old_in = '''                d["in_q"] = idx.is_quantized(f"{m}.in_proj")
                if d["in_q"]:
                    d["in_codes"] = self._u8(f"{m}.in_proj.weight")
                    d["in_scales"] = self._u8(f"{m}.in_proj.weight_scale")
                    d["in_g"] = idx.get_scalar(f"{m}.in_proj.weight_scale_2")
                else:
                    d["in_w"] = self._u16(f"{m}.in_proj.weight")'''
new_in = '''                d["in_k"] = idx.quant_kind(f"{m}.in_proj")
                d["in_q"] = d["in_k"] == "nvfp4"
                if d["in_k"] == "nvfp4":
                    d["in_codes"] = self._u8(f"{m}.in_proj.weight")
                    d["in_scales"] = self._u8(f"{m}.in_proj.weight_scale")
                    d["in_g"] = idx.get_scalar(f"{m}.in_proj.weight_scale_2")
                elif d["in_k"] == "fp8_tensor":
                    d["in_w8"] = self._u8(f"{m}.in_proj.weight")
                    d["in_s"] = idx.get_scalar(f"{m}.in_proj.weight_scale")
                else:
                    d["in_w"] = self._u16(f"{m}.in_proj.weight")'''

old_out = '''                d["out_q"] = idx.is_quantized(f"{m}.out_proj")
                if d["out_q"]:
                    d["out_codes"] = self._u8(f"{m}.out_proj.weight")
                    d["out_scales"] = self._u8(f"{m}.out_proj.weight_scale")
                    d["out_g"] = idx.get_scalar(f"{m}.out_proj.weight_scale_2")
                else:
                    d["out_w"] = self._u16(f"{m}.out_proj.weight")'''
new_out = '''                d["out_k"] = idx.quant_kind(f"{m}.out_proj")
                d["out_q"] = d["out_k"] == "nvfp4"
                if d["out_k"] == "nvfp4":
                    d["out_codes"] = self._u8(f"{m}.out_proj.weight")
                    d["out_scales"] = self._u8(f"{m}.out_proj.weight_scale")
                    d["out_g"] = idx.get_scalar(f"{m}.out_proj.weight_scale_2")
                elif d["out_k"] == "fp8_tensor":
                    d["out_w8"] = self._u8(f"{m}.out_proj.weight")
                    d["out_s"] = idx.get_scalar(f"{m}.out_proj.weight_scale")
                else:
                    d["out_w"] = self._u16(f"{m}.out_proj.weight")'''

for old, new, tag in ((old_in, new_in, "mamba in_proj"), (old_out, new_out, "mamba out_proj")):
    if old in src:
        src = src.replace(old, new, 1)
        changed.append(tag)

# --- lm_head: NVFP4 on 3.5 Lightning --------------------------------------
old_head = '''        self.lm_head = self._u16("lm_head.weight")'''
new_head = '''        self.lm_head_kind = idx.quant_kind("lm_head")
        if self.lm_head_kind == "nvfp4":
            # 3.5 Lightning ships lm_head as NVFP4: 198 MB instead of 704 MB.
            self.lm_head_codes = self._u8("lm_head.weight")
            self.lm_head_scales = self._u8("lm_head.weight_scale")
            self.lm_head_g = idx.get_scalar("lm_head.weight_scale_2")
            self.lm_head = None
        else:
            self.lm_head = self._u16("lm_head.weight")'''
if old_head in src:
    src = src.replace(old_head, new_head, 1)
    changed.append("lm_head")

# --- mamba forward dispatch ----------------------------------------------
old_mf = '''        if d["in_q"]:
            self.fused.gemv_into(self.proj, d["in_codes"], d["in_scales"], self.normed,
                                 d["in_g"], self.proj.size, self.hidden)
        else:
            k.mv_bf16(self.proj, d["in_w"], self.normed, self.proj.size, self.hidden)'''
new_mf = '''        if d["in_k"] == "nvfp4":
            self.fused.gemv_into(self.proj, d["in_codes"], d["in_scales"], self.normed,
                                 d["in_g"], self.proj.size, self.hidden)
        elif d["in_k"] == "fp8_tensor":
            k.mv_fp8_tensor(self.proj, d["in_w8"], self.normed, d["in_s"],
                            self.proj.size, self.hidden)
        else:
            k.mv_bf16(self.proj, d["in_w"], self.normed, self.proj.size, self.hidden)'''
if old_mf in src:
    src = src.replace(old_mf, new_mf, 1)
    changed.append("mamba in forward")

old_of = '''        if d["out_q"]:
            self.fused.gemv_into(out, d["out_codes"], d["out_scales"], self.gn,
                                 d["out_g"], self.hidden, self.d_inner)
        else:
            k.mv_bf16(out, d["out_w"], self.gn, self.hidden, self.d_inner)'''
new_of = '''        if d["out_k"] == "nvfp4":
            self.fused.gemv_into(out, d["out_codes"], d["out_scales"], self.gn,
                                 d["out_g"], self.hidden, self.d_inner)
        elif d["out_k"] == "fp8_tensor":
            k.mv_fp8_tensor(out, d["out_w8"], self.gn, d["out_s"],
                            self.hidden, self.d_inner)
        else:
            k.mv_bf16(out, d["out_w"], self.gn, self.hidden, self.d_inner)'''
if old_of in src:
    src = src.replace(old_of, new_of, 1)
    changed.append("mamba out forward")

# --- lm_head forward ------------------------------------------------------
old_lmf = '''        k.mv_bf16(self.logits, self.lm_head, self.normed, self.vocab, self.hidden)'''
new_lmf = '''        if self.lm_head_kind == "nvfp4":
            self.fused.gemv_into(self.logits, self.lm_head_codes, self.lm_head_scales,
                                 self.normed, self.lm_head_g, self.vocab, self.hidden)
        else:
            k.mv_bf16(self.logits, self.lm_head, self.normed, self.vocab, self.hidden)'''
if old_lmf in src:
    src = src.replace(old_lmf, new_lmf, 1)
    changed.append("lm_head forward")

R.write_text(src, encoding="utf-8")
print("wired:", ", ".join(changed) if changed else "nothing (already wired?)")
