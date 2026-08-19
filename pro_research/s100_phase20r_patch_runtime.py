from __future__ import annotations

import json
from pathlib import Path

from common import REPO, write_json_atomic, utc_now

RESULT = REPO / "pro_research" / "results" / "s100_phase20r" / "S100_PHASE20R_PATCH.json"
RUNTIME = REPO / "src" / "moe_lab" / "lightningstream_nemotron" / "runtime.py"
IDENTITY = REPO / "pro_research" / "s100_phase20a_identity.py"

LOAD_OLD = '''            elif ch == "*":\n                m = f"{p}.mixer"\n                for n in ("q_proj", "k_proj", "v_proj", "o_proj"):\n                    d[n] = self._u16(f"{m}.{n}.weight")\n'''
LOAD_NEW = '''            elif ch == "*":\n                m = f"{p}.mixer"\n                for n in ("q_proj", "k_proj", "v_proj", "o_proj"):\n                    d[n] = self._u16(f"{m}.{n}.weight")\n                # Phase20R: calibrated FP8 KV-cache dequant scales are target tensors.\n                d["k_scale"] = idx.get_scalar(f"{m}.k_scale") if f"{m}.k_scale" in idx else 1.0\n                d["v_scale"] = idx.get_scalar(f"{m}.v_scale") if f"{m}.v_scale" in idx else 1.0\n'''

ATTN_OLD = '''        scale = 1.0 / float(np.sqrt(self.head_dim))\n        if self.fp8_kv and self.graph_mode:\n'''
ATTN_NEW = '''        scale = 1.0 / float(np.sqrt(self.head_dim))\n        # Phase20R: checkpoint-calibrated scalar FP8 KV scales. Cache stores\n        # K/k_scale and V/v_scale. Scaling Q and final context is algebraically\n        # equivalent to dequantizing K/V inside the attention kernel.\n        if self.fp8_kv:\n            _ks = float(d.get("k_scale", 1.0))\n            _vs = float(d.get("v_scale", 1.0))\n            self.qv *= _ks\n            self.kv_ *= (1.0 / _ks)\n            self.vv *= (1.0 / _vs)\n        else:\n            _ks = _vs = 1.0\n        if self.fp8_kv and self.graph_mode:\n'''

GRAPH_OLD = '''            k.attention_fp8_gqa4_dp(self.ctx, self.qv, self.kc[i], self.vc[i],\n                                    self._pos_dev, self.n_heads, self.head_dim,\n                                    self.groups, self.max_ctx, scale,\n                                    self.part_acc, self.part_ml)\n            k.mv_bf16(out, d["o_proj"], self.ctx, self.hidden,\n                      self.n_heads * self.head_dim)\n'''
GRAPH_NEW = '''            k.attention_fp8_gqa4_dp(self.ctx, self.qv, self.kc[i], self.vc[i],\n                                    self._pos_dev, self.n_heads, self.head_dim,\n                                    self.groups, self.max_ctx, scale,\n                                    self.part_acc, self.part_ml)\n            self.ctx *= _vs\n            k.mv_bf16(out, d["o_proj"], self.ctx, self.hidden,\n                      self.n_heads * self.head_dim)\n'''

EAGER_OLD = '''            self.attn(self.ctx, self.qv, self.kc[i], self.vc[i], t,\n                      self.n_heads, self.head_dim, self.groups,\n                      self.max_ctx, scale, self.part_acc, self.part_ml)\n        else:\n'''
EAGER_NEW = '''            self.attn(self.ctx, self.qv, self.kc[i], self.vc[i], t,\n                      self.n_heads, self.head_dim, self.groups,\n                      self.max_ctx, scale, self.part_acc, self.part_ml)\n            self.ctx *= _vs\n        else:\n'''

AUDIT_OLD = '''        elif kind == "attention":\n            for field in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):\n                take(f"{m}.{field}")\n'''
AUDIT_NEW = '''        elif kind == "attention":\n            for field in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):\n                take(f"{m}.{field}")\n            # Phase20R: these are calibrated target KV-cache tensors, not metadata.\n            take(f"{m}.k_scale")\n            take(f"{m}.v_scale")\n'''


def replace_once(text, old, new, label):
    if new in text:
        return text, "already_patched"
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one patch point, found {text.count(old)}")
    return text.replace(old, new), "patched"


def main():
    audit = json.loads((REPO/"pro_research"/"results"/"s100_phase20r"/"S100_PHASE20R_KVSCALE_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("KVSCALE_SEMANTICS_GREEN") is not True:
        raise RuntimeError("refusing production patch: KV-scale semantics gate not green")

    rt = RUNTIME.read_text(encoding="utf-8-sig")
    statuses = {}
    for old,new,label in ((LOAD_OLD,LOAD_NEW,"scale_load"),(ATTN_OLD,ATTN_NEW,"attention_preamble"),
                          (GRAPH_OLD,GRAPH_NEW,"graph_context_scale"),(EAGER_OLD,EAGER_NEW,"eager_context_scale")):
        rt,status=replace_once(rt,old,new,label); statuses[label]=status
    RUNTIME.write_text(rt,encoding="utf-8",newline="\n")
    compile(rt,str(RUNTIME),"exec")

    ident = IDENTITY.read_text(encoding="utf-8")
    ident,status=replace_once(ident,AUDIT_OLD,AUDIT_NEW,"consumption_audit")
    statuses["consumption_audit"]=status
    IDENTITY.write_text(ident,encoding="utf-8",newline="\n")
    compile(ident,str(IDENTITY),"exec")

    out={"kind":"s100_phase20r_patch","created_utc":utc_now(),"runtime":str(RUNTIME),
         "identity_script":str(IDENTITY),"statuses":statuses,"PATCH_APPLIED":True}
    write_json_atomic(RESULT,out,archive=True)
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
