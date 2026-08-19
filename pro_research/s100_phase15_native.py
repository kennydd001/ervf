from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BF16Case:
    name: str
    family: str
    W: object
    rows: int
    cols: int
    weight_bytes: int

def collect_bf16_cases(rt):
    cases = []
    for layer in rt.mamba_layers:
        layer = int(layer)
        d = rt.layer[layer]
        if d.get("in_k") == "bf16":
            cases.append(BF16Case(
                f"mamba_{layer}_in", "mamba_in", d["in_w"],
                int(rt.proj.size), int(rt.hidden), int(d["in_w"].nbytes)
            ))
        if d.get("out_k") == "bf16":
            cases.append(BF16Case(
                f"mamba_{layer}_out", "mamba_out", d["out_w"],
                int(rt.hidden), int(rt.d_inner), int(d["out_w"].nbytes)
            ))

    for layer in rt.attn_layers:
        layer = int(layer)
        d = rt.layer[layer]
        hq = int(rt.n_heads * rt.head_dim)
        if d.get("q_kind", "bf16") == "bf16" and "q_proj" in d:
            cases.append(BF16Case(
                f"attention_{layer}_q", "attention", d["q_proj"],
                hq, int(rt.hidden), int(d["q_proj"].nbytes)
            ))
        for side, rows, cols in (
            ("k", int(rt.kv_dim), int(rt.hidden)),
            ("v", int(rt.kv_dim), int(rt.hidden)),
            ("o", int(rt.hidden), hq),
        ):
            key = f"{side}_proj"
            if key in d:
                cases.append(BF16Case(
                    f"attention_{layer}_{side}", "attention", d[key],
                    rows, cols, int(d[key].nbytes)
                ))

    if getattr(rt, "lm_head_kind", None) != "nvfp4" and hasattr(rt, "lm_head"):
        cases.append(BF16Case(
            "lm_head", "lm_head", rt.lm_head,
            int(rt.vocab), int(rt.hidden), int(rt.lm_head.nbytes)
        ))
    return cases

def case_map(rt):
    return {
        int(c.W.data.ptr): c
        for c in collect_bf16_cases(rt)
    }

def scope_match(case, scope: str) -> bool:
    if scope == "all":
        return True
    if case is None:
        return False
    if scope == "mamba":
        return case.family in ("mamba_in", "mamba_out")
    if scope in ("mamba_in", "mamba_out", "attention", "lm_head"):
        return case.family == scope
    if scope.startswith("name:"):
        return case.name == scope.split(":", 1)[1]
    raise ValueError(f"unknown scope {scope}")

class NativeBF16VariantDispatch:
    """Torch native BF16 path using the exact D2 MM geometry.

    We cache W^T as contiguous BF16 because D2 measured torch.mm(x_bf16, W^T).
    `mm_fp32out` requests FP32 output directly from torch.mm when supported.
    `mm_fp32out_comp2` decomposes FP32 x into two BF16 terms:
        x ~= bf16(x) + bf16(x - float(bf16(x)))
    and sums the two FP32 GEMM outputs.
    """
    def __init__(self, rt, variant="mm_fp32out", scope="all"):
        import torch
        self.rt = rt
        self.torch = torch
        self.cp = rt.cp
        self.original = rt.k.mv_bf16
        self.variant = variant
        self.scope = scope
        self.cases = case_map(rt)
        self.weights_t = {}
        self.calls_native = 0
        self.calls_original = 0
        self.fp32_out_supported = None

    def set_scope(self, scope):
        self.scope = scope

    def _weight_t(self, W, rows, cols):
        key = (int(W.data.ptr), int(rows), int(cols))
        wt = self.weights_t.get(key)
        if wt is None:
            raw = (
                self.torch.utils.dlpack.from_dlpack(W)
                .view(self.torch.bfloat16)
                .reshape(int(rows), int(cols))
            )
            wt = raw.t().contiguous()
            self.weights_t[key] = wt
        return wt

    def _mm_fp32(self, a, b):
        try:
            y = self.torch.mm(a, b, out_dtype=self.torch.float32)
            self.fp32_out_supported = True
            return y
        except TypeError:
            self.fp32_out_supported = False
            raise RuntimeError(
                "installed torch.mm does not expose CUDA BF16 -> FP32 out_dtype"
            )

    def __call__(self, out, W, x, rows, cols):
        case = self.cases.get(int(W.data.ptr))
        if not scope_match(case, self.scope):
            self.calls_original += 1
            return self.original(out, W, x, rows, cols)

        torch = self.torch
        cp = self.cp
        wtt = self._weight_t(W, rows, cols)
        stream = torch.cuda.ExternalStream(cp.cuda.get_current_stream().ptr)
        with torch.cuda.stream(stream):
            xt = torch.utils.dlpack.from_dlpack(x)
            if self.variant == "mm_bf16out":
                xb = xt.to(torch.bfloat16).reshape(1, -1)
                y = torch.mm(xb, wtt).float().reshape(-1)
            elif self.variant == "mm_fp32out":
                xb = xt.to(torch.bfloat16).reshape(1, -1)
                y = self._mm_fp32(xb, wtt).reshape(-1)
            elif self.variant == "mm_fp32out_comp2":
                hi = xt.to(torch.bfloat16)
                lo = (xt - hi.float()).to(torch.bfloat16)
                y = (
                    self._mm_fp32(hi.reshape(1, -1), wtt)
                    + self._mm_fp32(lo.reshape(1, -1), wtt)
                ).reshape(-1)
            else:
                raise ValueError(self.variant)
            torch.utils.dlpack.from_dlpack(out).copy_(y)
        self.calls_native += 1
        return None

def make_runtime(variant=None, scope="all", contexts_max=4096):
    import os
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    model = Path(os.environ["LS_MODEL_DIR"])
    rt = LightningRuntime(
        model, contexts_max=contexts_max, embed_on_host=True,
        fp8_kv=True, verbose=False,
    )
    rt.load_routed_bank()
    rt.deterministic_accum = True
    dispatch = None
    if variant is not None:
        dispatch = NativeBF16VariantDispatch(rt, variant=variant, scope=scope)
        rt.k.mv_bf16 = dispatch
    return rt, dispatch

def release_runtime(rt):
    import cupy as cp
    import torch
    try:
        rt.bank = {}
        rt.cache = {}
        rt._dev_cache = {}
    except Exception:
        pass
    del rt
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

def prompt_rows(split):
    import json
    import os
    from transformers import AutoTokenizer

    from common import REPO
    rows = json.loads(
        (REPO / "pro_research" / "S100_PHASE3_PROMPTS.json")
        .read_text(encoding="utf-8")
    )["prompts"]
    tok = AutoTokenizer.from_pretrained(
        os.environ["LS_MODEL_DIR"], local_files_only=True,
        trust_remote_code=True, use_fast=True,
    )
    suffixes = {
        "calibration": ("_01",),
        "validation": ("_02",),
        "heldout": ("_03", "_04"),
    }[split]
    out = []
    for row in rows:
        if row["id"].endswith(suffixes):
            out.append({
                "id": row["id"],
                "domain": row["domain"],
                "prompt": row["prompt"],
                "prompt_ids": [int(x) for x in tok.encode(
                    row["prompt"], add_special_tokens=False
                )],
            })
    return out

def logsumexp_np(x):
    import numpy as np
    x = np.asarray(x, np.float64)
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))

def snapshot_recurrent_state(rt):
    return {
        "pos": int(rt.pos),
        "conv": {int(k): v.copy() for k, v in rt.conv.items()},
        "ssm": {int(k): v.copy() for k, v in rt.ssm.items()},
        "kc": {int(k): v.copy() for k, v in rt.kc.items()},
        "vc": {int(k): v.copy() for k, v in rt.vc.items()},
    }

def restore_recurrent_state(rt, snap):
    rt.pos = int(snap["pos"])
    for name in ("conv", "ssm", "kc", "vc"):
        target = getattr(rt, name)
        for k, src in snap[name].items():
            target[int(k)][...] = src
