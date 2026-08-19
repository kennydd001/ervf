from __future__ import annotations

from dataclasses import dataclass
import numpy as np

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
                f"mamba_{layer}_in", "mamba", d["in_w"],
                int(rt.proj.size), int(rt.hidden), int(d["in_w"].nbytes)
            ))
        if d.get("out_k") == "bf16":
            cases.append(BF16Case(
                f"mamba_{layer}_out", "mamba", d["out_w"],
                int(rt.hidden), int(rt.d_inner), int(d["out_w"].nbytes)
            ))

    for layer in rt.attn_layers:
        layer = int(layer)
        d = rt.layer[layer]
        hq = int(rt.n_heads * rt.head_dim)
        # QFAST may replace Q. Do not reintroduce BF16 Q if the live candidate
        # says Q is quantized.
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

class NativeBF16Dispatch:
    """Eager numerical-fidelity dispatcher.

    It deliberately does not claim production graph compatibility. It shares
    the checkpoint BF16 weight storage through DLPack and changes only the
    native matrix reduction path.
    """
    def __init__(self, rt):
        import torch
        self.rt = rt
        self.torch = torch
        self.cp = rt.cp
        self.original = rt.k.mv_bf16
        self.weights = {}
        self.calls = 0

    def _weight(self, W, rows, cols):
        key = (int(W.data.ptr), int(rows), int(cols))
        wt = self.weights.get(key)
        if wt is None:
            wt = (
                self.torch.utils.dlpack.from_dlpack(W)
                .view(self.torch.bfloat16)
                .reshape(int(rows), int(cols))
            )
            self.weights[key] = wt
        return wt

    def __call__(self, out, W, x, rows, cols):
        torch = self.torch
        cp = self.cp
        wt = self._weight(W, rows, cols)
        stream = torch.cuda.ExternalStream(cp.cuda.get_current_stream().ptr)
        with torch.cuda.stream(stream):
            xt = torch.utils.dlpack.from_dlpack(x)
            yt = torch.mv(wt, xt.to(torch.bfloat16)).float()
            ot = torch.utils.dlpack.from_dlpack(out)
            ot.copy_(yt)
        self.calls += 1
