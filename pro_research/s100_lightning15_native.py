from __future__ import annotations

from dataclasses import dataclass
import types

class TorchMMOut32:
    def __init__(self):
        import torch
        self.torch = torch
        self.call_style = None
        self._preflight()

    def _preflight(self):
        torch = self.torch
        a = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(2, 8, device="cuda", dtype=torch.float32)
        errors = []
        for style in ("keyword", "positional"):
            try:
                if style == "keyword":
                    torch.mm(
                        a, b, out_dtype=torch.float32, out=out
                    )
                else:
                    torch.mm(a, b, torch.float32, out=out)
                torch.cuda.synchronize()
                if out.dtype != torch.float32:
                    raise RuntimeError("torch.mm output is not FP32")
                self.call_style = style
                return
            except Exception as exc:
                errors.append(f"{style}: {type(exc).__name__}: {exc}")
        raise RuntimeError(
            "torch.mm BF16->FP32 out_dtype preflight failed: "
            + " | ".join(errors)
        )

    def __call__(self, a, b, out):
        torch = self.torch
        if self.call_style == "keyword":
            return torch.mm(
                a, b, out_dtype=torch.float32, out=out
            )
        return torch.mm(a, b, torch.float32, out=out)

@dataclass
class Buffers:
    packed: object
    product: object
    residual: object
    decoded: object
    weight: object

class NativeSplitEngine:
    """BF16xN activation decomposition with FP32 output."""
    def __init__(self):
        import torch
        self.torch = torch
        self.mm = TorchMMOut32()
        self.buffers = {}
        self.streams = {}

    def _stream(self, cp):
        pointer = int(cp.cuda.get_current_stream().ptr)
        stream = self.streams.get(pointer)
        if stream is None:
            stream = self.torch.cuda.ExternalStream(pointer)
            self.streams[pointer] = stream
        return stream

    def prepare(self, weight_cp, rows, cols, batch, terms):
        torch = self.torch
        key = (
            int(weight_cp.data.ptr), int(rows), int(cols),
            int(batch), int(terms),
        )
        value = self.buffers.get(key)
        if value is None:
            weight = (
                torch.utils.dlpack.from_dlpack(weight_cp)
                .view(torch.bfloat16)
                .reshape(int(rows), int(cols))
            )
            value = Buffers(
                packed=torch.empty(
                    (batch * terms, cols),
                    device="cuda", dtype=torch.bfloat16,
                ),
                product=torch.empty(
                    (batch * terms, rows),
                    device="cuda", dtype=torch.float32,
                ),
                residual=torch.empty(
                    (batch, cols),
                    device="cuda", dtype=torch.float32,
                ),
                decoded=torch.empty(
                    (batch, cols),
                    device="cuda", dtype=torch.float32,
                ),
                weight=weight,
            )
            self.buffers[key] = value
        return value

    def run(self, weight_cp, x_cp, out_cp, rows, cols, batch, terms):
        torch = self.torch
        cp = __import__("cupy")
        buffers = self.prepare(
            weight_cp, rows, cols, batch, terms
        )
        stream = self._stream(cp)
        x_t = torch.utils.dlpack.from_dlpack(x_cp).reshape(batch, cols)
        out_t = torch.utils.dlpack.from_dlpack(out_cp).reshape(batch, rows)

        with torch.cuda.stream(stream):
            buffers.residual.copy_(x_t)
            for term in range(terms):
                start = term * batch
                stop = start + batch
                current = buffers.packed[start:stop]
                current.copy_(buffers.residual)
                if term + 1 < terms:
                    buffers.decoded.copy_(current)
                    buffers.residual.sub_(buffers.decoded)

            self.mm(
                buffers.packed,
                buffers.weight.t(),
                buffers.product,
            )
            out_t.copy_(buffers.product[:batch])
            for term in range(1, terms):
                start = term * batch
                out_t.add_(buffers.product[start:start + batch])
        return out_cp

class RoundInputERVF:
    """Quality attribution: BF16-round x, retain current ERVF/output."""
    def __init__(self, original):
        import torch
        self.torch = torch
        self.original = original
        self.buffers = {}
        self.streams = {}

    def _stream(self, cp):
        pointer = int(cp.cuda.get_current_stream().ptr)
        stream = self.streams.get(pointer)
        if stream is None:
            stream = self.torch.cuda.ExternalStream(pointer)
            self.streams[pointer] = stream
        return stream

    def __call__(self, out, weight, x, rows, cols):
        import cupy as cp
        torch = self.torch
        key = (int(x.data.ptr), int(cols))
        pair = self.buffers.get(key)
        if pair is None:
            pair = (
                torch.empty(cols, device="cuda", dtype=torch.bfloat16),
                torch.empty(cols, device="cuda", dtype=torch.float32),
            )
            self.buffers[key] = pair
        xb, xf = pair
        stream = self._stream(cp)
        with torch.cuda.stream(stream):
            xt = torch.utils.dlpack.from_dlpack(x)
            xb.copy_(xt)
            xf.copy_(xb)
        return self.original(
            out, weight, cp.from_dlpack(xf), rows, cols
        )

class SelectiveNativeDispatch:
    def __init__(self, rt, mode: str, families: set[str]):
        self.rt = rt
        self.mode = mode
        self.families = set(families)
        self.original = rt.k.mv_bf16
        self.engine = NativeSplitEngine() if mode.startswith("tc") else None
        self.rounder = (
            RoundInputERVF(self.original)
            if mode == "round_ervf" else None
        )
        self.pointer_family = {}
        self.pointer_layer = {}

        for layer in rt.attn_layers:
            data = rt.layer[int(layer)]
            for family, key in (
                ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj"),
            ):
                if key in data:
                    pointer = int(data[key].data.ptr)
                    self.pointer_family[pointer] = family
                    self.pointer_layer[pointer] = int(layer)

        if mode.startswith("tc"):
            self.terms = int(mode[2:])
            if self.terms not in (1, 2, 3):
                raise ValueError(mode)
        else:
            self.terms = None

        self.native_calls = 0
        self.original_calls = 0

    def __call__(self, out, weight, x, rows, cols):
        pointer = int(weight.data.ptr)
        family = self.pointer_family.get(pointer)
        if family not in self.families:
            self.original_calls += 1
            return self.original(out, weight, x, rows, cols)

        self.native_calls += 1
        if self.mode == "round_ervf":
            return self.rounder(out, weight, x, rows, cols)

        return self.engine.run(
            weight, x.reshape(1, cols), out.reshape(1, rows),
            int(rows), int(cols), 1, self.terms,
        )

    def install(self):
        self.rt.k.mv_bf16 = self
        return self
