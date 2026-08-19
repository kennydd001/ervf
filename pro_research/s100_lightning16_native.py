from __future__ import annotations

from dataclasses import dataclass

CUBLAS_WARMUP_DIAG = {
    "attempted": False,
    "ok": None,
    "error": None,
    "torch_version": None,
    "torch_cuda_version": None,
    "cupy_version": None,
    "cuda_runtime_version": None,
    "cuda_available": None,
    "purpose": (
        "Bind torch's cuBLAS BF16->FP32 handle before CuPy loads its "
        "own cuBLAS DLLs. Without it cublasGemmEx fails with "
        "CUBLAS_STATUS_INVALID_VALUE on torch 2.9.1+cu128 / "
        "cupy 14.1.1. Diagnostic bridge only: a production "
        "captureable path must not depend on Torch at all."
    ),
}

def _warm_cublas_before_cupy():
    """Run one tiny BF16->FP32 mm before CuPy's first CUDA use.

    Every outcome is recorded in CUBLAS_WARMUP_DIAG and surfaced in
    the phase JSON payloads; this must never fail silently.
    """
    diag = CUBLAS_WARMUP_DIAG
    diag["attempted"] = True
    try:
        import torch
        diag["torch_version"] = torch.__version__
        diag["torch_cuda_version"] = torch.version.cuda
        diag["cuda_available"] = bool(torch.cuda.is_available())
        if not diag["cuda_available"]:
            diag["ok"] = False
            diag["error"] = "cuda_unavailable"
        else:
            a = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(2, 8, device="cuda", dtype=torch.float32)
            torch.mm(a, b, out_dtype=torch.float32, out=out)
            torch.cuda.synchronize()
            diag["ok"] = True
    except Exception as exc:
        diag["ok"] = False
        diag["error"] = f"{type(exc).__name__}: {exc}"
    # Version probe AFTER the mm so CuPy cannot win the cuBLAS race.
    try:
        import cupy as cp
        diag["cupy_version"] = cp.__version__
        diag["cuda_runtime_version"] = int(
            cp.cuda.runtime.runtimeGetVersion()
        )
    except Exception as exc:
        diag["cupy_probe_error"] = f"{type(exc).__name__}: {exc}"

_warm_cublas_before_cupy()

class TorchMMOut32:
    def __init__(self):
        import torch
        self.torch = torch
        self.style = None
        a = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(2, 8, device="cuda", dtype=torch.float32)
        errors = []
        for style in ("keyword", "positional"):
            try:
                if style == "keyword":
                    torch.mm(a, b, out_dtype=torch.float32, out=out)
                else:
                    torch.mm(a, b, torch.float32, out=out)
                torch.cuda.synchronize()
                self.style = style
                break
            except Exception as exc:
                errors.append(f"{style}: {type(exc).__name__}: {exc}")
        if self.style is None:
            raise RuntimeError(
                "BF16->FP32 torch.mm unsupported: " + " | ".join(errors)
            )

    def __call__(self, a, b, out):
        if self.style == "keyword":
            return self.torch.mm(
                a, b, out_dtype=self.torch.float32, out=out
            )
        return self.torch.mm(a, b, self.torch.float32, out=out)

@dataclass
class Buffers:
    packed: object
    product: object
    residual: object
    decoded: object
    weight: object

class NativeEngine:
    def __init__(self, handoff: str = "context_first"):
        if handoff not in {"legacy", "context_first", "sync_control"}:
            raise ValueError(handoff)
        import torch
        self.torch = torch
        self.handoff = handoff
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
                .reshape(rows, cols)
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

    def _compute(self, buffers, x_t, out_t, batch, terms):
        buffers.residual.copy_(x_t)
        for term in range(terms):
            start = term * batch
            current = buffers.packed[start:start + batch]
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

    def run(self, weight_cp, x_cp, out_cp, rows, cols, batch, terms):
        import cupy as cp
        torch = self.torch
        stream = self._stream(cp)
        buffers = self.prepare(
            weight_cp, rows, cols, batch, terms
        )

        if self.handoff == "sync_control":
            cp.cuda.get_current_stream().synchronize()

        if self.handoff == "context_first":
            with torch.cuda.stream(stream):
                # Critical difference from Phase 15: the DLPack consumer is
                # created while the actual consumer stream is current.
                x_t = torch.utils.dlpack.from_dlpack(x_cp).reshape(
                    batch, cols
                )
                out_t = torch.utils.dlpack.from_dlpack(out_cp).reshape(
                    batch, rows
                )
                self._compute(buffers, x_t, out_t, batch, terms)
        else:
            # Legacy and sync-control intentionally preserve Phase-15 order.
            x_t = torch.utils.dlpack.from_dlpack(x_cp).reshape(
                batch, cols
            )
            out_t = torch.utils.dlpack.from_dlpack(out_cp).reshape(
                batch, rows
            )
            with torch.cuda.stream(stream):
                self._compute(buffers, x_t, out_t, batch, terms)
        return out_cp

class PointerDispatch:
    def __init__(
        self, rt, *, terms: int = 2,
        handoff: str = "context_first",
        enabled_cases: set[str] | None = None,
    ):
        from s100_lightning16_common import case_manifest
        self.rt = rt
        self.original = rt.k.mv_bf16
        self.engine = NativeEngine(handoff)
        self.terms = int(terms)
        self.enabled_cases = set(enabled_cases or ())
        self.by_pointer = {
            row["pointer"]: row for row in case_manifest(rt)
        }
        self.native_calls = 0
        self.original_calls = 0

    def __call__(self, out, weight, x, rows, cols):
        record = self.by_pointer.get(int(weight.data.ptr))
        if record is None or record["case"] not in self.enabled_cases:
            self.original_calls += 1
            return self.original(out, weight, x, rows, cols)
        self.native_calls += 1
        return self.engine.run(
            weight, x.reshape(1, cols), out.reshape(1, rows),
            int(rows), int(cols), 1, self.terms,
        )

    def install(self):
        self.rt.k.mv_bf16 = self
        return self
