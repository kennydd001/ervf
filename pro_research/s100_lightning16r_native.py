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

HANDOFFS = {"sync_control", "pair_sync_control"}

class TorchMMOut32:
    def __init__(self):
        import torch

        self.torch = torch
        self.style = None
        a = torch.randn(
            2, 16, device="cuda", dtype=torch.bfloat16
        )
        b = torch.randn(
            16, 8, device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty(
            2, 8, device="cuda", dtype=torch.float32
        )
        errors = []
        for style in ("keyword", "positional"):
            try:
                if style == "keyword":
                    torch.mm(
                        a, b,
                        out_dtype=torch.float32,
                        out=out,
                    )
                else:
                    torch.mm(
                        a, b, torch.float32, out=out
                    )
                torch.cuda.synchronize()
                self.style = style
                break
            except Exception as exc:
                errors.append(
                    f"{style}: {type(exc).__name__}: {exc}"
                )
        if self.style is None:
            raise RuntimeError(
                "BF16->FP32 torch.mm unsupported: "
                + " | ".join(errors)
            )

    def __call__(self, left, right, out):
        if self.style == "keyword":
            return self.torch.mm(
                left,
                right,
                out_dtype=self.torch.float32,
                out=out,
            )
        return self.torch.mm(
            left, right, self.torch.float32, out=out
        )

@dataclass
class Buffers:
    packed: object
    product: object
    residual: object
    decoded: object
    weight: object

class NativeEngine:
    """Native BF16/FP32 matmul after an external handoff decision.

    The caller is responsible for producer synchronization. This keeps the
    synchronization policy visible and countable in PointerDispatch.
    """

    def __init__(self):
        import torch

        self.torch = torch
        self.mm = TorchMMOut32()
        self.buffers = {}
        self.streams = {}

    def stream(self, cp):
        pointer = int(cp.cuda.get_current_stream().ptr)
        stream = self.streams.get(pointer)
        if stream is None:
            stream = self.torch.cuda.ExternalStream(pointer)
            self.streams[pointer] = stream
        return stream

    def prepare(
        self,
        weight_cp,
        rows: int,
        cols: int,
        batch: int,
        terms: int,
    ) -> Buffers:
        torch = self.torch
        key = (
            int(weight_cp.data.ptr),
            int(rows),
            int(cols),
            int(batch),
            int(terms),
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
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                product=torch.empty(
                    (batch * terms, rows),
                    device="cuda",
                    dtype=torch.float32,
                ),
                residual=torch.empty(
                    (batch, cols),
                    device="cuda",
                    dtype=torch.float32,
                ),
                decoded=torch.empty(
                    (batch, cols),
                    device="cuda",
                    dtype=torch.float32,
                ),
                weight=weight,
            )
            self.buffers[key] = value
        return value

    def run(
        self,
        weight_cp,
        x_cp,
        out_cp,
        rows: int,
        cols: int,
        batch: int,
        terms: int,
    ):
        torch = self.torch
        import cupy as cp

        stream = self.stream(cp)
        buffers = self.prepare(
            weight_cp, rows, cols, batch, terms
        )

        # Phase 16 proved that this legacy-order DLPack conversion is safe
        # after an explicit producer synchronization.
        x_t = torch.utils.dlpack.from_dlpack(x_cp).reshape(
            batch, cols
        )
        out_t = torch.utils.dlpack.from_dlpack(out_cp).reshape(
            batch, rows
        )

        with torch.cuda.stream(stream):
            buffers.residual.copy_(x_t)
            for term in range(terms):
                start = term * batch
                current = buffers.packed[
                    start:start + batch
                ]
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
                out_t.add_(
                    buffers.product[start:start + batch]
                )
        return out_cp

class PointerDispatch:
    """K/V/O pointer dispatch with explicit synchronization accounting.

    pair_sync_control skips V's host synchronization only when the immediately
    preceding native call was K from the same attention layer and both are
    enabled. K and V consume the same already-synchronized `normed` tensor and
    are adjacent in LightningRuntime._attention. Any intervening original call
    clears the pairing state.
    """

    def __init__(
        self,
        rt,
        *,
        terms: int = 1,
        handoff: str = "sync_control",
        enabled_cases: set[str] | None = None,
    ):
        if handoff not in HANDOFFS:
            raise ValueError(handoff)
        from s100_lightning16_common import case_manifest

        self.rt = rt
        self.original = rt.k.mv_bf16
        self.engine = NativeEngine()
        self.terms = int(terms)
        self.handoff = str(handoff)
        self.enabled_cases = set(enabled_cases or ())
        self.by_pointer = {
            row["pointer"]: row
            for row in case_manifest(rt)
        }
        self.native_calls = 0
        self.original_calls = 0
        self.sync_calls = 0
        self.paired_sync_elisions = 0
        self.last_native_case: str | None = None

    def reset_counters(self) -> None:
        self.native_calls = 0
        self.original_calls = 0
        self.sync_calls = 0
        self.paired_sync_elisions = 0

    def configure(
        self,
        *,
        terms: int,
        handoff: str,
        enabled_cases,
    ) -> None:
        if handoff not in HANDOFFS:
            raise ValueError(handoff)
        self.terms = int(terms)
        self.handoff = str(handoff)
        self.enabled_cases = set(enabled_cases)
        self.last_native_case = None
        self.reset_counters()

    def _should_sync(self, record: dict) -> bool:
        if self.handoff == "sync_control":
            return True
        if record["family"] != "v":
            return True
        paired_k = (
            f"attention_{record['layer']}_k"
        )
        can_pair = (
            paired_k in self.enabled_cases
            and self.last_native_case == paired_k
        )
        if can_pair:
            self.paired_sync_elisions += 1
            return False
        return True

    def __call__(self, out, weight, x, rows, cols):
        import cupy as cp

        record = self.by_pointer.get(int(weight.data.ptr))
        if (
            record is None
            or record["case"] not in self.enabled_cases
        ):
            self.original_calls += 1
            self.last_native_case = None
            return self.original(
                out, weight, x, rows, cols
            )

        if self._should_sync(record):
            cp.cuda.get_current_stream().synchronize()
            self.sync_calls += 1

        self.native_calls += 1
        result = self.engine.run(
            weight,
            x.reshape(1, cols),
            out.reshape(1, rows),
            int(rows),
            int(cols),
            1,
            self.terms,
        )
        self.last_native_case = record["case"]
        return result

    def install(self):
        self.rt.k.mv_bf16 = self
        return self
