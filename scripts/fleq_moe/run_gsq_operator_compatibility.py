from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from lion_pytorch import Lion

from moe_lab.fleq_moe.gsq_bridge import (
    GSQ_COMMIT,
    GSQ_FILE_HASHES,
    load_official_quantizer_module,
)
from moe_lab.reporting import ROOT


GSQ_ROOT = ROOT / "third_party" / "GSQ"
OUTPUT = ROOT / "reports" / "fleq_moe" / "gsq_operator_compatibility.json"
SEED = 260811


def run_once(kind: str) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rows, columns, groupsize = 64, 128, 128
    scales = torch.linspace(0.01, 0.04, rows, device=device).unsqueeze(1)
    if kind == "2bit":
        codes = torch.randint(-2, 2, (rows, columns), device=device)
        q = (codes * scales).to(dtype)
        class_name = "GumbelQuantizer2Bit"
    else:
        codes = torch.randint(-1, 2, (rows, columns), device=device)
        q = (codes * scales).to(dtype)
        class_name = "GumbelQuantizerTernary"

    module = load_official_quantizer_module(GSQ_ROOT, kind)
    quantizer_class = getattr(module, class_name)
    quantizer = quantizer_class(
        q,
        scales,
        groupsize,
        0.01,
        6,
        device,
        dtype,
        logits_dtype=dtype,
    )
    target = torch.randn(rows, columns, device=device, dtype=dtype) * 0.02
    x = torch.randn(32, columns, device=device, dtype=dtype)
    reference = torch.nn.functional.linear(x, target)
    logit_params = [p for name, p in quantizer.named_parameters() if name != "scales"]
    optimizer = Lion(
        [
            {"params": logit_params, "lr": 1e-4, "weight_decay": 1.0},
            {"params": quantizer.scales, "lr": 5e-5, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    losses = []
    gradients_finite = True
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(4):
        temperature = 2.0 + (0.05 - 2.0) * step / 3
        strength = 100.0 + (500.0 - 100.0) * step / 3
        optimizer.zero_grad(set_to_none=True)
        weight = quantizer(temperature, strength)
        loss = torch.nn.functional.mse_loss(
            torch.nn.functional.linear(x, weight), reference
        )
        loss.backward()
        gradients_finite = gradients_finite and all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in quantizer.parameters()
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    hard, learned_scales = quantizer.get_hard_weights()
    result = {
        "kind": kind,
        "losses": losses,
        "gradients_finite": gradients_finite,
        "hard_weights_finite": bool(torch.isfinite(hard).all()),
        "scales_finite": bool(torch.isfinite(learned_scales).all()),
        "hard_weight_sha256": __import__("hashlib").sha256(
            hard.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del optimizer, quantizer, hard, learned_scales, target, x, reference, q, scales
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pinned GSQ custom autograd operator")
    first = {kind: run_once(kind) for kind in ("2bit", "ternary")}
    second = {kind: run_once(kind) for kind in ("2bit", "ternary")}
    deterministic = {
        kind: first[kind]["hard_weight_sha256"] == second[kind]["hard_weight_sha256"]
        and first[kind]["losses"] == second[kind]["losses"]
        for kind in first
    }
    payload = {
        "kind": "fleq_moe_gsq_operator_compatibility",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "gsq_commit": GSQ_COMMIT,
        "gsq_file_hashes": GSQ_FILE_HASHES,
        "seed": SEED,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "process_rss_bytes": psutil.Process().memory_info().rss,
        },
        "first": first,
        "repeat": second,
        "deterministic": deterministic,
        "all_required_controls_pass": all(deterministic.values())
        and all(
            row["gradients_finite"]
            and row["hard_weights_finite"]
            and row["scales_finite"]
            for row in first.values()
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

