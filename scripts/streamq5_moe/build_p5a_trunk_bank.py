from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from moe_lab.reporting import ROOT


MODEL = ROOT / "models/qwen3-30b-a3b-base"
R = ROOT / "reports/streamq5_moe"
PREREG = R / "P5A_PHYSICAL_TRUNK_PREREGISTRATION.md"
OUT_DIR = ROOT / "reports/runs/streamq5_moe/p5a_int8_trunk_bank"
RESULT = R / "p5a_trunk_bank_result.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_specs():
    specs = []
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        specs.extend([
            (layer, "q", f"{prefix}.self_attn.q_proj.weight", 4096, 2048),
            (layer, "k", f"{prefix}.self_attn.k_proj.weight", 512, 2048),
            (layer, "v", f"{prefix}.self_attn.v_proj.weight", 512, 2048),
            (layer, "o", f"{prefix}.self_attn.o_proj.weight", 2048, 4096),
            (layer, "router", f"{prefix}.mlp.gate.weight", 128, 2048),
        ])
    specs.append((48, "head", "lm_head.weight", 151936, 2048))
    return specs


def quantize(value: torch.Tensor):
    rows, cols = value.shape
    codes = np.empty((rows, cols), dtype=np.int8)
    scale_bits = np.empty((rows, cols // 128), dtype="<u2")
    for begin in range(0, rows, 512):
        end = min(rows, begin + 512)
        work = value[begin:end].float().reshape(end - begin, cols // 128, 128)
        maximum = work.abs().amax(dim=-1)
        scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        stored = scale.to(torch.bfloat16)
        q = torch.round(work / stored.float().unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
        codes[begin:end] = q.reshape(end - begin, cols).numpy()
        scale_bits[begin:end] = stored.view(torch.int16).numpy().astype("<u2", copy=False)
    return codes, scale_bits


def main():
    if OUT_DIR.exists() or RESULT.exists():
        raise FileExistsError("refusing to overwrite P5A trunk bank")
    OUT_DIR.mkdir(parents=True)
    index = json.loads((MODEL / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    specs = matrix_specs()
    by_file = {}
    for spec in specs:
        by_file.setdefault(index[spec[2]], []).append(spec)
    records = []
    started = time.perf_counter()
    total_weights = total_codes = total_scales = total_bytes = 0
    for shard, shard_specs in by_file.items():
        with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
            for layer, name, key, rows, cols in shard_specs:
                value = handle.get_tensor(key)
                if tuple(value.shape) != (rows, cols):
                    raise ValueError(f"shape mismatch for {key}")
                codes, scale_bits = quantize(value)
                path = OUT_DIR / (f"layer_{layer:02d}_{name}.q8bin" if layer < 48 else "lm_head.q8bin")
                with path.open("xb") as output:
                    output.write(codes.tobytes(order="C"))
                    output.write(scale_bits.tobytes(order="C"))
                code_bytes = codes.nbytes; scale_bytes = scale_bits.nbytes
                records.append({"layer":layer,"name":name,"source_key":key,"source_shard":shard,"rows":rows,"cols":cols,"groups":rows*(cols//128),"weights":rows*cols,"code_bytes":code_bytes,"scale_bytes":scale_bytes,"bytes":path.stat().st_size,"artifact":str(path.relative_to(ROOT)).replace("\\","/"),"artifact_sha256":sha256(path)})
                total_weights += rows*cols; total_codes += codes.size; total_scales += scale_bits.size; total_bytes += path.stat().st_size
                print(json.dumps({"record":len(records),"layer":layer,"name":name,"bytes":path.stat().st_size}),flush=True)
    result={"kind":"streamq5_moe_p5a_physical_int8_trunk_bank","completed_utc":datetime.now(timezone.utc).isoformat(),"status":"p5a_trunk_bank_built_pending_verification","inputs":{"preregistration_sha256":sha256(PREREG),"model_index_sha256":sha256(MODEL/"model.safetensors.index.json")},"quantization":{"bits":8,"group_size":128,"qmax":127,"scale_storage":"BF16 bits little-endian","code_storage":"signed int8"},"aggregate":{"records":len(records),"weights":total_weights,"codes":total_codes,"scales":total_scales,"bytes":total_bytes,"gib":total_bytes/2**30},"records":records,"runtime_seconds":time.perf_counter()-started,"claim_boundary":"Physical INT8 projection-weight bank only; kernel and decode timing unproven."}
    RESULT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"aggregate":result["aggregate"],"runtime_seconds":result["runtime_seconds"]},indent=2))


if __name__=="__main__": main()
