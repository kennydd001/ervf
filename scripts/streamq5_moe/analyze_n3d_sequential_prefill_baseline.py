from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/streamq5_moe"
SOURCE = R / "p13c_evt_pm_32g_endurance.json"
OUTPUT = R / "n3d_sequential_prefill_baseline.json"
REPORT = R / "N3D_SEQUENTIAL_PREFILL_BASELINE_2026-08-12.md"
LENGTHS = (1, 7, 128, 512, 1024, 4096)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cycle(values, begin):
    rows = {}
    for length in LENGTHS:
        sample = np.asarray(values[begin:begin + length], dtype=np.float64)
        rows[str(length)] = {"wall_ms": float(sample.sum()), "effective_input_tokens_per_second": float(length * 1000.0 / sample.sum()),
                             "per_token_mean_ms": float(sample.mean()), "per_token_p95_ms": float(np.percentile(sample, 95))}
    return rows


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("N3D output exists")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    values = source["wall_ms"]
    first = cycle(values, 0); second = cycle(values, 4096)
    activation = float(source["activation_ms"])
    result = {"kind": "streamq5_moe_n3d_posthoc_sequential_prefill_baseline",
              "completed_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sha256(SOURCE),
              "source_tokens": source["tokens"], "prompt_tokens": 7, "domain_activation_ms": activation,
              "first_4k_cycle": first, "second_4k_cycle": second,
              "service_ready_ttft_ms": first["7"]["wall_ms"],
              "activation_plus_ttft_ms": activation + first["7"]["wall_ms"],
              "claim_boundary": "Post-hoc physical sequential-token baseline from P13C. It is not a batched GEMM prefill implementation, cold-process startup measurement or preregistered comparison."}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# N3D — TTFT en sequentiële prefillbaseline", "", "Datum: 2026-08-12", "",
             "Dit is een post-hoc herberekening van de fysieke P13C-run; geen nieuwe testclaim.", "",
             f"De service-ready tijd voor het 7-tokenprompt tot de eerste vrije voorspelling was **{first['7']['wall_ms']:.3f} ms**. "
             f"Inclusief de eenmalige domeincacheactivatie was dit **{activation + first['7']['wall_ms']:.3f} ms**.", "",
             "| tokens sequentieel | eerste cyclus wall | effectieve input tok/s | tweede cyclus wall |", "|---:|---:|---:|---:|"]
    for length in LENGTHS:
        a=first[str(length)]; b=second[str(length)]
        lines.append(f"| {length} | {a['wall_ms']:.3f} ms | {a['effective_input_tokens_per_second']:.3f} | {b['wall_ms']:.3f} ms |")
    lines += ["", "De 4K-invoer kost sequentieel honderden seconden. Een echte GEMM-prefillkernel is dus nog open; deze tabel voorkomt dat decode-tok/s als prefillprestatie wordt gepresenteerd.", "",
              "Claimgrens: geen cold-process-TTFT, geen batch-prefill, geen nieuwe ongeopende testpartition."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"service_ready_ttft_ms": result["service_ready_ttft_ms"], "activation_plus_ttft_ms": result["activation_plus_ttft_ms"], "first_cycle": first}, indent=2))


if __name__ == "__main__":
    main()
