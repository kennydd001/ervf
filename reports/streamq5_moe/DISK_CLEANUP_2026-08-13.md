# Disk cleanup — 2026-08-13

Purpose: reclaim workspace disk space without deleting source checkpoints,
research reports, preregistrations, scripts, manifests, raw result JSONs, route
captures, or the active PORT80B D9/D10 backing bank.

## Permanently removed reproducible bulk artefacts

| path | bytes | reason / reconstruction source |
|---|---:|---|
| `reports/runs/qwen3-30b-a3b-bf16.gguf` | 61,095,802,432 | Redundant converted BF16 copy. The 16 original `models/qwen3-30b-a3b-base/*.safetensors` shards and the Q5 GGUF remain. |
| `reports/runs/coretail_moe/p0_full_bank/` | 7,225,139,200 | Closed CORETAIL experiment bank. Results and format-verification evidence remain; the bank is reproducible with `scripts/coretail_moe/run_p0_full_bank_format.py`. |
| `reports/runs/streamq5_moe/p1d_q5_bank/` | 18,647,875,584 | Superseded P1D physical bank. Per-layer manifests, results and verification evidence remain; the bank is reproducible with `scripts/streamq5_moe/build_p1d_physical_bank.py`. |

Total selected: **86,968,817,216 bytes** (approximately **80.996 GiB**).

## Explicitly retained large artefacts

- `reports/runs/streamq5_moe/port80b_p0/port80b_p0_full_q5_bank.bin`
  (49,925,652,480 bytes): immutable backing store required by D9 and D10.
- Original DeepSeek and Qwen safetensor checkpoints.
- `reports/runs/qwen3-30b-a3b-q5_k_m.gguf`: runnable Q5 baseline.
- All reports, result JSONs, manifests, source code and route captures.

Removal is permanent rather than Recycle Bin based, because the explicit goal
is to recover disk capacity. All three removed targets were resolved beneath
`C:\Users\de_do\Documents\ChatGPT\New project` before deletion.
