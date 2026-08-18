# llama.cpp bench vs ERVF runtime — Nemotron 3.5 Lightning 30B-A3B (v35)

Date: 2026-08-18 · Branch: agent/s100-phase10-hypotheses ·
GPU: RTX PRO 2000 Blackwell Laptop (SM120, 8150 MiB) · Host: 64 GB RAM.

## What was done

1. The historical blocker (`GGML_ASSERT(d_inner % (n_group*n_embd) == 0)`,
   upstream issue #20570, see `LLAMA_CPP_ISSUE_DRAFT.md`) is **fixed upstream**:
   current master (`mamba-base.cpp:171-172`) asserts the dimensionally correct
   `d_inner % n_head == 0` and `d_inner % n_group == 0`. No local patch needed.
2. Checkpoint converted with this repo's `convert_hf_to_gguf.py`:
   `models/nemotron_3_5_lightning_v35` (ModelOpt mixed FP8/NVFP4) ->
   `gguf-out/nemotron35-lightning-v35.gguf`, 510 tensors, 20.91 GiB, ftype
   NVFP4. Conversion kept the quantization; load verified with coherent
   generation.
3. Benchmarked with prebuilt release **b10488 win-cuda-13.3** (no local
   toolchain on this machine), `llama-bench -p 512 -n 128 -r 2`.

## Results (llama-bench, decode = tg128)

| Config | pp512 t/s | tg128 t/s |
|---|---:|---:|
| A: `-ngl 99 -fitt 512` (auto fit, max GPU) | 319.55 ± 23.90 | **32.08 ± 2.69** |
| B: A + `-ncmoe 99` (all experts on CPU) | 376.21 ± 25.91 | 31.22 ± 2.68 |

Decode is identical within noise across both offload configs — the binding
constraint is streaming the active expert bytes from host DDR, exactly the
wall our own engine was built around.

## Comparison vs the ERVF runtime (same GPU, same checkpoint)

| Engine | decode tok/s | basis |
|---|---:|---|
| llama.cpp b10488, best config | 32.1 | llama-bench tg128, CUDA |
| **ERVF research runtime (current map)** | **53.7** | phase-10 base arms, 765+ samples, 18.59-18.65 ms/token |
| ERVF via OpenAI server, real chat | 48-49 | measured 2026-08-16, see LLAMA_CPP_INTEROP.md |

**The ERVF engine is ~1.7x faster in decode than the best llama.cpp
configuration**, and ~1.5x faster end-to-end in real chat. llama.cpp's role
stays what the interop doc said: correctness reference and external
yardstick, not a speed target. Chat remains via `CHAT.ps1` (ERVF,
OpenAI-compatible). If a llama.cpp chat is wanted anyway:
`llama-server.exe -m <gguf> -ngl 99 -fitt 512` (expect ~30 tok/s).

## Incident postmortem (laptop freeze)

The first sanity run used `llama-cli -p ...`, which in b10488 is a TUI chat
client: it answered the prompt (that is where the 8.2/3.7 t/s numbers came
from) and then stayed alive at the interactive `>` prompt. A timeout-killed
pipeline left one such client orphaned holding ~7.7 GB VRAM; the retry then
started a second one. Two 21 GB model instances plus two CUDA contexts on an
8 GB WDDM laptop GPU froze the desktop; a hard reboot was needed.

Mitigations now in force: benchmarks run only through `llama-bench`
(terminates by design), one process at a time, pre-run check that no
llama/python processes and no VRAM are left, all output to log files.
`llama-cli` interactive runs should only be done in a real terminal,
never in the background.
