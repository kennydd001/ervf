# S5 — column-selective down_proj via masked mapped-host reads (preregistration)

Datum: 2026-08-14
Registry: LIGHTNINGSTREAM_NEMOTRON
Status: PREREGISTERED, before any S5 code or measurement exists.
Motivation: S2 measured **90,69% exact zeros** in the ReLU² intermediates
(20/20 independently verified). Zeros do not cluster (16-col blocks 30,6%),
so selection must be column-accurate. S5 changes **what moves**, not how fast.

## Frozen design (one variable vs the reproduced N8 baseline)

1. **Bank transpose (host, bit-exact).** `down_proj` is stored per expert in
   column-major order: column j = 2688 NVFP4 codes = 1.344 B contiguous, plus
   its 168 group scales (2688/16) contiguous. `up_proj` stays row-major.
   The transpose is a pure byte permutation of the checkpoint values; a
   reconstruction check proves dequant(transposed) == dequant(original)
   exactly (integer codes and FP8 scales identical, reordered only).
2. **Masked column GEMV kernel (new, deterministic).**
   `out[i] = Σ_{j ∈ nz} dequant(W_col j)[i] · act[j]`, iterating nz in
   **ascending j**, block = 128 threads each owning one output row, acc in a
   register. Skipped terms are exact zeros (`relu2` output = +0,0; the skipped
   product is ±0 and cannot change a nonzero accumulator). Deterministic
   nonzero-list compaction (single-block scan, no atomic append).
3. **Cache holds `up` only; `down` is never copied H2D.** Miss = H2D copy of
   the up half only (2.806.272 B vs 5.612.560 B). Down is read by the masked
   kernel **directly from mapped pinned host memory** (UVA zero-copy) — on
   hits and misses alike — touching only the cachelines of nonzero columns.
   Capacity stays 31 slots/layer (capacity is NOT varied in this phase).
4. Everything else unchanged: same router, same fused up GEMV, same attention,
   same FP8 KV, same runner harness (n7b-style context sweep).

Explicitly NOT in this phase: capacity changes, batch>1, any quality-affecting
approximation, any change to up_proj/attention/Mamba paths.

## Frozen gates

Correctness (hard):
- **G-S5-C1**: greedy generation on prompts A ("The capital of France is") and
  B ("The history of computing began when") reproduces the frozen baseline
  token ids **exactly** (32 tokens per prompt, baseline captured from the
  unmodified runtime as `s5_baseline_generation.json` BEFORE any S5 code).
- **G-S5-C2**: transposed-bank reconstruction check passes on all 2.944
  records (code/scale multisets identical per expert, permutation-exact).
- **G-S5-C3**: masked-kernel output vs current fused path on real routed
  calls: rel_l2 ≤ 1e-6 per call, all calls checked during a 64-token rollout.

Performance (on the n7b-style sweep, capacity 31, warm windows):
- **G-S5-P3 no-regression**: ctx 0 p50 ≥ 21 tok/s (reproduction: 22,062).
- **G-S5-P1 minimum**: ctx 262100 p50 ≥ 15 tok/s (reproduction: 13,143).
- **G-S5-P2 primary**: ctx 262100 p50 ≥ 18 tok/s.

Arithmetic behind the performance gates (component-level, not a claim):
miss bytes 275 MB → ~166 MB/token (up-only misses 130 MB + masked down reads
~36 MB); at the measured 26,03 GB/s that is 10,6 → ~6,4 ms of link time.
Whether the dependent, column-scattered mapped reads realise that saving —
or eat it in latency — is precisely what this phase measures.

Fallback (preregistered, only if G-S5-C1/C3 fail for the mapped-read design
or P1 fails): variant B = D2H mask readback + batched H2D column gather
(`cudaMemcpyBatchAsync`-style) into staging, dense smaller GEMV. If executed,
it is reported as attempt 2 with its own input lock; gates unchanged.
Gates are not widened after seeing results; a failed gate is recorded.

## Claim boundary (pre-committed)

S5 may claim: measured decode tok/s of the modified runtime on this GPU at the
measured contexts, with bit-identical generation to the frozen baseline, and
the measured byte counts that moved. It may NOT claim: quality/benchmark
results, other hardware, batch>1, or any projection beyond the measured
contexts. If P1 fails, the outcome is a scientific negative for the
mapped-host masked-read design, not a licence to tune the gates.

---

## Addendum S5-R1 — design A2: SM-side wide gather (preregistered before the gated run)

Evidence from the component smoke and microbench (both component-level, no
tok/s claims):
- Variant A as written (masked kernel reading mapped host memory at 1
  byte/thread) measures **~320 µs/call**; the microbench isolates the cause:
  byte-per-thread mapped reads run at **1,78 GB/s**. At 138 calls/token this
  design cannot pass G-S5-P1.
- Wide coalesced mapped reads (uchar4/uint4) reach **25,05 GB/s** ≈ the copy
  engine's 26,03 GB/s — the SMs can DMA if the loads are wide.
- Scattered 1.344 B copies through the copy engine (variant B's mechanism via
  per-slice copies) measure **0,16 GB/s** (8,3 µs per copy) — variant B is
  dead on this evidence and will not be executed.

Frozen design A2 (replaces A's read mechanism; gates, cache policy, bank
layout and everything else in S5 unchanged):
1. `panel_scan` is extended to also emit an ascending nonzero-column list
   (`nz_list`, `nz_count`), still fully deterministic (single-thread scan).
2. New `gather_down_sparse` kernel: warp per nonzero column and warp per
   active panel, uchar4 loads from the mapped host bank, written to a
   device-resident mirror of the panel-major record (same offsets — a sparse
   memcpy). Only nonzero columns and active scale blocks cross PCIe.
3. The UNCHANGED masked GEMV (`gemv_down_masked_partial` + `reduce_partials`)
   then runs on the device mirror. One compute kernel for hits and misses.
4. Misses still H2D-copy only the up half into the LRU slot; down is never
   copied to device persistently.

All S5 gates (C1, C2, C3, P1, P2, P3) apply unchanged to A2. If A2 fails,
S5 closes as a negative with the component evidence above.
