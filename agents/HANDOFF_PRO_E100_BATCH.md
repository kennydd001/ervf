# Handoff — PRO E100 exact aggregate batch line

Active branch: **`pro-e100-batch`**.
Base: `ba132ad776d57e72f0429c0d8f76dce80181dbff` from `pro-v12-async`.
Do not write `main`. Claude's current V12/H-SCALE/Mamba work is intentionally separate and may exist only in his local working tree until pushed.

## Claim boundary

The primary target on this branch is **E100 aggregate exact decode**, not 100 tok/s single-stream latency.

A full E100 aggregate result must eventually mean:

- independent autoregressive sequences, each exact against its single-sequence greedy reference;
- total generated exact tokens / wall time >= 100 tok/s;
- >=1000 timed generated tokens in the verification campaign;
- per-sequence latency/gap distribution reported next to aggregate throughput;
- no relabelling aggregate throughput as single-stream throughput;
- independent CPU verifier recomputes the claim from raw result evidence.

## Important correction to the inherited batch narrative

`agents/PATH_TO_100_TOKS.md` and `BATCH_ARCHITECTURE_DESIGN.md` contain useful historical reasoning, but one conclusion is superseded by a later diagnostic already in this repo.

The old `proto_multi_seq_full_model_n8.json` reports 7.3896 aggregate tok/s and was interpreted as a 4x N=8 collapse. **Do not use that number as a physical result.**

`diag_n8_timing_reconcile.json` re-ran the identical N=8 loop back-to-back with both timing methods:

- replica solo: 30.1898 tok/s;
- CUDA-event method: 30.4806 aggregate tok/s, 1.0096x vs solo;
- perf_counter+synchronize: 30.3042 aggregate tok/s;
- timer ratio: 1.00582, explicitly agreeing within 20%.

Therefore the earlier 7.3896 result is a stale/invalid measurement, not evidence of an N=8 architectural collapse. N=8 still gives essentially **no aggregate scaling**, but it does not physically collapse by 4x.

This is consistent with `diag_n8_cache_hitrate.json`: N=8 device-cache hit rate was 77.08% vs solo 69.71%, so cache thrash is not the limiter.

## Existing batch facts re-audited

Naive full-model state-swap measurements:

- N=2: 31.6556 aggregate tok/s vs 31.0195 solo, +2.05%; 80 real tokens, exact.
- N=4 short run: 31.2154 aggregate vs 29.8199 solo, +4.68%; exact.
- N=4 cache cap 144: 19.0712 aggregate, a large regression; scaling cache capacity is not a free win.
- N=8 corrected timing: ~30.3-30.5 aggregate, essentially flat vs solo.

Cross-sequence routing overlap from 13,800 layer/step observations:

- N=2 mean union 11.58 / 12 slots;
- N=4 mean union 21.665 / 24;
- N=8 mean union 38.90 / 48;
- N=16 mean union 63.90 / 96.

So route-union fetch sharing is real but modest at N=4 and improves with larger N. It cannot by itself create E100.

Existing sequential-N common-matrix diagnostics are also correctly measured but were interpreted too narrowly:

- attention Q-style BF16 GEMV scales approximately linearly when launched N times;
- Mamba FP8 in-proj becomes ~15% more expensive per RHS at N=8-16 when launched N times;
- LM head NVFP4/production path becomes ~19-24% more expensive per RHS when launched N times;
- shared expert stays roughly linear/slightly better when launched N times.

Those measurements prove only that **N sequential GEMVs** do not amortise. They do not prove the matrices are intrinsically non-shareable.

## New primary insight: exact multi-RHS common-weight reuse

Attention, Mamba, routers, shared experts and LM head are common-weight operators. Every active sequence uses the same matrix. The inherited design said these components had “nothing to deduplicate” because they are not expert-selected. That is true for *routing identity* but false for *weight traffic*.

For batch N, N independent GEMVs stream the same weight matrix N times. An exact multi-RHS kernel can instead:

1. load one weight/dequantised scalar once;
2. apply it to N independent activation values;
3. maintain N independent accumulators;
4. preserve, for each RHS, the original virtual-thread MAC order and exact reduction tree.

No floating-point operation from two sequences is combined. This is weight reuse, not arithmetic approximation.

This is especially important for LM head and Mamba: the old route-to-100 document reduced its theoretical ceiling because these components became worse with N **under sequential launches**. MRHS directly attacks that assumption.

## Frozen E100-MRHS experiment

Files:

- `pro_research/E100_MRHS_PREREGISTRATION.md`
- `pro_research/mrhs_exact_kernels.py`
- `pro_research/e100_mrhs.py`
- `pro_research/verify_e100_mrhs.py`
- `pro_research/RUN_E100_MRHS.ps1`

The kernel uses one physical 32-lane warp per output row and emulates all 256 reference virtual threads as eight accumulators/lane. Fixed compile-time N in `{2,4,8}` avoids paying N=8 register pressure in the N=2/N=4 kernels.

Implemented storage kinds:

- BF16;
- FP32 router;
- FP8 tensor preserving production `uchar4` assignment and x/y/z/w FMA order;
- NVFP4 preserving production packed-code assignment, scale lookup, eight nibble FMA order, ReLU2 option and final output scale.

The N=4 primary gate is frozen before target execution: bit-exact all real checkpoint cases, weighted registered common-matrix speedup >=1.75x, LM head >=1.50x, Mamba input >=1.50x, no case <0.95x and reference drift <=7%.

A passing MRHS microbench is still only a primitive. No projected speedup is multiplied into V12 or any Claude result.

## Architecture if MRHS passes

The first serious E100 runtime should use fixed N=4, not N=8. N=4 needs <=40 ms per batch decode tick for 100 aggregate tok/s and has manageable state/VRAM cost.

Per layer, the target architecture is:

1. `[N, hidden]` hidden/norm/acc buffers and independent per-sequence Mamba/KV state;
2. common-weight projections through exact MRHS kernels (one matrix stream, N RHS);
3. router through F32 MRHS -> `[N,128]`, then one batched exact top-k kernel;
4. flatten N*top_k routed pairs; existing device `cache_assign` semantics can deduplicate repeated expert ids when arrays are sized to P=N*top_k;
5. one cache-fetch launch for the P list (`need=0` on duplicates/hits);
6. one up-proj pair-batch launch over all P pairs, selecting the proper sequence activation per pair;
7. one batched panel-scan/gather/down pipeline over P contributions;
8. accumulate six contributions per sequence in original route order, never across sequences;
9. one common MRHS LM head -> N independent argmax results;
10. capture the whole fixed-N tick in one CUDA graph; host only manages active slots/token delivery.

The key distinction from the old Python prototypes is that launch count should be approximately **constant in N** for each layer family. N increases grid dimensions / RHS work, not Python loops over 52 layers N times.

## Secondary routed-expert insight

Existing `UpProjBatchKernels` already proves top_k=6 expert GEMVs can be one launch for a single sequence. The multi-sequence extension should flatten P=N*6 pairs and add `seq_of_pair`/implicit `pair//top_k` addressing for X. This does not require expert overlap to reduce launch count. When two sequences select the same expert, a later expert-major MRHS subarm can additionally reuse that expert's weight stream; the measured N=4 overlap (~2.3 duplicate route slots on average) says that is a secondary, not primary, lever.

## Run order

Do not run this concurrently with Claude's GPU campaign.

When the target GPU is free:

```powershell
git fetch origin
git switch pro-e100-batch
git pull --ff-only origin pro-e100-batch
.\pro_research\RUN_E100_MRHS.ps1 -Mode smoke
```

Only if smoke is exact/clean:

```powershell
.\pro_research\RUN_E100_MRHS.ps1 -Mode full
```

If full returns `mrhs_candidate`, proceed to fixed-N=4 layer/runtime integration. If it returns `micro_null`, do not tune thresholds or choose N=8 post hoc; inspect register pressure / byte traffic as a new preregistered experiment.
