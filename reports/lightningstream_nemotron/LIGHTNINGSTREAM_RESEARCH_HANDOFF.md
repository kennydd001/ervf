# LIGHTNINGSTREAM_NEMOTRON — research handoff

**Date:** 2026-08-14 (local 11:52 +02:00, UTC 09:52Z)
**Author:** LIGHTNINGSTREAM_NEMOTRON line (new, independent registry)
**Status:** comprehension artifact. No Nemotron payload was downloaded, decoded or executed for this document.
**Protected-80B baseline:** `PROTECTED_80B_MANIFEST_BEFORE.json`, root digest
`7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`,
4,501 protected files, 193,299,000,498 bytes, 4 listing trees.

This document exists because the assignment forbids starting N2 payload work
before the entire prior research history has been read and mapped. Everything
below is sourced from the protected artifacts listed in §6; where a number is
quoted, it is quoted from those files rather than recomputed.

---

## 0. Environment of record

| item | value |
|---|---|
| repository | `C:\Users\de_do\Documents\ChatGPT\New project` |
| git branch / commit | `master` / **no commits exist**; entire tree untracked |
| git status | 24 untracked top-level entries at baseline |
| OS | Windows 11 Pro 10.0.26200 |
| CPU | Intel Core Ultra 9 285H, 16 cores / 16 logical |
| system RAM | 63.43 GiB total |
| dGPU | NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, 8,151 MiB, CC 12.0, driver 596.58 |
| iGPU | Intel Arc Pro 140T, driver 32.0.101.8517, PCI `0000:00:02.0` (per protected Intel evidence) |
| free disk (C:) | 272.57 GiB |
| Nemotron interpreter | `.venv-nemotron`, Python 3.12.10 |
| Nemotron pins | `numpy==2.2.6`, `huggingface_hub==0.35.3`, `safetensors==0.6.2`, `requests==2.32.5` |
| CUDA toolkit | `nvcc` not on PATH |

`numpy==2.2.6` is pinned deliberately, not incidentally: `README.md` records
that NumPy 2.5.2 was blocked by Windows Application Control on this machine when
loading `_bounded_integers`, while 2.2.6 survives the full runtime probe. The
Nemotron environment inherits that constraint as a fact about the host.

**GPU occupancy at start:** 0 MiB used, 7,899 MiB free, no Python process alive.
No PORT80B/STREAMQ5 run was active, so no phase had to be downgraded to CPU-only
for non-interference.

---

## 1. Chronological map of the project

The project is one continuous line of local MoE-inference research on a single
8 GiB laptop GPU, with a hard rule that every phase is preregistered, gated, and
independently verified before its result is allowed to be called anything.

### Phase A — baseline and instrumentation (2026-08-09)

Teacher `deepseek-ai/DeepSeek-V2-Lite`. Hardware/software/runtime preflight,
pinned revision, metadata-only fetch, synthetic MoE to validate that routing,
aggregation, baselines, metrics and reporting are correct. `README.md` is
explicit that the synthetic test proves **only** the harness, never that expert
compression works. Decision gates 0–3 were fixed up front (4× expert-byte
reduction → 8× with rollout → hundreds of stable tokens), and V4 Flash was
declared a scale test only after V2-Lite passed.

### Phase B — activation-space compression on V2-Lite (2026-08-09 → 08-10)

Shared basis, aggregate students, residual-basis students, weight quantization,
mixed quantization, streamed-model effects, all-layer router calibration,
storage accounting. Produced `RESULTS_2026-08-09.md` and a large `reports/baseline`
tree (~100 JSON result files covering layer-1 traces through model-wide
cache-aware bottom-1 substitution with CIs).

### Phase C — behavioral observability and cache routing (2026-08-09 → 08-10)

Layer-26 behavioral observability, reliability, dynamic-precision oracles and
predictors, quadratic mask predictor, progressive bit-plane predictor, route
equivalence, route-equivalent LRU cache, conformal cache selector, model-wide
cache-aware bottom-1 with exact LRU, Pareto screens. Adjudicated in
`EUREKA_VERDICT_2026-08-10.md`, then narrowed to the mass-budget policy in
`MASS_BUDGET_EUREKA_2026-08-10.md` with a preregistered confirmation at offset
4096.

### Phase D — named hypothesis families, mostly closed (2026-08-10 → 08-12)

CRAFT-MoE, RSIV/GhostWeights, FLEQ/GSQ, E2GQ, HERA → DHERA → DCHERA → LDHERA →
ADHERA, CORETAIL, BITFLOW, offload-roofline, STREAMQ4. Each has its own research
log in `docs/`, its own `reports/<family>/` tree, and its own terminal verdict.
`FINAL_EUREKA_VERDICT_2026-08-12.md` closes BITFLOW, offload-roofline and
CORETAIL together.

### Phase E — STREAMQ5 and the real local Eureka (2026-08-12)

Target model switches to **Qwen3-30B-A3B**. This is where the project produced
its strongest result: a complete physical custom decoder on the 8 GiB GPU. See §2.

### Phase F — ERVF/ERGV exact-kernel science (2026-08-12)

Exact-Reduction Virtual Fusion, then a generating compiler (ERGV). Bit-exact
speedups with zero changed CE values. See §2.

### Phase G — closure campaign and reality check (2026-08-12)

`ALL_IDEAS_FINAL_REPORT_2026-08-12.md` forced all 48 inventoried ideas to an
explicit status (14 verified pass, 3 quality-only pass, 19 verified negative,
3 superseded, 9 blocked, 0 queued). `BREAKTHROUGH_RESEARCH_FINAL_REPORT_2026-08-12.md`
invalidated a headline prior result (GaugePack/P9B) and set
`breakthrough_claim_allowed = false`.

### Phase H — PORT80B: Qwen3-Coder-Next 80B (2026-08-12 → 08-13)

Scale-up attempt. Physical 46.5 GiB host bank succeeded; the zero-cache transfer
gate failed; a mechanistic explanation was found (§3). D10 established that the
real blocker is the missing Qwen3-Next shell, not transport (§3).

### Phase I — HET-NEXT: heterogeneous Intel + NVIDIA (2026-08-13 → 08-14, ongoing)

Whole-expert partitioning across the Intel Arc Pro 140T iGPU and the NVIDIA
dGPU. **Intel half is a formal component PASS.** The NVIDIA half is stalled at
static-preflight audits N0→N5, all NO-GO, for source-gate reasons rather than
device reasons. This is the live front the other agent owns.

### Phase J — Nemotron reconnaissance (2026-08-12)

Two metadata-only phases, N0 and N1, executed inside the STREAMQ5 report
namespace. They are the seed of this line and are imported by hash in §7.

---

## 2. What was genuinely achieved

### 2.1 STREAMQ5 P6B — strict physical end-to-end decode

The single most important positive result. A complete custom runtime for
Qwen3-30B-A3B on the locked 8 GiB GPU, with the stopwatch starting *before*
the host embedding lookup:

| phase | quality | mean / p95 | throughput |
|---|---:|---:|---:|
| validation, 1,270 labels | +1.782862% CE | 50.112 / 59.352 ms | 19.955 tok/s |
| test, 1,270 labels | +0.048026% CE | 49.927 / 58.187 ms | 20.029 tok/s |
| rollout, 512 tokens | greedy feedback | 63.024 / 74.936 ms | 15.867 tok/s |

Physically co-resident: 4,977,623,040 B expert cache + 1,248,931,840 B Q8
trunk/head + 402,653,184 B KV, with 751,828,992 B free after fixed allocation.
The full 17.3671875 GiB Q5 expert bank (48 layers, 6,144 experts, 18,432 matrix
records, 28,991,029,248 Q5 codes, 226,492,416 BF16 scales) re-decoded with zero
code/scale/CRC/header/padding/hash errors. Independent verification 120/120.
Status token: `p6b_strict_end_to_end_eureka_pass`.

**The +0.048% was later corrected.** A 10× larger full-depth audit on 100
contexts / 12,700 labels gave **+1.4517% relative CE**, top-1 92.9528%, 95%
bootstrap CI [+1.1542%, +1.7619%]. The gate still passes; the headline number
was sampling. This correction is a required lesson: **a single small sealed test
is a gate, not an estimate.**

### 2.2 P13C — the memory-constrained endurance result

Qwen3-30B-A3B, batch 1, 4K context, under a hard Windows Job Object limit of
32 GiB, 10,000 tokens without OOM: 14.2348 tok/s, 69.862 ms mean, 91.984 ms p95,
100.498 ms p99, peak process commit 10.185 GB, peak working set ≈19.645 GiB.
Its component decomposition is directly instructive for Nemotron:

| component | ms | % |
|---|---:|---:|
| window penalty (mapped → pinned → H2D, 8 windows) | 24.12 | 34.5% |
| attention EVT-PM @4K | 12.93 | 18.5% |
| rest (norms, RoPE, router, residuals, KV write, sampling) | 16.37 | 23.4% |
| Q8 projections | 8.82 | 12.6% |
| Q5 experts ERVF | 7.61 | 10.9% |

**Under a 32 GiB cap, transfer is again dominant.** The project holds two
configurations with opposite bottlenecks — that is itself a result.

### 2.3 ERVF / ERGV — exact kernel acceleration and a compiler that reproduces it

ERVF processes sixteen rows per CUDA block while preserving the original 256
virtual accumulators and exactly the same reduction tree. Closed test
20.029 → 30.113 tok/s; 512-token rollout 15.867 → 20.915 tok/s; **0 differing CE
values** across 1,270 validation and 1,270 test labels; identical predictions,
routes, KV digests and all 512 rollout tokens; 48/48 verification gates.

ERGV then generated such kernels mechanically: `ExactReductionIR` over logical
accumulators, ordered add edges, round/cast points, FMA policy and physical
schedules. 7/7 CPU tests, 20/20 graph-isomorphic Q8/Q5 schedules, 2,680/2,680
CPU bitchecks, 6/6 corrupt mutations rejected, 115,496 generated-CUDA outputs
with zero bit differences. Autotuned against manual P7 it won 16.30% (Q8) and
7.87% (Q5) p50; against the hand-tuned N1C graph it reached parity. 63/63
independent checks over 960 raw events.

The defensible novelty boundary is stated in the source and must be preserved:
**preservation of a chosen source reduction DAG under a changed physical
topology** — not "first deterministic reduction."

### 2.4 Q5 quality transfers across model families

DeepSeek-V2-Lite full-depth over 26 MoE layers: validation +0.716% relative CE,
test +1.493%, test top-1 94.922%, median route overlap 96.745%. This supports
transferability of the uniform-Q5 quality result to top-6 routing and shared
experts — directly relevant to Nemotron, which is top-6 with one shared expert.
No physical DeepSeek bank/cache/runtime was built.

### 2.5 Mixed Q4/Q5

A fixed selection of twelve Q4 layers and thirty-six Q5 layers passed validation
and test and projects 5% fewer expert code bytes. Valid quality candidate; not a
speed result.

### 2.6 CORETAIL P0/P1 — format and microkernel (quality failed later)

Exact full-bank format at 1.993759 bpp, 7.725844 GiB resident formula, audit
26/26; exact fused kernel 30.738 Gweight/s vs a 27.2 gate, 72/72 correct, audit
13/13. The representation and microkernel are real; the model semantics failed
(§3).

### 2.7 HET-NEXT Intel component PASS

One official real expert-50 MLP, natural D2R3 `p0_n16` input, on the Intel Arc
Pro 140T: gate, up, normative-LUT SiLU, exact BF16 activation and down stages
reproduced the frozen CPU oracle **exactly at every retained stage**, with
preregistered host-USM, ownership, lifecycle, cleanup, resource and control
contracts. 10/10 verifier checks, 20/20 numerical checks, all 18 physical gates,
102 execution-ledger rows, 95 ownership rows, 21 successful releases, all six
forbidden-API counters zero.

Frozen stage hashes (these are the cross-device correctness target):

| stage | SHA-256 |
|---|---|
| gate | `e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867` |
| up | `f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08` |
| SiLU | `a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8` |
| activation | `762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f` |
| down | `142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc` |

### 2.8 PORT80B P0 — host bank hosting is solved

A non-sparse, uncompressed, Q5-compatible bank of exactly 49,925,652,480 bytes
(46.496887 GiB) was built, full-SHA verified, 132 records sampled, and run
uninterrupted for 3,600.093 s on 64 GB RAM. Peak process commit 6.075 GB. No
CUDA/thermal/runner errors.

### 2.9 The protocol itself

Append-only preregistration, input/runner/verifier source locks, validation-only
selection with once-opened tests, independent verifiers written against the
result rather than the runner, explicit claim boundaries, and terminal
classification that distinguishes *scientific negative* from *verifier-protocol
negative* from *invalid*. The Intel R8A5 case is the clearest example: a verifier
with a topology bug was correctly classified as a verifier-protocol negative and
was **not** retroactively rewritten; a new verifier independently adjudicated the
same immutable bundle.

---

## 3. What was falsified, and the mechanism of each failure

| family | outcome | mechanism |
|---|---|---|
| **RSIV / GhostWeights** | `falsified_rank_working_set` | The storage bound `S_layer <= (2d+3m)·top_k·prompt_tokens` is exact algebra, but the empirical premise fails: routed input and SwiGLU-intermediate rank grows to ~the observation bound (98.98–100% utilization) instead of saturating. Rank-32 double-fast was 0.000–1.742% against a 92% gate; even non-deployable rank-128 reached 5.762%. Two model families, so terminal. |
| **CRAFT-MoE** | closed | Route substitution as a primary method. Closed alongside RSIV; negatives may not be relabeled or post-hoc tuned. |
| **BITFLOW-MoE** | negative, closed | Linear branch: validation/test CE-damage recovery −645.95% / −395.02%. Independent audit 23/23. |
| **CORETAIL-MoE P2** | negative, closed | Universal ternary core + sparse exact tail is a real, physically built, bit-exact format with a fast fused kernel — but as *model semantics* it costs +35.953% validation and +42.943% test relative CE against a 2% gate and a 10% hard stop. A microkernel projection may not be promoted to tok/s. |
| **Expert pruning (GaugePack/P9B)** | premise invalidated, then falsified | The original "50% pruning quality pass" was a **no-op bug**: `weight[boolean_mask].zero_()` mutates an advanced-indexing copy, not the parameter. A mutation audit proved unchanged weight hashes and bit-exact expert output. Corrected: 50% → +47.804% CE, group-balanced 50% → +47.186%, 25% → +22.846%. Roughly linear in the removed fraction, so 2% pruning already consumes the whole quality budget. **Pruning is dead for MoE experts without retraining, at every fraction.** The structural explanation is that an expert seeing 6.25% of tokens is already specialized and lacks dense-FFN redundancy. |
| **Naive compaction of a pruned bank** | negative | Even where a logical zero-mask passed, repacking selected channels into new dense Q5 groups gave +48.027% CE / 60.472% top-1. Lesson: **channel selection and quantization-group layout are not independent.** |
| **TierFlow** | traffic feasible, quality not | `r=1` route-edit oracle: 4.1577× fewer critical expert bytes, 8× fewer worst-case new loads, 20/20 verification — but only 67.97% mean route overlap, 32.03% of router outputs substituted, and exact top-8 set match on just 8.99%. Rational only as a separately budgeted training study. |
| **GPU router** | closed | Bit-exact ids and BF16 weights on 480 real router vectors, yet the full route barrier was 4.392× slower p50 / 2.064× slower p95. While cache planning lives on the host, this is closed. |
| **CPU miss-compute** | falsified | CPU p50 7.447 ms vs GPU all-cold 1.122 ms = 6.64× slower. |
| **offload-roofline** | mixed, no Eureka | LFU and permutation negative; local H2D roofline verified but the K3 claim stayed conditional. |
| **PORT80B zero-cache transfer** | gate failed, mechanism found | p95 73.544 ms vs a ≤45 ms gate, and post-warmup page reads non-zero. See §3.1. |
| **PORT80B T0Q5 S0-R5** | formally negative | Its frozen integrity mutation on natural `p0/n8` shared-down changed **0 BF16 output words**, so the conjunct failed — even though the numerical arm passed 96/96 at `relL2 ≤ 0.08`. A neighbor witness at `p0/n15` does change one word but does not override the frozen result. |
| **The +0.048% CE headline** | superseded by better sampling | → +1.4517% [+1.1542, +1.7619] on 12,700 labels. |
| **"Glue" as a named cost** | falsified reasoning | A residual term `G = total − E − P − T = 14.961 ms` at ctx=128 was named "launch overhead"; the attention cost was hiding inside it and is context-dependent (negligible at 128, 96.626 ms at 4K). **Never name a residual without measuring it separately.** |
| **A launch-count analysis** | arithmetically wrong | P3A already had four launches per layer = 192 per token, not 768. CUDA Graph then gave 17.243 vs 17.113 ms p50 and failed its 10% gate. |

### 3.1 The PORT80B transfer smoking gun — the single most reusable diagnosis

The P0 failure was **not** a PCIe wall. The path was `mmap → pinned staging →
GPU` for 480 records per token:

```
record size          2,027,520 bytes
records/token              480
active bytes/token   973,209,600 bytes
measured pinned H2D  26.158915 GB/s
raw PCIe floor       37.204 ms/token
observed p50         63.034 ms
observed p95         73.544 ms
```

`63.034 − 37.204 = 25.830 ms`, and moving 973.210 MB through host DRAM in
25.830 ms is 37.68 GB/s — a plausible CPU memcpy rate. So the observed p50 is
almost exactly **one host gather pass plus one PCIe pass**. The p95 adds
10.510 ms of dispatch/paging/sync tail.

The corrective conclusion matters more than the failure: against a 100 ms token
budget with a 28.077 ms dense shell, the expert path may take 71.923 ms, needing
only 13.531 GB/s effective — and zero-cache p95 already delivers 13.233 GB/s.
**The gap was ~2%, not 1.634×.** Three exact remedies were preregistered:
registered batched copy (`cudaMemcpyBatchAsync`, 480 per-record calls → 48
per-layer batches), mapped-host direct kernel reads, and TMA host→SMEM.

### 3.2 The D10 blocker — architecture, not transport

For the 80B line the remaining blocker is the absence of an exact, physically
validated Qwen3-Next shell: 36 Gated-DeltaNet layers, 12 dimension-correct
Q-gated full-attention/KV layers (16 Q heads, 2 KV heads, head dim 256 — the
existing kernels are hardcoded to 32/4/128 and are *not* valid unchanged),
shared-expert sigmoid gating, top-10 routed aggregation, and **stateful layer
chaining**. D9 stages 480 records and evaluates every layer against the same
input vector; it never feeds a layer's output into the next. That compositor is
the blocker. There is also no natural 80B route trace; the safe interim is
explicitly named `representative_lifted_p4d`, never `natural_80b`.

---

## 4. Reusable mechanisms (portable without changing model semantics)

Everything here is ported **into the new namespace by copy**, recording source
path and source hash, never by modifying the original.

| mechanism | proven by | why it transfers to Nemotron |
|---|---|---|
| host-resident low-bit expert bank, GPU holds only trunk/shared/cache | STREAMQ5 P6B | Nemotron routed bank is 15.389 GiB — same shape of problem, 1.13× smaller than Qwen3-30B's 17.367 GiB |
| validation-selected static + causal dynamic LRU cache | STREAMQ5 (20 static + 15/14 dynamic beat a failed 32-static policy) | the two-timescale split is model-agnostic; slot counts must be re-derived, never inherited |
| causal H2D/compute overlap with bit-identical end state | P4A: 1.402× test speedup, identical LRU misses, identical bytes, bit-identical final states | applies to any per-layer expert stream |
| registered/batched host transfer (`cudaMemcpyBatchAsync`) | PORT80B DirectPath diagnosis | Nemotron top-6 needs 138 records/token; batching per layer is 23 submissions instead of 138 calls |
| mapped-host direct kernel reads | PORT80B DirectPath path 2 | batch-1 expert weights are read once and are coalescible — the stated precondition |
| ERVF exact-reduction virtual fusion | 0 differing CE values, 48/48 gates | preserves semantics by construction; the correct way to go faster without a quality argument |
| ERGV generated exact kernels + width autotuning | 115,496 outputs, zero bit differences; 63/63 | the NVFP4 group-16 shape needs a width search; do it mechanically |
| vectorized low-bit code loads, subwarp width search, multi-row blocks | P7/ERGV | direct analogues for NVFP4 |
| exact route-weighted reduction from **official** router output | STREAMQ5, and the explicit "recomputed topk is not authoritative" rule | Nemotron top-6 + shared expert |
| grouped expert GEMM for prefill | project prefill work | required for the 128K/256K context goals |
| strict full-depth + autoregressive gates; sealed once-only tests | whole project | the reason the +0.048% error was caught |
| append-only preregistration, source locks, independent verification, terminal-state taxonomy | whole project | non-negotiable |
| PDH page telemetry (`Memory/Page Reads/sec`, `Pages Input/sec`), process commit/RSS/peak, nvidia-smi power/temp/clocks, 1 Hz with idle baseline | P0 / P13C / D10 | the memory gates require exactly this |

---

## 5. Forbidden: closed hypotheses that must not re-enter the Nemotron baseline

Per assignment §2 and confirmed by the archive:

1. shared or low-rank expert surrogates — falsified by RSIV rank census;
2. post-hoc hidden-state correction;
3. Q2 CORETAIL as model semantics — +42.943% test CE;
4. CRAFT route substitution as a primary method;
5. dynamic Q3/Q4 selectors;
6. atom skipping / neuron sparsity as the baseline — pruning is linear-cost and dead at every fraction;
7. speculative block-coalescing before an ordinary decoder exists;
8. multiplying local oracle gains together without one full-depth candidate;
9. recomputed `topk` treated as authoritative when the official block already returned IDs.

Two further self-imposed rules, drawn from failures above rather than from the
assignment text:

10. **never name a residual term** (a "glue", "overhead" or "misc" bucket) without measuring it independently first;
11. **never promote a microkernel or component projection to tokens/s.**

---

## 6. Source-to-claim table

Paths are repo-relative. Byte counts and SHA-256 are as frozen in
`PROTECTED_80B_MANIFEST_BEFORE.json` at the start of this research line. All are
protected/read-only for the Nemotron line.

| # | protected path | bytes | SHA-256 |
|---:|---|---:|---|
| 1 | `README.md` | 3,990 | `7da71c63313db8eefe1c65334d58cdc2dd0117f9c14e242e3013e0fc06a82905` |
| 2 | `docs/RESEARCH_LOG.md` | 40,622 | `6ed90bba9f006bbde40ae2cbeaa40166ead729ccfd479b3b72ddd4b59a6718b9` |
| 3 | `docs/PRIOR_ART.md` | 12,420 | `ace2c3db18b0249ac600ebade4ad2c82011b67a08027e2e67ce5c83f4d7ca067` |
| 4 | `docs/CRAFT_MOE_PRIOR_ART.md` | 16,693 | `a18ba18e553c2bd73d081da4787285f629a2fef5319cd952bb1448956f86f310` |
| 5 | `docs/RSIV_MOE_RESEARCH_LOG.md` | 14,816 | `43aa64385c8f5bf88e0f57e3dd39ee2554abd5b8775abc82ef5a5b3aebeee63b` |
| 6 | `docs/CORETAIL_MOE_RESEARCH_LOG.md` | 3,081 | `df448eb44a74d37b7ce5174092a2791a4e3419bcfd6f4b30c89c7ec0f0dc0236` |
| 7 | `docs/OFFLOAD_ROOFLINE_RESEARCH_LOG.md` | 3,068 | `e1c8a548529f6cae8f6994109b3bf7d50488bf3378bfe9e8d4bc3cfced75ccf5` |
| 8 | `reports/BASELINE_2026-08-09.md` | 3,558 | `98e47c7eab1abdb2f72a0326114b2b86be3400de970cc3884b8a6c14f3569399` |
| 9 | `reports/RESULTS_2026-08-09.md` | 8,642 | `600fdcc4f32a6895bc7c7187e231146e5214d2a6817a7bf23778f6af19f706ba` |
| 10 | `reports/EUREKA_VERDICT_2026-08-10.md` | 13,749 | `7c7e210620f13da397b32fb1fbd72da999fa6852e96a2e343440fc467fb38455` |
| 11 | `reports/MASS_BUDGET_EUREKA_2026-08-10.md` | 14,985 | `37cad2afde4fa72160f70205907689274ec9e8cea08e0f0e42a021519b073028` |
| 12 | `reports/PREREGISTERED_MASS_BUDGET_CONFIRMATION_2026-08-10.md` | 3,700 | `f37c29669eb025f85f71acff2e197d11744aff53877d5f74c1c4c1d3223ee0b6` |
| 13 | `reports/FINAL_EUREKA_VERDICT_2026-08-12.md` | 2,659 | `4c6d7739cfb41f3fa4695e2a98a6d33a6110802bda90b7cab7337089960668b4` |
| 14 | `reports/rsiv_moe/RSIV_MOE_FINAL_VERDICT.md` | 6,154 | `c5a77b3c767255d93d02d6bc7ecf19660cd2078e8482b89db5f391df543a840a` |
| 15 | `reports/streamq5_moe/FINAL_VERDICT.md` | 8,904 | `90409e8c2056f2b7f5c3b748454dbe46843e82793c8e9e6b637050451d3cc83a` |
| 16 | `reports/streamq5_moe/EXPERIMENT_REGISTRY.yaml` | 14,989 | `a3440108fe9d83538f42df3d06a7f09965df3eb15538e0d5cf5adfc3dd538241` |
| 17 | `reports/streamq5_moe/ALL_IDEAS_FINAL_REPORT_2026-08-12.md` | 5,765 | `a4ecfe501cbb84b0958e0d7fba1ce4e9bf3f890d2ce61d6b0ae884fd287f50f0` |
| 18 | `reports/streamq5_moe/ALL_IDEAS_CLOSURE_REGISTRY_2026-08-12.yaml` | 11,027 | `0f07ceded503a9b8f20170e68f81e3c47f7a2be2f9d35b7d7f8c73ecf628c616` |
| 19 | `reports/streamq5_moe/BREAKTHROUGH_RESEARCH_FINAL_REPORT_2026-08-12.md` | 7,016 | `203a8adce7c019782da6a98bc22268a6be7cb315ac521376114e794a734c1658` |
| 20 | `reports/streamq5_moe/BREAKTHROUGH_PHASE_REGISTRY_2026-08-12.yaml` | 4,993 | `db54aada8bc1d673d8419f4ebd782d7b57fa38a35b70a72723be0641c23db5bc` |
| 21 | `reports/streamq5_moe/DATAPLANE_ONTLEDING_VERDICT_2026-08-12.md` | 8,299 | `2313822975e54c674aed12a9a2e0d859f45c8ea8f7dd72672c957e757ea6caf6` |
| 22 | `reports/streamq5_moe/PORT80B_D10_ARCHITECTURE_AUDIT_AND_DESIGN_2026-08-13.md` | 11,197 | `558cb016f4a97dbdb4982a498e500fd37f55c8413e417065a9c115ec1fe8ec69` |
| 23 | `reports/streamq5_moe/PORT80B_T0Q5_S0R5_C1R2A_COMBINED_REPORT_2026-08-13.md` | 2,101 | `cd91e9226f99e1177caff83a62a438c29651af462a53d95891fdbc95b9477e06` |
| 24 | `reports/streamq5_moe/PH1_INTEL_R8A5_FINAL_COMPONENT_REPORT_2026-08-14.md` | 5,723 | `d40fc5f628d2d2465e61a6105db3ed1ff104bc6430a28724b7d2ec6d9eed5a5f` |
| 25 | `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8v1r1a_independent_verification.json` | 8,098 | `42cd69582a47b8b5f8f4b7f24a696f1d3fcc6fbd49c05d0f61354a57cefc052d` |
| 26 | `reports/streamq5_moe/HET_NEXT_L0_PH1_NVIDIA_FULL_EXPERT_N5_IMPLEMENTATION_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md` | 9,200 | `c6474a55d2fe3a0567917dd7f47e63e147c0099f5ef3bd67cc99ad04c5c47d13` |
| 27 | `reports/streamq5_moe/NEMOTRON_N0_METADATA_GATE_REPORT_2026-08-12.md` | 446 | `0aad902cfb98644689420349505cfb827b726592204730fe8df87107f3e8cc76` |
| 28 | `reports/streamq5_moe/nemotron_n0_metadata_gate.json` | 5,885 | `28d4660af02da40f712fe21fb1f284ad260f76de715461e5cfb95009564e00d7` |
| 29 | `reports/streamq5_moe/NEMOTRON_N1_HEADER_INVENTORY_PREREGISTRATION.md` | 829 | `2d38753c96d36283a7eb1f58abdb1e8ef5554563ecca6667e6b8d07572235f7e` |
| 30 | `reports/streamq5_moe/NEMOTRON_N1_HEADER_INVENTORY_REPORT_2026-08-12.md` | 476 | `0f019beaeaad7cad514560eac3277e541b55076ff373a3c3868dd55c6eb54779` |
| 31 | `reports/streamq5_moe/nemotron_n1_header_inventory.json` | 3,266 | `4a501c79a4608c3e6aff53b9b027a546416a41b98bbafb20004fffef65aad348` |
| 32 | `reports/streamq5_moe/P7_ERVF_FINAL_REPORT_2026-08-12.md` | 3,407 | `56d49408eadac91e4a999608700143d3806b333ead41e9162b28ea41534173af` |
| 33 | `reports/streamq5_moe/p7_ervf_independent_verification.json` | 4,466 | `95d8691b7972135d4ded66b00c9c2e2b75427a0902f5840082b33343731d4e03` |
| 34 | `info/ERVF_EUREKA_2026-08-12.md` | 1,087 | `1d0161aa5885bdc0cd70a81f56ab6689965982b2091eb1f6c4606a112e00cafb` |
| 35 | `info/RAND_VAN_WAT_2026-08-12.md` | 11,247 | `a049ff53681c45ebaf9d9086f6ac5aaf5e4226e5a1730d3e85457b003250a266` |
| 36 | `info/NA_P13C_VIER_HEFBOMEN_2026-08-12.md` | 12,283 | `95afc82e4c9da106cf6724bb4f7260ee273372ff003fdcab2cb6f075b9199de6` |
| 37 | `info/KERNEL_INVERSIE_2026-08-12.md` | 11,530 | `97d260bf2fa4e7e1f92a0324e7158db744126ad407f6aabf59a83eb1ccbaeea9` |
| 38 | `info/RICHTINGEN_DOORGEREKEND_2026-08-12.md` | 11,711 | `f41b19bf2bf14d75a085380aa696778941ff254bae7c2ff7433baec9832e3e55` |
| 39 | `info/BITBREEDTE_ANALYSE_2026-08-12.md` | 10,372 | `f936096a328e3b7558394d3f8995c9f1b34d061a20fadb8dd681c6c30d996cae` |
| 40 | `info/PORT80B_DIRECTPATH_PACK_2026-08-12/PORT80B_DIRECTPATH_SMOKING_GUN_REPORT_2026-08-12.md` | 4,712 | `f1972b0cd024d7c52d4ff9bd1bd7b5feb8808ced48ee45d33ee4f9c91b010d2a` |
| 41 | `info/PORT80B_DIRECTPATH_PACK_2026-08-12/PORT80B_DIRECTPATH_CALCULATIONS_2026-08-12.json` | 3,715 | `6eb0461c3ed276c103a20a4dbeee867f0df9a8b1a296a801938abcbc65e3e491` |
| 42 | `research_docs_nuttige_info_2026-08-14_111334_compleet.zip` | 3,610,321 | `7253ba79913447d0d29297076f686dd8fa372d2f8da8b7233b4bd5717fbe01fc` |

The archive in row 42 was opened read-only and enumerated: 2,010 entries
(1,938 `reports/`, 52 `info/`, 15 `docs/`, plus root files). Every entry is also
present in the working tree, so the tree is the superset and was read directly.

---

## 7. Imported Nemotron evidence (by hash; not rewritten)

Rows 27–31 above are the two completed metadata phases. Their established facts,
at pinned Hugging Face commit `ce1b118ae66ec705d02c241525192832eb045fd3` of
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`:

| quantity | value |
|---|---:|
| shards | 5 |
| tensors | 24,147 |
| indexed tensor payload | 19,339,781,632 B = 18.011575 GiB |
| routed experts | 16,523,376,640 B = 15.389 GiB |
| shared experts | 258,177,392 B = 0.240 GiB |
| trunk / other | 2,558,227,600 B = 2.383 GiB |
| MoE layers | 23 (layer ids 1, 3, 6, 8, 10, 13, …) |
| routed experts per MoE layer | 128, uniform |
| routed records | 2,944 |
| one routed expert record | 5,612,560 B (uniform across all 2,944) |
| top-6 all-cold records/token | 138 |
| top-6 all-cold bytes/token | 774,533,280 B |
| transfer floor @ 26.158915 GB/s | 29.608769 ms/token |

dtype byte split: U8 15,245,905,920 · BF16 2,156,424,064 · F8_E4M3 1,905,738,240 · F32 31,713,408.

N1 gates all true: five headers, index key count, no offset failures, tensor
bytes equal index total, 23 MoE layers, 128 experts each, uniform routed records.
Claim boundary preserved: header/index inventory only; no payload downloaded,
decoded or executed; the NIM alias-to-HF weight identity was **not** proven.

**Consistency check performed here (arithmetic only, no new data):**
`2,944 × 5,612,560 = 16,523,376,640` ✓ matches the routed bucket exactly;
`23 × 6 = 138` ✓; `138 × 5,612,560 = 774,533,280` ✓;
`774,533,280 / 26.158915e9 = 29.6088 ms` ✓. The N1 inventory is internally
consistent.

---

## 8. Why Nemotron is a mechanistically independent next target

This is the part that decides whether this line is legitimate research or a
rerun of closed work.

**8.1 The quantization format is native, not ours.** Every prior line built its
own codec (Q5 RTN with BF16 scales, Q8 trunk, CORETAIL ternary+tail, GaugePack
pruned layout) and then had to defend model quality against a BF16 reference.
Nemotron 3 Nano NVFP4 is **quantization-aware-trained by the publisher**: the
NVFP4 checkpoint *is* the target model. The quality question changes from "does
our codec preserve the model?" (repeatedly answered *no* — CORETAIL +42.9%,
pruning +47.8%) to "does our runtime reproduce the published checkpoint's
semantics?" That is a correctness question with a bit-exact answer, not a
quality-budget question. **This removes the single most common cause of death in
the archive.**

**8.2 The active set is genuinely smaller and the layer count is halved.**
23 MoE layers × top-6 = 138 records/token at 774.533 MB, against Qwen3-Coder-Next's
480 records at 973.210 MB. That is 20.4% less routed traffic per token *and*
72% fewer per-layer expert dispatches — and the PORT80B diagnosis showed
per-record dispatch, not bandwidth, was a major term. Only six full-attention
layers (versus twelve in Qwen3-Next, forty-eight in the Qwen3-30B runtime) shrinks
the attention plane that P13C measured at 96.626 ms unoptimized @4K.

**8.3 The hybrid-shell blocker is inherited but strictly smaller.** Nemotron is
Mamba-2 + attention, so the D10 lesson applies in full: the stateful
layer-composition path is the real work, not the transport. But the Nemotron
shell is 23 Mamba-2/MoE modules and 6 attention modules on a fully public
30B checkpoint, versus 36 Gated-DeltaNet + 12 specialized attention layers on an
80B model whose weights the project has never held. **The same class of problem
at roughly a third of the scale, with the reference implementation public.**

**8.4 It is a genuine second model for the exact-kernel work.** ERGV's stated
open item is "a second GPU, second model, matched public kernels." NVFP4 group-16
with FP8 block scales and FP32 global scales is a different code layout from
Q5-with-BF16-scales, so porting ERVF/ERGV to it is a real generalization test,
not a repetition.

**8.5 It does not compete with the 80B line for anything.** Different model,
different registry, different venv, different report tree, different GPU
schedule. The Intel/NVIDIA HET-NEXT work and the PORT80B D10 shell work continue
untouched; this line consumes only disk and, when the GPU is idle, GPU time.

**8.6 What is deliberately *not* claimed.** Nemotron being newer or hybrid does
not by itself imply better tok/s, and the 1M context figure is a metadata claim
from NVIDIA that this line has measured nothing about. Per §5 rule 11, the
29.609 ms transfer floor is a floor, not a projection of achievable latency.

### The naming rule this line will obey

| use | when |
|---|---|
| `LOCAL_PUBLIC_WEIGHTS` = `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`, described as **"public Nemotron 3 Nano NVFP4 checkpoint"** | always, for anything measured locally |
| `LIGHTNING_SERVICE` = `nvidia/nemotron-3.5-nano-30b-a3b`, described as **a separate NIM endpoint** | only for service-side observations |
| "Nemotron 3.5 Lightning local weights" | **never**, unless H0 returns `identity_proven` |

NVIDIA's public metadata is currently inconsistent about top-5 versus top-6
routing (NIM model card says top-5 in one place; the public checkpoint config
says top-6). For the local checkpoint the pinned config, tensor index, model
code and one intercepted official routing call are authoritative — and N1's own
138-records/token figure already assumes top-6, which must be re-derived from the
config in N2 rather than inherited.

---

## 9. Isolation status at handoff

| control | state |
|---|---|
| protected manifest built | yes — `PROTECTED_80B_MANIFEST_BEFORE.json`, root digest `7c992ce2…46ba` |
| manifest self-test | `PROTECTED_80B_INTACT`, 0 removed / 0 modified / 0 added / 0 listing changes |
| write allowlist created | 7 directories, all empty at creation |
| isolated interpreter | `.venv-nemotron` created from system Python 3.12.10; the project `.venv` and `.venv-next-ref` were not touched and are covered by listing digests |
| protected processes | none killed, suspended, reniced or altered; a `codex` process (PID 11108) and its host were left running |
| GPU interference | none; GPU was idle at 0 MiB used |
| `git` operations | none. The tree has no commits; no branch or worktree was created because branching an uncommitted 193 GB tree would be more disruptive than isolating by path |
| `.gitignore` | **deliberately not modified.** It is a protected file, so adding `.venv-nemotron/` and `.cache/` to it would itself be a protected-byte change. Recorded here as a known, accepted deviation |

Any future phase that reports a changed protected byte is a hard stop under
assignment §0.6.

---

## 10. Next step

`N0R_IDENTITY_REFRESH`, then preregistration of
`N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS`. No runtime code is written until the
registry entry, input lock and gates for the phase in question exist.
