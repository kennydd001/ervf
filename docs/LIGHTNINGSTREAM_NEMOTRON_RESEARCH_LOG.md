# LIGHTNINGSTREAM_NEMOTRON — research log

Append-only. One section per working session. Numbers are quoted from the
artifacts named beside them; nothing here is a claim that its artifact does not
already support.

**Line identity**

- `LIGHTNING_SERVICE` = `nvidia/nemotron-3.5-nano-30b-a3b` — a NIM endpoint.
- `LOCAL_PUBLIC_WEIGHTS` = `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` — the
  public Nemotron 3 Nano NVFP4 checkpoint, and the only thing measured locally.
- The phrase "Nemotron 3.5 Lightning local weights" is forbidden until H0
  returns `identity_proven`. As of N0R it has not.

**Isolation.** The Qwen3-Coder-Next / PORT80B / STREAMQ5 line belongs to another
agent and is still active. Everything outside the Nemotron allowlist is
read-only, fingerprinted in `PROTECTED_80B_MANIFEST_BEFORE.json`
(root digest `7c992ce2…46ba`, 4,501 files, 193,299,000,498 bytes) and re-verified
after every phase.

---

## 2026-08-14 — line opened; handoff, N0R, N2 started

### Setup

- Inventoried the repository and running processes without changing anything.
  GPU idle at 0 MiB used, no Python process alive, so no 80B run was disturbed
  and no phase had to be downgraded to CPU-only.
- Built the protected manifest and self-tested its verify path
  (`PROTECTED_80B_INTACT`). Tiered fingerprints: full SHA-256 below 32 MiB,
  size + mtime + both 4 MiB edges above it, listing digests for the venvs,
  `.pytest_cache` and `third_party`. Whole pass runs in ~17 s, which is what
  makes per-phase re-verification affordable.
- Created the write allowlist and `.venv-nemotron` (Python 3.12.10) with
  independent pins. `numpy==2.2.6` is pinned deliberately: the protected
  `README.md` records that NumPy 2.5.2 was blocked by Windows Application
  Control on this machine.
- Did **not** create a git branch (the repo has zero commits and 193 GB of
  untracked content, so branching isolates nothing) and did **not** edit
  `.gitignore` (it is a protected file, so editing it would itself be a
  violation). Both recorded as accepted deviations in the registry.

### Research handoff

`reports/lightningstream_nemotron/LIGHTNINGSTREAM_RESEARCH_HANDOFF.md`. Maps the
project from the DeepSeek-V2-Lite baseline through CRAFT/RSIV/FLEQ/E2GQ/HERA/
CORETAIL/BITFLOW/STREAMQ4, the STREAMQ5 Eureka, ERVF/ERGV, the 48-idea closure
campaign, PORT80B and the live HET-NEXT work. 42 source files bound by exact
path, byte count and SHA-256.

Load-bearing lessons carried into this line:

1. **The +0.048% CE headline was sampling.** A 10× larger audit gave +1.4517%
   [+1.1542, +1.7619]. A single small sealed test is a gate, not an estimate.
2. **GaugePack/P9B was a no-op bug** — `weight[mask].zero_()` mutated a copy.
   Corrected, pruning costs ~linearly in the removed fraction (+47.8% at 50%,
   +22.8% at 25%), so it is dead at every fraction without retraining.
3. **Never name a residual term without measuring it.** "Glue" turned out to be
   context-dependent attention (negligible at ctx 128, 96.6 ms at 4K).
4. **Never promote a microkernel projection to tok/s** (CORETAIL's lesson).
5. **PORT80B's zero-cache failure was a host gather pass, not a PCIe wall** —
   p50 63.034 ms ≈ 37.204 ms PCIe + 25.830 ms DRAM copy. The remedy list
   (registered batched copy, mapped-host reads, TMA) transfers directly to H3.
6. **D10's blocker is the stateful shell, not transport.** Nemotron inherits the
   same class of problem (hybrid recurrent + attention) at roughly a third of
   the scale, with public reference code.

### N0R_IDENTITY_REFRESH — `service_only_unknown_payload`, branch 3

Preregistered, then executed; all seven gates pass. The outcome was **predicted
in writing before execution**, and the preregistration also explicitly removed
`behaviorally_close_identity_unproven` from reach because no prompt suite runs
in a metadata phase.

- HF pin `ce1b118a…` still resolves and `main` has **not** drifted since N0.
  Last modified 2026-03-15. 18 siblings; five shards totalling 19,342,796,520 B
  with LFS SHA-256 recorded for each.
- Architecture (authoritative for the local checkpoint): `nemotron_h`,
  52 layers as `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` =
  **23 Mamba-2 + 23 MoE + 6 attention**, hidden 2688, moe_intermediate 1856,
  shared 3712, 128 routed experts, **top-6**, 1 shared expert,
  `mlp_hidden_act = relu2`, 32 Q heads / 2 KV heads / head_dim 128,
  vocab 131,072.
- **Routing conflict resolved:** the pinned config says `num_experts_per_tok = 6`.
  Re-derived from config, 23 × 6 = 138 records/token and 138 × 5,612,560 =
  774,533,280 B/token, both matching N1 exactly. The NIM card's top-5 is the
  service-side inconsistency and has no authority here.
- **Second conflict, raised early:** the NIM service advertises 1M context; the
  pinned config declares `max_position_embeddings = 262,144`. So the planned
  `N13_1M_STRETCH` cannot be a plain context extension of this checkpoint. That
  is recorded now rather than discovered at N13.
- Quantization declared: ModelOpt 0.29.0, NVFP4, group 16, FP8 KV, and 63
  `exclude_modules` whose pattern is structural — all six attention layers'
  q/k/v/o, the `in_proj`/`out_proj` of the Mamba layer immediately preceding
  each attention layer, every Mamba `conv1d`, and `lm_head`. These are
  incompressible BF16 costs H4 must carry.
- Declared languages are en, es, fr, de, ja, it. **Dutch is absent**, which
  confirms the instruction to evaluate Dutch explicitly rather than assume it.
- Derived an NVFP4 record-layout hypothesis and found it reproduces the frozen
  N1 routed record bytes, record count, routed bucket and shared bucket
  **exactly**. Recorded as `derived_hypothesis_not_layout_proof` precisely
  because a likely-but-unverified premise is what killed GaugePack.

Protected check after the phase: `PROTECTED_80B_INTACT`, root digest unchanged.

### N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS — started

Preregistered with a falsification-first design: the phase exists to confirm
**or destroy** the N0R layout hypothesis against real tensor entries, and the
decoder validation rules and tolerances were fixed before any result was opened.

- Download of the five official shards into `models/nemotron_3_5_lightning/`
  began after a non-interference check (no Python process, GPU idle, 272 GiB
  free against an 18.01 GiB budget).
- Built the NVFP4 codec as **two independent implementations** — table-driven
  and bit-arithmetic — that share no decode-time helper, per the preregistration.
  16/16 codec tests pass, run with `-p no:cacheprovider` because `.pytest_cache`
  is inside a protected listing digest.
- First real header read already confirms the layout hypothesis in detail:
  `up_proj.weight` is U8 `[1856, 1344]` (1344 = 2688/2), `up_proj.weight_scale`
  is F8_E4M3 `[1856, 168]` (168 = 2688/16), and the two FP32 globals are
  `weight_scale_2` and `input_scale`. Per expert that is
  2 × 2,494,464 + 2 × 311,808 + 4 × 4 = **5,612,560 bytes**, the frozen N1
  record size.
- The router carries `gate.e_score_correction_bias` alongside `gate.weight`, and
  the config sets `norm_topk_prob = true` with `routed_scaling_factor = 2.5`.
  Route selection is therefore not a plain top-k over a linear projection, and
  N3 must capture the official routing call rather than reimplement it.

### N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS — PASS, layout hypothesis confirmed

All five shards verified byte-exact **on the first attempt**; 18.014383 GiB
against a 25 GiB artifact gate. All nine frozen gates pass. Protected check
after the phase: `PROTECTED_80B_INTACT`.

- **Every N1 bucket reproduced exactly** from the local copy: 24,147 tensors,
  19,339,781,632 bytes, routed 16,523,376,640, shared 258,177,392, trunk
  2,558,227,600, 2,944 uniform 5,612,560-byte records. The index reconciles
  fully and the layer roles derived from real tensors match the
  `hybrid_override_pattern` and N1's MoE layer list.
- **Layout confirmed on all 2,944 routed and all 23 shared experts** — dtypes,
  shapes and per-field byte counts, not just totals. Grouping runs along the
  contraction dimension in both matrices, which is the kernel-friendly
  orientation.
- **Provenance cross-check.** Shard 1's header reproduces N1's remote read from
  two days earlier. The hash initially mismatched; the cause was that N1 hashed
  the 8-byte length prefix together with the body. Both conventions are now
  recorded per shard so it cannot recur. Worth noting as a general lesson: a
  hash mismatch is a question, not a verdict.
- **Random access, the H3-relevant result.** Every one of the 2,940 single-shard
  routed experts needs exactly **three contiguous ranges** — 4,988,928 B of
  codes, 623,616 B of scales, 16 B of globals — because within each dtype region
  an expert's two matrices are adjacent. Four experts straddle a shard boundary
  and need a two-file gather. Recorded per §6 as a design input, not a defect.
- **Incompressible BF16 is 2.008 GiB**, of which `lm_head` and `embeddings` are
  704,643,072 B each — together 1.312 GiB, or 65% of all BF16 bytes, against an
  8 GiB budget. They have opposite runtime profiles (embedding is a one-row
  gather per token; the LM head is a full matvec per token), so the preregistered
  H4 host-placement ablation is likely decisive rather than bookkeeping.
- **Decoder validated** on 9,977,856 real codes: range invariance, bit-exact
  round trip, agreement between the two independent implementations, and
  structure. 28/28 checks.

Two open assumptions are recorded rather than claimed: **nibble order**
(`low_first`) and the **dequant grouping** including `input_scale`'s role. Both
pass every self-consistency rule while remaining falsifiable only against the
official code — a wrong nibble order would produce a wrong model silently. N3
must settle them first.

*Process note:* the inventory runner initially carried a contiguity check as a
hard gate that the preregistration does not list. It was moved to observations
to match the frozen document — a correction toward the preregistration, not a
relaxation of it.

### N3_ONE_MODULE_REFERENCE — PASS, and the nibble question is closed

CPU-only (a CPU torch build makes GPU contention with the 80B agent structurally
impossible, not merely unlikely). All twelve gates pass, every tolerance met with
4×–900× margin. Protected check: `PROTECTED_80B_INTACT`.

**Nibble order confirmed as `low_first`.** First, the negative result that
mattered: *nibble order cannot be falsified by any per-block statistic* — a
nibble swap permutes elements only within a byte, a byte lies wholly inside one
group of 16, so every block's value multiset is invariant. Block-amax,
histograms and scale checks are all blind to it. Resolved instead against
torchao 0.18.0's published `unpack_uint4`/`f4_unpacked_to_f32` on 2,097,152 real
elements: `low_first` matches codes and values exactly, `high_first` matches
neither. `torch.float4_e2m1fn_x2` exists but has no CPU conversion kernel, so
attempt 1 could not decide. Reading torchao's `NVFP4Tensor.dequantize` also
confirms the grouping structurally.

**Module comparisons** (mine vs the checkpoint's own `modeling_nemotron_h.py`,
identical inputs, relative L2):

| module | rel L2 | tol |
|---|---:|---:|
| RMSNorm | 6.84e-08 | 1e-06 |
| router logits | 2.52e-07 | 1e-06 |
| routed expert | 3.01e-07 | 1e-05 |
| shared expert | 1.75e-07 | 1e-05 |
| MoE aggregate | 1.89e-07 | 1e-05 |
| attention | 7.67e-07 | 1e-05 |
| **Mamba-2** | **5.44e-07** | 1e-04 |
| mixed block | 1.10e-07 | 1e-04 |

The Mamba-2 number is the strongest evidence in the phase: my implementation is
a **sequential recurrence**, the official one is a **chunked SSD factorisation**.
Different algorithms agreeing to 5e-07 is real evidence; two copies of one
algorithm agreeing would not be.

**Router cleared.** Top-6 index set matches exactly on all tokens, weights to
1.1e-07, and the minimum tie margin is 1.19e-03 — a thousand times above the
`tie_ambiguous` escape threshold, so the match did not depend on that clause. The
three traps were reproduced deliberately: selection on `scores + bias` but
weighting on **raw scores**, `sigmoid` not `softmax`, and the group-mask branch
that is a no-op at `n_group = 1` but implemented anyway.

**Two architectural findings that change later phases:**

1. **There is no RoPE.** `apply_rotary_pos_emb` appears nowhere; `rope_theta` is
   vestigial. The six attention layers are NoPE and positional information comes
   entirely from the Mamba layers. Long context here is not a RoPE-scaling
   problem, and `N13_1M_STRETCH` cannot be reframed as one.
2. **The shared expert is ungated** — a plain add. Qwen3-Next gates its shared
   expert and the project's D10 notes describe that variant; it must not be
   carried over.

**Per-sequence state is small.** From real shapes: KV is 3,072 elements/token
because only six layers carry one, and Mamba state is 47.078 MiB *constant* in
context. At 262K the whole per-sequence state is ~815 MiB. For scale, STREAMQ5's
Qwen3-30B needed 402 MiB of KV at **4K**. A 64× context ratio for ~2× the state.
This is a projection about state bytes only — it says nothing about attention
compute, expert streaming, or tok/s.

**One honest gap.** `mamba_ssm` needs CUDA and cannot be installed, so its gated
RMSNorm was supplied by us and used by *both* sides — that single op is therefore
**not** independently validated by N3. Everything else in the mixer, including
the whole SSD scan, is. Deferred to N6 coherence.

*Process note:* the runner first expressed the no-GPU gate as
`not cuda_available or True`, which is vacuously true. Replaced with
`not torch.cuda.is_available()` and re-run. A gate that cannot fail is not a gate.

### N4_ZERO_CACHE_DATAPLANE — NEGATIVE on G4, and the bottleneck moved

Terminal state `n4_zero_cache_screen_fail_unfused_decode_dominates`.
Independent verification **32/32**. Protected check `PROTECTED_80B_INTACT`.
First phase run on the real GPU (RTX PRO 2000 Blackwell, torch 2.9.1+cu128,
sm_120); free device memory before the run was 7,385,120,768 B, the exact figure
the protected STREAMQ5 verdict records for this machine.

| stage | p50 | p95 |
|---|---:|---:|
| transport, best arm | **29.756 ms** | 29.848 ms |
| decode only | 353.133 ms | 361.664 ms |
| composed token | 376.244 ms | **403.649 ms** |

**Transport is solved and is not the problem.** 774,533,280 B move at
26.03 GB/s — **99.51% of N1's assumed roofline**, 100.50% of its floor time —
with no cache and a 0.902 GiB device footprint. The 45 ms screen would pass on
transport alone with 15.2 ms to spare.

**Pre-pinning the bank eliminates PORT80B's host-gather term.** That line's
63.034 ms p50 decomposed into 37.204 ms of PCIe plus 25.830 ms of `mmap→pinned`
gather. Building the bank pinned once removes that 41% entirely. Transferable
mechanism, not a Nemotron accident.

**Batching is not the lever here.** 414 copies versus 3 differ by 1.232 ms
(3.98%), about 3 µs per copy. So `cudaMemcpyBatchAsync`, mapped-host and TMA —
all listed in the preregistration as remedies — are **not worth testing**: the
remaining gap to the roofline is 0.49%. That is a genuine saving of effort,
bought by measuring first.

**The wall is the unfused NVFP4 decode:** ≥10× transport, 93.9% of the composed
token. It expands 4,988,928 packed bytes into 9,977,856 float32 weights per
matrix through an int64 gather, a LUT lookup, a `repeat_interleave` and two
multiplies — ~120 MB of intermediate traffic per matrix, 276 times per token, to
feed two GEMVs that consume it once.

Correctness held throughout: bank records bit-identical to a fresh checkpoint
read, GPU decode **bit-identical** to the CPU float32 decode on both matrices
(max abs diff 0), expert output `rel_l2 = 1.859e-07` against the N3 reference.

**The residual is deliberately unnamed.** transport 29.756 + decode_only 353.133
= 382.889 > composed 376.244, so the arms do not sum — the decode-only arm forces
a scalar readback per matrix to prevent elision, adding syncs the composed loop
lacks. GEMV/allocation/loop overhead is left unattributed rather than guessed.
This is the rule the project learned when the "glue" term turned out to be
attention.

**The architectural stop was NOT declared**, and that matters. The preregistered
stop requires p95 > 60 ms *after registered/batched transfer **and a correct
fused kernel***. Batched transfer was tested; a fused kernel was not built. The
precondition is unsatisfied, so the stop must not fire however large 403.649 ms
is. Firing it would have condemned the architecture for the cost of an
implementation the preregistration itself designates as future work. The runner
records the precondition and the verifier checks it was honoured.

Two process corrections, both recorded in the registry:

1. The non-interference guard first blocked on "a foreign python process
   exists" — a false positive, since this machine runs short-lived python
   helpers with changing PIDs and a CPU-only process contends for nothing. It
   now blocks on `nvidia-smi --query-compute-apps` (does any other PID hold a
   CUDA context) plus a memory threshold, and fails closed on query error.
2. An intermediate run evaluated G4 on transport alone, which would have passed.
   G4 as preregistered is the routed-expert *path*, so the composed measurement
   was added and G4 re-evaluated against it. It fails.

This is a useful negative with a specific, falsifiable cause: the physical
architecture is not implicated, one implementation is. The next phase now has a
number to beat rather than a question to explore — **decode must fall from
~353 ms to under ~15 ms**.

Also adopted this session, matching the house convention: a **separate
independent verifier** per phase that re-reads the raw arrays, recomputes every
percentile, re-reads checkpoint bytes, recomputes the CPU decode from scratch,
and re-evaluates every gate without importing the runner.

### N4-R1 — fused NVFP4 kernel: decode repaired 25.66×, screen missed by 2.421 ms

Terminal state `n4r1_fused_screen_fail`. Independent verification **28/28**.
Toolchain: CuPy 14.1.1 + NVRTC, cc 120 — no host CUDA toolchain needed, and the
same tool the protected line uses.

| arm | p50 | p95 |
|---|---:|---:|
| transport only | 32.273 ms | 32.683 ms |
| **fused compute** | **13.762 ms** | 13.991 ms |
| composed token | 46.954 ms | **47.421 ms** |

The N4 diagnosis was right and the repair works: 353.133 ms of unfused decode
becomes **13.762 ms**, beating the ~15 ms target N4 had set. The kernel assigns
one block per output row, streams that row's packed bytes and block scales, and
decodes in registers — never materialising the dequantised matrix.

That claim is settled by measurement, not assertion: **peak device pool exceeds
the working set by 68,448 B**, while a single materialised `[1856, 2688]`
float32 matrix would be 19,961,856 B.

The composed token still missed the 45 ms screen by 2.421 ms, because transport
and compute ran strictly serially. Bit-exactness of the fused *output* was
deliberately not claimed — the kernel uses a warp-shuffle reduction tree, not
the reference's sequential order.

### N4-R2 — causal overlap: PASS, and bit-identical

Terminal state `n4r2_overlapped_zero_cache_screen_pass`. Independent
verification **34/34**. Protected check `PROTECTED_80B_INTACT`.

| arm | p50 | p95 |
|---|---:|---:|
| transport only | 29.888 ms | 30.017 ms |
| composed serial | 45.853 ms | 50.917 ms |
| **composed overlapped** | **31.506 ms** | **39.714 ms** |

**The decisive gate was O3, not the speed.** Overlap is admissible only if it is
semantically invisible:

```text
differing words   0 of 2,688
serial SHA-256    == overlap SHA-256
```

A speedup with even one changed bit would have been a failure, not a trade-off.
This reproduces on Nemotron what STREAMQ5 P4A established on the protected
line — the mechanism was ported, not invented.

Cumulative: **11.94×** against N4's unfused composed token, **1.490×** against
R1's serial path.

### H3 closed — the whole sequence

1. **Transport is at the roofline.** 774,533,280 B in 29.888 ms; 26.03 GB/s against an assumed 26.158915.
2. **Pre-pinning eliminates the 80B line's 25.830 ms host-gather term.**
3. **Batching is not the lever**: 414 copies vs 3 differ by 3.98%.
4. **Fusing decode into the GEMV removes 96.1% of decode cost** for +68,448 B.
5. **Causal overlap closes the gap and changes 0 of 2,688 words.**
6. Complete cache-free routed path: **31.506 ms p50 / 39.714 ms p95**, 0.721 GiB,
   **no expert cache of any kind**.

**Honest caveats.** The p95 margin is 5.286 ms with a wide tail — the fused
compute arm moved 13.762 → 18.454 ms p50 between R1 and R2, a 34% run-to-run
shift on identical work. The cause is *not* diagnosed; naming it without
measuring it is the error this project recorded over the "glue" term. Thermal
and steady-state characterisation is N12 work. Routes remain synthetic-input
routes. And this is one component — attention, Mamba-2, trunk, LM head and
sampling are not in the number, so it is never promoted to tok/s.

### Protected-line activity — isolation checked, not asserted

The post-N4-R2 verification reported **66 added files, 0 modified, 0 removed, 0
listing changes**. All 66 are `HET_NEXT_L0_PH1_NVIDIA_NC0..NC13`
preregistrations, design audits, errata and locks dated today — the other
agent's own work, in a directory this line has never written to.

So the protected line advanced from the N5 static-preflight NO-GO to NC0–NC13
compile-only work while this line ran four GPU phases, and not one protected
byte was modified or removed. Additions by the owning agent are informational by
design; only modification, removal or a listing change triggers the hard stop.
No GPU contention was ever observed — every GPU phase checked
`nvidia-smi --query-compute-apps` and found zero foreign CUDA contexts.

### N5_PHYSICAL_RESIDENT_SHELL — PASS, it fits with 2.99 GiB to spare

Independent verification **32/32**. Every prior memory statement in this line was
arithmetic; N5 allocates and **touches**, measured as the `cuMemGetInfo` driver
delta so context overhead and fragmentation are inside the number.

| variant | embedding | lm_head | peak device | free |
|---|---|---|---:|---:|
| **A** | device | device | **4.9664 GiB** | **3065 MiB** |
| B | host | device | 4.9664 GiB | 3737 MiB |
| C | host | host | 4.9664 GiB | 4409 MiB |

All three fit, holding simultaneously: the 2,816,404,992 B shell, the
774,533,280 B expert staging buffer, 49,364,992 B of Mamba state and
402,653,184 B of FP8 KV at **131,072** context.

**H4's "embedding/LM head when physically justified" is answered: both are
justified on device.** The runner labels C "best" because it minimises bytes —
that label is misleading. Variant A is the right choice: memory-minimal is not
design-optimal when there is headroom, and C would trade 1.34 GiB of unused
space for two host round-trips per token.

The sharpest consistency test passed exactly: moving the embedding to host saves
precisely 704,643,072 B, and moving the head saves precisely 704,643,072 B.

Two things worth carrying: the **1,161,363,456 B driver reserve** exists before
any allocation of ours and matches the protected D10 figure for this GPU —
budgeting against `total` instead of `free` would overstate memory by 1.08 GiB.
And **teardown leak is exactly 0**.

**Cache budget is now measured, not estimated:** 3,214,082,048 B free =
**572 routed expert slots**, 19.4% of the 2,944-record bank.

Two process corrections: the first run reported a 1.56 GB teardown leak that was
a *runner* defect (clearing the list didn't drop local references); and S3 read
`peak_commit` as 0, making the gate vacuously true — it now uses
`K32GetProcessMemoryInfo` and **fails closed** when unmeasurable.

### N6-A — the model speaks

Independent verification **32/32**. Terminal state `n6a_full_depth_coherent`.

| prompt | top-1 | p |
|---|---|---:|
| **`The capital of France is`** | **` Paris`** | **0.9603** |
| `1, 2, 3, 4,` | `' '` | 0.9255 |
| `def add(a, b):\n    return` | **` a`** | 0.9865 |

All 52 layers, real NVFP4 weights, CPU float64, weights dequantised per layer
and released — peak commit 6.734 GiB against a 32 GiB gate, where the full model
in float32 would be ~117 GB.

**This pays the debt N3 recorded.** N3 could not validate the gated RMSNorm —
`mamba_ssm` needs CUDA, so our implementation sat on both sides of that
comparison — and deferred it explicitly to "end-to-end coherence at N6". A
52-layer forward emitting ` Paris` at p=0.96 is a **joint** confirmation of the
gated RMSNorm, the `low_first` nibble order, the dequant grouping and the
`up → ReLU² → down` / ungated-shared / router-bias-split semantics at once. The
caveat is recorded: passing supports the set as a whole; failing would not have
said which member was wrong.

**First natural routes.** Every earlier phase used synthetic-input routes. N6-A
captured 552 real route rows across all 23 MoE layers — and the statistic
matters for H5: **all 128 experts appear**, usage max 61 / min 7, a spread of
only 8.7×. A cache of N5's 572 slots cannot be filled by "the popular experts",
so STREAMQ5's static+dynamic split must be **re-derived here, not inherited**.
Three prompts are not a routing distribution, and the artifact says so itself.

A pleasing confirmation: the first run crashed on
`backbone.layers.4.mixer.in_proj.weight_scale`, which does not exist — layers 4,
11, 18, 25, 32 and 41 keep those in BF16, exactly as the N0R `exclude_modules`
analysis predicted. The crash **confirmed** that reading.

### Stand van zaken

Two halves now exist and both work: a **correct** full-depth graph (N6-A, CPU)
and a **fast, memory-feasible** routed dataplane (N4-R2 + N5, GPU). Joining them
is engineering, not research — every component question it depends on has an
answer.

### Next

`N6-B`: the GPU decode loop — N5's resident shell plus N4-R2's overlapped routed
path as a single-token device forward, gated on reproducing N6-A's logits within
a declared tolerance and on the same natural routes. Only then is H5 meaningful.

---

## Sessie 2026-08-14 (avond) — nieuwe wetenschap voor 50 tok/s: census S1–S4 + S5 start

**Opdracht.** Niet verder optimaliseren; nieuwe wetenschap voor 50 tok/s bij
lange context. N8-eindstand gereproduceerd vóór alle nieuwe werk:
22,062 / 20,147 / 16,686 / 13,143 tok/s bij ctx 0 / 32K / 131K / 262100
(runner `n7b_cached_decode.py`, capacity 31, FP8 KV) — binnen ruis gelijk aan
het N8-rapport. Omgeving is dus gezond.

**S1–S4 hypothese-census** (preregistratie, input lock, runner, aparte
verifier die alles herberekent zonder de runner te importeren, 20/20):
- S1 route-voorspelling (temporele bigram): recall@12 = 0,611, recall@24 =
  0,724 < 0,80-sluitpoort → **weerlegd**. Prefetch op routevoorspelling is
  met deze predictorklasse dood. Adjacent-layer overlap 0,050 — vrijwel nul.
- S2 ReLU²-sparsity: **90,69% exacte nullen** in de intermediates
  (72.174 expert-calls, 2 prompts × 256 greedy tokens). Poort ≥ 0,45 ruim
  gehaald → **geopend**. Cruciaal: nullen clusteren NIET (16-kolomsblokken
  volledig nul: 30,6%; 64-koloms: 4,9%) → selectie moet kolom-nauwkeurig.
- S3 code-entropie: nibble-entropie min 3,9671/4 bits; H(B|A) tussen experts
  3,9663 → lossless én delta-codering **weerlegd**.
- S4 MTP-scan: 0 van 24.147 gewichten draft-achtig → speculatief decoden via
  uitgever-gewichten **gesloten** (er bestaan geen draft-gewichten).

Belangrijke modelobservatie: de Nemotron-MoE-expert heeft maar TWEE matrices
(up → ReLU² → down), geen gate. Kolom j van down_proj met h[j]=0 doet niet mee
→ tot ~45% van de miss-bytes beweegt voor niets.

**S5 start — kolom-selectieve down_proj.** Preregistratie
`S5_COLUMN_SELECTIVE_DOWN_PROJ_PREREGISTRATION_2026-08-14.md`, baseline-generatie
bevroren (`s5_baseline_generation.json`, 32 tokens × 2 prompts, ongemodificeerde
runtime). Implementatie: down_proj panel-major getransponeerd in de bank
(116 panels × 16 kolommen; per panel 2688 scale-bytes + 16 kolommen à 1344 B
contiguous), masked GEMV-kernel met deterministische scan. Smoke
(`s5_masked_smoke.py`): transpose exact (permutatie-bewijs), masked kernel
rel_l2 = 1,77e-07 vs fused referentie, mapped-host == device bit-identiek.

**Component-negatief binnen S5 (variant A zoals ontworpen):** de masked kernel
leest host-geheugen met 1 byte/thread → microbench
(`s5_mapped_read_microbench.py`): byte-per-thread 1,78 GB/s → ~320 µs/call →
onhaalbaar. Maar dezelfde microbench toont: **uchar4-brede coalesced reads
halen 25,05 GB/s — vrijwel de PCIe-piek van 26,03 GB/s**, en verspreide
1344 B-copies via de copy engine zijn dood (0,16 GB/s, 8,3 µs/copy).
Conclusie: de SM kan zelf de DMA zijn als de loads breed zijn. Ontwerp A2
(warp-per-kolom gather-kernel naar device-staging, daarna dezelfde masked
GEMV vanaf device) volgt als S5-R1 met eigen preregistratie-addendum; poorten
blijven ongewijzigd.

**S5 afgerond — gemengd verdict.** Design A2 gebouwd en gegate gemeten
(`s5_masked_decode.py`, rapport `S5_COLUMN_SELECTIVE_DOWN_PROJ_REPORT_2026-08-14.md`,
verifier 9/9). Correctheid volledig: generatie bit-identiek aan de bevroren
baseline (2×32 tokens), transpose exact over alle 2.944 records, per-call
rel_l2 ≤ 1,88e-07 op 2.208 echte calls, geen regressie bij ctx 0 (21,759 vs
poort 21). Prestatiepoorten P1/P2 bij 262K **gefaald**: 13,678 tok/s (poort
15/18), effect +4,1% t.o.v. baseline 13,143. Mechanisme-conclusie: de
byte-besparing is echt maar bij 262K domineert attention (~39 ms vs 3,2 ms
vloer, ~12× boven roofline), en bij ctx 0 was de transfer al verborgen — de
premisse "MoE-miss-PCIe domineert" is voor deze runtime weerlegd. Bijkomend:
cache nu 1,86 GiB (up-only), 1,4 GiB VRAM vrij bij 262K. Eerste verifier-run
faalde op een spy-bug (verifier-protocol-negatief, hersteld, zelfde artefacten
opnieuw beoordeeld: 9/9). Protected manifest: PROTECTED_80B_INTACT na elke fase.

**Volgende stap: S6 componentbreakdown @262K op de masked runtime** — meten
waar de 73 ms zit vóór een nieuwe hypothese wordt gekozen.

**S10-A — MTP-acceptatiegraad gemeten, poort gehaald.** Preregistratie
`S10A_MTP_ACCEPTANCE_PREREGISTRATION_2026-08-15.md` bevroren vóór uitvoering.
Eerst de bestaande meting gereproduceerd (27,574 / 25,523 / 21,794 / 18,358
tok/s tegen 27,743 / 26,200 / 21,699 / 18,424 — alle vier binnen 2,6%, shell en
cache identiek), zodat een afwijking later niet op de omgeving kon worden
geschoven.

*A1, wiring-resolutie.* De MTP-bedrading stond niet vast: `transformers` 5.15.0
negeert `mtp.*` expliciet en de modelmap heeft geen `modeling_*.py`, dus er is
geen referentie. Vier combinaties (concat-volgorde × bron van `h`) gescoord op
bevroren teacher-forced WikiText-2, 632 posities per variant, beslisregel vooraf
= laagste gemiddelde NLL met afbreekplafond 7,0 nats. Uitkomst:
`eh_proj( concat( enorm(embed), hnorm(norm_f(h)) ) )` — dus de **genormaliseerde**
hidden, ná `norm_f`. NLL 3,262 tegen 4,018 voor de ruwe hidden en 10,46/10,53
voor de omgekeerde concat (uniform = 11,78). Anker: de backbone haalt 2,473 nats
op next-token, het MTP-blok 3,262 op next-next-token.

*A2, de poort.* `D = 4`, geketend op de eigen hidden state, greedy vergeleken met
wat de backbone zelf produceert. Gepoold **`A` = 2,114** over 360 stappen en 3
prompts — **G-S10-1 (≥ 1,5) gehaald**. Spreiding tussen domeinen is groot:
narrative 1,208 (zou alleen falen), expository 1,850, code 3,283. De
acceptatieladder is opvallend vlak (0,786 / 0,753 / 0,728 / 0,710) en `A = 4` is
met 110 van 360 de grootste klasse, dus `D` kapt af — `D > 4` is een reële
kandidaat. Secundair, geen poort: bij 4.096 tokens context `A` = 2,083, dus de
acceptatie stort niet in met diepte. Verifier 48/48 `VERIFIED`, protected
0 modified / 0 removed.

*Kostenkant, gemeten en tegenvallend.* De MTP-keten van vier drafts kost
**19,10 ms** (p50) mét alle 128 MTP-experts device-resident — de gunstigst
mogelijke opstelling — tegen de ~2 ms per forward die de S10-preregistratie
aannam, dus 2,4× duurder. Residentie is wel betaalbaar: shell 1,795 + cache
1,924 (capacity 32) + MTP 2,379 GiB paste met 0,577 GiB over; bij 262K komt dat
neer op capacity ≈ 30, tegen de 72 van nu.

*Wat stap 2 eerst moet meten.* De MoE-term is 39,5 van de 54,3 ms per token en is
precies de term die **niet** meeschaalt bij verificatie van D+1 tokens: hij volgt
de **unie** van de routes. Uit N7-A's gemeten overlap (2,011 van 6) volgt een
bovengrens van 21,96 unieke experts per laag bij 5 tokens tegen 6 nu (≤ 3,66×).
Attention, Mamba-projecties en `lm_head` amortiseren wél. Die unie is met
`step(capture_routes=...)` te tellen zonder ook maar één kernel te schrijven, en
moet vóór elke bouw gemeten worden. `G-S10-C1` blijft staan.

**S11 — volledig-record caching weerlegd bij gelijke bytes.** De open vraag uit
S9. Drie armen in één proces, één modelload: A₁ up-only @72, B full-record @36,
A₂ up-only @72. Een full-record-slot is exact twee keer een up-only-slot
(2.806.272 B elk, onafhankelijk nagerekend uit de safetensors-headers), dus beide
armen houden exact 4,328 GiB vast en de enige variabele is wát erin ligt.
Uitkomst: B verliest overal — ctx 0 25,551 vs 27,078, ctx 131.072 20,903 vs
21,466, ctx 262100 **17,346 vs 18,227 (−4,84%)** tegen een adoptiepoort van +3%.
**G-S11-P1 gefaald, up-only blijft.** G-S11-C1 gehaald: de generatie is
bit-identiek in alle drie de armen, dus de dataplane-wissel is exact. G-S11-D1:
A₂ wijkt 0,23% van A₁ af tegen een effect van 4,84%, met identieke hit/miss-tellingen
(4.530) — conclusief.

Mechanisme: halvering van de capacity kost hitrate 0,879 → 0,690, dus 2,57× zoveel
misses, elk met het dubbele aantal bytes = 5,1× het miss-verkeer. Dat de 37.531
weggevallen `gather_down_sparse`-calls dat niet compenseren is het antwoord.
Bijvangst die verder reikt dan de poort: arm B verplaatst **2,9× zoveel PCIe-bytes
en verliest maar 4,8%** — als de MoE-term transfergebonden was geweest, was dat
catastrofaal geweest. Tweede onafhankelijke bevestiging van S8. Verifier 41/41.

**S12 + S12-R1 — in-lus attributie van de MoE-term.** S8's les omgezet in
methode: nooit een component apart timen, maar de échte lus draaien met precies
één extra aanroep van één component naar een kladbuffer, en het end-to-end
verschil lezen. De probe zit in een subklasse in het runnerscript; `runtime.py`
blijft onaangeraakt en de verifier controleert dat het woord `probe` er niet in
voorkomt. Generatie bit-identiek in alle armen.

De eerste run **faalde op zijn eigen driftpoort** (base₂ − base₁ = 5,057 ms bij
262100, groter dan de kleinste marginaal) en is zo gerapporteerd. Oorzaak
zichtbaar in de herhaling, die temperatuur meelogt: **86–87 °C**; de probe-armen
verwarmen de GPU en base₂ draaide na alle vijf. Poort niet verruimd, schema
veranderd: S12-R1 zet elke probe tussen twee basislijnen en meet tegen hun
gemiddelde, met een eigen lokale ruisvloer per probe.

Marginale in-lus ondergrenzen bij 262100: `down` +7,478 · `up` +4,756 ·
`shared` +3,298 · `accum` +0,533 (onder de ruisvloer). Samen dekken de drie
schone marginalen **15,53 van de 39,52 ms** MoE-term — meer dan S9's
microbenchmark-grens van 9,0 ms, wat klopt omdat de in-lus marginaal van `down`
ook `panel_scan`, de gather en de reductie bevat. Geen enkele component verklaart
de term ook maar half: de 39,5 ms zit niet op één plek waar een kernelherschrijving
hem weghaalt.

De `router`-probe (+8,156 ms) is **niet vergelijkbaar** met de andere vier en
telt niet mee: hij repliceert ook de device→host-readback, en die is een sync die
overlap vernietigt op de plek waar hij staat — aan het einde van de laag, waar
zes experts in de wachtrij staan, terwijl de échte readback vroeg in de laag zit
juist zodat de shared expert eroverheen loopt. Ontwerpfout van deze probe, hier
benoemd in plaats van weggepoetst. Wat het getal wél corroboreert: 8,156 ms over
23 lagen = 0,355 ms per laag, tegen de 0,339 ms per laag die de runtime zelf uit
een eerdere meting citeert. Correcte vervolgmeting = dezelfde probe met
**gematchte plaatsing**, eigen preregistratie. Verifier 99/99.

**S13 — expert-unie over speculatieve vensters: bouwpoort negatief, speculatieve
lus weerlegd vóór de bouw.** De beslismeting die S10A §5 voorschreef en die het
SWEEPSPEC_50-pack (LIGHTNINGFLASH_50) als make-or-break erft: hoeveel unieke
experts raakt een verificatie-sweep over W opeenvolgende tokens per MoE-laag.
Vier armen (drie bevroren S10A-gate-prompts à 124 stappen + een 4K-prompt à 64
stappen), routes uit de échte greedy generatie via het bestaande
`step(capture_routes=...)`; `runtime.py` onaangeraakt, generatie bit-identiek
aan de S10A-sequenties (G-S13-C1) zodat unie en acceptatie uit dezelfde
toestanden komen.

Gepoolde unie-curve (van 128): W1 6,000 · W2 9,895 · W3 13,421 · W4 16,659 ·
**W5 19,512** · W6 22,180 · W8 27,204. Elke extra token voegt ~2,5–3,9 experts
toe die de vorige niet raakte. Poort G-S13-U1 (≤ 12,0 bij W=5) **gefaald** —
niet verruimd. De betekenis: per geverifieerde token daalt de MoE-bytekost wél
(3,90 records versus 6,00), maar een W=5-sweep commit bij de gemeten acceptatie
(A = 2,114) slechts 3,114 tokens, dus per gécommitteerde token beweegt de sweep
**6,27 records tegen 6,00** zonder speculatie — méér, niet minder — vóór de
19,10 ms MTP-draftketen (S10A) meegeteld is. En het domein met de hoogste
acceptatie (code, A = 3,283) heeft ook de hoogste unie (20,862): de twee
parameters die speculatie zouden redden werken tegen elkaar. Grotere bomen
vergroten de unie alleen maar. Beslissing: **niet bouwen**; LIGHTNINGFLASH_50
gesloten voor dit model in deze vorm. Verifier 43/43, manifest 0 modified /
0 removed. Rapport: `S13_EXPERT_UNION_REPORT_2026-08-15.md`.

**S14 — GPU-event-tijdslijn van de MoE-laag: de restpost heeft een naam.** S12's
marginale methode kon niet zien wáár de ~24 ms zat die S8's MoE-term (39,5 ms)
niet toeschreef. Deze fase zette CUDA-events mét timing tussen de fasen van
`_moe_cached` (subklasse in de runner; `runtime.py` onaangeraakt) — timestamps
op de stream, geen extra host-sync, dus zonder S8's meetartefact. Armen
base0·probed·base1, ctx 0 en 262.100, 16 tokens per context. Poorten C1
(bit-identiek), P1 (probelast +3,0–5,0 ms < 20%), S1 (boekhouding klopt per
token ≤ 0,05 ms) allemaal gehaald; verifier volledig groen.

Resultaat: de MoE-lagen bezetten **27,7 ms stream-wandtijd per token bij beide
contexten** — context-onafhankelijk, zoals de dataplane bedoeld is. Segmenten
@262K: expert-werk `up` 6,55 + `down_masked` 8,39 + `accum` 1,00 · `route` 3,50
(launch-gebonden: 344 kFLOP in tien launches) · shared 3,58 · **`host_gap` 4,67
ms GPU-stilstand** terwijl de host readback-afhandeling, LRU en copy-issue doet.
De rest van S8's 39,5 was nooit MoE-werk: `readback_host` is bij 262K 18,0 ms
groter dan bij ctx 0 (31,0 vs 13,0) terwijl elk MoE-segment context-onafhankelijk
is — dat is de in de wachtrij staande attention (S8: 18,6 ms) die bij de enige
sync van de lus afdraait. Twee onafhankelijke metingen, dezelfde grootheid.

Gevolg voor 50 tok/s, als vloer-aritmetiek (expliciet geen meting): MoE-vloer
~16 + attention-vloer 3,3 (KV eenmaal lezen, S7) + Mamba ~8,3 + lm_head 2,1 ≈
**30 ms/token ≈ 33 tok/s bij 262K**. 50 tok/s vereist 20 ms en ligt daarmee
buiten de gemeten fysica van dit model op deze GPU zolang de semantiek (KV-bytes,
expert-bytes per token) gelijk blijft. Rapport:
`S14_MOE_LAYER_TIMELINE_REPORT_2026-08-15.md`.

**Synthese — de 50-tok/s-grens is nu aantoonbaar.** Zie
`reports/lightningstream_nemotron/SYNTHESIS_50TOKS_CEILING_2026-08-15.md`.
Kern: alle "wat beweegt"-assen zijn gemeten en afgesloten (S1–S14);
vloer-aritmetiek op gemeten componenten geeft ~30 ms/token ≈ **33 tok/s bij
262K** als fysieke grens bij ongewijzigde semantiek; 50 vereist 20 ms. Resterende
ruimte (~25–30 tok/s) is engineering tegen bekende vloeren: host_gap (4,7 ms),
router-fusie (3,5 ms), in-lus GEMV-vertraging (~6 ms), attention richting de
KV-leesvloer (~15 ms). Geen daarvan is nieuwe wetenschap en geen haalt 50.

**Versie-notitie (2026-08-15).** Deze lijn draait sinds de v35-overstap op
`models/nemotron_3_5_lightning_v35` — **Nemotron 3.5 Lightning** 30B-A3B NVFP4,
niet Nemotron 3 Nano uit de oorspronkelijke opdrachttekst. De bevroren
opdracht-baseline (21,4 / 20,2 / 16,7 / 13,2 tok/s) is het 3.0-model; alle
fasen vanaf N2R/S10A (en de huidige baseline 27,7 / 26,2 / 21,7 / 18,4) zijn
3.5 Lightning. Versie-afhankelijk feit met wetenschappelijke relevantie: het
MTP-blok bestaat alleen in 3.5 Lightning — S10A/S13 zijn dus per definitie op
het 3.5-checkpoint gemeten, en de speculatie-weerlegging geldt mét de sterkst
beschikbare drafter (het eigen MTP-blok van het model).

**K0/K1/K2 — Kimi's LightningSpec-50 P0 gemeten.** Kimi's correctie op mijn
S10-A-drempel is overgenomen en klopte: mijn "unie boven 12 = negatief" was fout
gerekend; de pariteit is `U* = top_k x (A+1) = 18,683`. Er is verder geen drempel
verzonnen — de AR-basislijn van 6 records per uitgestoten token is het criterium.

*K0, op de officiele routes* (`step(capture_routes=...)`, 3 domeinen, >=300
vensters per B, verifier 62/62). Gemiddelde unie: B=2 10,06 · B=3 13,63 ·
**B=5 19,88** · B=7 25,23 · B=9 30,01 · B=13 38,24. Bij B=5 is dat **6,384
records per uitgestoten token tegen 6 autoregressief — G-K0-1 gefaald**, nipt
boven pariteit. Alleen D=1 (5,631) en D=2 (5,734) staan aan de goede kant, en
dat is 6,2% respectievelijk 4,4%. Structureel: de unie groeit met 2 a 3,6
experts per extra token terwijl de uitgestoten tokens geometrisch afvlakken, dus
dieper drafting verergert het. LRU-replay in ronde-orde kost bij elke capacity
(32..72) **1,62x** zoveel misses per uitgestoten token; **G-K0-2 gefaald**.
Meetnotitie: de runner deelde door de gemiddelde serielengte i.p.v. tokens x
prompts, dus zijn `*_per_emitted`-velden staan een factor 3 te hoog; de poort is
een verhouding en blijft ongewijzigd, en de verifier herberekent het correct
(24,63 misses/token bij cap 72 = 23 x 6 x (1 - 0,8215) ✓).

*K1, waar de draft-keten zijn tijd laat* (gebracketeerd, globale drift 0,192 ms
over elf armen — de stabielste meting van de lijn). Marginalen per keten van vier
drafts: **`lm_head` +10,508** · experts +3,546 · attention +1,262 · shared +0,824
· `eh_proj` +0,733, samen 16,87 van een basislijn van 20,66 ms. **De LM-kop is
~51% van de draft-keten, de experts 17%.** Dat herschikt Kimi's H1: kwantiseren
van de MTP-experts valt de kleinste post aan, het actieve vocabulaire de grootste.

*K2, actief vocabulaire (MicroSpec), een variabele.* Top-N rijen van de logits die
de backbone zelf al berekende, per commit-positie. N=4096: recall 0,932, gepoolde
`A` 2,003 (van 2,114), keten **18,83 -> 10,52 ms (-44,9%)**. N=2048: 0,889 /
1,928. N=1024: 0,820 / 1,814. **G-K2-T1 gehaald (-44,9% tegen -30%), G-K2-R1
(>=99,5% recall) en G-K2-A1 (acceptatieverlies <=0,05) gefaald; geen enkele N
haalt alle drie.** De tijdwinst is precies wat K1 voorspelde: 8,31 ms over vier
drafts = 2,08 ms per `lm_head`, tegen de 2,106 ms die S8 los mat — twee
onafhankelijke methoden, hetzelfde getal. Het is de *selector* die faalt, niet
het idee; top-N van de huidige logits is de simpelste denkbare.

*De spanning die overblijft.* P0 komt negatief uit op zijn eigen criterium, maar
P0 meet **records** en S12 heeft gemeten dat de MoE-term niet expert-load-gebonden
is (per-expert marginalen 12,23 ms van de 39,523). Als de resterende ~24 ms per
laag is en niet per expert, kost een B=5-sweep 21,8 ms per uitgestoten token
tegen 39,5 nu; schaalt de hele term met het expert-aantal, dan 42 ms. Factor twee
verschil, en alleen een meting scheidt ze. Zelfde les als S8 en S11: op deze
runtime voorspelt byte-boekhouding geen tijd. **Volgende meting, afgebakend:** de
*tijd* van een MoE-laag over de unie van vijf routes met expert-major groepering,
tegen 5x het huidige per-token-pad — niet de hele P1-verifier.

**X1 — SweepSpec gebouwd, exact, en weerlegd.** Beide agent-packs (ExactFlow
hypothese E / prioriteit 2, LightningSpec P1) zetten de expert-major
blockverifier voorop. Gebouwd in `sweepspec.py`, buiten `runtime.py`: een
gebatchte NVFP4-GEMM die het gewicht een keer leest en tegen B activatievectoren
vermenigvuldigt, plus een gebatchte masked down-GEMV over de unie-panelmasker.
Node-selectie en scatter via index-arrays, bijdragen per (node, slot) en pas
daarna sommeren in route-volgorde.

*Exactheid volledig.* **460 laag-blokken, 0 mismatches, worst rel_l2 0,000e+00**,
en de gebatchte kernel is los getest bit-identiek aan `gemv_nvfp4_rows` voor B
t/m 8. De unie-mask voegt alleen kolommen toe waar de activatie exact nul is en
`fmaf(w,0,acc)=acc`; bij `nchunks=1` is er ook geen chunk-herverdeling. G-X1-C1
en G-X1-C2 gehaald.

*Prestatie weerlegd.* Verhouding sweep/sequentieel: B=1 1,1465 · B=2 1,0753 ·
B=3 1,0343 · B=4 1,0192 · **B=5 1,0017**, tegen een poort van 0,6228. Het
sequentiele pad schaalt 4,959x bij B=5, dus vrijwel perfect lineair in het aantal
posities, en de sweep die elk record 18,85 i.p.v. 30 keer per laag laadt komt op
dezelfde tijd uit. **De MoE-term is lineair in het aantal (node,
expert)-toewijzingen, niet in het aantal unieke expert-records.** Dezelfde
conclusie als S11 (2,9x PCIe kost 4,8%) en S12 (per-expert marginalen 12,23 van
39,5 ms), nu langs een derde onafhankelijke weg.

*Gevolg.* Alleen het routed-expert-deel van een B=5-verificatie kost 111,5 ms,
bij korte context en met alles resident — 1,8x het hele rondebudget van 62,280 ms
voor 50 tok/s, voor router, shared, Mamba, attention, LM-kop en draft. Per
uitgestoten token: B=2 +9,7% · B=3 +24,7% · B=4 +40,8% · B=5 **+59,3%** t.o.v.
autoregressief. Speculatieve verificatie maakt de dominante term bij **elke**
blokgrootte duurder, want de kosten volgen B terwijl de opbrengst 1+A is. Dat
sluit SweepSpec, ExactFlow-E en LightningSpec-H2/H3 als route naar 50 tok/s.

*Kernelnotitie.* Eerste versie faalde met ILLEGAL_ADDRESS zodra B>=4; drie probes
(over B, over rows/cols, over shared size) toonden dat het puur van B afhing en
niet van een resourcelimiet (B=4 faalde bij 16 KB shared terwijl B=3 bij 32 KB
slaagde). Oorzaak: de lokale array `float acc[MAX_B]` met runtime-index. De
b-lussen zijn compile-time ontrold met `if (b >= B) break;`, `local_size_bytes`
ging van 32 naar 0. Exactheidspoorten daarna opnieuw gedraaid, niet ervoor.
Verifier 35/35.

**Y1/Y2 — de readback is 6,7 ms waard; bytes snijden koopt 34% van de GEMV.**

*Y1, de host-round-trip.* De routed lus synchroniseert per MoE-laag omdat de
expert-ids naar de host moeten (23 syncs per token); Kimi's S14 mat daaromheen
`host_gap` 4,7 ms GPU-idle plus een launch-gebonden `route` van 3,5 ms. Oracle,
geen bouw: arm B draait de echte lus, laat `_route_device` volledig op device
draaien maar leest hem niet terug — de ids komen uit een capture van dezelfde run,
dus per constructie identieke routes. **Generatie bit-identiek** (G-Y1-C1).
Winst: **+4,359 ms bij ctx 0 (11,2%, drift 2,121)** en **+6,656 ms bij 262100
(11,3%, drift 0,919)**, beide conclusief en beide binnen S14's `host_gap+route`-
grens, wat bevestigt dat de opzet de sync meet en niet iets anders. Grootste
niet-semantische winst die in dit model gevonden is. Arm B is geen implementatie:
LRU-bookkeeping en copy-issue blijven host-werk, ze overlappen alleen in plaats
van te stallen.

*Y2, de byte-premisse onder ExactFlow A/B/C/D.* Eerste pass was fout gemeten —
sync na elke call, dus een vaste ~7 us launch bovenop een ~40 us kernel — en dat
was zichtbaar: de curve werd niet-monotoon (75% trager dan 100%, 25% trager dan
50%). Opnieuw met 200 calls per sync: 100% 34,472 us · 87,5% 30,822 · 75%
26,703 · 50% 22,692 · 25% 16,781, monotoon. **Halvering bespaart 34,2%** tegen
een poort van 40% — **gefaald, maar nipt**. Lineaire fit **8,13 us/MB + 10,90 us
vast**, dus de kernel is voor **31,6% byte-onafhankelijk**. Effectief 81,4 GB/s
tegen ~250 GB/s roofline, consistent met S9's 86,5. Verifier 33/33.

*Bovengrens, aritmetiek op gemeten componenten.* Op S14's segmenten bij 262K:
readback weg 6,66 + expert-bytes halveren 5,11 + shared idem 1,22 = **12,99 ms**.
54,28 -> 41,3 ms bij 262K; 36,05 -> 25,4 ms bij ctx 0. Dat veronderstelt al dat
een 2-bits codec bestaat en zijn kwaliteitspoorten haalt. De 20 ms die 50 tok/s
vraagt zit er niet in: factor 1,27 tekort bij ctx 0, factor 2,07 bij 262K.

*Stand van beide packs.* Geen hypothese meer open die niet gemeten is of begrensd
door een meting. SweepSpec/H2/H3/E gebouwd en weerlegd (X1). P0/S13 gemeten,
gefaald. H1 punt 4 gemeten, recall gefaald; punten 1-3 begrensd door K1 (experts
17% van de draft-keten). H4/H6/H7/H8/F begrensd door X1 (kosten volgen B, dus
perfecte acceptatie breekt hooguit quitte). A begrensd door Y2-R1 (12-20%
packreductie -> 4-7% van de GEMV). B/C/D begrensd door Y2-R1 (halvering -> 34%).

**Z1 — TreeSweep Oracle A: het plafond van elke boomverifier.** Het derde pack
(TREESWEEP_200) vraagt 200 tok/s en sluit dat programma als de gezamenlijke
oracle faalt: `max_N A_oracle(N)/T_v(N) >= 250 tok/s`. Omdat een boom van N
geverifieerde posities er hoogstens N kan committen, geldt voor **elke** drafter
en **elke** topologie `A_oracle(N)/T_v(N) <= N/T_v(N)`. Is `T_v` lineair in N,
dan is dat `1/c` — een constante die geen boomgrootte kan overtreffen.

Gemeten over N = 1/2/4/8/16, alle 23 MoE-lagen, echte hidden states en echte
officiele routes, elke expert resident: 16,630 · 16,260 · 15,755 · 16,352 ·
16,468 ms per positie. Lineaire fit **16,505 ms/positie**, **R2 = 0,99986**,
`T(16)/T(1) = 15,845` tegen 16. **G-Z1-L1 gehaald.** Plafond **60,59 tok/s**;
X1's onafhankelijke meting van dezelfde grootheid geeft 44,90. **G-Z1-P2C
gefaald met een factor 4,1 tot 5,6.** Verifier 27/27.

`G-Z1-S1` (sanity tegen X1's 22,454 ms) **faalt** en dat blijft staan: X1 middelde
over acht blokken met doorstromende LRU, Z1 gebruikt een warm blok. Beide zijn
echte metingen van hetzelfde pad onder andere cachedruk; voor de conclusie maakt
het niets uit, want beide plafonds liggen ver onder 250.

Het plafond veronderstelt al perfecte dekking, nul draftkosten en gratis Mamba,
attention, LM-kop, router, shared expert en state-commit. Het werkelijke
resultaat ligt er dus ruim onder. Daarmee is TreeSweep's eigen uitkomst A
bereikt: 200 tok/s is gefalsifieerd voor dit model op deze GPU, en hun eigen
voorschrift geldt — doorgaan richting 50-100, geen nieuwe drafter trainen voor
200. Alle twaalf TreeSweep-hypotheses zijn nu gemeten of begrensd door een
meting.

**W1 — het hostpad goedkoper: gebouwd, bit-identiek, poort gefaald op diepte.**
Niet de device-side router: die stuit op een echte ontwerpvraag (de sync bestaat
omdat de host bij een miss de H2D-kopie uitgeeft; missers uit mapped host lezen
verplaatst PCIe-verkeer naar het kritieke pad). Wel het hostwerk eromheen.
`_moe_cached_fast` staat naast `_moe_cached`, achter `fast_host` (default False),
met dezelfde kernels, argumenten en volgorde. Weggehaald: ~400 cupy-slices per
token (nu views uit `enable_cache`/`_alloc_state`), 276 numpy-scalarextracties
(nu Python-float-lijsten), per-call pointerrekenwerk (nu een lijst), twee
numpy-allocaties per laag (nu een `tolist()`), 138 dict-increments (nu lokale
ints), de copy-stream-context als er geen misser is, en twee list-comprehensions.

**G-W1-C1 gehaald: generatie bit-identiek in alle drie de armen.** Winst:
**ctx 0 +2,206 ms (+5,5%) tegen een drift van 0,047 — verhouding 47:1**, dus
scherp. 131072 +0,439 (drift 2,591) en 262100 +0,511 (drift 4,520): beide **niet
conclusief**. **G-W1-P1 (>= 1,0 ms bij 262100) gefaald met +0,511**; de poort
wordt niet verlaagd, dus `fast_host` blijft opt-in en de default verandert niet.
Verifier 39/39.

De ctx-0-winst is 44% van S14's `host_gap` van 5,058 ms op die diepte: het
Python-werk was ongeveer de helft van die leegloop, de rest is de
readback-wachttijd zelf, die deze ingreep niet raakt. Bij 262K is het token 58 ms
in plaats van 40 en is de extra tijd attention-werk waarachter de host zich
verstopt — hostkosten tellen alleen zolang de GPU niets te doen heeft. De drift
bij diepe context (2,6 en 4,5 ms tegen 0,047 bij ctx 0, GPU op 86-87 C) is het
obstakel voor de *meting*, niet voor de ingreep; kortere, vaker afgewisselde
armen zouden de 0,5 ms kunnen oplossen, maar dat is een nieuwe preregistratie.

**V1 / W1-R1 — de router is niet haalbaar, de 0,5 ms blijft onopgelost.**

*V1.* Een device-side router kan een cache-miss alleen zonder host afhandelen door
de GEMV rechtstreeks uit mapped pinned host te laten lezen via de UVA-pointer die
`gather_down_sparse` al gebruikt. Gemeten op een echt `up_proj`-record, dezelfde
kernel, dezelfde bytes: **device 32,66 us/call (85,9 GB/s), mapped host 417,44
us/call (6,7 GB/s)**. G-V1-C1 gehaald — de twee armen zijn bit-identiek, dus de
tijden meten hetzelfde werk. **G-V1-F1 gefaald**: het budget is
`6,656 ms / (0,1785 x 138) = 270,21 us` per miss (Y1's gemeten sync, K0's gemeten
miss-rate) en de meting geeft **384,78 us** — factor 1,42 te duur, nog voor de
slechtere hitrate van een direct-mapped device-cache. Waarom niet S5's 25 GB/s:
dat was een kopieerkernel met warp-per-kolom coalescing; de GEMV heeft een
block-per-outputrij patroon met schaal-lookups, dat 85,9 GB/s haalt op device en
instort tot 6,7 over PCIe. Latency per toegang, niet bandbreedte. Sluit de router
**zoals ontworpen**; een variant die de miss eerst met een gecoalesceerde gather
naar het slot haalt is een ander ontwerp met eigen preregistratie.

*W1-R1.* Idee: `base fast base` op opeenvolgende stappen, zodat buren hun
thermische toestand delen. Uitkomst: effect p50 +1,525 ms bij ctx 0 en +2,877 bij
262100, maar **triplet-drift p50 7,609 resp. 6,779 ms** — **slechter** dan W1's
arm-drift van 4,520. **G-W1R-R1 gefaald.** De redenering was fout en de meting
toont waarom: pairing verwijdert thermische drift maar geen per-stap-ruis, en elk
triplet gebruikt een enkel sample per arm terwijl W1's armen 32 samples middelden.
Per preregistratie wordt bij een gefaalde resolutiepoort **niets geconcludeerd
over de effectgrootte**, dus de +2,877 ms is geen resultaat en **G-W1-P1 blijft
onopgelost bij diepte**. **G-W1R-E1 slaagt** en hangt niet van de drift af: 71%
van 24 tripletten bij 262100 heeft `fast < gemiddelde van de buren` (17/24,
binomiaal tweezijdig p ~ 0,03), dus het teken is positief, de grootte niet
vastgesteld. G-W1R-C1 gehaald: generatie bit-identiek. Verifier 115/115.

Een derde opzet zou korte blokken van ~8 samples per arm snel moeten afwisselen —
middelen per arm en armen dicht bij elkaar in de tijd, wat W1 en W1-R1 elk half
deden. Twee ontwerpen aan een halve milliseconde is genoeg; of een derde het waard
is, is een afweging en geen meetvraag.

**C1 — CertiPlane: het bewijs klopt, maar bewijst vrijwel niets.** ExactFlow
hypothese C en TreeSweep H8, allebei nog ongemeten, en anders van aard dan alles
wat al sneuvelde: niet hoe vaak je het target gebruikt of hoeveel bytes een
gewicht kost, maar welke bytes je aantoonbaar niet hoeft te lezen.

Opzet: splits het 4-bits NVFP4-woord in een core (tekenbit + bovenste c-1
magnitudebits) en een exacte residual; bereken de preactivatie uit de core en
begrens de residual per schaalgroep van 16 met Cauchy-Schwarz. Is `y0 + B <= 0`,
dan is `ReLU^2` exact nul en hoeven die staartbits nooit gelezen. 240 (expert,
activatie)-paren over 15 lagen en 108 experts, echte gewichten en activaties,
referentie in float64.

**G-C1-S1 gehaald: nul valse certificaten over 890.880 beslissingen** — de grens
is aantoonbaar sound. **G-C1-R1 en G-C1-B1 gefaald met een factor ~90**: 0,33%
van de rijen gecertificeerd tegen een poort van 30%, en 0,37% van de werkelijke
nullen (die zelf 90,64% zijn, consistent met S5's ~91%).

Oorzaak is structureel: de grens is **9x (core 3 bits) tot 31x (core 2 bits)
groter dan de preactivatie waar hij iets over moet zeggen**. De som loopt over 168
groepen en telt die als worst case op alsof alle residuals met `x` meebewegen; in
werkelijkheid middelen ze uit. Meer corebits verscherpt de grens precies zoals
verwacht (31x -> 9x) maar beweegt de certificatiegraad niet, want 9x is nog steeds
hopeloos ruim. Repareren met een geleerde schatting vervalt tegen het pack's eigen
verbod op "een onveilige learned gate als vervanging van een gefaalde
certificatebranch", en hun eigen regel geldt: "Mislukt de oracle, dan geen kernel."
Verifier 18/18.

Blijft open, maar als wiskundige vraag en niet als kernelvraag: bestaat er een
**sound** bound die ordes scherper is dan Cauchy-Schwarz per groep? Wie die tak
wil openen moet dat eerst laten zien.

**W1-R2 / C1-R1 / O1 — de derde opzet, de scherpere bound, en OrbitANS.**

*W1-R2.* W1 middelde per arm maar zette de armen minuten uit elkaar; W1-R1 zette
ze naast elkaar maar middelde niet. Deze opzet doet allebei: blokken van 8 samples
per arm, direct afgewisseld. **Best oplossend van de drie: drift 2,220 ms tegen
W1's 4,520 en W1-R1's 6,779.** Bij 262100: effect **+0,106 ms, tekenfractie
0,50** — een muntworp, dus **geen effect**. Bij ctx 0: +1,854 ms, tekenfractie
0,75, consistent met W1's +2,206. Generatie bit-identiek. G-W1R2-R1 gefaald
(2,220 > 1,0), dus formeel niets over de grootte, en zoals vastgelegd stop ik met
deze vraag. W1-R1's 0,71 was vermoedelijk een **positie-artefact**: daar zat de
`fast`-arm altijd op de middelste stap van elk triplet. Eindstand over drie
opzetten: het effect is reeel en scherp bij korte context, en bij diepte niet aan
te tonen — mechanistisch consistent, want bij 262K is het token 18 ms langer en
dat is attention-werk waarachter hostwerk zich verstopt. `fast_host` blijft
opt-in.

*C1-R1.* C1's grens gebruikte de **exacte** residual-normen en kende de staart dus
al. C1-R1 gebruikt alleen de core: het tekenbit zit erin en bit-truncatie kan een
magnitude alleen laten groeien, dus `dy_j <= sum_k dmax(core_jk) * max(s_jk*x_k,
0)` — alleen de adverse helft telt, met exacte maxima, en `dmax` volgt uit de
core-code, dus **nul opgeslagen certificaat-metadata** (het pack begroot 0,15
bit/gewicht). Sound (nul valse certificaten), maar **0,14% gecertificeerd tegen
een poort van 30%**. C1's 0,33% en dit zijn geen like-for-like; de **inzetbare**
grens haalt 0,14%, en dat is de eerlijke bovengrens op CertiPlane.

Bug onderweg, en hoe hij gevonden werd: de eerste versie las het teken af uit
`e2m1[core] < 0.0`, maar code 8 decodeert naar **-0,0** en `-0.0 < 0.0` is False,
dus elk negatief gewicht met kleine magnitude kreeg het verkeerde teken en de
grens was niet sound. Er kwam geen enkel vals certificaat uit, dus de poort ving
hem niet. Wat hem ving was een **incoherentie**: scherpere grens (9,06 -> 4,96)
maar minder certificaten (0,33% -> 0,13%) kan niet allebei. Na correctie (teken
uit bit 3) verandert het cijfer nauwelijks (0,14%) maar is de grens wel een bewijs.

*O1.* Codes: 3,9373 bit marginaal, **3,9092 bit** gegeven de schaal-exponent (het
conditionele model dat het pack zelf voorstelt wint 0,028 bit). Scales: 4,2820
bit van 8, delta 4,2502. Record 2.806.272 -> 2.603.515 B = **7,23% packreductie**,
tegen een doorgangspoort van 12% en een sterke poort van 20%. **Beide gefaald.**
Scherper beeld dan S3 alleen: de codes zijn vrijwel incompressibel, de scales wel,
maar die zijn 11% van het record dus de winst verdunt. Via Y2-R1's helling is
7,23% bytes hooguit 4,9% van de GEMV-tijd, voor ANS-decodekosten. Entropie-
ondergrens, dus bovengrens op wat OrbitANS kan. Verifier 38/38.

**N1-N5 — vijf eigen hypotheses, en een correctie op mijn eigen conclusie.**
Niet uit de packs; gekozen op de termen die deze sessie als grootst heeft gemeten,
plus een vraag die niemand had gesteld.

*N5, de roofline van een token.* Gemeten streaming-leesbandbreedte **338,4 GB/s**
(eigen kernel, 256 MiB, geen datasheet). Bytes die een correcte forward moet
lezen: ctx 0 **1.953,3 MiB -> vloer 6,05 ms -> plafond 165,2 tok/s**; 262K
**2.721,2 MiB -> 8,43 ms -> 118,6 tok/s**. **1000 tok/s is uitgesloten** (6x onder
de vloer). **100 en 50 tok/s zijn dat niet.** Daarmee moet ik mijn eerdere
uitspraak dat 50 tok/s "buiten de gemeten fysica" ligt **terugnemen**: te sterk.
De runtime draait op **16,8% van de roofline** bij ctx 0 en 15,5% bij 262K; 50
tok/s vraagt 30-42%. Het gat is efficientie, geen natuurwet.

*N1, graph-plafond.* Dezelfde kernelreeks eager tegen CUDA-graph-replay (routes
bevroren, want capture verbiedt sync): **36,714 -> 28,023 ms = 23,7%
verwijderbaar**. Dat is de bovengrens van elk ontwerp dat werk van host naar GPU
verplaatst — megakernel, device routing, persistente kernels. S9 schatte de
launch-overhead op 13% van de MoE-term; over het hele token is het 23,7%.

*N2, het down-pad gesplitst.* Marginalen: `panel_scan` +0,305 (drift 4,669, onder
de ruisvloer), **`gather_down_sparse` +8,192 ms met drift 0,040** — 205:1, de best
opgeloste marginaal van de sessie — masked GEMV +12,429 (drift 4,825), heel
`down_masked_into` +12,649. **scan+gather = 67,2% van het down-pad, G-N2-1
gehaald.** De gather haalt ~35 MB/token uit mapped host op ~4,3 GB/s effectief
tegen de 25,05 die S5 geisoleerd mat: 6x slechter in de lus. Verklaart S11
achteraf.

*N3, exacte ReLU2-prefilter.* Rang-r benadering plus sound residualgrens om
bewijsbaar-nulle rijen over te slaan, bit-identiek. Sound (nul valse
certificaten) maar **0,01% gecertificeerd tegen 30%**. Oorzaak in de laatste
kolom: **rang 64 vangt maar 21,8% van de spectrale energie** van een matrix van
rang 1856 — de expertgewichten zijn vrijwel vol-rang. Onafhankelijke bevestiging
van waarom de RSIV-laag-rang-lijn sneuvelde, en het sluit die deur nu ook voor
*exact* gebruik.

*N4, attention tegen KV-bytes.* Fit **21,48 ms per GB + (-0,033) ms vast, R2 =
0,9964**: volledig byte-gebonden. Halvering van de KV-bytes bij 262K: 17,047 ->
8,615 ms (-49,5%), dus een FP4-KV zou 8,4 ms schelen (met eigen kwaliteitspoort).
Belangrijker: de kernel draait op **47,2 GB/s waar het apparaat 338,4 haalt** —
factor **7,2**. Zou attention de helft van de roofline halen, dan zakt hij van
17,0 naar 4,8 ms zonder een byte te besparen en zonder semantiekwijziging.

*Waar het gat zit, voor het eerst opgeteld:* uitgifte-overhead 23,7% (N1),
attention 7,2x onder roofline (N4), GEMV 4,2x onder roofline (Y2-R1), gather 6x
slechter in de lus dan geisoleerd (N2). Alle vier efficientie, geen bytes en geen
semantiek. Verifier 36/36.

---

## 2026-08-15 (latere sessie) — E4: attention-roofline herstel, kernelfase

*Namespace treesweep200 (pack NEMOTRON_TREESWEEP_200_ROOFLINE_V2_AGENT_PACK_2026-08-15,
agent 21). Preregistratie bevroren, geen poort verruimd. Protected manifest na
fase: 0 modified / 0 removed.*

**P0/E0 herhaald en PAS**: identiteit + roofline-reproductie, 8/8 claims,
onafhankelijke verifier 0 failed. Detail: reports/treesweep200/P0_E0_REPORT_*.

**E4 kernelfase**: vijf kandidaten gebouwd en gemeten tegen v1
(attn_decode_warp_fp8_gqa). Bevindingen:

- v2 (2 lanes/head) was nooit geregistreerd en bevatte een OOB-writeback-bug;
  na reparatie correct (1,3e-5) maar structureel trager (16x redundante
  LUT-decode -> shared-pipe-bound).
- sm_120-hardwarekennis: `fma.rn.f32x2` bestaat, is bitwise == scalair fmaf
  (2048/2048), throughput 1,63x scalair. `redux.sync.add.f32` bestaat NIET op
  sm_120. Hardware fp8->f16x2 `cvt` is exact (e4m3 past exact in f16).
- Kernel is issue-bound (~1,2 warp-inst/cyclus/SM), niet HBM-bound: ablaties
  @262144: full 2,27 / -shuffles 1,42 / -exp 2,19 / -PV 1,75 ms. Shuffle-pipe
  (~80 warp-shuffles/positie-bezoek) is de grootste enkelvoudige kost.
- **Beste kandidaat v4** (cvt-decode + double-buffered loads + 2-pos-ILP):
  **bitwise identiek aan v1** (5 contexten x 3 seeds), 2,304 vs 2,803 ms/laag
  @262144 (-17,8%), -33% @4K. Effectief 58,3 GB/s.
- Poorten: fit R2=0,9981 PASS; profiel PASS; correctheid PASS; **S1 (100 GB/s)
  en S2 (169 GB/s) FAIL**. Exact-fp32-vloer geschat ~1,2-1,5 ms/laag (~90-110
  GB/s); S2 structureel onhaalbaar zonder fp16/tensor-cores (verworpen:
  exactheid). Registry: E4 = gate_failed, v4 bevroren als beste exacte
  kandidaat voor E6-integratie.

*Open:* in-lus adoptiemeting v4 (G-E4-T1: attention-component @262100 +
64-token-pariteit vs s5-anker) en de onafhankelijke E4-verifier. Daarna E2
(gatherloze downflow, 8,2 ms/token-hefboom), E5 (GEMV-suite), E1
(graph-resident), E6 (integratie/E50). Volledige handoff:
reports/treesweep200/HANDOFF_E4_EN_VERDER_2026-08-15.md.

**E4 in-lus (treesweep200-lijn) — v4-attention conclusief sneller, absolute poort
gefaald.** Kimi's openstaande punt uit `HANDOFF_E4_EN_VERDER_2026-08-15.md`
opgepakt. Methode als `s14_moe_layer_timeline.py`: CUDA-events zetten
tijdstempels op de stream rond elk van de zes attention-lagen **binnen** de echte
lus, dus geen host-sync toegevoegd en de overlap blijft intact. Kernelwissel is
een monkeypatch (identieke signaturen). Drie armen v1/v4/v1.

| ctx | attention v1 | v4 | winst | drift | token v1 | v4 | winst | drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,474 | 2,430 | +0,044 | 0,022 | 38,147 | 37,520 | +0,628 | 2,264 |
| 131.072 | 10,771 | 9,028 | +1,743 | 0,258 | 46,791 | 45,509 | +1,281 | 1,115 |
| 262.100 | 18,211 | 14,554 | **+3,658** | 0,544 | 55,915 | 51,164 | **+4,751** | 3,014 |

Attention-winst overschrijdt op alle drie de diepten haar drift. E4's
geisoleerde cijfers kloppen in de lus: 6 x 2,803 = 16,82 tegen 18,21 gemeten, en
6 x 2,304 = 13,82 tegen 14,55 — beide binnen 10%, en de voorspelde ~3,0 ms komt
uit op 3,658.

**G-E4-T1 gefaald**: eist attention <= 6,0 ms bij 262100, meet 14,554. Niet
verruimd; het is een **absolute roofline-poort**, geen regressietest. De
arm-tegen-arm-pariteit (2 x 64 tokens bit-identiek in v1/v4/v1) is gehaald, en de
verifier herbevestigt bitwise-identiteit los van de runner op willekeurige FP8-KV
(3/3).

**De ankerclausule faalde op een verouderd artefact, niet op v4.**
`s5_baseline_generation.json` is bevroren 2026-08-14T20:02Z, de v35-layoutopname
staat op 20:52Z — het anker komt dus van Nemotron 3 Nano en kan per constructie
niet matchen ('...No extra punctuation?' tegen '...The capital of Germany is
Berlin.'). Het ontbrekende v35-anker is nu bevroren als
`reports/treesweep200/V35_GENERATION_ANCHOR.json` (2 x 64 tokens, met runtime- en
kernelhashes) zodat E2/E5/E1/E6 er tegen kunnen gaten.

Verifier 52/52, protected 0 modified / 0 removed, registry gevalideerd (30
experimenten). Adoptie van v4 als default is niet omgezet: dat is een
productiebeslissing die bij E6-integratie hoort.

**NERVF — ERVF gerepliceerd op Nemotron (tweede-modelreplicatie van P7).** Nieuwe
namespace `NERVF_NEMOTRON`. Archiefinspectie: ERVF bestaat niet in deze runtime,
en `gemv_nvfp4_rows` is structureel exact de Qwen-vorm van vóór ERVF (1 block van
256 threads per rij, shared-memory reductie, `__syncthreads`). P7 zelf noemde
"P8E tweede modelreplicatie" al als openstaand punt; dit is die.

*NERVF-1, geometrie.* Eerste ronde was INCONCLUSIVE door een L2-defect in mijn
eigen referentie-armen (één record van 2,81 MiB past in L2; `RAW_SCAN` zwaaide
5,3x tussen runs). Verholpen zoals N5 het deed: alle armen cyclen door een pool
van 95 replicas (254 MiB) boven de gemeten L2 van 32 MiB -> spreiding van 5,3x
naar **0,1%**. Armen: RAW 12,43 us / 225,8 GB/s · ROW_PATTERN 17,08 ·
DECODE_SCALE 20,80 · FULL_GEMV 38,58 / 72,7 GB/s. **Beide poorten open**:
bandbreedte-efficientie **0,322** <= 0,40 en reductie+sync **46,1%** >= 25%. De
instorting van 226 naar 73 GB/s zit niet in geheugen (4,6 us) en niet in decode
(7,7 us) maar voor 46% in reductie/sync — exact de Qwen-signatuur.

*NERVF-2, de kernel.* WIDTH lanes per rij, 256/WIDTH rijen per block, 256/WIDTH
gescheiden virtuele accumulatoren per lane, referentieboom exact herbouwd. Twee
vondsten: de offset-16-stap van de referentie wordt bij WIDTH<=16 **lane-lokaal**
(twee virtuele accumulatoren van dezelfde fysieke lane), en die lane-lokale
stappen moeten in **butterfly**-volgorde — sequentieel vouwen gaf bij w=4/w=8
72/72 mismatches terwijl w=16 toevallig goed uitkwam. Na correctie **alle vier de
breedtes bitexact** (0/72 elk, over 3 lagen x 3 experts x 4 activatieregimes x 2
ReLU2-standen). Speedups: w=4 1,336x · w=8 1,686x · **w=16 1,936x** · w=32
1,897x. Primaire (1,35x) en sterke (1,75x) poort gehaald, moonshot (2,0x) net
niet. **Gekozen breedte 16 — dezelfde als Qwen P7**, op een architectonisch ander
model met andere quantisatie.

*NERVF-3, integratie.* ERVF additief achter `use_ervf` (default uit). Drie armen
base/ervf/base. **Exact: alle armen bit-identiek, ook tegen het bevroren
V35-anker.** Tokenwinst conclusief op alle diepten: ctx 0 **−3,701 ms**
(37,660 -> 33,959), 131K **−3,102**, 262100 **−4,505** (55,640 -> 51,135). De
componentpoort G-NERVF-3P faalde (1,144x tegen 1,35x) omdat mijn venster het hele
routed-expert-blok omsluit terwijl ERVF alleen de rij-GEMV vervangt — verdund per
constructie, niet geherinterpreteerd. Verifier 66/66, protected 0/0.

Volgende: NERVF-4 gatherless-ERVF down, die samenvalt met Kimi's E2.


## 2026-08-15 -- E6, E2-afsluiting en A1-adoptie

- **E6** geintegreerde fysieke A/B (ERVF w16 + attention v4 + D1), 3 domeinen x
  512 causale tokens: **41,980 -> 37,490 ms per token**, bit-identieke uitvoer,
  VRAM ongewijzigd. Eindpoort >=50 tok/s niet gehaald (26,7).
- **E2** gatherloze downflow formeel gesloten als **weerlegd** -- hetzelfde
  experiment als NERVF-4, -5,70 / -7,56 / -7,38 ms per token.
- **A1** de bewezen stack is **default** geworden, na een preregistratie met een
  controle-arm die moest falen (en faalde). Nieuw anker
  `V36_DETERMINISTIC_ANCHOR.json`; V35 blijft staan en is niet bit-vergelijkbaar.
- De takenlijst en het logboek verhuizen naar `agents/`:
  `agents/STATE_OF_THE_WORK.md` (startpunt), `agents/TODO.md` (afvinken),
  `agents/RESEARCH_NOTEBOOK.md` (logboek), `agents/AGENT_HANDOFF_PROMPT.md`.

Details: `reports/treesweep200/E6_INTEGRATED_REPORT_2026-08-15.md`,
`reports/treesweep200/E2_GATHERLESS_DOWNFLOW_REPORT_2026-08-15.md`,
`reports/treesweep200/A1_ADOPTION_REPORT_2026-08-15.md`.

## 2026-08-15 — E1 fase 2.1 (device-resident routing) afgerond; fase 2.2 (graph-replay) gebouwd, ongemeten

- **E1-2.1 PASS (alle 5 poorten, verifier 14/14):** routerkop + device-LRU + bulk-staging (24,93 GB/s, DMA-pariteit) halen elke device→host-sync uit de MoE-laag. p50 41,540 → 36,998 ms/token (−4,542 ms), pariteit vs bevroren A1-ids, capaciteit 56 ≡ 72, controle-arm faalde zoals vereist. Opt-in via `rt.device_cache=True`; nog niet geadopteerd als default. Bugfix onderweg: `enable_cache` reset nu ook `_dev_cache`. Rapport: `reports/treesweep200/E1F21_DEVICE_ROUTING_REPORT_2026-08-15.md`.
- **E1-2.2 GEBOUWD, ONGEMETEN:** preregistratie bevroren (`E1F22_GRAPH_CAPTURE_PREREGISTRATION_2026-08-15.md`), graph-API gesmoketest, dp-kernels + `setup_graph`/`step_graph`/`ring_harvest` geschreven (syntax OK). Runner/verifier/rapport ontbreken nog — pickup-stappen in `agents/TODO.md`.
- Protected-manifest: nieuwe baseline `PROTECTED_80B_MANIFEST_AFTER_USER_COMMIT_2026-08-15.json` (oude was pre-git; .gitignore-vlag was de eerste commit van de eigenaar, geen schending).
