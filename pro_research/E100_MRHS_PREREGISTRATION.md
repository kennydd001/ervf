# E100-MRHS preregistration — exact common-weight reuse across sequences

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: **no target-GPU result exists for this experiment**.

## Claim boundary

This experiment tests an aggregate-throughput primitive for exact multi-sequence decode. It does **not** claim 100 tok/s single-stream latency.

The observation under test is narrower than the earlier route-union work: attention, Mamba, router, shared-expert and LM-head matrices are common to every active sequence. Earlier N-scaling diagnostics launched N independent GEMVs, therefore the same matrix was streamed N times. E100-MRHS instead loads each matrix element once inside a kernel and applies it to N independent right-hand-side activation vectors.

Arithmetic for each sequence is deliberately unchanged. Each sequence keeps the same virtual thread assignment, the same per-thread MAC order and the same two-level reduction tree as the exact production/ERVF GEMV. Operations from other sequences may be interleaved in instruction issue, but no accumulator or reduction is shared between sequences.

## Frozen hypotheses

**H0:** exact multi-RHS reuse provides no useful aggregate amortisation once register pressure and extra RHS reads are included.

**H1:** for at least N=4, common-weight matrices obtain substantial aggregate amortisation while remaining bit-identical per RHS. This would invalidate the old working assumption that non-MoE layers are intrinsically linear in N; only the *sequential implementation* was linear in N.

The experiment is a component/oracle test. Even a large win here is not an E100 runtime claim until integrated into a fixed-N graph-resident full-model step with exact causal rollouts.

## Kernels and exactness contract

Test N in `{2,4,8}`. The first implementation uses a 32-lane physical subwarp and reconstructs the original 256-virtual-thread reduction:

- virtual tid = `lane + 32 * vi`, `vi in [0,7]`;
- BF16/F32: virtual tid walks scalar `k = tid, tid+256, ...` exactly as the reference;
- FP8 tensor: virtual tid walks the production `uchar4` index and executes q.x/q.y/q.z/q.w FMAs in the original order;
- NVFP4: virtual tid walks the production `uchar4` packed-code index and executes all eight nibble FMAs in the original order;
- each loaded weight/dequantised scalar is reused across RHS accumulators only; RHS accumulators never mix;
- each RHS is reduced independently with the original warp tree and the original 8-warp second-stage tree.

Any bit mismatch closes integration. There is no tolerance-based exception.

## Real-checkpoint cases

The runner must use the local `nemotron_3_5_lightning_v35` checkpoint, not synthetic weight matrices. At minimum:

1. attention Q BF16;
2. attention O BF16;
3. MoE router F32;
4. Mamba input projection in its stored kind (FP8 tensor on Lightning);
5. shared-expert up NVFP4;
6. shared-expert down NVFP4;
7. LM head NVFP4.

If a named key/storage kind is absent, record an explicit `unsupported_case` rather than silently substitute a different matrix.

For each supported case and N, test at least three deterministic activation batches in full mode. Bitwise compare every RHS output against N sequential calls through the currently adopted exact baseline kernel.

## Timing protocol

Timing is CUDA-event device time. Full mode uses interleaved `REF, MRHS, MRHS, REF` blocks after warmup; report all raw samples. `REF` means N sequential exact baseline GEMVs using the same X batch. `MRHS` means one multi-RHS launch.

Primary performance quantities are:

- `aggregate_speedup = sequential_reference_ms / mrhs_ms`;
- `mrhs_ms_per_rhs = mrhs_ms / N`;
- effective matrix-byte amplification avoided: N reference matrix streams versus one logical MRHS matrix stream.

No component speedup may be multiplied by V12/H-SCALE/Mamba or any other projected gain.

## Gates

Correctness gates, all mandatory:

- all supported real-checkpoint cases bit-identical for every RHS and activation batch;
- deterministic repeat of MRHS output bit-identical;
- at least 6 of the 7 frozen case families supported; LM head and Mamba input are mandatory;
- no NaN/Inf in compared outputs.

Performance gates for an **E100-worthy primitive** in full mode:

- N=4 weighted-common-matrix aggregate speedup >= **1.75x** using frozen `calls_per_token` weights;
- N=4 LM-head aggregate speedup >= **1.50x**;
- N=4 Mamba-input aggregate speedup >= **1.50x**;
- no supported N=4 family regresses by more than 5% (`speedup >= 0.95x`);
- timing drift between the two REF blocks <= 7% per case (microkernels are short; the tighter 1 ms token-level drift rule is not meaningful here).

N=2 and N=8 are shape/pressure diagnostics and do not rescue a failed N=4 primary gate post hoc.

## Interpretation

- `correctness_failed`: any exactness/determinism gate fails. Stop.
- `micro_null`: exact, but the N=4 performance gate fails. Do not integrate.
- `mrhs_candidate`: exact and all N=4 primitive gates pass. Proceed to graph-resident N=4 integration.

A later full-model E100 claim is separate and requires, at minimum, exact per-sequence causal parity, >=100 aggregate generated tok/s over >=1000 timed generated tokens, stable baseline/control arms, and an independently recomputed verifier. Per-sequence latency must be reported next to aggregate throughput; aggregate throughput must never be relabelled as single-stream tok/s.

## Deliberate non-combination

Claude's current V12 async-harvest, H-SCALE and Mamba work is a separate experimental line. E100-MRHS must first stand on its own. Only independently verified primitives may later be combined in one integrated A/B run; projected speedups are never multiplied.
