# HET-NEXT-L0-C0-R1 — real-weight whole-expert Intel + NVIDIA component gate

## Status, supersession and claim boundary

This is an immutable design preregistration. It supersedes C0 only for future work; the C0 documents and their independent NO-GO audit remain immutable. C0-R1 authorizes no executable preflight, compilation, device enumeration, allocation, kernel launch, timing run, GPU/iGPU use or output creation. No C0-R1 runner exists.

The only admissible positive claim is:

> On four frozen real-weight, natural-route Qwen3-Coder-Next layer-0 rows, a fixed whole-expert Intel/NVIDIA partition reproduced an independent exact Q5/ERGV CPU component oracle and met the preregistered concurrent-hybrid/dGPU-only inclusive-latency ratios.

It cannot claim official layer/logit equality, held-out quality, end-to-end tokens/s, full-model acceleration, cache-resistant throughput, endurance, production readiness or an industrial breakthrough. D2-R3 is diagnostic evidence. R5 remains formally verifier-negative because of its natural p0/n8 control conjunct. C1-R2A proves only synthetic control sensitivity. A C0-R1 positive is a real-weight natural-route heterogeneous component result only.

## Immutable evidence

Every later source and lock must bind these exact bytes:

- D2-R3 raw/result/audit/interpretation: `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`, `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb`, `a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33`, `be603f4edc648939aa86b2fcec16df802f4e778c6ab14256aecdc48f347da7f0`.
- R5 raw/result/commit and control diagnosis: `fcf49479396682634e4a5b9faa3fd3e76c17ba7cfc389e711931996f5e3efbd8`, `56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91`, `d784ded5e7893095e2f27b75695e635c9cc880109736c87496138e3188509372`, `b22808626e45178cb917cebde5aac789ba720d091ef143099948c94f243bf2e0`.
- C1-R2A raw/result/commit and verifier source: `d7272ce6aec3533b487829360e40398f1d5fa9d3b766c2593acad01faedca89a`, `bcaa5b2531d422e7eabd09b92fc8f6659c44cbd879614d5d83ee2b9bbc24a736`, `a6736d2b12307cd6cde462513235b4c6f7517289ed89acfde8441832ebcce875`, `ccf7bda4dcc13135a5e43a9c8ee35d79182a61b2b1a0192a6a37337365e99a11`.
- Official `Qwen/Qwen3-Coder-Next` revision `a19358a7659bd1f564300250ee189120c49a562f`, shard 1 exact size `3,999,619,288` and SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- ST2-mini report/source: `af23a4fbffb18028ce1a88b3c73f21546cae7ee397d118c186094f985ee4ac49`, `6472de274fa68a9f577b1483ef1225607f8425ac8587cb348d0c328cff7126ca`.
- D7 report/source/audit: `8fce019bb4d51ff5e04a0c91d1cfa43679caf54e86f884acb4fb6a5225df5e86`, `26d4daba81d5f132857f9b584dfb12f3634874a2e9ee2290f9221c486ef4059a`, `acf925757f097e16d62567a64ee95c5974bc95241ee7f0cc5377795eb4d9d1d5`.
- C0 prereg/design/audit: `5ba80f6c8f3a5b144192146dde32a1e3b8e0439a60e0370afa049623a6e8cd63`, `86a8f60eb7779ff9951c4bf7611406518d37a86626fb70ecf4e4cff36b4e7495`, plus the independently rehashed C0 audit file.

ST2's `18.540 GB/s` and D7's synthetic `19.473033 GB/s` are motivation, never inherited performance passes.

## Rows, validation seal and split

Only last position index 15 of each frozen 16-token D2-R3 whole capture is in scope. The public, hash-locked route-ID metadata are:

| role | row | rank-ordered IDs |
|---|---|---|
| validation | `p0/n16` | `50,199,237,474,245,374,239,8,168,12` |
| test | `p1/n16` | `42,162,267,299,467,307,326,145,297,182` |
| test | `p2/n16` | `474,232,382,80,31,450,103,372,286,206` |
| test | `p3/n16` | `26,159,28,176,253,84,431,294,386,356` |

Before a committed p0 validation pass, a guarded offset reader may open only p0 input/post-norm, native BF16 route weights, shared-gate and reference tensors, and only the official p0-required expert triplets plus shared weights. It must not mmap the full D2 payload. It must not read any p1-p3 input, weights, shared-gate, oracle, control or timing array, nor any additional shard tensor selected solely by p1-p3. A p0 expert that also appears in public test metadata remains a p0-authorized source read; that overlap does not open any test row.

Every read is logged before access as `(phase,row,key,absolute_offset,byte_count,expected_sha256)` and marked completed with observed SHA. Before validation adjudication, independent verification requires zero completed test-payload ledger entries and zero intersection with test-only byte ranges. A committed p0 pass atomically changes `tests_opened:false` to `true`; only a new clean test-source process may then read p1-p3 payloads and test-only official weights. The test source/oracle manifests are committed before test timing. No lock or threshold is edited after p0.

Dispatch ownership is frozen by original route rank:

- Intel Arc Pro 140T owns ranks `0..3`.
- NVIDIA owns ranks `4..9` plus shared.
- No expert migrates; no gate/up/down matrix or reduction is split.
- Each device returns separate BF16 `[2048]` routed down outputs; NVIDIA returns BF16 shared-raw `[2048]`.

## Exact Q5 and official component arithmetic

Only necessary official layer-0 tensors may be reread. Recreate the R5 codec exactly: row-major groups 128; FP32 `max_abs/15`; q round/clamp `[-15,15]`; stored BF16 scale; zero group has scale BF16 1 and q0; field `q+15` in `[0,30]`; 31 forbidden; eight little-order fields/five bytes. There is no persistent bank.

The independent CPU ERGV/Q5 oracle shares no device decoder, GEMV or merge helper. For each expert it freezes these points:

1. gate and up ERGV reductions accumulate in the frozen ERGV order; each linear result is cast to BF16;
2. `silu(gate_bf16)` is evaluated with the frozen implementation and cast to BF16, then multiplied by `up_bf16`, producing BF16 activated values;
3. down ERGV reduction consumes that BF16 activation and its output is cast to BF16;
4. the captured native BF16 route weight is multiplied with the BF16 down output and the contribution is cast to BF16;
5. routed contributions are added into a BF16 zero buffer in **ascending expert-ID order**, exactly matching the official `expert_hit.nonzero()` loop and BF16 `index_add_`; route rank determines ownership and selects its captured weight but does not determine addition order;
6. shared gate is `sigmoid(captured_shared_gate_linear_bf16)` cast to BF16; `sigmoid_gate_bf16 * shared_raw_bf16` is cast to BF16 in that operand order;
7. routed aggregate plus shared-gated is one final BF16 add.

For p0 the routed addition rank order is therefore `7,9,0,8,1,2,6,4,5,3`, not `0..9`. The implementation derives and retains the analogous expert-ID-sorted original-rank permutation for every opened row. Oracle evidence retains all gate/up/activation/down, weighted contributions, each sequential routed accumulator state, shared raw/gate/gated and final component arrays. Every device and arm intermediate must be bitwise equal to this oracle.

## Arms and exact inclusive clock

Each opened row has the same host merge and three arms:

- `A = dGPU_only`: all ten routed experts plus shared on NVIDIA.
- `S = hybrid_sequential`: Intel ranks 0-3 completes, NVIDIA ranks 4-9 plus shared completes, then host merge.
- `B = hybrid_concurrent`: both device partitions are released from one host barrier, then host waits for both completion events and merges.

Inclusive wall uses one invariant host clock. It starts immediately before the first submission or sample-specific copy and ends after all submissions, waits, output copies and the host BF16 merge. Device events are telemetry only. One-time quantization/allocation/compile is excluded and separately reported. Every timed sample must execute all Q5 reads and GEMVs; graph/result reuse is forbidden.

## Frozen schedule and statistics

The only schedule seed is unsigned decimal `2026081302`. Define three 12-observation templates by concatenating these four three-arm groups:

- `T0 = ABS | BAS | SAB | SBA`
- `T1 = ASB | BSA | ASB | BSA`
- `T2 = SAB | SBA | ABS | BAS`

For block `b=0..29`, use `T[(2026081302+b) mod 3]`; `2026081302 mod 3 = 0`. This yields exactly 30 blocks, 360 timed observations and **120 samples per arm per row**. Each template's A/B projection is `ABBAABBA`; every group has its explicit reverse in the same template. Across every T0/T1/T2 cycle, each arm occupies each within-group position exactly four times. No PRNG, shuffle or alternative order is allowed. Store and SHA-256 hash the canonical UTF-8 comma-separated 360-arm schedule before capability work.

Warmups use the first 30 observations of the same infinite template cycle in a separate warmup ledger, yielding exactly ten warmups per arm. Warmups are never included in statistics. Timed samples are indexed independently `0..359` from a fresh T0 start. A `pair_id` is `(block,group)`; it binds each three-arm group and its named reverse group. Pairing controls order only: gates are ratios of arm quantiles, never per-pair ratios.

For each arm separately, sort all 120 raw FP64 nanosecond-to-millisecond samples. NumPy-linear quantile is frozen as `h=(n-1)q`, `lo=floor(h)`, `hi=ceil(h)`, `Qq=x[lo]+(h-lo)*(x[hi]-x[lo])`, evaluated in IEEE FP64, for `q=0.50` and `0.95`. Thus p50 interpolates indices 59/60 and p95 uses index 113 plus `0.05*(x[114]-x[113])`, zero-based.

The four performance formulas are exactly:

- `Q0.50(B) / Q0.50(A) <= 0.90`;
- `Q0.95(B) / Q0.95(A) <= 0.95`;
- `Q0.50(B) < Q0.50(S)`;
- `Q0.95(B) < Q0.95(S)`.

No percentile of paired ratios, pooling, trimming, retry, outlier removal or alternate quantile is admissible. Validation p0 must pass every correctness/control/resource gate and all four formulas before test payloads open. Each test row passes independently by the same gates; aggregate pooling cannot rescue a failure.

## Exact cache-thrash, paging, clock and thermal protocol

Allocate exactly `268,435,456` bytes by Windows `VirtualAlloc`, 4096-byte aligned. The pinned timing thread first-touches every 4096-byte page on its recorded NUMA node. Initial byte `i` is the low eight bits of SplitMix64 applied to `2026081302 XOR i`; retain pre-run SHA-256. Immediately before every warmup/timed observation, outside the inclusive arm clock, visit every 64-byte line exactly once in cyclic order starting at line `SHA256("2026081302:row:phase:observation")[0:8] little-endian mod 4,194,304`; read its first byte into a volatile 64-bit XOR accumulator and replace it with `(old + low8(SplitMix64(seed XOR global_line_visit))) mod 256`. Retain per-observation start line, accumulator and rolling digest, plus final full-buffer SHA. No device maps this buffer. Capability inventory must prove 256 MiB exceeds each relevant reported last-level cache; otherwise C0-R1 is `blocked_capability`, not resized.

An independent monitor calls `PdhAddEnglishCounterW` for `\\Memory\\Page Reads/sec`, `\\Memory\\Pages Input/sec`, and `\\Paging File(_Total)\\% Usage`, sampling every 100 ms from two seconds before warmups until two seconds after cleanup. Store timestamped raw/formatted values and PDH status. Paging blocks the attempt if any PDH read is invalid, if Page Reads/sec exceeds 64 or Pages Input/sec exceeds 1024 for three consecutive samples, or if paging-file usage rises more than 0.25 percentage points over the median of the first ten valid pre-warmup samples. `GetProcessMemoryInfo.PageFaultCount`, working set and peak working set are telemetry, not substituted thresholds.

After exactly ten warmups/arm, compute for each device the FP64 median of its reported effective clock over warmups in which it performed work; this is that device's immutable validation clock baseline. During timing, only active-device samples count. Before starting a new observation, block as `blocked_thermal_or_device` if either temperature is `>=90 C`, any device error/reset occurred, or either device has five consecutive completed active observations whose effective clock is `<0.70 * its warmup baseline`. This rule is live during p0 because the baseline is complete before timing. Store every raw clock, temperature and reset/error observation.

## Controls, resources and lifecycle

Before enqueue, safe dispatch rejects wrong expert, wrong device slot, gate/up swap, wrong down shape, source/codes/scales digest mismatch and field31. A create-new call ledger proves rejection before any queue counter changes. Unsafe diagnostic bypass must change the exact CPU-predicted output. C1's one-hot field control may establish field sensitivity, but each device also needs a real assigned-expert wrong-slot/projection control with observable unsafe change.

Hard gates: exact device/driver/runtime/compiler identity; Intel host-USM and NVIDIA buffer/pinned semantics with no timed hidden full-weight copy; source-record-row-rank-slot binding; all finite arrays; deterministic correctness repeat; no competing device process; frozen CPU affinity/runtime; start available RAM `>=16 GiB`, reserve `>=2 GiB`, process peak working set `<=12 GiB`; Intel allocations `<1 GiB`; NVIDIA VRAM `<256 MiB`; retained evidence `<10 MiB`; no paging/thermal/clock trigger. Every allocation, mapping, registration, queue and event is released exactly once and pre/post counts are equal. Results use create-new atomic raw/result/commit or failure evidence, Windows writable-handle fsync, directory durability/recovery and no physical retry.

## Adjudication

- `heterogeneous_component_positive`: p0 and all p1-p3 rows pass every gate.
- `validation_performance_negative`: p0 correctness passes but a performance formula fails; tests stay sealed.
- `test_performance_negative`: p0 passes and any opened test misses a performance formula.
- `correctness_negative`: any oracle/device/arm bit mismatch or control-observability failure.
- `blocked_capability`, `blocked_resource`, `blocked_paging`, `blocked_thermal_or_device`, `invalid_protocol`: fail-closed named outcomes.

A positive result opens only a separately preregistered rotating-real-record endurance/cache-resistance phase.
