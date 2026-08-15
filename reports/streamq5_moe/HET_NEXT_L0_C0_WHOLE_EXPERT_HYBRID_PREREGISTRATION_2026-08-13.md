# HET-NEXT-L0-C0 — real-weight whole-expert Intel + NVIDIA component gate

## Status and claim boundary

This document preregisters a design only. It authorizes no compilation, device enumeration, allocation, kernel launch, timing run, GPU/iGPU use, or output creation. No runner exists yet. The only admissible positive claim is:

> On four frozen real-weight, natural-route Qwen3-Coder-Next layer-0 rows, a fixed whole-expert Intel/NVIDIA partition reproduced an independent exact Q5/ERGV CPU oracle and achieved the preregistered concurrent-hybrid/dGPU-only latency ratios as a component test.

It cannot claim official layer/logit equality, held-out quality, end-to-end tokens/s, full-model acceleration, cache-resistant throughput, endurance, production readiness, or an industrial breakthrough. D2-R3 is diagnostic evidence, R5 remains formally verifier-negative because of its natural p0/n8 control conjunct, and C1-R2A proves only synthetic control sensitivity. Those boundaries remain immutable.

## Bound evidence

All implementations and later locks must bind exact bytes for:

- D2-R3 official layer-0 capture raw: `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`.
- D2-R3 result: `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb`.
- D2-R3 independent artifact audit: `a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33`.
- D2-R3 independent interpretation: `be603f4edc648939aa86b2fcec16df802f4e778c6ab14256aecdc48f347da7f0`.
- R5 raw/result/commit: `fcf49479396682634e4a5b9faa3fd3e76c17ba7cfc389e711931996f5e3efbd8`, `56eaac7367da14b060b2c17574a5d36046dca79fafb991de059d6e7b95eb4f91`, `d784ded5e7893095e2f27b75695e635c9cc880109736c87496138e3188509372`.
- R5 control diagnosis: `b22808626e45178cb917cebde5aac789ba720d091ef143099948c94f243bf2e0`.
- C1-R2A raw/result/commit: `d7272ce6aec3533b487829360e40398f1d5fa9d3b766c2593acad01faedca89a`, `bcaa5b2531d422e7eabd09b92fc8f6659c44cbd879614d5d83ee2b9bbc24a736`, `a6736d2b12307cd6cde462513235b4c6f7517289ed89acfde8441832ebcce875`.
- C1-R2A independent verifier source: `ccf7bda4dcc13135a5e43a9c8ee35d79182a61b2b1a0192a6a37337365e99a11`; its persisted verification artifact must be created and bound before implementation freeze if one is later produced.
- Official `Qwen/Qwen3-Coder-Next` revision `a19358a7659bd1f564300250ee189120c49a562f`, shard 1 size `3,999,619,288`, SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- ST2-mini report/source: `af23a4fbffb18028ce1a88b3c73f21546cae7ee397d118c186094f985ee4ac49`, `6472de274fa68a9f577b1483ef1225607f8425ac8587cb348d0c328cff7126ca`.
- D7 report/source/independent audit: `8fce019bb4d51ff5e04a0c91d1cfa43679caf54e86f884acb4fb6a5225df5e86`, `26d4daba81d5f132857f9b584dfb12f3634874a2e9ee2290f9221c486ef4059a`, `acf925757f097e16d62567a64ee95c5974bc95241ee7f0cc5377795eb4d9d1d5`.

ST2-mini established exact Intel host-USM Q5/ERVG but a conservative tail of only `18.540 GB/s`, below its older `21.63 GB/s` gate. D7 established a staged NVIDIA component on synthetic identical payloads at `19.473033 GB/s` effective rate and did not prove differentiated routing. C0 therefore tests a new concurrency/component question; it must not inherit either result as a performance pass.

## Frozen rows and partition

Use only the last position (`index 15`) of each frozen 16-token D2-R3 whole-sequence capture. `p0` is validation. `p1`, `p2`, and `p3` are tests and remain unopened until validation passes. Native BF16 route weights are used bitwise as captured.

| row | rank-ordered routed expert IDs |
|---|---|
| validation `p0/n16` | `50,199,237,474,245,374,239,8,168,12` |
| test `p1/n16` | `42,162,267,299,467,307,326,145,297,182` |
| test `p2/n16` | `474,232,382,80,31,450,103,372,286,206` |
| test `p3/n16` | `26,159,28,176,253,84,431,294,386,356` |

The split is immutable:

- Intel Arc Pro 140T: routed ranks `0..3`.
- NVIDIA device: routed ranks `4..9` and the shared expert.
- No expert may migrate devices after validation or timing.
- No matrix, gate/up/down partial, or reduction tree is split across devices.
- Each device returns a separate BF16 `[2048]` down-output for every assigned routed expert; NVIDIA also returns the BF16 shared-raw `[2048]` output.
- Host code multiplies each routed output by its captured BF16 weight and accumulates in exact official rank order `0..9`, then adds `sigmoid(shared_gate_linear) * shared_raw` using the frozen operand/dtype order. Shared gate linear is taken from the bound D2-R3 capture; it is not recomputed on a device.

## Real Q5 records and oracle

Reread only the required official layer-0 triplets from shard 1. Recreate the R5 codec exactly: contiguous row-major groups of 128; FP32 `max_abs/15` for q selection; q clamped `[-15,15]`; stored BF16 scale; zero group uses BF16 scale 1 and q 0; stored field `q+15` in `[0,30]`; field 31 forbidden; eight little-order fields per five bytes. No persistent bank is permitted. Records reside in anonymous/page-locked/USM memory and device buffers only for the current row or frozen union.

An independently implemented CPU ERGV/Q5 oracle, sharing no device decoder/reduction helper, must produce:

1. every routed expert's ungated BF16 down-output;
2. shared-raw BF16 output;
3. each weighted routed contribution;
4. exact rank-ordered routed aggregate;
5. exact gate-first shared-gated output;
6. final component aggregate.

All intermediate arrays are retained for the four rows. Every Intel result, NVIDIA result, dGPU-only result, sequential-hybrid result, concurrent-hybrid result and host aggregate must be bitwise identical to this CPU oracle. Any mismatch is a correctness negative and test rows stay closed if it occurs in validation.

## Arms and inclusive timing

For each row implement three execution arms with identical decoded semantics and the same host merge:

1. `dGPU_only`: all ten routed experts plus shared on NVIDIA.
2. `hybrid_sequential`: Intel ranks 0–3 completes, then NVIDIA ranks 4–9 plus shared completes, then host merge.
3. `hybrid_concurrent`: Intel ranks 0–3 and NVIDIA ranks 4–9 plus shared start from a coordinated barrier and overlap, then host merge after both completion events.

The primary ratio is concurrent-hybrid inclusive wall time divided by dGPU-only inclusive wall time. Inclusive wall begins immediately before host-side submissions/copies needed for that sample and ends only after both devices, output copies and host merge are complete. It includes launch, synchronization, transfers and merge. Device-event timing is telemetry only. Setup, source quantization and one-time buffer creation are excluded but separately timed and reported. No graph capture or cached result may skip Q5 reads or GEMV work.

After exactly 10 untimed warmups per arm, collect 120 paired timed samples per opened row in a deterministic seeded balanced Williams/ABBA schedule. Every 12-sample block contains equal arm counts and reversed orders; store the complete schedule before first timing. Between samples rotate across at least all row-required real records and write/read a preregistered cache-thrash buffer large enough to exceed combined relevant last-level caches. No timing outlier removal, retry or retuning is permitted. Store every inclusive wall sample, device events, submission/completion timestamps, clocks, temperatures, errors and order.

Validation opens tests only if all correctness/control/resource gates pass and:

- `concurrent_hybrid / dGPU_only` wall `p50 <= 0.90`;
- `concurrent_hybrid / dGPU_only` wall `p95 <= 0.95`;
- concurrent hybrid p50 and p95 are each strictly below sequential hybrid;
- no device-error, thermal hard stop, paging or clock-collapse rule triggers.

All three test rows must independently satisfy the same frozen gates. Ratios use independently recomputed quantiles from raw paired samples; aggregate pooling cannot rescue a failing row. No absolute latency threshold is preregistered.

## Controls and hard failure gates

Before timing, the safe dispatch path must reject wrong expert, wrong device slot, gate/up swap, down-shape mismatch, source/codes/scales digest mismatch and any field 31 before any device enqueue. Retain a call ledger/counters proving rejection ordering. Unsafe bypasses are diagnostic-only and must produce a CPU-oracle-predicted different output. C1's one-hot control may be reused for field sensitivity, but each device path also needs at least one real assigned-expert wrong-slot or projection control whose unsafe output changes.

Additional hard gates:

- exact device identities, driver/runtime/compiler hashes and capability inventory;
- Intel host-USM access and NVIDIA mapped/pinned/device-buffer semantics explicitly proven; no hidden full-weight copy in a timed arm;
- exact Q5 source/record/dispatch binding for every row, rank, expert, projection and device slot;
- finite input, weight, intermediate and output arrays;
- deterministic repeat of every correctness row before timing;
- no overlapping process, unrelated GPU workload or dynamic clock policy change;
- CPU affinity/threads/runtime locked; CUDA and Intel queues created only after all static/source gates pass;
- start available RAM at least 16 GiB, reserve at least 2 GiB throughout; process peak working set at most 12 GiB;
- combined Intel allocations below 1 GiB, NVIDIA VRAM below 256 MiB, retained evidence below 10 MiB;
- pre/post free RAM, Intel/NVIDIA memory, handle/registration counts and cleanup recorded; every allocation, queue, event, mapping and registration released exactly once;
- create-new atomic result/raw/commit or failure evidence with Windows-safe writable-handle fsync and recovery; no retries after a physical attempt.

Thermal hard stop is triggered before a new sample at either device temperature `>=90 C`, a device error/reset, or sustained effective clock below 70% of its validation median for five samples. A hard stop is `blocked_thermal_or_device`, never an outcome-based performance stop.

## Adjudication

- `heterogeneous_component_positive`: validation and all three tests pass every gate.
- `validation_performance_negative`: validation correctness passes but either frozen ratio fails; tests remain unopened.
- `test_performance_negative`: validation opens tests but any test row misses a ratio.
- `correctness_negative`: any CPU/device/arm bit mismatch or unsafe control fails observability.
- `blocked_capability`, `blocked_resource`, `blocked_thermal_or_device`, or `invalid_protocol`: named fail-closed outcomes.

A positive result opens only a separately preregistered cache-resistant rotating-workset endurance/performance phase. It does not by itself justify full-layer or full-model integration.
