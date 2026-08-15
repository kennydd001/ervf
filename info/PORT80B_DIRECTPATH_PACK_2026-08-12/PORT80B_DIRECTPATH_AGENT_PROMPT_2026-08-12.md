# Agent prompt — PORT80B DirectPath

Open a new independent registry `PORT80B_DIRECTPATH`.

## Immutable parent result

`PORT80B_P0_ACTIVE_SET` remains negative under its frozen gates:

- zero-cache H2D p95 73.544 ms > 45 ms;
- episodic post-warm-up page reads;
- no real 80B model, quality or decode result.

Do not rewrite that verdict. DirectPath tests a new transfer mechanism.

## P0 — device capability probe

Before creating a CUDA context, query and record:

- compute capability;
- `canMapHostMemory`;
- `hostRegisterSupported`;
- `hostRegisterReadOnlySupported`;
- `canUseHostPointerForRegisteredMem`;
- `pageableMemoryAccessUsesHostPageTables`;
- async engine count;
- a compile/run probe for TMA remote access.

Create the context with host mapping enabled where required.

## P1 — isolate the current 73.544-ms path

Use the exact same 46.496887-GiB bank, route digest and 10,000-token protocol.

Measure independently:

1. mmap -> ordinary host buffer;
2. mmap -> eight pinned windows;
3. one contiguous 973,209,600-byte pinned H2D;
4. 480 ordinary registered H2D copies;
5. 48 layer batches of ten copies via `cudaMemcpyBatchAsync`.

Save raw p50/p95/p99, CPU time, H2D events, wall time, page telemetry and
thermal clocks. No model quality is involved.

## P2 — exact direct-host Q5 kernel

Adapt the verified ERGV Q5 kernel so that selected records are addressed
through mapped page-locked host pointers.

Requirements:

- no full HBM staging;
- no materialized dequantized matrix;
- exactly the same Q5 codes, BF16 scales and ordered reduction DAG;
- bit equality against resident-HBM ERGV on random, adversarial and real
  records;
- one-pass coalesced host reads.

Test zero-cache only first.

Primary gate:

```text
p95 <= 65 ms/token for all 480 records
effective host bandwidth >=15 GB/s
zero output bit differences
```

## P3 — optional DAK/TMA path

Open only when the TMA device probe succeeds.

Use a producer/consumer kernel:

- producer warp fetches host Q5 tiles directly into double-buffered SMEM;
- consumer warps execute the exact ERGV reduction;
- cap in-flight requests to avoid interconnect congestion;
- no HBM bounce buffer.

Compare against P2 and the batch-copy path.

## P4 — 2-MiB expert pages

Open only after P2 or P3 is correct.

Allocate one 2-MiB nonpageable large page per expert record at startup:

- 24,576 records;
- exactly 48 GiB;
- `SeLockMemoryPrivilege`;
- no pageable fallback;
- full content digest.

Compare 4-KiB versus 2-MiB mapping with identical records and routes.

## P5 — 64 versus 96 GiB A/B

Authorize a 96-GiB purchase or test only when:

- full-bank registration/large-page allocation fails on 64 GiB; or
- hard-fault tails remain after removing the bounce copy.

Use the exact winning P2/P3 path on both memory sizes. RAM is useful only if
it changes residency or tail latency.

## P6 — real 80B port

Open only after one transfer path passes.

Use official Qwen3-Coder-Next weights and exact official routing. Start with
zero cache. The final model gates remain:

```text
relative CE <=2%
VRAM <=8 GiB
no host paging
4K context
>=10 tok/s mean
p95 <=120 ms
512-token rollout
```

## Claim boundary

- CUDA zero-copy, batch copy, large pages and DAK-style direct access are
  prior art.
- A synthetic transfer pass is not an 80B inference result.
- The possible contribution is an exact quantized-MoE direct-execution
  dataplane on an 8-GB discrete laptop GPU.
