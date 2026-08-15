# PORT80B DirectPath — the transfer smoking gun

**Date:** 2026-08-12  
**Source SHA-256:** `40e37bc4fa92e179b6955a493a49906c43a30ec8e7fb8bb9aaa334490ba9dc61`  
**Status:** new independent hypothesis; no 80B quality or decode claim yet.

## Smoking gun

The failed PORT80B P0 did not hit a fundamental PCIe bandwidth wall. It used
an `mmap -> pinned staging -> GPU` path for 480 expert records per token.

Locked facts:

```text
record size             2,027,520 bytes
records/token            480
active bytes/token       973,209,600 bytes
measured pinned H2D      26.158915 GB/s
raw PCIe floor           37.204 ms/token
observed p50             63.034 ms
observed p95             73.544 ms
```

The unexplained p50 is:

```text
63.034 - 37.204
= 25.830 ms
```

Moving the same 973.210 MB through host DRAM in
25.830 ms corresponds to 37.68 GB/s,
a plausible CPU memory-copy rate.

Thus the observed p50 is almost exactly:

```text
one host gather pass + one PCIe pass
```

The p95 adds another 10.510 ms of tail overhead,
consistent with paging, dispatch and synchronization. This is the first
mechanistic explanation that fits the measured magnitude.

## Why the failed 45-ms gate does not kill 10 tok/s

The locked synthetic dense-shell p95 projection is 28.077 ms.
A 100-ms token therefore permits an expert path of:

```text
100 - 28.077
= 71.923 ms
```

The required effective remote bandwidth is only:

```text
0.973210 GB / 0.071923 s
= 13.531 GB/s
```

Current zero-cache p95 already delivers
13.233 GB/s. The gap to the actual
end-to-end 10-tok/s bandwidth is about two percent, not 1.6x.

With perfect expert transfer/compute overlap, the separate projections are:

| Case | Projected p95 | Projected tok/s |
|---|---:|---:|
| Existing zero-cache p95 | 101.621 ms | 9.840 |
| Existing zero-cache p50 | 91.111 ms | 10.976 |
| Existing synthetic 4K cache p95 | 94.225 ms | 10.613 |

These are not full-model measurements, but they prove the P0 failure is
close to the desired operating point.

## Three exact paths to test

### 1. Registered batched copy

CUDA 13.2 provides `cudaMemcpyBatchAsync` specifically to amortize dispatch
overhead. Register a stable host-bank range and submit the ten independent
expert copies for a layer in one batch. This reduces 480 per-record API calls
to 48 per-layer batches and removes the CPU gather when the bank itself is
registered.

### 2. Mapped-host direct ERGV

Map page-locked host memory into the GPU address space and let the exact Q5
kernel read it directly. NVIDIA recommends this on discrete GPUs when data is
read once and accesses are coalesced. Batch-1 expert weights satisfy both:

- each selected weight is consumed once per token;
- every Q5 row is contiguous and coalescible.

This path eliminates both the bounce copy and HBM staging.

### 3. TMA direct host-to-SMEM

DAK demonstrates that direct GPU access to host memory can outperform
prefetch-to-HBM, including up to 1.8x over PCIe offloading baselines. Adapt the
exact ERGV producer/consumer kernel only if a device and compile-time TMA probe
succeeds on the mobile Blackwell GPU.

## The page-layout clue

One expert record is exactly:

```text
2,027,520 / 4096 = 495 four-KiB pages
```

A zero-cache token touches:

```text
495 × 480
= 237,600 four-KiB pages
```

Padding each record to one 2-MiB Windows large page changes the bank from
46.497 to 48.000 GiB, only
1.503 GiB extra. It reduces record-level mapping
granularity to 480 large pages/token. This is a separate measured ablation,
not an assumed speedup.

## Hardware decision

Do not buy RAM before the path-isolation test.

96 GiB becomes rational only when it enables one of these otherwise impossible
conditions:

- full 46.5-GiB host registration;
- a 48-GiB nonpageable large-page bank;
- elimination of hard-fault tails after the double-copy path is gone.

RAM cannot repair per-record dispatch or a serial bounce copy.

## Preregistered order

1. Query CUDA host-mapping and registration device attributes.
2. Benchmark mmap-to-pinned gather only.
3. Benchmark one contiguous 973-MB pinned H2D transfer.
4. Benchmark 480 ordinary copies.
5. Benchmark 48 per-layer `cudaMemcpyBatchAsync` submissions.
6. Benchmark direct mapped-host Q5 ERGV.
7. Only then try TMA and 2-MiB large pages.
8. Repeat the winning path on 96 GiB only when the 64-GiB run shows a
   memory-residency limitation.

## Pass gates

A path passes the hardware screen only when:

```text
bit-exact Q5 outputs
expert-path p95 <= 65 ms
effective remote bandwidth >= 15 GB/s
no hard page faults after warm-up
one-hour thermal stability
```

A real 80B port remains gated on quality, official routing, the hybrid
DeltaNet shell and a 512-token rollout.
