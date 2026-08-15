# HET-NEXT-L0-C0-R2 — real-weight whole-expert Intel + NVIDIA component gate

## Status, inheritance and claim boundary

This immutable design supersedes C0-R1 for future work. It authorizes no executable preflight, compilation, device enumeration, allocation, kernel launch, timing, GPU/iGPU use or output creation. No C0-R2 runner exists.

Except where made more exact below, C0-R2 incorporates the C0-R1 preregistration and capability design byte-for-byte by SHA-256 `7596e2083ce8ca07490d1582b9f9fcb3b6ddc36cc29580f06c4b4f80ec763ea0` and `dc22fb5b65dcfd16d4899e2f0ecb00f6100d155d717bd48bafbb6167df859e92`. It also binds the prior independent C0 audit exactly as `d2d33e0131b56fee2432c6945226998058495ec06bc44639bf42cba1d9767fed`. The original C0 and C0-R1 remain immutable negatives.

The only admissible positive claim remains:

> On four frozen real-weight, natural-route Qwen3-Coder-Next layer-0 rows, a fixed whole-expert Intel/NVIDIA partition reproduced the frozen exact CPU Q5/ERGV component arrays and met the preregistered concurrent-hybrid/dGPU-only inclusive-latency ratios.

This is not official-layer/logit equality, held-out quality, tokens/s, a full-model result, cache-resistant throughput, endurance, production readiness or a breakthrough. D2-R3 remains diagnostic; R5 remains formally verifier-negative; C1-R2A establishes only synthetic control sensitivity.

All evidence hashes, official shard identity, p0 validation/p1-p3 sealed tests, rank ownership, Q5 codec, resource/control/lifecycle gates and adjudication in C0-R1 remain binding. In particular, before committed p0 PASS only p0 payloads and p0-required official tensors may be opened; p1-p3 route IDs are public hash-locked metadata only. Test payloads and test-only weights remain byte-range sealed until a separate clean post-PASS process.

## Frozen CPU activation oracle

Q5 ERGV gate/up/down and BF16 casts/adds remain exactly as C0-R1. The formerly abstract SiLU/sigmoid operations are replaced by this frozen source oracle:

- Python `3.12.10`, PyTorch `2.12.1+cu132`, build commit `7269437d655783a26cba32aa88195b741ff496aa`.
- `torch.nn.functional.silu(input, inplace=False)` from `torch/nn/functional.py` SHA-256 `e409a97896241e0dfb8c23fbf1f09967ecf5e65ec9626aec0d97d9cc5d727d50`.
- `torch.sigmoid(input)` from the same frozen ATen CPU binary set.
- `_C.cp312-win_amd64.pyd` SHA-256 `0948fb62c5e58866a485077cf54f8cfd907fcd8482bf8f139823d1d0a724c7d2`.
- `torch_cpu.dll` SHA-256 `56aaff6d76ee7ba9573e88fd8e920acb170e5c0a8d9d2ee94e8a20ed480aa32b`; `c10.dll` `9aa3fb6fe82d9b3a0ccd6d406d59b61140a65990d3ffd3929b9ee0b6f4954866`; `libiomp5md.dll` `2299b0460e8118e8187fd57a8d17df836c2a3d59f2639c3681582070da66b7be`.
- CPU capability AVX2; one Torch intra-op and one inter-op thread; deterministic algorithms enabled; float32 matmul precision `highest`; MKLDNN enabled; flush-denormal disabled. These are asserted before oracle work and retained.

For every opened row, the CPU source process retains raw contiguous little-endian BF16 words and SHA-256 for: gate linear, up linear, SiLU(gate), SiLU-times-up, down, route-weighted down, every ascending-expert-ID accumulator state, shared gate linear, FP32 sigmoid result, BF16 sigmoid result, shared raw, shared gated and final component. FP32 sigmoid words are retained before BF16 conversion. The CPU arrays—not a reimplementation of the transcendental on either device—are the bitwise target. Intel/NVIDIA kernels consume the retained BF16 activation/shared-gate operand where relevant and must match every device-computed Q5 linear/down contribution and the resulting CPU merge arrays bitwise. No claim is made that native device transcendental instructions are bitwise cross-backend equivalent.

Official routed accumulation is ascending expert ID with original-rank BF16 weights, exactly as C0-R1; ownership remains rank0-3 Intel and rank4-9 plus shared NVIDIA.

## Frozen three-arm schedule and reverse mapping

Arms are `A=dGPU_only`, `S=hybrid_sequential`, `B=hybrid_concurrent`. Seed is unsigned decimal `2026081302`. The 12-observation block templates remain:

- `T0 = ABS | BAS | SAB | SBA`
- `T1 = ASB | BSA | ASB | BSA`
- `T2 = SAB | SBA | ABS | BAS`

Block `b=0..29` uses `T[(2026081302+b) mod 3]`, starting T0. This is 360 timed observations and exactly 120/arm/row. Warmup is the first 30 observations of the same infinite template cycle and yields exactly 10/arm; timed order restarts at T0.

Within a template, three-arm group index `g=0..3` has exact reverse mapping:

- T0: `0<->3`, `1<->2` (`ABS<->SBA`, `BAS<->SAB`);
- T1: `0<->1`, `2<->3` (`ASB<->BSA` for both occurrences);
- T2: `0<->3`, `1<->2` (`SAB<->BAS`, `SBA<->ABS`).

The canonical unordered reverse-pair ID is `(row,phase,block,min(g,reverse[g]),max(g,reverse[g]))`; both constituent groups store that ID plus their group index and exact arm string. Pair IDs are order-balance evidence only. Performance uses separate-arm NumPy-linear p50/p95 and the four C0-R1 formulas; per-pair ratios remain forbidden.

## Exact CPU cache perturbation

This is explicitly a **CPU cache perturbation only**. It does not claim or attempt to evict Intel device caches, NVIDIA LLC or VRAM and cannot support a cache-resistant device-throughput claim.

Allocate exactly `N=268,435,456` bytes by `VirtualAlloc`, base aligned to 4096 bytes, on the timing coordinator's NUMA node. Define unsigned 64-bit wraparound and:

```
SM64(x):
  z = (x + 0x9E3779B97F4A7C15) mod 2^64
  z = ((z xor (z >> 30)) * 0xBF58476D1CE4E5B9) mod 2^64
  z = ((z xor (z >> 27)) * 0x94D049BB133111EB) mod 2^64
  return z xor (z >> 31)
```

Set `seed=2026081302`; initialize every byte `buf[i]=SM64(seed xor i)&255` for `i=0..N-1`, then retain full SHA-256. Set `global_line_visit=0` once immediately after initialization and never reset it during an opened physical attempt, including across p0 warmup/timed and, after p0 PASS, p1/p2/p3 warmup/timed phases.

Canonical row tokens are ASCII `p0`, `p1`, `p2`, `p3`; phase tokens are ASCII `warmup`, `timed`; the phase-local observation index is zero-based decimal with no leading zeros. Before observation `j` of a row/phase, compute `H=SHA256(UTF8("HET-NEXT-L0-C0-R2|"+row+"|"+phase+"|"+decimal(j)))`; interpret `H[0:8]` as unsigned little-endian and set `start_line mod L`, where `L=N/64=4,194,304`. Then for `k=0..L-1`:

```
line = (start_line + k) mod L
offset = 64 * line
old = buf[offset]
volatile_xor = volatile_xor xor old
buf[offset] = (old + (SM64(seed xor global_line_visit) & 255)) mod 256
global_line_visit += 1
```

`volatile_xor` starts unsigned64 zero for each observation. Perturbation finishes before the inclusive arm clock. Retain row/phase/index, start line, volatile_xor, global counter before/after and a SHA-256 rolling digest of these ledger rows; retain final full-buffer SHA after cleanup. The buffer is never device-mapped. Capability must show N exceeds each reported relevant CPU LLC; otherwise `blocked_capability`.

PDH paging paths, 100 ms cadence, thresholds and warmup-derived per-device clock baseline remain exactly C0-R1.

## Frozen host/device submission topology

Capability must prove logical processors 0, 2 and 4 exist, belong to three distinct physical cores, and are in the same processor group. Otherwise C0-R2 is `blocked_capability`; affinity is never retuned.

- Coordinator/monitor thread: Windows processor-group affinity logical processor 0.
- Persistent Intel submission thread: logical processor 2; it alone creates/owns the single in-order Intel queue.
- Persistent NVIDIA submission thread: logical processor 4; it alone creates/owns the single in-order CUDA stream.

No other thread may submit device work. Worker threads and queues are created before warmup; per-sample allocations are forbidden. Each worker has create-new `ready`, `start` and `done` sequence counters plus a fixed descriptor slot. The coordinator fills both descriptors, waits until required workers have published `ready==epoch`, executes a full memory fence, samples inclusive `t0`, and releases start counters in the fixed order Intel then NVIDIA. Each released worker observes its epoch, submits input copy/Q5 kernels/output copy in that exact order to its owned in-order queue, records its device event, waits/synchronizes its own output, then publishes `done==epoch` with a full fence.

- A: only NVIDIA descriptor is armed; coordinator releases NVIDIA and waits for NVIDIA done.
- S: both are armed; coordinator releases Intel, waits Intel done, then releases NVIDIA and waits NVIDIA done.
- B: both are armed and ready; after `t0`, coordinator releases Intel then NVIDIA without an intervening wait, then waits with `WaitForMultipleObjects(bWaitAll=TRUE)` on the fixed handle array `[intel_done,nvidia_done]`.

After required done states, coordinator copies worker outputs from their fixed host slots, merges in ascending expert-ID/BF16 order, samples inclusive `t1`, then records telemetry. The Intel-then-NVIDIA release and wait-handle order never changes across arms or ABBA groups. Store coordinator/worker processor group, logical CPU, OS thread ID, queue/stream identity, epoch, ready/start/submit/device-event/done/merge timestamps and submission ledger for every observation. A worker affinity/ownership/ordering mismatch is `invalid_protocol`, not performance data.

## Next gate

Independent no-device design audit must pass C0-R2 before any executable Phase-0 preflight is written. Subsequent capability, p0 source-build and one validation attempt each require their own audited immutable authorization exactly as C0-R1 specifies.
