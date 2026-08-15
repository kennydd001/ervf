# HET-NEXT-L0-C0-R3 — exact activation and concurrency revision

## Status and inheritance

This immutable design supersedes C0-R2 for future work and authorizes no executable preflight, compiler, device enumeration, allocation, kernel launch, timing, GPU/iGPU use or output creation. No C0-R3 runner exists.

C0-R3 incorporates the complete C0-R1 preregistration/design by SHA-256 `7596e2083ce8ca07490d1582b9f9fcb3b6ddc36cc29580f06c4b4f80ec763ea0` and `dc22fb5b65dcfd16d4899e2f0ecb00f6100d155d717bd48bafbb6167df859e92`, then the C0-R2 preregistration/design by SHA-256 `c29c4fe51e8bb7207a67f7cd47fd06371480f4243f0b7bdc63eb6238be98bddb` and `5f4c41af370b0705aa1c181788a841ba2621f8f064c602fa0d3743e41049d7bd`, except for the exact supersessions below. It binds the original independent C0 design audit SHA-256 `d2d33e0131b56fee2432c6945226998058495ec06bc44639bf42cba1d9767fed`.

The claim remains only a real-weight natural-route heterogeneous whole-expert component result. It is not official-layer/logit equality, held-out quality, end-to-end speed, full-model acceleration, cache-resistant throughput, endurance, production readiness or a breakthrough. All C0-R1 validation seals, route IDs, ownership, official expert-ID accumulation, exact schedule/statistics, CPU-only cache perturbation, paging/resource/control/lifecycle gates and adjudication remain binding unless explicitly refined here.

## Normative official BF16 activation semantics

The normative CPU oracle runtime remains Python `3.12.10`, PyTorch `2.12.1+cu132`, build `7269437d655783a26cba32aa88195b741ff496aa`, AVX2, one intra-op and one inter-op thread, deterministic algorithms enabled, float32 matmul precision `highest`, MKLDNN enabled and flush-denormal disabled. Its exact bound files remain:

- `torch/nn/functional.py` `e409a97896241e0dfb8c23fbf1f09967ecf5e65ec9626aec0d97d9cc5d727d50`;
- `_C.cp312-win_amd64.pyd` `0948fb62c5e58866a485077cf54f8cfd907fcd8482bf8f139823d1d0a724c7d2`;
- `torch_cpu.dll` `56aaff6d76ee7ba9573e88fd8e920acb170e5c0a8d9d2ee94e8a20ed480aa32b`;
- `c10.dll` `9aa3fb6fe82d9b3a0ccd6d406d59b61140a65990d3ffd3929b9ee0b6f4954866`;
- `libiomp5md.dll` `2299b0460e8118e8187fd57a8d17df836c2a3d59f2639c3681582070da66b7be`.

For each routed or shared expert, the CPU oracle executes exactly:

1. BF16 `x` and BF16 Q5-decoded gate/up weights enter the frozen ERGV linear reductions; gate and up results are stored BF16.
2. `torch.nn.functional.silu(gate_bf16, inplace=False)` is called directly. Its returned tensor must be BF16 and is retained as raw BF16 words.
3. `activation_bf16 = silu_bf16 * up_bf16` uses PyTorch BF16 elementwise multiply and returns BF16; no explicit FP32 promotion is inserted.
4. BF16 activation and BF16 Q5-decoded down weights enter the frozen ERGV down reduction; down output is stored BF16.
5. Routed down is multiplied by the captured BF16 route weight and converted/stored BF16; contributions are `index_add_`-equivalent into a BF16 zero buffer in ascending expert-ID order, retaining every accumulator state.
6. Shared gate uses the direct official call `shared_sigmoid_bf16 = torch.sigmoid(shared_gate_linear_bf16)`. Input and returned tensor must both be BF16. Only its raw BF16 words are normative.
7. `shared_gated_bf16 = shared_sigmoid_bf16 * shared_raw_bf16` uses this exact operand order and returns BF16. Final routed-plus-shared add returns BF16.

An optional diagnostic may compute `torch.sigmoid(shared_gate_linear_bf16.float())` and retain FP32 words under an explicitly named `diagnostic_fp32_*` key. That diagnostic is never an oracle, gate input, device input or acceptance criterion and cannot replace the official BF16 call.

Every normative intermediate is retained as contiguous little-endian BF16 words plus dtype/shape/bytes/SHA-256. The independent verifier reruns these exact CPU operations from source input and decoded weights; it does not trust stored summaries.

## No timed activation shortcut

Each timed device arm is a whole-expert computation. For every assigned expert, that device must in the same sample:

1. read/decode its Q5 gate/up weights;
2. compute its own BF16 gate and up ERGV outputs;
3. compute its own SiLU(gate) and BF16 SiLU-times-up activation with the source-bound device implementation;
4. pass that device-produced activation directly into its own Q5 down ERGV without a host or CPU activation roundtrip;
5. return the BF16 down output.

For shared, NVIDIA likewise computes its own gate/up activation and shared down. The captured official shared-gate-linear tensor is only the separate outer shared gate, as already preregistered.

CPU-oracle gate/up/SiLU/activation arrays are comparison evidence only. Timed device queues must not read, map, copy or alias them. Source audit and call ledgers must prove the down input pointer/hash is the immediately preceding on-device activation buffer and that gate/up Q5 decode, activation and down kernels occur between inclusive `t0` and worker `done`. Any Intel or NVIDIA gate, up, SiLU, activation, down or aggregate bit mismatch against the normative CPU BF16 words is `correctness_negative`. If either backend cannot implement the exact BF16 activation semantics, C0-R3 fails; tolerances and CPU substitution are forbidden.

## Exact native Windows synchronization

Capability must prove logical processors 0, 2, 4 and 6 are present, in one Windows processor group and on four distinct physical cores. Otherwise `blocked_capability`; no affinity retuning is permitted.

- LP0: coordinator only.
- LP2: persistent Intel worker and sole owner/submission thread of one in-order Intel queue.
- LP4: persistent NVIDIA worker and sole owner/submission thread of one in-order CUDA stream.
- LP6: persistent independent PDH monitor only; it performs no thrash, submission, merge or device API call.

All synchronization below uses Win32 kernel event handles created by `CreateEventW`; Python `threading.Event`, futures and thread pools are forbidden. For each worker `i in {intel,nvidia}`:

- `command_i`: auto-reset, initially nonsignaled;
- `ready_i`: manual-reset, initially nonsignaled;
- `done_i`: manual-reset, initially nonsignaled.

One common `start`: manual-reset, initially nonsignaled. Descriptor fields and unsigned64 epoch are protected by one `SRWLOCK`; publication/observation happens under exclusive/shared acquire-release respectively. Worker loop waits `command_i`, reads its descriptor+epoch under the lock, performs no device submission, sets `ready_i`, waits `start`, submits the exact input-copy/gate-up/activation/down/output-copy sequence to its owned queue, synchronizes its own output, records telemetry, then sets `done_i`. Events are never pulsed.

Before each stage the coordinator resets `start`, relevant `ready_i` and relevant `done_i`, writes descriptor+epoch under the SRW lock, then signals commands in Intel-then-NVIDIA order where both are relevant. It waits for all relevant ready handles. Inclusive `t0=QueryPerformanceCounter()` is sampled immediately before the first `SetEvent(start)` of the arm:

- A: arm only NVIDIA; signal `command_nvidia`, wait `ready_nvidia`, sample `t0`, set `start`, wait `done_nvidia`.
- S: signal Intel command and wait ready; sample `t0`, set `start`, wait Intel done; reset `start`; signal NVIDIA command, wait ready, set `start`, wait NVIDIA done.
- B: signal Intel command then NVIDIA command; use `WaitForMultipleObjects([ready_intel,ready_nvidia], bWaitAll=TRUE)`; sample `t0`; one `SetEvent(start)` releases both; use `WaitForMultipleObjects([done_intel,done_nvidia], bWaitAll=TRUE)` with that fixed handle order.

After required done states, LP0 copies from fixed output slots and performs the exact expert-ID-sorted BF16 host merge. Inclusive `t1=QueryPerformanceCounter()` is sampled immediately after the final BF16 add and before telemetry serialization. The coordinator resets completed events only when both workers have left the prior epoch; an atomic acknowledged-epoch per worker proves this before descriptor reuse. Timeouts are fixed at 30 seconds per wait and yield `blocked_device`, never a retry.

Every observation retains QPC frequency and all command/ready/t0/start/submit/event/done/merge/t1 timestamps, epoch, OS thread IDs, group affinities, handle identities, queue/stream identities and submission sequence. Any ownership, affinity, event-state, epoch or ordering mismatch is `invalid_protocol`.

## Independent PDH monitor

LP6 creates and owns the PDH query/counters. For each opened row it starts exactly 2.000 seconds before that row's first warmup and stops exactly 2.000 seconds after that row's final cleanup. It uses an LP6-owned waitable timer with due times `monitor_start_QPC + k * round(QPC_frequency/10)` for integer `k`, calls `PdhCollectQueryData` at each due time, and retains scheduled/actual QPC and lateness.

No coordinator wait or CPU thrash runs on LP6. If any interval between successive actual collections is outside `[80 ms,120 ms]`, any lateness exceeds 20 ms, or any PDH status is invalid, the row is `invalid_protocol`. The exact English counter paths, baseline and thresholds remain C0-R1. The monitor exposes atomic latest raw/formatted samples to LP0, but LP0 does not call PDH. At row end LP0 requests monitor stop only after final cleanup; LP6 continues for 2.000 seconds, commits its ledger, closes every PDH/timer handle once and reports cleanup equality.

## Next gate

A new independent no-device design audit must return GO before executable Phase-0 source is written. Capability, p0 source-build and physical validation remain separately audited and authorized future phases.
