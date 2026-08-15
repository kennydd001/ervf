# HET-NEXT-L0-C0-R4 — per-worker epoch lifecycle revision

## Status and exact scope

This immutable design supersedes C0-R3 only for the worker epoch/acknowledgement lifecycle below. It authorizes no implementation, executable preflight, device enumeration, compilation, allocation, kernel launch, timing or device use.

All C0-R3 science, activation dataflow, schedule/statistics, validation seal, controls, resources, Win32 event topology, affinities, queue ownership, QPC boundaries and independent LP6 PDH monitor remain frozen by:

- C0-R3 preregistration SHA-256 `5957fff1dff54627f8e7cd81a3456d7320988ff43949471266420546685b5dd8`;
- C0-R3 capability/preflight design SHA-256 `6d50092b595ff635faa969ab3883617b8c26927819f684f4adebc5c420835035`.

This revision makes no scientific, threshold, schedule, timing-boundary or claim change.

## Per-worker command and acknowledgement state

For each persistent worker `i in {intel,nvidia}`, allocate distinct cache-line-aligned unsigned64 atomics:

- `last_command_epoch_i`, initialized 0;
- `ack_epoch_i`, initialized 0.

Global observation/stage epochs are strictly increasing positive unsigned64 integers and never wrap during the process. A worker is **active** in a stage only if the coordinator publishes a descriptor to it and signals its `command_i`. An inactive worker receives no descriptor, command or epoch for that stage, and is excluded from every ready/done/ack predicate for that stage.

Before publishing a new descriptor to active worker `i`, the coordinator must establish, with an acquire `InterlockedCompareExchange64(&ack_epoch_i,0,0)`, that `ack_epoch_i == last_command_epoch_i`. If not equal within the frozen 30-second wait budget, the attempt is `blocked_device`; descriptor/event reuse is forbidden. It then resets that worker's `ready_i` and `done_i`, writes the descriptor under the existing exclusive SRWLOCK, writes `last_command_epoch_i = new_epoch` with `InterlockedExchange64` after the descriptor write, releases the lock, and signals `command_i`.

The worker waits for `command_i`, acquires the SRWLOCK shared, reads its descriptor and `last_command_epoch_i`, releases the lock, checks the epoch is strictly greater than its prior acknowledged epoch, then sets `ready_i`. It waits for the common `start`, executes its exact frozen queue work, synchronizes its output, writes all device/output/error/telemetry fields for that epoch, executes a full `MemoryBarrier`, publishes `ack_epoch_i = observed_epoch` using `InterlockedExchange64`, and **immediately afterward** calls `SetEvent(done_i)`. No work or evidence write for that epoch may occur between ack publication and `SetEvent(done_i)`.

The coordinator waits only for `done_i` of active workers. After a signaled done handle, it performs an acquire `InterlockedCompareExchange64(&ack_epoch_i,0,0)` and requires exact equality with `last_command_epoch_i` and the stage epoch before reading that worker's outputs. A mismatch is `invalid_protocol`; no retry. Only after this check may it merge/output-read and later reset that worker's `ready_i`/`done_i` for a future descriptor.

The common manual-reset `start` event is reset only after all workers active in the immediately preceding stage have signaled done and passed their exact ack checks. Inactive workers are never waited on and never block reset or the next stage. Before release of a new stage, all its active workers must have published ready for that new epoch.

Arm application is exact:

- A: NVIDIA alone is active; only NVIDIA command/ready/done/ack advances. Intel state remains unchanged and is excluded.
- S stage 1: Intel alone active and acknowledged; reset common start after Intel ack; S stage 2: NVIDIA alone active and acknowledged.
- B: Intel and NVIDIA active with the same stage epoch; both ready required before one common start release; both done and both exact per-worker acks required before output merge/start reset.

For every stage retain active-worker mask, global epoch, per-worker prior/new `last_command_epoch`, prior/observed `ack_epoch`, all Interlocked return values, ready/start/done event states and QPC timestamps. The independent verifier reconstructs the state machine and proves no inactive-worker wait, descriptor change, event reset or epoch advance.

## Required next gate

The future static preflight must TEMP-simulate at minimum the real schedule prefix `A -> B -> S(Intel) -> S(NVIDIA) -> A`, proving Intel is not required to acknowledge A, both workers are required for B, and each S stage requires only its active worker. It must also inject stale/mismatched acks and reject output access/reset. Independent no-device design audit must return GO before any executable source is written.
