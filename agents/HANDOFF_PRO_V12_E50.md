# Handoff — PRO V12 exact E50 scheduler line

Active branch: **`pro-v12-async`**.
Do not work on `main`. `pro-research` contains the PRO-MAX V2 result intake;
this branch contains the next experiments.

## Starting fact

Verified V6 on the correct Lightning checkpoint:

- 21.0923 ms/token p50
- 47.4107 tok/s
- 765 timed decode samples
- exact causal parity, deterministic replay, sabotage control pass

PRO-MAX V2 then found that separately queued exact child graph replays run at:

- K=2: 18.7580 ms/token
- K=4: 19.0660 ms/token

The parent child-graph itself was slower. The new signal is therefore not
"child graphs are fast"; it is that blocking `ring_harvest()` after every token
starves the existing causal graph queue.

## Branch experiments

### V12 — fixed-K baseline

`queue_stream_v12.py`

Separates synchronous per-token harvest, fixed-K queued harvest, and fixed-K
per-token event-query delivery. Use it to reproduce the ~19 ms queued signal and
establish whether fixed epochs can individually stream tokens.

### V12B — rolling busy-query credit

`credit_stream_v12b.py` via `credit_stream_v12b_entry.py`.

Never intentionally drains the queue. Windows 1/2/4/8/16/32. As soon as the
oldest D2H completion event is query-ready, deliver that ring token and append
one new replay. Independent verifier: `verify_v12b.py`.

### V12C — rolling blocking-event credit

`credit_wait_v12c.py` via `credit_wait_v12c_blocking_entry.py`.

Same credit algorithm but the token event is explicitly
`cupy.cuda.Event(block=True, disable_timing=True)` and the CPU waits only on the
oldest event, **not on the stream**. CUDA documents that this blocks the CPU
thread through the event while later stream work remains outside that wait.
Independent verifier: `verify_v12c.py`.

### AddNorm diagnostic

PV2-10 was micro-bitexact but one graph rollout diverged only at generated token
124. `diag_addnorm_late_divergence.py` compares production add+RMSNorm with the
fused kernel at every layer on the real troublesome trajectory while continuing
model state with production outputs. It is diagnostic only, never a speed claim.

## One-command order

```powershell
git fetch origin
git switch pro-v12-async
git pull --ff-only origin pro-v12-async
.\pro_research\RUN_E50_LADDER.ps1 -Mode smoke -IncludeAddNormDiagnostic
```

If all technical/correctness checks are clean:

```powershell
.\pro_research\RUN_E50_LADDER.ps1 -Mode full
.\pro_research\PUSH_V12_RESULTS.ps1
```

The ladder writes `pro_research/results/v12_async/V12_E50_REPORT.md`.

## Frozen interpretation

Three claims are distinct:

1. **SYNC E50** — blocking per-token p50 <=20 ms.
2. **QUEUED E50** — one exact autoregressive sequence generates >=50 tok/s when
   host harvest is batched. This is generation throughput, not per-token host
   round-trip latency.
3. **STREAMED E50** — exact tokens are individually host-visible at >=50 tok/s,
   every prompt has steady p50 delivery gap <=20 ms, full arm >=500 tokens,
   baseline drift <=1 ms, independent verifier agrees.

Do not rename QUEUED E50 to interactive latency. If V12B or V12C verifies the
third claim, freeze that scheduler as the new serving baseline and run 10k-token
plus one-hour thermal validation before declaring E50 durable.

## After E50

Highest-value exact follow-ups:

- remeasure the exact mixed Q/K/V one-launch candidate under a stable interleaved
  harness; previous full causal parity passed, performance attribution failed
  only because BASE_A/B drift was 1.8577 ms;
- Mamba conv/dt overlap;
- attention O-projection + residual write;
- graph/event/fence census on the new scheduler;
- mapped-host no-bounce/TMA physical microbenchmark before any TMA claim;
- for aggregate E100, do **not** increase N in the existing Python loop. Build a
  fixed-N_MAX graph-resident batch step with device route union and shared expert
  fetch; the previous N=8 Python loop collapse was launch/orchestration overhead,
  not cache thrashing.
