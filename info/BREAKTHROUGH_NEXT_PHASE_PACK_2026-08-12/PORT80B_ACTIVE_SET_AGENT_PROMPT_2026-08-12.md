# Agent prompt — PORT80B Active-Set Dominance

Open an independent registry `PORT80B_ACTIVE_SET`.

Do not alter or reopen STREAMQ5, CRAFT, RSIV, HERA, CORETAIL or the next-wave
registry.

## P0 — physical 46.497-GiB host-bank gate

Build the exact final-size synthetic Q5 bank and benchmark:

- memory map plus pinned staging;
- optional full host registration;
- zero-cache top-10 traces;
- 4K and 32K cache budgets;
- 10,000 tokens;
- one-hour stability.

No model quality is involved.

Pass:

```text
no hard page faults after warmup
process commit <=58 GiB
H2D p95 <=45 ms
```

Failure due memory pressure authorizes 96 GB RAM. No other hardware purchase.

## P1 — real Q5 bank

Use official revision-locked BF16 shards, one shard at a time. Hash before
conversion, append to an immutable bank, verify every record, then delete the
source shard.

Build:

- routed Q5 experts;
- resident shared Q5 experts;
- Q8 dense shell;
- tokenizer and exact official routing semantics.

## P2 — quality

Validation before test. Require relative CE <=2%, full-depth live routing and
512-token stability.

## P3 — hybrid shell

Reuse the official Qwen3-Next reference or llama.cpp shell. Implement and
verify:

- 36 Gated DeltaNet layers;
- 12 full-attention layers;
- causal Conv1D;
- recurrent state;
- shared expert;
- official top-10 and weights.

## P4 — zero-cache full decode

The primary claim must not depend on cache locality.

Pass:

```text
4K context
mean >=10 tok/s
p95 <=100 ms
VRAM <=8 GiB
no host paging
```

## P5 — cache and 32K

Only after zero-cache closes:

- validation-only cache allocation;
- 32K context;
- separate prefill and decode metrics.

## P6 — prefill

Implement grouped expert GEMM and chunk Gated DeltaNet. Report TTFT for
128/512/2048/4096 tokens.

## Claim boundary

A synthetic shape gate is not an 80B runtime. Active-Set Dominance is proven
only if the real 80B model meets the same-hardware quality and speed gates.
