# Post-V6 route to 50 and 100 tok/s

Base: `pro-research@5c699300`.

## Physical state

The verified single-stream record is `21.0923 ms/token = 47.4107 tok/s`.
The immediate E50 deficit is only `1.0923 ms/token`. The exact final-mile pack
therefore tests add+norm, QKV launch aggregation, and LM-head+top-1 before any
new model approximation.

The single-stream E100 deficit is `11.0923 ms/token`. The repository's own
component work and roofline analysis do not currently support claiming that the
remaining batch=1 execution can lose half its latency. E100 is therefore split
into two paths.

## Path A — exact single-stream

1. Run PV2-10/11/12 and physical V10.
2. If V10 reaches <=20 ms, perform 10k-token and thermal E50 validation.
3. Profile the new graph, not V6, before opening another fusion.
4. Remaining exact candidates in priority order:
   - Mamba conv/dt overlap;
   - O-projection + residual write;
   - KV append + attention producer/consumer fusion;
   - graph fence/event audit;
   - context-specific attention split policy;
   - persistent mapped-host down producer/consumer.
5. Do not claim a path to E75/E100 until a physical run is below 16.7/12 ms.

## Path B — aggregate E100 through a real batch graph

The existing prototypes proved expert sharing but also proved that a Python
loop over N sequences collapses under launch overhead. The next implementation
must therefore be graph-resident from its first meaningful end-to-end test.

### B0 — fixed design

- fixed `N_MAX` in `{2,4,8}`;
- device active mask;
- per-slot token and position buffers;
- state `[N_MAX,...]` for Mamba/KV;
- no Python per-sequence layer loop;
- no device allocation inside a decode step;
- one graph replay advances all active slots exactly once.

### B1 — dense shell

Batch the non-MoE shell first with exact outputs:

- embedding gather `[N,H]`;
- BF16/FP8 GEMMs with a small-N matrix RHS;
- per-sequence Mamba state;
- GQA attention with independent positions;
- shared expert;
- final norm, LM-head and per-sequence argmax.

Gate: N independent outputs equal N single-sequence V10 outputs for 64 tokens.
Do not add expert sharing yet.

### B2 — routed expert union

- router emits `[N,top_k]` ids/weights;
- device deduplication creates unique expert work items;
- cache assign runs once over the union;
- each cached up expert processes every assigned sequence;
- down gather uses union-of-column masks, while each sequence keeps its own
  exact mask and reduction;
- route-order accumulation remains per sequence.

Gate: bitexact versus B1 with shared fetch disabled, then physical aggregate
speedup >=1.15x.

### B3 — overlap and graph

- double-buffer expert fetch;
- overlap gather/fetch with already-ready dense/shared work;
- capture fixed N_MAX graph;
- active slots write outputs; inactive slots are neutral and do not mutate
  sequence state.

Gate: aggregate throughput grows from N=1 to N=2 and N=4 without the measured
N=8 Python collapse. Report both aggregate tok/s and per-sequence token latency.

### B4 — E100 validation

An aggregate E100 claim requires:

- >=100 generated tokens/s summed over active sequences;
- exact outputs against independent single-sequence references;
- >=10,000 total output tokens;
- p50/p95/p99 and per-sequence fairness;
- one-hour thermal run;
- fixed power/performance mode;
- no model-quality or routing change.

## What is not a valid shortcut

- multiplying isolated 1.7x MoE fetch numbers into end-to-end speed;
- calling K-token queued graph epochs single-token latency;
- increasing N in the current Python loop;
- reopening the current linear MTP design;
- treating a capability probe as a TMA/no-bounce performance result.
