# S100 Phase31 — exact residual-sink fusion

## Frozen parent

`codex/s100-phase30e-breakthrough@f51d207914ccd32bc7c3133d8826ab70b747fca1`
is the only performance parent.  Its context-1024 thermal median is 71.10845
ms/H4 (56.252 target-only tok/s), with exact IDs/state and four positive
rounds.

## New mechanism

The parent terminal path of every one of the 23 MoE layers is:

1. `reduce_routes(partials -> route_down)`;
2. wait for the overlapped shared expert;
3. copy `shared_out -> acc`;
4. `accumulate_h4(acc, route_down, route_w)`;
5. four separate `add_(h[token], acc[token])` launches.

Phase31 removes the `acc` materialization and writes the mathematically
complete MoE contribution directly into the residual stream.

- `sink`: keep `reduce_routes` before the shared wait, then fuse steps 3–5.
- `reduce_sink`: fuse steps 1 and 3–5, removing `route_down` traffic too, at
  the cost of moving reduction behind the shared wait.

Both kernels retain chunk order `0..nchunks-1`, route-slot order `0..5`, the
same explicit `fmaf`, and one final FP32 residual addition.  No quantization,
routing, cache, gather, down-projection, Mamba, attention, KV or head arithmetic
is changed.

## Falsifiable hypotheses

- H31-1: `sink` is bit/state exact and saves at least 3% in the initial matched
  context-1024 screen.
- H31-2: `reduce_sink` is bit/state exact; it beats `sink` only if the removed
  route-down round-trip is worth more than the lost reduction/shared overlap.
- H31-3: the selected arm has no local-memory spills and uses at most 64
  registers/thread.

## Fixed execution ladder

1. static Python compile;
2. CUDA compile/resource audit;
3. one-block parent/sink/reduce_sink exact smoke;
4. parent-vs-each-candidate state capture at context 1024;
5. matched screen in rotating order, 8 measured H4 blocks after 4 warmups;
6. only an exact arm with at least 3% screen gain proceeds to four thermal
   rounds of 16 measured blocks after 8 warmups;
7. adoption requires all four rounds positive, median gain at least 5%, and a
   positive bootstrap lower-95% bound;
8. adopted code must remain positive and exact at contexts 128 and 4096.

## Stop rules

- Any ID, SSM, convolution, KV or logits mismatch closes that arm.
- Any local-memory allocation or resource-audit failure closes that arm.
- If neither arm reaches 3% in the matched screen, terminal fusion is recorded
  as measured-but-insufficient; no thermal claim is made.
- Timing-only diagnostics may guide a later phase but cannot promote code.

## Claim boundary

Target-only exact H4 verification on the local RTX PRO 2000 Blackwell Laptop
GPU.  No drafter, acceptance, rejection, fallback or end-to-end speculative
decoding cost is included.

## Phase31B preregistered addendum — multiplicity-lane pipeline

The residual-sink micro ladder is allowed to close without ending Phase31.
Before running the next code, Phase31B adds one new dependency experiment.

Phase30E launches routed-UP for multiplicities M1-2 and M3-4 sequentially,
then waits for both before one global scan and the gather/down pipeline.  The
group sets and route rows of those multiplicity buckets are disjoint.

Phase31B therefore uses two independent producer/consumer lanes:

1. main stream launches M1-2 and records `up12_ready`;
2. lane 12 scans, gathers and computes sparse-down for count 1-2 groups;
3. concurrently, main launches M3-4 and records `up34_ready`;
4. lane 34 scans, gathers and computes sparse-down for count 3-4 groups;
5. main joins both lanes before the unchanged reduction and accumulation.

Each lane retains Phase27R's three gather batches and uses a separate gather
and down stream so batch `n+1` transfer can overlap batch `n` compute.  The
subset predicates only choose ownership; each active route executes the exact
parent scan/gather/down body once.  The two lanes write disjoint route masks,
group mirrors and partial rows.

Additional gates:

- both lane kernels must compile with zero local-memory spills;
- one-block IDs must be exact before any screen;
- state must pass before thermal promotion;
- at least 3% matched screen gain is still required;
- if lane duplication is slower, a later compact-list kernel needs a separate
  preregistration rather than post-hoc reinterpretation.

## Phase31C preregistered addendum — device-mirror grouped sparse-down

Phase31B was token-exact but its two-block micro median (75.53125 ms/H4) did
not beat the matched parent micro median (75.46145 ms/H4).  The lane split is
therefore closed before this code is measured.

Phase31C keeps Phase30E's shared branch, routed-UP, scan and three-batch PCIe
gather unchanged.  It changes only the consumer of the gathered device mirror.
The parent launches one sparse-down calculation per route, so routes selecting
the same expert reread the same gathered NVFP4 bytes.  Phase31C launches by
expert group and chunk.  A block loads each required 128-row code tile once
into shared memory and applies it to all one-to-four routes in that group.

Exactness is structural: each route retains its own accumulator; its panel
sequence remains `pi = chunk + step * nchunks`; its active columns remain
ascending bit order; and reduction plus route-slot accumulation are unchanged.
Cross-route reuse changes data movement only, never a route's FMA sequence.
Two specialized kernels cover counts 1-2 and 3-4, and the existing three group
ranges remain the gather/down ownership boundary.

Phase31C must have zero local-memory spills, exact one-block IDs, and at least
3% matched-screen gain before state or thermal testing.  If it misses that
threshold, grouped device-mirror reuse is closed in this form.

## Phase31D preregistered addendum — direct-L2 BF16 attention M4

Phase31C was token-exact but its two-block micro median (75.6618 ms/H4) did
not beat the parent.  Device-mirror group reuse is closed before Phase31D is
measured.

The earlier Phase24 BF16 M4 kernel staged four complete activation rows in
dynamic shared memory and measured slower than four production GEMVs.  That
experiment predates Phase30B's key occupancy result: for the shared expert,
removing activation staging and relying on L2 changed an otherwise rejected
M4 design into the adopted Phase30E kernel.

Phase31D applies that mechanism to the six attention layers.  One CTA still
owns one output row and retains production's `k = threadIdx.x + 256*n` FMA
sequence plus the same warp/two-level reduction.  It loads each BF16 weight
once and updates four independent position accumulators, reading the small
activation rows directly through L2.  Q, K, V, KV writes, causal attention and
O arithmetic are otherwise unchanged.

Gates: bit-exact attention outputs and one-block IDs; zero local-memory spills;
at most 64 registers/thread; and at least 3% matched full-verifier gain before
state/thermal promotion.  A compile or exactness failure closes the arm.

## Phase31E preregistered addendum — direct-L2 FP32 router M4

Phase31D passed its balanced screen at 73.087875 ms/H4 parent midpoint versus
66.654175 ms/H4 candidate midpoint (8.80% gain), and its SSM, convolution, KV
and logits captures are bit-identical.  It becomes the comparison parent for
this incremental experiment, while Phase30E remains the adoption baseline.

Phase31E applies the same staging-free four-row mechanism to every FP32 router
matrix.  Four production-order GEMVs share each FP32 weight load; each token
retains its own accumulator and the production two-level reduction.  The four
unchanged `route_topk` calls consume the four output rows in the original token
order.  No expert, cache or MoE arithmetic changes.

The router addition must compile with zero local memory and at most 64
registers/thread, keep exact one-block IDs, and improve the Phase31D micro
median by at least 1% to proceed.  Full Phase31 adoption still requires at
least 5% over Phase30E and the original state/thermal/generalization gates.

## Phase31F preregistered addendum — direct-L2 NVFP4 LM head

Phase31E was token-exact but measured 70.96965 ms/H4 versus Phase31D's
69.8558 ms/H4 micro median, so router M4 is closed before this experiment.

The selected Phase22 head executes four production ERVF passes and measured
about 6.01 ms/H4.  Its rejected generic M4 alternative staged 43 KiB of H4
activations independently in every vocabulary-row CTA and cost 21.31 ms.  The
Phase30B direct-L2 NVFP4 kernel removed exactly that occupancy defect for the
shared expert and is already bit-exact in the same ERVF arithmetic.

Phase31F applies that proven kernel to the fixed LM head after Phase31D:

- `head_m4_direct`: one weight pass and four accumulators, 96 registers;
- `head_m2_direct`: two weight passes over rows 0-1 and 2-3, 64 registers.

Both retain each output's production virtual-thread FMA and reduction order;
argmax remains unchanged.  One-block IDs must be exact and the best arm must
improve Phase31D by at least 1% to proceed to state/thermal testing.
