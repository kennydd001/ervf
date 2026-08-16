# PRO-MAX V2 — post-V6 preregistration

Frozen before any target-hardware run of this pack.

Base: `pro-research@5c699300da2d10552f5037426c1607119b2239b4`.
Target checkpoint: `models/nemotron_3_5_lightning_v35`.
Current verified reference: `PRO_V6_FULL_STACK.json`, p50 `21.0923 ms/token`
(`47.4107 tok/s`) over 765 timed causal samples.

## Claim boundary

This pack has two distinct objectives.

1. **E50 single-stream:** remove at least `1.0923 ms/token` from the verified
   V6 point while preserving target token semantics.
2. **Architecture probes toward E75/E100:** test exact graph amortisation and
   CUDA capabilities. These probes are not themselves a 75/100 tok/s claim.

`100 tok/s aggregate` and `100 tok/s for one sequence` are never conflated.

## Fixed experimental order

1. provenance/budget lock;
2. add+next-RMSNorm candidate;
3. mixed-shape Q/K/V candidate;
4. LM-head+top-1 candidate;
5. physical composition according to the frozen adoption rule;
6. low-level child-graph epoch probe;
7. capability census;
8. independent verifier and report.

## Candidate adoption rule

A candidate enters the V10 composition only if its own raw result proves:

- production-output bit equality in the direct microcheck;
- exact causal token parity in its graph A/B/A arm;
- baseline A/B p50 drift <= 1.0 ms;
- no p50 regression greater than `0.2%` versus the A/B midpoint;
- the candidate kernel appears in the captured graph DOT representation.

This deliberately allows a tiny neutral candidate into composition only when it
is exact; it does not allow a known regression. The V10 full causal result is
the only source of a combined speed claim.

## PV2-10 — add + next RMSNorm

The original transition is:

```text
add_inplace(h, residual)
rmsnorm_bf16w(h, next_weight, normed)
```

The candidate assigns every norm element to the same virtual thread as the
production norm kernel. That thread first performs the same float32 add, stores
`h[i]`, then performs the same `fmaf(v,v,acc)` sequence. The 256-thread warp and
cross-warp reduction order is copied verbatim. It must reproduce both the
updated hidden state and the normalized output bit-for-bit.

The full graph starts with the ordinary layer-0 norm. After each layer, the
candidate computes `h += acc` and the next layer norm in one launch. After the
last layer it produces the final model norm. No layer arithmetic is reordered.

## PV2-11 — mixed Q/K/V one-launch kernel

The candidate uses one 256-thread launch grid with three regions:

- Q rows: the already verified width-16 ERVF reduction;
- K rows: the production one-block-per-row BF16 reduction;
- V rows: the production one-block-per-row BF16 reduction.

The three matrices remain separate pointers. Q/K/V values and their reductions
must be bit-identical to the current V6 selective dispatch. Only launch
aggregation changes.

## PV2-12 — LM-head ERVF + hierarchical exact top-1

The first kernel performs the same NVFP4 ERVF MAC assignment and reduction tree
as V6. In debug mode it writes every logit, allowing a complete bit comparison.
Each 16-row physical block also writes its exact local `(value,index)` winner.
A second kernel performs a deterministic low-index-tie top-1 scan and writes
`tok_dev` directly.

In the captured candidate graph the full logits buffer is not materialised or
reread solely for greedy top-1. This candidate is valid only for greedy decode;
a future API requesting logits must use the original path.

## PV2-13 — V10 physical composition

V10 uses the unchanged V6 stack and adds only candidates satisfying the frozen
adoption rule. It runs BASE_A / V10 / BASE_B, determinism, and the existing
`bad_pick` sabotage control.

Smoke mode is diagnostic. Full mode requires:

- 3 prompts x 256 generated tokens;
- >=500 timed decode samples;
- exact causal ids for BASE_A/V10/BASE_B;
- deterministic V10 replay;
- sabotage diverges on at least one prompt;
- extra VRAM <64 MiB;
- baseline A/B drift <=1.0 ms.

Milestones are reported mechanically:

- E50: p50 <=20.000 ms;
- E75: p50 <=13.333333 ms;
- E100 single-stream: p50 <=10.000 ms.

## PV2-20 — exact child-graph epochs

The previous G2 attempt tried to capture `cudaGraphLaunch()` and correctly hit
`cudaErrorStreamCaptureUnsupported`. PV2-20 is a different mechanism: it uses
`cudaGraphAddChildGraphNode` to clone the captured token graph into a parent
DAG. Device-to-device copies record each intermediate token after each child.

For K in `{2,4,8,16}`, compare:

- K host-issued child graph launches with one final synchronize;
- one parent graph launch containing K dependent child nodes.

Generated ids must be exactly equal. This measures queued/offline throughput;
it does not reduce the user-visible latency of the first token.

## Verification

`verify.py` imports no experimental runner. It reads raw JSON and recomputes the
status and every numerical gate. A technical failure remains a technical
failure. Missing results are marked missing, never interpreted as a negative
hypothesis result.
