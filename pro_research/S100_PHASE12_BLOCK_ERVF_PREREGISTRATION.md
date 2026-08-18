# S100 Phase 12 preregistration — lossless mini-prefill verifier

## Phase 12A: verifier economics before drafter training

Draft tokens are initially taken from the known target continuation so acceptance is artificially 100%. This is not a throughput claim; it isolates the target verifier's minimum possible cost.

Frozen block sizes: B={2,4,8}. For each B:

- verify the known next B tokens in one block forward;
- produce logits for all positions;
- compare every position with B sequential baseline steps;
- preserve exact Mamba state, attention KV state, MoE routing and cache semantics;
- report full cycle time including route grouping and state commit.

A drafter may be trained only if the perfect-draft verifier satisfies at least one break-even gate:

- B=2 verifier <=18 ms;
- B=4 verifier <=28 ms;
- B=8 verifier <=40 ms.

## Phase 12B: route-union census

On at least 10,000 frozen target tokens, report per layer and B:

- B*6 route slots;
- unique experts;
- tokens per unique expert histogram;
- cache hits/misses before and after deduplication;
- bytes fetched under current and grouped policies;
- routed-up and routed-down useful M distribution.

Grouped MoE opens only if median routed-weight bytes per token fall by >=20% at B=4.

## Phase 12C: exact ERVF-M

Every candidate must be bit-identical to B independent baseline calls for real checkpoint matrices and adversarial inputs. Weight rotation must exceed 4x L2. Integrate only candidates meeting useful-row throughput gates specified in the Phase-12 plan.

## Phase 12D: drafter

Train only lightweight auxiliary modules with the target frozen. Compare FastMTP, PARD and DFlash-style parallel blocks on disjoint calibration/validation/heldout splits. Record acceptance-length distribution by domain, not only the mean.

## Final claim

The final measurement is useful accepted target tokens divided by complete wall time. S100 requires a lower 95% confidence bound above 100 tok/s and exact greedy token identity against the frozen quality-green parent. Component results, perfect-draft verifier results and draft-only throughput cannot make the claim.
