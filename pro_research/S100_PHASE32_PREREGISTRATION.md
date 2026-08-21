# S100 Phase 32 preregistration — Phase31-rebased H8 verifier

Frozen before any Phase32 GPU timing.

## Question

Can the exact Phase25 H8 route-union verifier become economically better than
two launches of the adopted Phase31 `attention_head_m4` parent when the dense
and shared paths reuse weights across all eight rows?

This is a target-only perfect-draft ceiling. It is not achieved speculative
throughput and it excludes drafter, selector, rejection and rollback cost.

## Parent and arms

- Parent: two consecutive exact Phase31 `attention_head_m4` H4 launches.
- `dense_m8`: Phase25 `direct8_route` MoE plus direct-L2 M8 BF16 Q/K/V/O,
  direct-L2 M8 FP32 router, two exact direct-L2 M4 shared-expert waves and two
  exact direct-L2 M4 LM-head waves.
- `dense_split4`: identical except Q/K/V/O and router use two direct-L2 M4
  waves. This isolates any M8 register/occupancy penalty.

The candidate keeps Phase25's exact direct8 routed-UP, route-wise sparse DOWN,
H-SCALE planes, cache policy, Mamba block scan, attention recurrence and final
route accumulation order. No approximate arithmetic or quality relaxation is
allowed.

## Frozen execution protocol

- Context screen: 1024.
- Fresh process per arm.
- Screen: 4 warm-up plus 8 measured H8 windows.
- State capture: one H8 window from canonical position 1024, repeated for
  deterministic token identity; compare against two Phase31 H4 launches.
- Promotion: fastest state-green candidate only.
- Thermal: four position-paired alternating rounds, each 4 warm-up plus 12
  measured windows.
- Context robustness: 128 and 4096 after thermal promotion.

## Gates

Correctness requires exact greedy token IDs, deterministic replay, finite
outputs, SSM NRMSE <= 5e-5, convolution NRMSE <= 1e-5, KV NRMSE <= 5e-6 and
logit NRMSE <= 5e-4 against the Phase31 parent.

Resource gate: every new kernel must report zero local-memory bytes. A kernel
with spills is rejected even if its screen is fast.

Adoption requires median position-paired gain >= 5%, at least three of four
rounds positive, parent and candidate robust CV <= 5%, and all correctness and
resource gates green.

Economic labels:

- `< 127.0625 ms/H8`: better than two Phase31 median H4 windows.
- `<= 100 ms/H8`: strong verifier screen.
- `<= 80 ms/H8`: zero-cost perfect-draft S100 ceiling opens.
- `<= 64 ms/H8`: at least 16 ms of full-acceptance draft headroom.

MTP and DFlash2 are re-adjudicated only from measured candidate wall time and
their existing empirical acceptance/draft data. No projected component sum may
open training or support a throughput claim.
