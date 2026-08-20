# S100 Phase 25 — H8 Best-of-All Full Verifier

## Frozen parent

- Phase24 H4 + H4 official baseline: `148.3404 ms / 8 tokens`.
- Absolute adoption gate: `<= 140.91 ms/H8`.

## Selection

- Selected variant: `direct8_route`.
- Screen latency: `140.4044 ms/H8`.
- Screen target-only: `56.98 tok/s`.
- Full state green: `True`.

## Thermal adoption

- H8 adopted: `False`.
- Thermal median-of-rounds: `n/a ms/H8`.
- Thermal target-only: `n/a tok/s`.
- S100 target-only <=80 ms: `False`.

## State parity

- `{"logits_nrmse": 0.0, "max_conv_nrmse": 0.0, "max_kv_nrmse": 0.0, "max_ssm_nrmse": 0.0}`

## Profile

- Stage totals: `{"cache_group": 1.7331360038369894, "down_compute_reduce": 6.78679995983839, "down_gather": 22.797824010252953, "mask_union": 2.0384160056710243, "routed_up": 28.105472147464752, "router": 8.606752000749111, "shared": 40.2671038210392}`
- Expert streams: `{"ideal_unique": 704.5, "selected": 704.5, "selected_vs_split4_reduction_fraction": 0.03294440631434459, "split4": 728.5}`

## Final

- H8 active parent: `False`.
- S100 target-only achieved: `False`.
- S100 single achieved: `False`.
- Next route: `PROFILE_H8_ECONOMICS_AND_REDUCE_DOMINANT_STAGE`.

> Claim boundary: target-verifier timing is not true end-to-end single-stream throughput until drafter, rejection, and fallback costs are included.
