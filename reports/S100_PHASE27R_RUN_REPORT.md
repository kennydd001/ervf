# S100 Phase 27R — Thermally Stable Pipeline Adjudication

Phase27R changes no kernel or model math. It remeasures the frozen Phase27 candidate under a balanced thermal/order protocol because Phase27's candidate A/B was stable while the parent A/B bracket exceeded the stability gate.

Frozen candidate:

```text
gather_y       = 4
batches        = 3
shared_overlap = true
```

| Round | Parent ms | Candidate ms | Gain | Aligned |
|---:|---:|---:|---:|:---:|
| 1 | 74.9629 | 72.0306 | 3.912% | yes |
| 2 | 76.1569 | 71.8161 | 5.700% | yes |
| 3 | 74.7420 | 72.3331 | 3.223% | yes |
| 4 | 75.3148 | 73.6884 | 2.159% | yes |

- Median round gain: `0.035672788223393315`
- Median 64-position paired gain: `0.03698393747952772`
- Positive rounds: `4/4`
- Parent robust CV: `0.005651095023112088`
- Candidate robust CV: `0.005310562558316099`
- Adopted: `False`

- TARGET_100_TARGET_ONLY_OPEN: `False`
- DRAFTER_SHOOTOUT_OPEN: `False`
- NEXT_ROUTE: `FUSE_GATHER_DOWN_AND_ELIMINATE_MIRROR_TRAFFIC`
- S100 SINGLE ACHIEVED: `False`
