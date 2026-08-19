# S100 Phase 20B — Full H=4 Perfect-Draft Verifier

Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`

Grouped MoE H4 green: **True**
Full H4 state parity green: **True**
Full verifier correctness green: **True**

| context | baseline H4 ms | candidate H4 ms | speedup | ms/useful token | target-only tok/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 156.3425000058487 | 171.74689999228576 | 0.9103075514776164 | 42.93672499807144 | 23.29008558628811 |
| 1024 | 156.3397999998415 | 174.41595000127563 | 0.896361829286244 | 43.60398750031891 | 22.933682383811487 |
| 4096 | 162.93319999385858 | 185.04304999805754 | 0.8805151017321048 | 46.260762499514385 | 21.616591382610636 |

`TARGET_H4_40MS_OPEN = False`
`DRAFTER_SHOOTOUT_OPEN = False`
`NEXT_ROUTE = FULL_VERIFIER_CORRECT_BUT_TOO_SLOW_PROFILE_MOE_ATTENTION_HEAD`

`S100_SINGLE_ACHIEVED = False`

Phase20B is perfect-draft target-only timing. Drafter generation, acceptance loss and rejection recovery are not included.

## H=4 route-union census

```json
{
  "layers": 23,
  "median_unique_experts": 16.0,
  "min_unique_experts": 13,
  "max_unique_experts": 21,
  "median_repeat_rate": 0.33333333333333337,
  "total_cache_hits": 439,
  "total_cache_misses": 113,
  "total_up_bytes_loaded": 317108736,
  "total_down_sparse_bytes_loaded": 173283264
}
```
