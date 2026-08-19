# S100 Phase 23R — Thermal Adjudication

Phase23 GPU-grouped MoE was already correctness-green. This repair run tests whether its measured transport savings survive a balanced thermal/order protocol.

| Round | Parent ms | Grouped ms | Gain |
|---:|---:|---:|---:|
| 1 | 84.306 | 82.187 | 2.51% |
| 2 | 84.495 | 78.331 | 7.30% |
| 3 | 92.703 | 81.269 | 12.33% |
| 4 | 84.616 | 80.100 | 5.34% |

- Median round gain: `0.06316303072633983`
- Median 64-block paired gain: `0.05544972434762163`
- Parent robust CV: `0.0027204007535860802`
- Grouped robust CV: `0.019171379455973664`
- GPU grouped adopted: `True`
- Target <=40 ms/H4: `False`
- Drafter shootout open: `False`
- Next route: `PROFILE_POST_GROUPED_GRAPH_AND_ATTACK_NEXT_DOMINANT_FAMILY`
- S100 single achieved: `False`
