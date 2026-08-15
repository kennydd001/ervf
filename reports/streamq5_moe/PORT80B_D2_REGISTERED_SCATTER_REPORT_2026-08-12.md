# PORT80B-D2 — registered-bank scatter report

Verdict: **registered_scatter_negative**. Mechanism pass: **False**; full-bank pass: **False**.

| prefix | registered GiB | status | p50 ms | p95 ms | GB/s at p95 | mismatches |
|---:|---:|---|---:|---:|---:|---:|
| 60.0% | 27.826 | timed | 51.88979148864746 | 52.37285137176514 | 18.582329861930518 | 0 |
| 69.9% | 32.448 | timed | 51.781856536865234 | 52.04604606628418 | 18.69901123248731 | 0 |
| 80.1% | 37.161 | timed | 51.76128005981445 | 52.06400489807129 | 18.692561240828645 | 0 |
| 100.0% | 46.406 | registration_or_timing_failed | — | — | — | — |

The immutable P0 negative result is unchanged. See the JSON and independent verifier for raw samples and gates.
