# TierFlow-F0 — bounded route-edit trace feasibility

## Verdict

**Traffic-feasible on the aggregate held-out P4D trace; quality and runtime are
untested.** With a top-8 persistent route and at most one replacement per
token/layer (`r=1`), the held-out trace oracle reduced steady-state critical
expert traffic by **4.158x** and reduced worst-case new expert loads from
**8 to 1 (8x)**. Those are the two requested trace gates.

This is not evidence that a trained TierFlow model retains language quality.
The oracle substituted **32.03%** of observed router-output slots on test, and
only **8.99%** of token/layer route sets matched the original top-8 exactly.
The original <=1% quality-regression condition therefore remains a large,
unmeasured training question.

## Experiment boundary

- CPU-only analysis of the frozen real Qwen30 P4D top-8 route captures;
- 48 layers, five domains, 1,024 tokens/domain, 128 experts;
- unchanged P4D split: validation 512:768, test 768:1024;
- one replacement is one route edit;
- critical traffic counts only experts newly needed versus the preceding
  token's route state;
- each new expert is exactly 3,035,136 bytes;
- warm-start state is token 511 for validation and token 767 for test;
- the oracle is non-causal inside each partition and is therefore an optimistic
  feasibility bound, not a deployable controller.

The oracle maximizes current-token overlap conditional on its preceding state.
When several admissions/evictions are equivalent at the current token, it uses
next use and remaining frequency inside the same partition. It never crosses
the validation/test boundary.

## Validation selection

| Edit budget | Critical-byte reduction | Worst-case new-load reduction | Mean top-8 overlap | Substitution | Exact route-set matches | Traffic gates |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 4.148x | 8.000x | 68.86% | 31.14% | 9.34% | pass |
| 2 | 2.166x | 4.000x | 79.91% | 20.09% | 29.90% | fail |
| 4 | 1.274x | 2.000x | 92.69% | 7.31% | 67.24% | fail |

Only `r=1` met both preregistered aggregate traffic gates, so exactly that
candidate was opened on test.

## Held-out test result (`r=1`)

Across 61,440 token/layer transitions:

| Metric | Observed P4D routes | TierFlow oracle |
|---|---:|---:|
| Mean new experts / token / layer | 4.1169 | 0.9902 |
| p95 new experts | 7 | 1 |
| p99 new experts | 8 | 1 |
| Worst case | 8 | 1 |
| Critical expert bytes | 767,710,334,976 | 184,648,568,832 |
| Critical expert GiB | 714.986 | 171.967 |

Aggregate reduction was **4.1577x**. Mean route-set overlap was **67.97%**
(5.438 of 8 experts); substitution was **32.03%**. Cold-start traffic was
reported separately as 5.427 GiB and did not enter the steady-state gates.

### Domain robustness

| Domain | Critical-byte reduction | Worst-case reduction | Overlap | Substitution | Exact sets |
|---|---:|---:|---:|---:|---:|
| general | 3.639x | 8x | 74.56% | 25.44% | 14.47% |
| code | 4.072x | 8x | 69.18% | 30.82% | 9.45% |
| math | 5.081x | 8x | 58.74% | 41.26% | 2.77% |
| multilingual | 3.750x | 8x | 71.71% | 28.29% | 11.36% |
| instruction | 4.234x | 8x | 65.66% | 34.34% | 6.90% |

The preregistered gate was aggregate, so the formal traffic verdict is pass.
However, a stronger every-domain 4x gate would fail on general and
multilingual. Math supplies the largest traffic gain but also requires the
largest behavioral change. This makes the aggregate pass promising but not
robust enough for an industrial claim.

## What this establishes

The real route traces do not make the TierFlow traffic targets arithmetically
impossible. A one-edit top-8 state can simultaneously meet 4x aggregate
critical-byte and 8x worst-case-load targets on held-out tokens. Budgets of two
or four edits preserve more of the original router output, but necessarily miss
the traffic gates on these traces.

## What remains unproven

- LM quality regression <=1%; no model was trained or evaluated;
- whether a causal controller can approach this clairvoyant oracle;
- measured p95 >=2x and absence of p99 collapse;
- progressive `W2 + D3 + D4 + D5` page quality/traffic interaction;
- replication under a second memory hierarchy;
- end-to-end throughput, VRAM, host commit, and output correctness.

Accordingly, TierFlow-R1 is now justified as a **small controlled training
experiment**, not as a breakthrough claim. The next defensible step is the
preregistered 100M–300M route-only comparison. Progressive pages should remain
closed until route-only meets both quality and measured-latency gates.

## Artifacts

- preregistration: `reports/streamq5_moe/TIERFLOW_F0_PREREGISTRATION_2026-08-12.md`
- validation: `reports/streamq5_moe/tierflow_f0_validation.json`
- held-out result: `reports/streamq5_moe/tierflow_f0_result.json`
- runner: `scripts/streamq5_moe/run_tierflow_f0_trace_feasibility.py`

