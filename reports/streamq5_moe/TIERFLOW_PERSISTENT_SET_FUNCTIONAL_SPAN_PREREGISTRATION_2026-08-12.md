# TierFlow persistent-set functional-span oracle — preregistration

Date: 2026-08-12

## Question and claim boundary

TierFlow-F0 obtained its traffic target with only 67.97% mean top-8 overlap.
This experiment tests whether that route-ID substitution exaggerates the
functional change: can the natural routed MoE output be reconstructed from the
eight expert outputs in the frozen `r=1` persistent set by nonnegative weights
whose sum is one?

This is a non-causal feasibility oracle. It uses the natural routed output to
fit coefficients separately for every token and therefore is not a controller,
training method, runtime or deployable quality result. It changes no weights
and performs no training or download.

## Locked local artifacts

- model: `models/qwen3-30b-a3b-base`, all 16 local safetensor shards addressed
  by `model.safetensors.index.json`;
- P4D input IDs:
  `reports/runs/streamq5_moe/p4d_fresh_route_input_ids.safetensors`, expected
  SHA-256 `32838e94887f8572445159925e815f5353f55a20a954f9adc2f8cef48427af08`;
- P4D capture manifest, expected SHA-256
  `7ebfcf30eceed76e2615e11702ca162eb43bf4236d6099cc307ec5cb4bcd74bb`;
- all 48 P4D route tensors with hashes from that manifest;
- model semantics: Q5 group-128 experts and INT8 group-128 trunk, with codes
  selected against FP32 max-absolute scale and the stored scale rounded to
  BF16 before dequantization, exactly as in P4D route capture;
- Transformers `4.51.3`, local-files-only, CUDA required.

Teacher hidden states are reconstructed layer by layer from the locked input
IDs and local weights. Their validity requires every natural top-8 route to
equal the stored P4D route capture exactly.

## Partitions and layers

The existing P4D partitions remain fixed:

- calibration `[0,512)` is unused by coefficient fitting;
- validation `[512,768)`;
- test `[768,1024)`.

Validation and test are strictly disjoint but not globally fresh: their route
statistics were previously opened by TierFlow-F0. This is a new functional
metric on reused inputs, not a fresh-dataset confirmation.

The fixed sentinel layers are `0`, `24`, and `47`. For next-token metrics,
positions `[start,end-1)` predict labels `[start+1,end)`, giving exactly 255
labels per domain without crossing a partition boundary.

## Frozen persistent sets

For every domain and layer, reproduce the TierFlow-F0 `r=1` state on the
active partition, warm-started from the natural route at `start-1`. Preserve
requested resident experts and admit at most one missing requested expert.
Admission/eviction ties use the same within-partition non-causal next-use,
remaining-frequency and expert-ID order as F0. The full 48-layer aggregate
must reproduce at least `4x` critical-byte reduction and `8x` worst-case
new-load reduction before model work is adjudicated.

## Functional-span oracle

At a sentinel layer, let `f_e(h)` be the unweighted Q5 expert output for the
exact post-attention normalized teacher state, `S_t` the persistent eight-set,
and `y_t` the official natural Q5 routed output. Solve independently per token:

`min ||sum_(e in S_t) alpha_e f_e(h) - y_t||_2^2`

subject to `alpha_e >= 0` and `sum alpha_e = 1`.

The solver is exhaustive active-support enumeration over all 255 nonempty
subsets. Each support uses the float64 equality-constrained KKT system;
infeasible negative solutions are rejected. Ties use the lower eight-bit
support mask. Coefficients, the weighted sum and local errors are retained.
The fitted routed vector is rounded once to BF16 and substituted only at the
sentinel layer. All downstream layers then run normally on the changed state;
no later route or state is forced to teacher values.

Controls:

1. coefficient nonnegativity and unit sum;
2. normalized KKT violation `<= 1e-7`;
3. official natural route IDs exactly match all P4D captures;
4. a manually decomposed natural sentinel layer is bit-identical to the
   official layer forward;
5. finite expert outputs, coefficients, hidden states and logits.

## Metrics

For each sentinel and domain report:

- tokenwise routed-output relative L2 error;
- coefficient support size and natural/persistent route overlap;
- full-vocabulary natural-to-candidate KL after the untouched downstream tail;
- next-token cross-entropy and relative CE versus the natural Q5/INT8 path;
- natural/candidate top-1 agreement.

The original BF16 final norm and LM head are shared by natural and candidate
arms. The experiment measures only the incremental persistent-set effect over
the P4D Q5/INT8 hidden path.

## Hard validation and test gates

Validation passes only when all controls pass and, separately for every
sentinel layer:

1. aggregate mean routed-output relative L2 `<= 0.05`;
2. aggregate p95 routed-output relative L2 `<= 0.10`;
3. aggregate mean full-vocabulary KL `<= 0.001`;
4. relative next-token CE regression `<= 1%`;
5. top-1 agreement `>= 99%`;
6. every-domain relative CE regression `<= 2%`.

The full-partition traffic gates (`>=4x`, `>=8x`) and every numerical/control
gate must also pass. No sentinel may compensate for another.

If validation fails, test stays closed. If validation passes, exactly the same
algorithm and gates are applied once to test; no coefficient rule, support
enumeration, layer, threshold or arithmetic may change. A test pass establishes
only a non-causal sentinel-layer functional-span ceiling and can at most
justify a separately preregistered full-48-layer or trained-controller study.

## Stop conditions

- Missing/corrupt model shard, input, route tensor, pinned software/CUDA, or
  irreproducible natural route: `blocked_artifact` or failed control, with no
  test opening.
- Validation gate failure: `verified_negative`, test closed.
- Validation pass/test failure: held-out negative.
- No training, download, central-registry edit or silent fallback is allowed.
