# S100 Phase 13I — decision-directed margin screen

This test records real validation-token logit margins and applies controlled
synthetic perturbations at 1%, 5%, and 10% of the logit standard deviation.
It compares top-1 stability for high-margin and low-margin tokens.

The result tests whether a margin gate is a plausible signal. It does not yet
connect the gate to an approximate layer implementation, so it cannot claim
end-to-end speed or quality and remains non-promoting.

Across 320 validation tokens, the median split kept 50% of tokens on the
candidate fast side. Under 10%-of-logit-standard-deviation noise, high-margin
tokens were 100% stable versus 95.0% for the low-margin half. At the stricter
90th-percentile margin gate, 10% of tokens were fast and remained 100% stable
under the same perturbation. This supports margin as a useful fallback signal,
but not yet as evidence that an approximate layer path is safe.
