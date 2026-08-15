# Qwen GPTQ Bank — semantics erratum

Locked on 2026-08-11 before any full-bank quantized output was produced.

## Discovered historical name collision

The FLEQ helper constructs upstream `GPTQ` with the layer name `fleq_projection`. The pinned upstream `fasterquant` implementation enters its 2,000-epoch Gumbel optimization branch whenever `"q_proj" in self.name`. Because `fleq_projection` contains that exact substring, artifacts reported as `gptq_2bit` in the historical locked-16 set are actually the post-Gumbel return, not the pure GPTQ return computed immediately beforehand.

This was exposed by the preregistered small batched-equivalence test: group scales returned by the historical helper differed from the scales assigned by the GPTQ loop even for a one-expert batch. Inspection with the pinned class showed two identical pure-GPTQ `find_params` calls followed by different returned scales, precisely because execution continued into the name-gated Gumbel branch.

## Canonical rule from this point

- The full bank represents **pure GPTQ**, matching the stated algorithm and hyperparameters in the original preregistration.
- The official reference instantiates the same pinned upstream `GPTQ` and `Quantizer`, but uses the neutral name `expert_projection`, which contains none of `q_proj`, `k_proj`, or `in_proj_qkv`.
- The returned value is therefore the direct `Q, group_scales` result of the pinned sequential GPTQ loop; no Gumbel optimizer or learned logits run.
- The accelerated implementation must match that name-neutral pinned reference exactly in integer codes and BF16 scale bits.
- The historical locked-16 CORETAIL size projection is retained as historical evidence only. It cannot be used as the full-bank result; CORETAIL P0 must be rerun physically from all 6,144 new pure-GPTQ experts.

No route, calibration, memory, quality, or runtime gate is weakened. RTN substitution remains forbidden.
