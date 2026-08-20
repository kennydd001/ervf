# Phase24 preregistration

## Identity
Require:
- correct Lightning snapshot;
- Phase23R GPU_GROUPED_MOE_ADOPTED=true;
- Phase23 correctness green;
- Phase22 graph correctness green.

## 24A diagnostic profile
Context1024, two canonical H4 blocks, eager synchronized diagnosis only.

Per MoE layer report:
- router
- cache/group + up miss staging
- shared expert
- routed up
- mask/union scan
- down gather
- down arithmetic/reduce/accumulate
- scale bytes and code bytes
- resident plane VRAM cost
- score = scale bytes avoided / resident plane bytes

Profiler timing is never a throughput result.

## 24B exact component screens
Real H4 activations from context1024.

Attention BF16 M4:
- every q/k/v/o matrix;
- bit-exact output versus four production gemv_bf16 calls;
- aggregate speedup >=1.05.

Router FP32 M4:
- all 23 routers;
- bit-exact logits and identical route ids;
- aggregate speedup >=1.05.

Shared NVFP4 M4:
- all 23 shared experts, up + down;
- bit-exact shared output;
- aggregate speedup >=1.05.

Only green components enter synthesis.

## 24C scale-resident ladder
K = 0/4/8/12/16/23 ranked layers.

- Actual cp.zeros allocation after all H4 graph buffers exist.
- No mem_info pre-rejection.
- CUDA OOM => infeasible_vram, not scientific failure.
- Selected layer scale planes are populated from the existing V6 device-cache
  mapping after every canonical prefill.
- Misses update up weights and the resident scale plane together.
- Gather copies code columns only.
- Arithmetic reads the exact same scale byte from the resident plane.

Every arm:
- fresh process;
- context1024;
- 4 warmup + 8 measured H4 blocks;
- exact tokens.

Select fastest correctness-green feasible arm; ties within 1% prefer less VRAM.

## 24D direct state gate
Selected synthesis versus adopted Phase23 graph from identical context1024:
- ids exact;
- SSM <=5e-5;
- conv <=1e-5;
- KV <=5e-6;
- logits <=5e-4;
- deterministic replay.

## 24E thermal adoption
If selected screen arm is not baseline:
four balanced fresh-process rounds after primer, same Phase23R schedule.
Adopt only if:
- median round gain >=5%;
- median matched-block gain >=5%;
- >=3/4 rounds positive;
- robust CV <=5% each arm;
- exact tokens.

## 24F promoted contexts
128/1024/4096, 12 H4 blocks.

H4_BEATS_V18_PARENT:
context1024 H4 < 4 * 19.5729 = 78.2916 ms.

PHASE24_TARGET_40MS_OPEN:
all contexts <=40 ms/H4.

## 24G generalization and H8 census
Eight prompt classes:
factual, code, math/reasoning, conversational, technical, Dutch, translation,
structured JSON.

Generate 8 exact tokens per prompt and record all 23 x 8 x 6 routes.

Report H4 and H8:
- unique experts;
- M histogram;
- stream reduction;
- prompt worst/median/best.

This is route generalization, not final latency generalization.

PHASE25_H8_BUILD_OPEN when:
- Phase24 H4 target40 is false;
- all target correctness is green;
- median H8 expert-stream count is <=1.85 times two-H4 average streams
  (>=7.5% additional cross-block reuse);
- no prompt has invalid route multiplicity.

S100_SINGLE_ACHIEVED=false.
