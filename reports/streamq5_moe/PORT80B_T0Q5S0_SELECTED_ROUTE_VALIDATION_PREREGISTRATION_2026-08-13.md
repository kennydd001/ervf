# PORT80B-T0Q5-S0 — selected-natural-route real-weight Q5 validation

Date: 2026-08-13  
State: design preregistration; no runner/model/shard access until independent source audit  
Purpose: minimal falsifiable validation after stopping the over-complex R4/R5 transaction lineage.

## Exact question and claim boundary

Using the already independently verified D2-R3 whole-sequence tensors and natural layer-0 routes, does exact in-memory Q5 quantize/dequantize of the official selected routed-expert weights plus shared preserve layer-0 expert-plane numerics under the same full-16 CPU graph shape and operation order?

This is a **validation-only** experiment on previously observed inputs/routes. It cannot produce a held-out pass, breakthrough claim, full-model claim, GPU/transport claim, performance result, or evidence for unselected experts. A positive result means only that the exact codec and graph are numerically viable on these four natural D2-R3 validation prompts. The next gate would require fresh disjoint held-out prompts.

## Immutable existing evidence

- D2-R3 raw: `reports/runs/streamq5_moe/port80b_t0r12d2r3_cloned_serialization/t0r12d2_raw.safetensors`, 171,696,126 bytes, SHA-256 `f773853573129b3d560654c9faa62c2f5304a1151208f299c0ed8c103d5385cd`.
- D2-R3 result: SHA-256 `694b45004c9dea6827e201c80198d7f63a8fa7b90deea97198879d17162d2acb`.
- Independent artifact audit: SHA-256 `a048450b10c9ab2a06fa00629eb5089bb67333c36879da814afcaafac4538c33`, all 15 checks true.
- Independent interpretation: SHA-256 `be603f4edc648939aa86b2fcec16df802f4e778c6ab14256aecdc48f347da7f0`.
- Official Qwen3-Coder-Next revision `a19358a7659bd1f564300250ee189120c49a562f`; shard 1 exactly 3,999,619,288 bytes, SHA-256 `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.

D2-R3 is diagnostic, not a quality pass. Its whole-sequence captures are nevertheless independently schema-, hash-, finiteness-, route- and metric-verified immutable validation inputs.

## Frozen rows and selected union

Use only `p0_whole` through `p3_whole`, all 16 tokens, with D2-R3 captured `mlp_input`, official BF16 route weights and int64 route IDs. Positions 0–7 are graph calibration; positions 8–15 are the 32 scored validation rows. Build the selected routed-expert union deterministically by ascending unique IDs from all `4×16×10` captured routes. Report its exact size and IDs; never add an unselected routed expert for quality scoring. Shared expert is always included.

No route, prompt, row or threshold changes after reading outputs. The frozen D2-R3 prompt set is explicitly not held out.

## Source graph-control arm

Read only the selected expert matrices and shared matrices from official shard 1. For each prompt, execute the full 16-token source-BF16 graph:

1. `expert_mask = one_hot(ids, 512).permute(2,1,0)`;
2. `expert_hit = greater(sum(mask,(-1,-2)),0).nonzero()` (ascending expert ID);
3. for each hit: `top_k_pos, token_idx = torch.where(mask[expert])` exactly;
4. gather `mlp_input[token_idx]`;
5. concatenate official gate then up matrices into `[1024,2048]`, one BF16 `F.linear`, `.chunk(2,-1)` gate then up;
6. `F.silu(gate) * up`, BF16 down `F.linear`, multiply by captured BF16 route weight;
7. `index_add_` into BF16 zero output;
8. shared over all 16 tokens with official gate/up/down; shared gated is **captured `sigmoid_gate * shared_raw`** in that operand order;
9. complete MLP is routed + gated shared.

Retain actual source graph routed/shared raw/shared gated/complete arrays. Strict bitwise equality is required against D2-R3 `p*_whole_experts`, `p*_whole_shared`, `sigmoid(p*_whole_shared_gate)` times shared in gate-first order, and `p*_whole_layer_output - pre-MLP residual` only if an exact captured complete tensor exists. Because D2-R3 raw schema is authoritative, source audit must identify exact available keys before implementation. If actual complete MLP or actual residual is absent, S0 narrows to routed/shared/gated validation and must not synthesize a layer reconstruction. No subtraction-based quality oracle is allowed.

Any source arm mismatch is `graph_control_negative`; Q5 metrics cannot pass validation.

## Exact in-memory codec

For each selected matrix independently, from source BF16 bytes:

- contiguous row-major groups of 128;
- nonzero scale `max(abs(group))/15`; zero group scale exact BF16 1.0 and q=0;
- round-to-nearest ties-to-even, clip q to [-15,15]; stored field=q+15 in 0..30, 31 forbidden;
- eight fields in little-order 40-bit/5-byte words; scales little-endian BF16;
- decoded value is BF16 cast of `(field-15) * widen(scale_bf16)`.

No persistent bank, record headers or transaction protocol. Each selected source matrix is quantized to in-RAM immutable codes+scales, decoded once to a BF16 matrix for graph execution, and freed after the prompt or union plan permits. Retain only compact per-matrix evidence: official key/dtype/shape/source SHA, codes/scales SHA, decoded-weight SHA, group counts, zero-group count, min/max q and maximum source-vs-decoded weight error/rel-L2. An independent verifier reopens each selected official source tensor and independently reconstructs byte-identical codes/scales and decoded digest.

## Q5 graph and unchanged R3 thresholds

The Q5 graph has exactly the source graph's full-16 batches, selected-route dispatch, gather order, fused gate-up, SiLU, casts, route weighting, increasing-ID accumulation and gate-first shared order. Retain complete raw routed/shared raw/shared gated/complete tensors when source schema permits.

For each of 32 scored rows compute the exact R3 metric definitions: flattened BF16 to FP64; fixed row-major loop accumulation; max-abs; rel-L2 with explicit zero-reference behavior; cosine with explicit zero-norm behavior; unequal BF16-word count; monotone signed BF16 ULP distance. No BLAS reduction for metrics.

Thresholds are unchanged from T0Q5-R3:

- routed rel-L2 <= 0.08;
- shared raw rel-L2 <= 0.08;
- shared gated rel-L2 <= 0.08;
- complete MLP rel-L2 <= 0.08 when exact complete reference exists;
- candidate layer thresholds are **not evaluated** unless an exact captured pre-MLP residual and complete official MLP are both present;
- if eligible: layer rel-L2 <= 0.02, cosine >=0.999, max-abs <=0.125 and 32-row mean layer rel-L2 <=0.01.

S0 status remains `validation_positive` or a named negative; never `pass`.

## Deterministic controls

Run on prompt 0 positions 8 and 15 only, using the normal gated complete-MPL baseline if eligible, otherwise the narrowest available routed/shared tensor. Retain unsafe raw arrays, not just counts.

1. Wrong selected expert: replace rank-0 with the smallest selected-union expert absent from that row; if none, blocked.
2. Projection swap: use rank-0 up bytes where gate is expected; source key/shape/projection identity checker must reject before graph.
3. Code mutation: shared-down, choose row-major first q!=0 with corresponding nonzero captured shared activation, step q toward zero without updating stored codes digest; digest check rejects.

For each control, independent verifier reconstructs selection/mutation from raw/source, verifies real rejection and unsafe raw bytes, and requires >=1 changed BF16 word versus exact baseline. Controls are integrity evidence only.

## Resources, files and outcomes

CPU only; CUDA must remain uninitialized. Start available RAM >=16 GiB; Windows peak working set <=12 GiB; minimum available RAM >=2 GiB. No persistent weights/codes bank. Retained output target <=256 MiB and hard maximum 512 MiB. One create-new raw safetensors, one create-new result JSON and one create-new failure JSON; atomic temp+fsync+rename, no overwrite. A simple single-bundle writer is sufficient; no reusable transaction framework.

Frozen outcomes:

- `selected_route_validation_positive` (not a scientific pass),
- `selected_route_q5_quality_negative`,
- `graph_control_negative`,
- `codec_or_control_negative`,
- `blocked`,
- `invalid`.

Only `selected_route_validation_positive` opens a separately preregistered fresh-input held-out S1.
