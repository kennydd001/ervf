# S100 Phase50 — official Ornith NVFP4 parity confirmation

## Question

Do the real routed-expert bytes in the official
`ornith-ai/Ornith-1.5-35B-A3B-NVFP4` checkpoint satisfy the same complete
SwiGLU correctness and route-adaptive M2-through-M8 performance contract as the
Pottokao abliterated checkpoint?

## Frozen inputs

- Official revision: `0f0b1b59b879ccde1353e6ebd0fb10c204d4c544`.
- Official tensor: `model.language_model.layers.20.mlp.experts.0` from shard 2.
- Control: Phase49 Pottokao layer-20 expert-0 result.
- Measurement engine: the unchanged Phase49 exact-size kernel family and its
  independent byte-level decoder; only the official layer-root spelling is
  parameterized.

## Gates

1. The official Phase49 run passes every checkpoint, correctness,
   determinism and M2-through-M8 speed gate.
2. Every official candidate median is between 0.65x and 1.35x the corresponding
   Pottokao median. This is a broad shape-parity guard, not an equality claim.
3. The first beneficial official route multiplicity is 2.

## Claim boundary

Passing establishes routed-expert kernel portability between the two
checkpoints. It excludes the official vision tower, MTP head, whole target
decoder and DFlash acceptance.

