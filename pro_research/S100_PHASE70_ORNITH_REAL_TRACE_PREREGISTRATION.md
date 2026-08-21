# S100 Phase70 — Ornith real route and final-activation trace preregistration

Date: 2026-08-21

## Question

Do real Pottokao Ornith-1.5 target activations preserve the two assumptions that
still separate the Phase69 component floor from a custom end-to-end result:

1. a 52-slot-per-layer expert cache has a sufficiently small miss set over H4;
2. the Phase64 native shortlist contains the exact top-32 logits when driven by
   real final normalized activations rather than deterministic synthetic inputs?

## Frozen capture path

- Model: `pottokao/Ornith-1.5-35B-A3B-abliterated-NVFP4-DFlash-GGUF`, target
  GGUF from snapshot `09cec755dab944bddc60bc068ae01bd75271dae8`.
- Runtime: local unmodified llama.cpp commit `9558fa4`, CPU callback build. CPU
  is used only to expose graph tensors; the S100 timing claims remain the
  separately measured CUDA results from Phases58–69.
- Callback tensors: every `ffn_moe_topk-*`, every
  `ffn_moe_weights_norm-*`, and `result_norm`.
- H4 probe text: `The quick brown fox`, already tokenizer-checked as exactly
  four tokens `[760, 3841, 13477, 37550]`.
- The trace runner marks every input token as an output so `result_norm`
  contains all four final activations.
- A longer fixed text trace may be added after the H4 capture succeeds, but it
  must use the same model, callback tensor set and cache policy.

## Frozen analyses

### Route/cache replay

For each of 40 layers, maintain 52 expert slots. Replay tokens in order. A hit
occurs when a selected expert is resident before that token. On a miss, insert
the expert and evict the resident expert whose next use is furthest in the
future (Belady oracle). This is an explicit optimistic upper bound; LRU is
reported separately as an implementable comparator. Within one token, all
eight selected experts are queried against the pre-token cache state and then
inserted.

Report selected assignments, unique experts, hit/miss counts and miss sets per
token/layer for cold, Belady-52 and LRU-52 replay. No synthetic router logits
may replace missing trace tensors.

### Final-head replay

Feed each captured 2048-wide `result_norm` row to the exact Phase64 reference
and native-shortlist paths using the same real output matrix and shortlist
width. Report exact selected-ID agreement, top-32 recall, score NRMSE and the
measured Phase64 latency unchanged. A layout/type mismatch is a technical
failure, not permission to reshape or regenerate activations.

## Gates

- Exactly 40 layer-indexed top-8 route tensors are captured for the four-token
  probe, each with shape `[8, 4]` and IDs in `[0, 255]`.
- When present, normalized route tensors are finite and each token sums to
  `1 +/- 5e-4` across its eight selected experts.
- `result_norm` has shape `[2048, 4]`, is finite, and is reproducible on a
  second fresh-context capture within maximum absolute error `1e-5`.
- Phase64 real-activation top-32 shortlist recall is 1.0 for all four rows and
  exact selected IDs match the full reference.
- Trace parsing and both cache policies are covered by deterministic unit
  fixtures.

If llama.cpp cannot expose the frozen tensors, record `technical_failure` with
the exact tensor inventory. A failed cache or shortlist gate is
`measured_fail`; it is not repaired by changing slot count, shortlist width or
prompt after inspection.
