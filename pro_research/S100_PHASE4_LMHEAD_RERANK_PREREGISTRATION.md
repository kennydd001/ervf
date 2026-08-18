
# Phase 4 follow-up — exact-reranked native FP4 lm_head

Shortlist sizes are frozen at 2, 4, 8, 16 and 32.

For each target activation:

1. quantize A with the CEIL block-scale convention;
2. native FP4 full-vocabulary logits produce top-K ids;
3. gather those original checkpoint rows;
4. evaluate exact current weight-only logits with the original FP32 activation;
5. select the exact best shortlisted token.

Timing includes A quantization, full-vocabulary candidate logits, top-K,
checkpoint-row gathering and exact rerank.

An arm opens integration only with 100% exact-token recall and 100% reranked
top-1 agreement on at least 50,000 held-out activations. Anything less is an
approximate model and must use the model-quality protocol.
