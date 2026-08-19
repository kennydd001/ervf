# S100 Phase 14K2 — real witness handoff

Phase 13K used synthetic score noise and found K=16 retained the true top-1.
Phase 14K2 replaces that synthetic approximation with the real native-BF16
candidate runtime.

The runtime is teacher-forced on the same frozen target ids, so candidate and
parent see the same token sequence. Their internal states may differ.

Therefore:
- shortlist inclusion is meaningful evidence;
- exact parent re-ranking of the shortlist is a diagnostic witness;
- it is NOT a cheap correctness certificate for upstream approximation.

A future cheap witness is only valid if the approximation is confined to the
scored operation (for example lm_head only), or if a stronger error bound is
proved.
