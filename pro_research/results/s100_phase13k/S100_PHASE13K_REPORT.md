# S100 Phase 13K — exact top-K witness

This test uses real validation logits, perturbs a fast shortlist score, and
then exactly re-ranks only the shortlisted original rows. It measures true
top-1 inclusion and exact witness recovery over multiple shortlist sizes.

The perturbation is controlled and does not represent a finished compressed
`lm_head`; no GPU shortlist/rerank kernel or end-to-end latency claim is made.
Promotion remains closed.

Over 160 validation logits, even the harshest tested perturbation (10% of
logit standard deviation) retained the exact top-1 in a K=16 shortlist and the
exact witness recovered the same winner in every sample. This is a positive
signal for a witness design, but it uses synthetic score noise rather than a
real compressed lm_head score, so it still needs a causal kernel test.
