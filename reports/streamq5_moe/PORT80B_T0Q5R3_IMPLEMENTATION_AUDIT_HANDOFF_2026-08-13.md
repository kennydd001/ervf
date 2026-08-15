# T0Q5-R3 implementation-audit handoff

The runner and independent verifier are source candidates only. No prompt lock was generated, no preflight executed, and no tokenizer/model/shard payload/forward/bank action occurred. Consequently the current runner lock must intentionally bind `prompt_lock_sha256` to `__ABSENT_PENDING_IMPLEMENTATION_AUDIT__`, and executable `lockcheck` must return false. After a source GO, generate the canonical prompt lock exactly once, preserve this blocked implementation immutable, and create a new lock-bound revision.

Audit especially:

- actual `post_attention_layernorm` pre-hook residual identity;
- official `expert_mask = one_hot(ids).permute(2,1,0)`, ascending `expert_hit`, and `top_k_pos,token_idx=torch.where(expert_mask[expert])` order in both graph arms;
- source graph strict bitwise comparisons and BF16 residual-first add;
- independent verifier imports neither runner nor codec and independently rereads/requantizes all 1,539 records;
- transaction/failure create-new behavior and peak/reserve/disk gates;
- controls operate on exact baselines and verifier truly reconstructs their unsafe effects rather than trusting runner summaries.

Any gap requires a new immutable revision; do not execute this candidate.
