# S100 Phase 38 preregistration — official NVIDIA DFlash

Frozen before the first Phase38 GPU run.

## Question

Does NVIDIA's official
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash` checkpoint provide
enough real greedy acceptance, at low enough standalone draft cost, to reopen
speculative decoding against the adopted exact Phase31 H4 and screened Phase32
H8 verifiers on this 8 GiB RTX system?

DSpark is outside this phase. No trained, synthetic or proxy drafter may be
substituted for the official DFlash checkpoint.

## Frozen identities

- Target snapshot:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, revision directory
  `e8f3c7c4de75ad84fe1bcef95d38eca76214480b`.
- Drafter snapshot:
  `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash`, revision
  directory `7fc1f1ff4b82b917efbd0710df0872c2bb89caa5`.
- Trace: measured Phase20B canonical greedy trace, unchanged.
- DFlash target hidden-state layer IDs, zero based: `[1, 5, 19, 29, 41, 51]`.
- DFlash block: eight input positions: one real anchor token followed by seven
  copies of mask token ID 990. `sample_from_anchor=false`, so only output
  positions 1 through 7 are draft tokens.

Every artifact records file hashes and source revision. A checkpoint or trace
identity mismatch is a technical failure, not a result.

## Stage A — exact target-state capture

Replay the first 512 token inputs of the canonical trace through the exact V6
single-token target stack. At every input token, capture the FP32 residual after
each of the six frozen target layers. Teacher-force the registered prompt. From
the final prompt token onward, the target argmax must equal the next canonical
generated token exactly; first continuation divergence aborts the capture.

Store `[512, 6, 2688]` little-endian FP32 target residuals and the corresponding
513 little-endian INT32 trace tokens. DFlash consumes a BF16-rounded copy of the
captured residuals, matching its W4A16 activation contract. The FP32 source is
retained for audit and alternative numerical validation.

The capture is correctness-only and untimed. It makes no throughput claim.

Technical erratum frozen after the first capture attempt, before any DFlash
forward or acceptance result: the canonical trace begins with ten externally
supplied prompt tokens. The first implementation incorrectly required token 0
to predict prompt token 1 and stopped at that comparison. The corrected rule
above checks the greedy continuation beginning with the final prompt token. No
captured array from the failed attempt was written or used.

## Stage B — official DFlash reference forward

Implement the checkpoint's six non-causal Qwen3 attention/MLP layers directly
from its published config and weights:

1. concatenate the six target residuals and apply the checkpoint `fc` projection
   plus `hidden_norm`;
2. use each DFlash layer's K/V projections of projected committed target states
   as the causal prefix cache;
3. embed `[anchor, 990, 990, 990, 990, 990, 990, 990]` with the DFlash
   checkpoint's own embedding rows (`has_embed_tokens=true`);
4. let all eight block queries attend to the full committed prefix and the full
   eight-position draft block (non-causal within the block);
5. apply the target's shared LM head to block positions 1 through 7 and take
   greedy argmax.

NVFP4 weights are decoded with group size 16, low-nibble-first E2M1 elements,
E4M3 scales and the checkpoint's scalar `weight_scale_2`. The first reference
arm dequantizes to BF16 so acceptance can be established independently of a
later optimized native W4A16 kernel.

The embedding residency audit may share rows only when their BF16 bytes equal
the target embedding. The DFlash mask row 990 is checked explicitly and must be
retained if it differs. The target LM head remains shared and packed NVFP4.

Evaluation begins at canonical position 128. At each round, acceptance is the
longest prefix (0 through 7) matching the canonical next tokens. The committed
length is `1 + accepted_drafts`; the next anchor advances by that amount. Rounds
continue while the entire seven-token comparison fits inside the 512-position
capture.

Report accepted-draft and committed-length histograms, mean, median, p10/p90,
full-block rate, zero-draft rate, and exact round anchors.

## Stage C — timing and economic gates

Time after warm-up, with synchronization around each measured block:

- target-hidden projection and per-layer prefix K/V append cost per committed
  target row;
- seven-draft block forward excluding target verification;
- shared target LM-head portion;
- peak allocated and reserved GPU memory.

The BF16-dequant reference time is a correctness baseline and upper bound, not
an optimized DFlash throughput claim. Native W4A16 integration is permitted
only if measured acceptance leaves a positive zero-cost economic ceiling.

For each measured acceptance interpretation, report end-to-end ceilings against
the measured Phase31 H4 and Phase32 H8 verifier medians. A path opens only when:

- target replay and DFlash forward are finite and deterministic;
- the measured committed length gives a zero-drafter-cost ceiling above the
  adopted 62.961 target-only tok/s baseline; and
- adding measured draft cost still leaves at least 5% projected margin before
  any resident integration work.

Failure of the economic gate closes integration but does not invalidate the
measured acceptance result.

## Post-result contract diagnostic addendum

Added after the first official-checkpoint result reported 1.0106 accepted
drafts per round, before running the diagnostic below. This is explicitly a
post-hoc contract diagnostic and cannot replace the frozen Stage A/B result.

The published DFlash checkpoint was distilled from a BF16 W4A16 target runtime,
whereas the adopted LightningStream target keeps residuals and most operator
outputs in FP32. To test whether that activation-domain difference explains the
acceptance gap, run one separately named `bf16_residual_proxy` target arm:

- keep the registered prompt, model, target layers and greedy decoding policy;
- round the target's normalized layer input, branch output and post-add residual
  to BF16 (round-to-nearest-even, then widen back to FP32) at every target layer;
- round the final normalized head input the same way;
- generate this proxy target's own continuation after the prompt and capture its
  six post-layer residuals;
- evaluate the unchanged official DFlash checkpoint against that self-consistent
  continuation with the unchanged Stage B forward.

This arm is only an activation-precision sensitivity proxy: internal Mamba/MoE
temporaries and recurrent states still use the custom runtime's native storage,
so it is not labeled an official vLLM acceptance reproduction. Regardless of
its acceptance, also report the acceptance-independent upper bound at perfect
seven-of-seven drafting. If even that bound cannot clear the frozen 5% gate,
native DFlash integration remains closed.
