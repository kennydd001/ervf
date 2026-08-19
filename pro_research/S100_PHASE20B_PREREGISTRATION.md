# Phase 20B preregistration

## Preflight
Correct Lightning identity/snapshot, Phase20S full-verifier gate true,
fp8_kv=False, H=4.

## MoE H4 validation
First/middle/last MoE layers on real normalized activations.
Grouped candidate versus 4 production calls:
- same route ids;
- route weights max abs <= 2e-6;
- output NRMSE <= 5e-4;
- finite.
All three must pass.

## Full verifier correctness
Canonical exact sequential trace generated with fp8_kv=False. Candidate inputs
are canonical future tokens. Initial prefix logit predicts draft1; candidate
block row logits predict drafts2-4 and next-cycle seed. Every id must match the
canonical trace. Nonfinite state/logits fail. Repeated C arms must commit the
same ids.

## Timing
Contexts 128, 1024, 4096. 12 advancing H4 blocks per arm. Fresh-process order:
B1 -> C1 -> C2 -> B2. Wall-clock begins after stream synchronize and ends after
argmax ids are on host.

## Gates
TARGET_H4_40MS_OPEN: correctness green and median candidate H4 <=40ms at every
context.
DRAFTER_SHOOTOUT_OPEN: correctness green and <=32ms at every context.
S100_SINGLE_ACHIEVED = False always in Phase20B.
