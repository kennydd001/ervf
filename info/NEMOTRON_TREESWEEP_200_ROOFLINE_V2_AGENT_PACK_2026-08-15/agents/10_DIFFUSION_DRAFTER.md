# Agent 10 — Diffusion/Block Drafter

## Mission

Test whether a parallel diffusion drafter can produce a deeper, more diverse tree than autoregressive/MTP drafting at acceptable cost.

## Sources to reproduce/compare

- DEER diffusion drafter + AR verifier;
- D2SD dual diffusion prefix recovery;
- PRESTO prefix-aligned tree scoring;
- Nemotron-TwoTower as architecture/teacher evidence;
- Nemotron-Labs-Diffusion tri-mode design.

## Phases

1. target-informed diffusion-tree oracle;
2. tiny/student diffusion drafter feasibility;
3. block draft cost;
4. prefix-tree construction;
5. exact Lightning target verification.

## Resource gate

Do not download/train a full 60B TwoTower merely to test the idea. Start with published weights only if storage/compute are approved, or distill a much smaller denoiser from captured Lightning trajectories.

## Gate

Draft plus verifier must meet 200 tok/s. Paper-reported acceptance lengths are not evidence for this target.
