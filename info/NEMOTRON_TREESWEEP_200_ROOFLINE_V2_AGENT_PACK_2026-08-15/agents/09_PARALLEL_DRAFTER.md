# Agent 09 — Dependent Parallel Drafter

## Mission

Train the cheapest drafter capable of breaking the linear-chain acceptance ceiling.

## Candidate families

- Hydra-style sequentially dependent heads;
- FastMTP position-shared recursive head;
- EAGLE-3 direct-token head with multi-layer features;
- FastEagle one-pass layer cascade;
- ReDrafter recurrent beam/tree drafter;
- Parallel Token Prediction (PTP).

## Experimental discipline

Use the same train/validation/test prompts and target verifier for all candidates. Match parameter count, training tokens and wall-clock where possible. Report:

- draft ms;
- node count;
- accepted depth/output tokens;
- tree expert union;
- end-to-end output/ms;
- training cost.

## Gate

The candidate must meet the registered coverage gate and integrated 200 tok/s gate. Higher acceptance with excessive draft cost is a failure.
