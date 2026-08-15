# Agent 08 — Native MTP Tree

## Mission

Extract maximum useful coverage from the existing Lightning MTP path before training a new drafter.

## Work

- identify whether heads are independent, recurrent or otherwise conditioned;
- capture top-k distribution at every draft depth;
- construct fixed and dynamic trees from head outputs;
- compare chain, beam, Sequoia, OPT-Tree, DySpec and cost-aware tree selection;
- use confidence/entropy to vary branch width;
- measure draft latency separately.

## Controls

No target logits may be used for deployable tree construction. The target-informed tree remains an oracle baseline.

## Gates

Primary: real integrated output/ms beats the existing linear MTP path.

Strong: complete draft+verify throughput reaches 200 tok/s.
