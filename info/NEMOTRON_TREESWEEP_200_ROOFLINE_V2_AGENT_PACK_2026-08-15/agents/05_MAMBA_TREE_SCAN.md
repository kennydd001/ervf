# Agent 05 — MambaTree Exact State Scan

## Mission

Adapt STree-style accumulated transitions to the exact Nemotron Mamba-2 implementation.

## Required comparisons

- individually unrolled branches;
- batched unrolled branches;
- packed tree scan;
- accepted-state reconstruction/activation replay.

## Correctness

Match sequential target outputs and accepted Mamba states under the frozen numerical reference. Document any unavoidable non-bitdeterminism by changing batch shape; do not hide it.

## Performance

Profile tree sizes 5/15/31/63, state memory, kernel launches and temporary buffers. The packed scan must beat unrolled verification at the registered budgets.

## Stop

If exact state commit is not possible or packed tree scan is slower after reasonable kernel optimization, close this branch.
