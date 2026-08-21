# S100 Phase65: Ornith shared/routed overlap preregistration

## Question

Can the resident shared SwiGLU expert be hidden behind the Phase59 routed bulk
dispatch, as in the selected Nemotron H4 executor?

## Frozen experiment

- Real Pottokao layer-20 shared expert and routed experts 0 through 31.
- H4 shared expert uses the exact M4 kernel; 32 unique routed assignments use
  the Phase59 bulk kernel.
- Serial control executes routed then shared on the main stream.
- Candidate forks the shared expert to one non-blocking stream while routed
  work stays on main, then joins before timing ends.
- Inputs and all weights are already GPU resident. Buffers are disjoint.

## Gates

1. Candidate routed and shared outputs are separately bit-identical to serial
   and repeat exactly.
2. Outputs are finite.
3. Candidate is at least 5% faster than serial.
4. Candidate latency is no more than 5% above the slower isolated branch.

This is a one-layer hot overlap primitive. Full graph integration and cache
miss interaction remain outside the claim.
