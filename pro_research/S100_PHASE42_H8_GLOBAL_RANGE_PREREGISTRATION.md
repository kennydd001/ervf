# S100 Phase42 — explicit-global H8 range pipeline

Frozen before Phase42 GPU timing.

## Opens from Phase41

Phase41's pointer-sliced group metadata failed canonical token exactness at the
first H8 window. Its serial control produced the identical divergence, proving
the failure was group-index semantics rather than cross-stream ordering.

Phase42 keeps all group, route, union and output arrays at their original base
addresses. New exact-subset UP and scan kernels compute
`global_group = g0 + blockIdx.{x|y}` explicitly. Their bodies are generated
from the existing Phase25 UP kernels or copied from its scan kernel; only the
global group-index expression and range launch are changed.

## Gates and sequence

1. Compile/resource audit: every new UP/scan and Phase40 gather/down kernel must
   have zero local-memory bytes.
2. `SERIAL_SMOKE`: three ranges execute UP, scan, gather and down serially for
   one canonical context-1024 H8 window. It must be token-exact before overlap.
3. `OVERLAP_SMOKE`: the three-stage range pipeline executes one canonical H8
   window and must be token-exact.
4. Only after both smokes pass: fresh-process `BASE_A`, `GLOBAL_PIPELINE_B3`,
   `BASE_B`, each four warmup plus sixteen measured H8 windows.

Promotion gates: all tokens exact, zero local memory, baseline drift <=5%, and
candidate gain >=3%. Candidate <=120 ms/H8 is the strong milestone. This is
target-only verifier timing; S100 remains <=80 ms/H8 before drafter cost.

