# Agent 12 — OrbitANS and PathQ Modifiers

## Exact subtrack: OrbitANS

Census and compress official NVFP4 codes/scales with random-access tile entropy coding. Optional safe physical ordering may use expert/group permutations only when logical IDs, quantization groups and reduction semantics are preserved.

Gates:

- byte-exact codes/scales;
- expert random access;
- no full dequantized bank;
- at least 5% integrated verification-round improvement.

## Quality subtrack: PathQ

Choose precision per layer/expert/projection using full-depth student-state damage, route frequency, p95/p99 damage and measured kernel/H2D cost.

Objective: minimize active critical-path bytes, not file size.

Gates:

- conservative test relative CE <=0.5% or registered research <=1%;
- active bytes reduction >=12%;
- integrated round improvement >=5%;
- maximum three formats on the critical path.

Never mix exact and PathQ claims.
