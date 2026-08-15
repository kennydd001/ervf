# TreeSweep-200 Roofline V2 changelog

This pack supersedes the original `NEMOTRON_TREESWEEP_200_AGENT_PACK_2026-08-15` without modifying it.

Added:

- N1–N5 imported evidence manifest and independent reproduction phase;
- corrected measured one-token byte ceilings;
- exact-efficiency milestones at 50/75/100 tok/s;
- graph-resident token execution;
- gather-free sparse downflow;
- attention and GEMV roofline-recovery agents;
- integrated persistent token fabric;
- optimized target-tree roofline rerun;
- revised hard-stop rule: an unoptimized verifier can no longer close 200 tok/s by itself;
- explicit closure of the low-rank ReLU² prefilter branch.
