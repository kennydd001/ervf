# S100 Phase 10B — Mamba FP8 ERVF-v2 bandwidth

Phase 9 leaves dense resident reads as the main first-principles target. Mamba FP8 in/out projections account for roughly 892 MB/token.

Old microbenchmarks could be warm-L2. Phase 10B therefore times each candidate by streaming all real Mamba in/out matrices in sequence: roughly 892 MB of different checkpoint weights per token-equivalent.

ERVF remains the baseline. Candidates preserve the same 256 virtual reference threads, per-thread FMA sequence and final reduction tree while exploring 8/16/32 physical lanes per output row, default/L2/streaming load policies, and one-step software prefetch.

Only bit-exact cold-stream candidates >=5% faster enter causal integration. End-to-end promotion requires fresh A/C/C/B, exact token parity, <=1 ms drift, >=765 samples and >=0.25 ms/token gain.

Pack hashes frozen 2026-08-18:
- Phase10A ZIP: f01489086beb2ff702e217c72ab95be39b190d0ba16b93c700c3e6bbedad060c
- Phase10B ZIP: c0746ae65507dc25709ab4f645702e468b556bff7bb1373c8f7102c97d7bb016
