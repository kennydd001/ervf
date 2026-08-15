# CAP0X-R2 independent retrospective diagnostic

## Verdict

The stored CAP0X-R2 artifacts support a positive **exploratory process-lifetime diagnostic**, not a formal CAP0 result.

CPU-only recomputation passed every replayable check:

- the Intel 1024-word procedural input and expected output hashes reproduce exactly;
- the stored Intel result reports 1000 launches, a valid 4096-byte host-USM allocation, zero differing words and an observed digest equal to the independently recomputed expected digest;
- the NVIDIA child retained a D7 strong synthetic-component pass with all nine stored gates true;
- both child PIDs are distinct, both exited zero, neither survived, and the monitor observed both processes alive in 29 samples;
- one both-alive monitor sample falls inside the stored 8.2214 ms Intel submit-to-complete interval.

This does **not** prove Intel/NVIDIA kernel or device-work overlap. The NVIDIA child did not retain a host submit/complete QPC interval. It only proves that the NVIDIA process was alive while the Intel device interval occurred.

## Recomputed Intel evidence

The frozen recurrence begins at `0xC0A080B1`; each state advances with `state = 1664525*state + 1013904223 mod 2^32`, then XORs the index-dependent term. The independently reconstructed hashes are:

- input: `3b910e7576ea08a85f3fa9962a371e59b57a9472393ec54cd6845f6d66aab7ab`;
- expected output: `379c1d4348822339d228f806c4b1c1709f80c5e4b6b12f3a2aa648a3aff5d4f1`.

They equal the stored input and expected hashes. The stored observed digest also equals the expected digest, and the stored result reports bitwise equality with zero differences. Submit QPC `260674590628700` and complete QPC `260674598850100` give an 8.2214 ms interval for 1000 queued launches followed by `clFinish`.

The exact observed 1024-word array was not retained. Consequently, the observed digest cannot be independently rehashed; the exactness conclusion is bound to the immutable stored result and matching stdout artifact.

## Recomputed NVIDIA D7 evidence

For the CAP0X-R2 NVIDIA child artifact:

- correctness: 1,474,560 elements, zero differing bits, equal resident/staged digest;
- validation: 24 samples, p50 `47.0830554962` ms, p95 `47.7547533035` ms;
- test: 120 samples, p50 `47.1325912476` ms, p95 `47.9173538208` ms;
- effective staged payload rate: `20.3101699572` GB/s;
- projected component total: `75.9945808208` ms;
- all nine gates recompute true, including both strong gates;
- `full_bank_pass` remains false.

The full NVIDIA output arrays were not retained. Its equal output digests and zero-difference count are therefore immutable stored evidence, not independently replayable output bytes.

## Process evidence

The stored process intervals intersect for `19083.2357` ms, but their end timestamps were sampled by the coordinator only after the monitor loop and are not exact child-exit timestamps. Stronger direct evidence is the 100 ms monitor series: 29 samples recorded both PIDs alive across `2809.4504` ms. Both final exit codes are zero, both `alive_after_wait` values are false, stderr is empty, and stdout hashes bind to the retained results.

One both-alive sample lies within the Intel submit/complete interval. This demonstrates overlapping Intel device activity and NVIDIA **process lifetime**, not NVIDIA device activity.

## Claim boundary

Admissible conclusion:

> In this exploratory run, an exact 4 KiB Intel host-USM sentinel and a strong synthetic NVIDIA D7 component run completed without child-process error while their process lifetimes overlapped.

Not established: kernel/device-work overlap, same-process coexistence, a formal CAP0 pass, speedup, end-to-end performance, full-bank behavior, differentiated routing integrity, real-model quality, deployment readiness or a breakthrough.

The independent verifier and machine-readable audit are:

- `scripts/streamq5_moe/verify_het_next_cap0x_r2_exploratory_diagnostic.py`
- `reports/streamq5_moe/het_next_cap0x_r2_independent_diagnostic.json`
