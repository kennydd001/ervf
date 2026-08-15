# PH1 Intel execution R7P — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only source audit. No preflight, payload, compiler, OpenCL, or device call was executed.

## Verdict

**GO for exactly one execution of the frozen R7P no-device static preflight.**

This is a preflight-only authorization. It does not authorize payload access, OpenCL loading, compilation, device allocation, kernel execution, or the R7 physical run. Those remain closed pending a complete R7P PASS artifact and a separate authorization review.

## Frozen identity and absence

Observed SHA-256 values match the handoff:

- R7P preflight: `8fa368146ceb659ac008ebe60453603d4a7d1ae58e65266159e91b36855aede0`
- R7P preregistration: `3a7b8e9e93ebee2a1c8e5287b39cc88e8e81414b405cab328f2a550bb4df0fb6`
- R7P closed lock: `356ec1807d7bd46afa5051b393b80a2214a939940393a070a226d607537ea413`

The bound R7 files remain unchanged:

- runner `8fae33f8fab9b3331caef23b04895990c0ea270e7f42b1cc7f23d2e4a8fe566a`
- verifier `549e3d9574ebea0d5c58f2d919608dee2c0f211ea9acbafad678670cebf2a39d`
- original R7 preflight `7b4e5552ef0ddd38ed0db40ca711bd1e1c9ece84c1345766709a66ceb2dd61c6`
- backend/common `8bbfa1a6...` / `d6abe579...`

The R7 output directory, R7 independent-verification output, original R7 preflight result, and R7P preflight result are absent. The R7P lock is closed/PENDING and binds the R7 lock/source/preregistration/audit plus the immutable R6P1 evidence chain.

## Closure of R7 audit blocker 1

The write-after-loop mutant no longer relies on `np.empty` contents:

- the inspected frozen R7 `linear()` source must contain the corrected inside-loop assignment and the original `np.empty` allocation;
- the mutant changes only its test allocation to `np.full(..., 0x7e00, uint16)` and moves the row assignment outside the loop;
- an AST walker proves exactly one `out[...]` assignment is inside the R7 row loop and outside the mutant row loop;
- both production shapes are tested: 512 rows × width 2048 and 2048 rows × width 512;
- two fresh mutant runs per shape must retain poison in every prefix row `0..r-2`, compute the correct last row, produce identical complete-byte digests, and differ from the independently constructed target digest.

The poison word `0x7e00` is distinct from both target words (`0x3f80`, `0x4000`). The Boolean/digest chain is coherent: first digest equals second, and the repeated mutant digest differs from the target. This makes the negative control deterministic and directly sensitive to the original indentation defect.

The existing corrected positive sentinel remains: two complete executions per production shape must equal every independently selected row word and the full expected SHA-256 digest.

## Closure of R7 audit blocker 2

The new TEMP-only simulations invoke `run_het_next_l0_ph1_intel_execution_r7`, not R6:

- `transaction_sim_r7()` builds and verifies an actual R7 result/manifest/commit tuple, exercises recovery of a valid commit, stale in-progress quarantine, and an R7 structured failure archive;
- `lifecycle_sim_r7()` drives the actual R7 `main()`/`execute_authorized()` paths with R7 ACK, R7 paths, and R7 failure kinds;
- cases cover authorization rejection without filesystem pollution, RAM failure, payload failure, ordinary post-device failure, telemetry-bearing failure, oversized-attempt quarantine/cap, and an existing valid completed negative bundle without a spurious failure artifact;
- all runner/base globals, monkeypatches, argv, backend/package replacements, paths, verifier callback, and `psutil` module state are restored in `finally`.

The fixtures use fake telemetry/backend/package objects and temporary directories only. They do not construct the real backend or call OpenCL.

## Retained gates

R7P retains the full R7 gate set: provenance/hash closure; call-based no-device AST sentinel; forbidden surface; 22 synthetic controls; codec/FMA/width-8 production-shape fixtures; compile/CPU package and R7 bundle mutations; 21-release cleanup and ownership faults; corrected all-row sentinel; deterministic write-after-loop negative control; a production-sized independent-verifier baseline with no false checks; and exactly 28 named production mutations, each rejected.

No deterministic source blocker was found. The full-shape baseline plus 28 replays may be CPU-heavy, but this is expected and does not expand authorization.

## Authorized next action

Execute the exact SHA-bound R7P preflight once. Accept it only if all 18 top-level gates are true, the positive and poisoned-negative per-shape evidence is complete, `baseline_false_names` is empty, and all 28 exact mutation names are present in `rejected_mutations`. Physical execution remains unauthorized afterward until separately audited.
