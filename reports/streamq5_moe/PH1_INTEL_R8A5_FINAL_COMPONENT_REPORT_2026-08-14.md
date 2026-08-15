# PH1 Intel R8A5 — final independent component report

Date: 2026-08-14  
Scope: final read-only audit of the immutable R8A5 physical bundle and completed R8V1R1A verifier-only adjudication. No physical rerun, model forward, compiler, OpenCL, or device call was made. A fresh CPU-only numerical oracle was recomputed from the frozen official source ranges, D2 input, and LUT.

## Final verdict

**FORMAL COMPONENT PASS.** The completed R8V1R1A verifier is terminal positive with 10/10 checks true and independently confirms the immutable R8A5 bundle as positive. The earlier R8A5 verifier remains, correctly and immutably, a verifier-protocol negative caused only by its topology bug. R8V1R1A does not rewrite or reclassify that historical artifact; it independently adjudicates the physical bundle.

This closes the Intel half of the frozen PH1 single-real-expert experiment for exactly one official expert/input.

## Final verifier artifact

- path: `reports/streamq5_moe/het_next_l0_ph1_intel_execution_r8v1r1a_independent_verification.json`;
- bytes: 8,098;
- SHA-256: `42cd69582a47b8b5f8f4b7f24a696f1d3fcc6fbd49c05d0f61354a57cefc052d`;
- kind: `ph1_intel_execution_r8v1r1a_independent_verification`;
- `pass=true`, `passed=10`, `total=10`;
- `terminal_state=positive`, `terminal_valid=true`;
- `prior_verifier_outcome=verifier_protocol_negative`;
- `bundle_adjudication=positive`;
- `model_forward=false`, `compiler_opened=false`, `opencl_opened=false`, `device_opened=false` for this verifier-only run.

The exact ten checks are all true: open lock, topology, topology mutations, exact prior verifier, prior-verifier mutations, immutable bundle, authorization, direct physical evidence, 20-check numerical oracle, and 31-case terminal matrix.

## Immutable physical bundle

The R8A5 directory still contains exactly three files. The manifest and commit independently reconstruct exactly, with hashes:

- `result.json`: `9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50`;
- `manifest.json`: `2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b`;
- `commit.json`: `07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965`.

The result remains `intel_execution_positive`, `positive=true`, with the exact one-expert/input claim.

## Fresh independent CPU recomputation

A new read-only invocation of the frozen R7A numerical verifier independently reread the three allowlisted official source ranges, D2 input and canonical LUT, rebuilt the Q5 records and integer BF16 graph, and returned all 20 named checks true:

`allocations`, `args`, `authorization`, `compile_package`, `controls`, `counters`, `extensions`, `forbidden`, `identity`, `initialization`, `launch_finish_read`, `ledger_order`, `oracle_outputs`, `ownership`, `positive_schema`, `records_input_lut`, `release`, `resources`, `runner_gates`, and `writes`.

The exact output-stage hashes remain:

- gate: `e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867`;
- up: `f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08`;
- SiLU: `a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8`;
- activation: `762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f`;
- down: `142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc`.

## Physical evidence recomputation

All 18 frozen result gates are present and true. Independent parsing of the retained raw result confirms:

- Intel Arc Pro 140T, driver `32.0.101.8517`, PCI `0000:00:02.0`;
- 102 execution-ledger rows with exact operation cardinalities;
- 95 ownership rows;
- 14 host-USM allocations and 18 pointer arguments;
- four launches, one finish, and nine post-finish direct reads;
- 21 successful releases and cleanup complete with zero live resources/errors;
- 22/22 predevice controls true;
- 12 resource samples with zero telemetry errors;
- all six forbidden API counters zero.

## Historical verifier and topology

The first verifier remains SHA-256 `d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917`, with exactly 27/29 checks true. Only `topology` and downstream `terminal_contract` are false; its terminal state is invalid and `pass=false`. Its 31-case mutation matrix is entirely positive. Its correct classification is **verifier-protocol negative**, not physical or scientific negative.

The post-run live family is exactly 17 case-preserving entries: the exact 16 pre-run entries plus only the new R8V1R1A output. There are no casefold collisions, failures, quarantines, or in-progress artifacts. This report and its companion JSON deliberately use names that do not begin with the frozen R8V1 family prefix, preserving that historical topology.

## Strict claim boundary and next gate

What is now proven: one official real expert/input Intel Q5 correctness component reproduced the frozen CPU oracle exactly at every retained stage, with the preregistered host-USM, ownership, lifecycle, cleanup, resource, and control contracts.

What is not proven: throughput, latency advantage, full-layer or full-model correctness, cross-device merge, heterogeneous execution, production readiness, industrial superiority, or an LLM-world breakthrough.

The next scientific gate is the separately preregistered **PH1 NVIDIA single-real-expert correctness component** for the same frozen package. Per the existing R1/R2 contract it must consume the committed Intel PASS, use the direct no-FTZ `sm_120` cubin and primary-context lifecycle, reproduce the same five stage hashes/counters, pass its exact allocation/copy/launch/release ledger and controls, and undergo an independent verifier. No concurrency or performance claim should be attempted before that NVIDIA correctness gate passes.

