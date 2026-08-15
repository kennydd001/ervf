# PH1 Intel execution R8A5 — independent post-run adjudication audit

Date: 2026-08-14  
Scope: read-only physical-bundle, device-evidence, CPU-oracle, first-verifier, and topology-failure audit. The physical runner was not rerun; no compiler, OpenCL, or device call was made. A frozen CPU-only numerical oracle was recomputed once from the immutable result and official source slices.

## Verdict

The immutable R8A5 physical bundle is **substantively positive** and independently CPU-recomputes positive. The first standalone verifier is **formally invalid only because of its Windows case-insensitive live-topology glob**. This is not a physical, numerical, authorization, device, ownership, lifecycle, cleanup, or resource failure.

Do not rerun R8A5. The correct next step is a fresh, separately preregistered **verifier-only R8V1** that binds and re-adjudicates the existing immutable bundle. Until R8V1 passes, describe R8A5 as “physical positive with verifier-topology adjudication pending,” not as a final formal component PASS.

## Immutable artifact hashes and bundle contract

- `result.json`: 99,483 bytes, SHA-256 `9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50`;
- `manifest.json`: 167 bytes, SHA-256 `2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b`;
- `commit.json`: 210 bytes, SHA-256 `07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965`;
- first verifier: 1,724 bytes, SHA-256 `d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917`;
- existing topology diagnosis: 3,916 bytes, SHA-256 `e3be1fa3d05fe8a6437f0b3fbb047bc99ee83000c7c67dde725bfb9254715f1a`.

The bundle directory contains exactly `result.json`, `manifest.json`, and `commit.json`. The manifest is exactly the one-row R7A manifest for the retained result bytes/hash. The commit exactly binds the manifest-byte SHA and result-byte SHA. Result kind/status are `ph1_intel_execution_r7a` / `intel_execution_positive`, with `positive=true` and the frozen one-expert/input claim.

## Physical and independent numerical gates

All 18 physical result gates are true:

`allocations`, `args`, `compile_identity`, `controls`, `counters`, `extensions`, `finish_reads`, `forbidden_static_and_runtime`, `identity`, `initialization`, `launch`, `ledger_order`, `ownership`, `release`, `resource_samples`, `resources`, `stages`, and `writes`.

A fresh read-only CPU invocation of the frozen R7A numerical verifier independently recomputed all 20 checks true:

`allocations`, `args`, `authorization`, `compile_package`, `controls`, `counters`, `extensions`, `forbidden`, `identity`, `initialization`, `launch_finish_read`, `ledger_order`, `oracle_outputs`, `ownership`, `positive_schema`, `records_input_lut`, `release`, `resources`, `runner_gates`, and `writes`.

Exact output-stage SHA-256 values are:

- gate: `e8a00c17f2ea66f4fc933103eeaf2429c9c1b63fd903720eabaa5b7513acc867`;
- up: `f8dc1dc2c9f19e2012ce806ea121d07135e70d383354ff8faa777377595def08`;
- SiLU: `a83041f1517b31f6b2a81b5d98c3f9a128b5bdc5602b57000453a57b036295e8`;
- activation: `762384a50598dc67aca0963b1e9ed52f5eda71ec9643aeb18a6750ab92fe3d5f`;
- down: `142607c8defe588a2833ce65a774515aeb9691dd7008e4ff6b32488af9bf10fc`.

## Device/lifecycle evidence

- device: `Intel(R) Arc(TM) Pro 140T GPU (32GB)`;
- vendor/driver/PCI: `Intel(R) Corporation`, `32.0.101.8517`, `0000:00:02.0`;
- 102 retained execution-ledger rows and 95 ownership rows;
- 14 host-USM allocations, 18 pointer arguments, four launches, one finish, nine direct reads, and 21 release attempts;
- cleanup complete, zero live owned resources, no cleanup error;
- exact extension calls: host allocation 14, free 14, pointer-argument 18, allocation-info 42;
- every forbidden buffer/copy/read/write/migrate/advice API count is zero;
- 22/22 predevice controls pass;
- 12/12 ordered resource samples are telemetry-error-free;
- peak retained working set 154,890,240 bytes;
- final available RAM 48,636,030,976 bytes.

The retained evidence therefore demonstrates that the intended Intel device route executed, produced the exact frozen stage/counter outputs, and completed its required cleanup.

## Exact first-verifier failure

The first verifier has 29 top-level checks. Exactly 27 are true. The only false checks are:

- `topology`;
- `terminal_contract`, downstream of the invalid topology state.

Its authorization, lock, history, live invocation and mutations, 31-case production matrix, committed-adjudicator mutations, and all 20 numerical checks are true.

The faulty production expression is:

```python
FAMILY_PARENT.glob(FAMILY_PREFIX + "*")
```

On Windows, `pathlib` wildcard matching is case-insensitive. At the instant of first verification it matched the three intended lowercase runtime entries plus two mandatory uppercase provenance files:

- `HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_PREREGISTRATION_2026-08-14.md`;
- `HET_NEXT_L0_PH1_INTEL_EXECUTION_R8A5_INDEPENDENT_SOURCE_AUDIT_2026-08-14.md`.

The verifier's allowed set contained only the bundle, lock, and verifier result. It therefore marked topology false even though the two “unexpected” paths were precisely the frozen documents required to authorize the run. After diagnosis, the matching family naturally also includes diagnosis/audit files; R8V1 must bind the exact final preregistered set rather than repeat a broad implicit glob.

## Exact R8V1 verifier-only contract

R8V1 must be a fresh namespace and must not call the physical runner, compiler, OpenCL, backend, or device. Its preregistration and source lock must freeze the following before execution.

### Immutable bindings

Bind by exact byte count and SHA-256:

1. all three R8A5 bundle files;
2. R8A5 runner, first verifier, preregistration, lock, and source GO audit;
3. the first invalid verifier result;
4. the existing topology diagnosis SHA `e3be1fa3d05fe8a6437f0b3fbb047bc99ee83000c7c67dde725bfb9254715f1a`;
5. this independent audit and its companion JSON;
6. frozen R7A numerical verifier and its source/compile/CPU provenance chain.

The physical bundle and first verifier remain immutable. R8V1 writes only a new create-only R8V1 verification bundle/result.

### Casefold-aware exact topology

Do not decide topology with an allowed set built only from existing runtime paths. Enumerate the report directory once, filter names whose `casefold()` begins with the exact R8A5 family prefix, and compare the resulting **case-preserving names** to a frozen explicit set containing every required R8A5 runtime and uppercase provenance/diagnostic file.

The frozen present set must include the committed bundle directory, R8A5 lock, first verifier result, uppercase preregistration, uppercase source audit, existing topology diagnosis, and this post-run audit/JSON. The frozen absent set must include both failure roots, both quarantine roots, every R8A5 in-progress form, and any unlisted casefold-colliding or prefix-matching entry.

Mutation tests must drop each required member individually and add lowercase, uppercase, mixed-case, suffix, temp, failure, quarantine, file, and directory extras. Each mutation must fail exact topology. The unchanged baseline must pass.

### Independent adjudication

R8V1 must:

- recompute the exact three-file manifest/commit contract;
- independently recompute all 20 frozen CPU numerical checks;
- revalidate all 18 physical gates, stage/counter digests, 102-row ledger, 95-row ownership ledger, device identity, extension/forbidden counts, resource samples, and cleanup;
- retain the R8A4 31-case terminal matrix and R8A2 protected-gate mutation suite;
- explicitly assert that the first verifier's exact false set is `{topology, terminal_contract}` and that every other check is true;
- emit `terminal_state=positive`, `terminal_valid=true`, and `pass=true` only if every R8V1 gate passes.

R8V1 is a correction of verifier topology only. It may formally validate the already completed one-expert/input result; it cannot enlarge the claim to performance, full layer/model, heterogeneous execution, industrial readiness, or breakthrough.

## Final disposition

- R8A5 physical rerun: **forbidden**.
- Existing physical bundle: **immutable positive evidence**.
- First R8A5 verifier: **immutable invalid result caused solely by topology enumeration**.
- Next authorized research action: **fresh preregistered, independently audited, CPU-only R8V1 verifier**.
