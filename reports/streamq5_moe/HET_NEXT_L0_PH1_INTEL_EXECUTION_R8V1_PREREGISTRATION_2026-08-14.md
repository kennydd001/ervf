# HET-NEXT-L0 PH1 Intel execution R8V1 verifier erratum

Date: 2026-08-14. Status: frozen, execution closed pending independent source audit. R8V1 is CPU/verifier-only and cannot run payload construction, compiler, OpenCL or a device.

## Immutable input

R8V1 adjudicates the already committed R8A5 physical bundle without rerunning it:

- `result.json` SHA-256 `9d1ac21f4fdd9657160e877f267369b5e831ff9f7a65e998f27895947c9cad50`, 99,483 bytes;
- `manifest.json` SHA-256 `2d13137f143ff183be3ffe89a3b85754cb2f35b52f92885580f49676e5fcfb7b`, 167 bytes;
- `commit.json` SHA-256 `07d9f03e8907a029d8bc31e40da6298de080b6bc0f0914769f8d52517b2dd965`, 210 bytes;
- immutable first verifier SHA-256 `d6b630658c59e1c6913ba099bb8d617fe1b451e14e31ee38b68d351fb9fde917`;
- local topology diagnosis SHA-256 `e3be1fa3d05fe8a6437f0b3fbb047bc99ee83000c7c67dde725bfb9254715f1a`;
- independent postrun audit SHA-256 `218ceb07f599bd7b7cad32c3da42373256f927c04b21f70a599c977411e4ae0b`;
- independent postrun JSON SHA-256 `01aba30e31db65d8b42c6dd047202391eb9a3da67fa434f944c8bbf1bf46978c`.

The first verifier was verifier-negative solely because a Windows case-insensitive prefix glob treated the mandatory uppercase R8A5 preregistration and source-audit filenames as unexpected. No failure, quarantine or temporary path exists. The physical bundle remains immutable and cannot be rerun.

## Exact topology

R8V1 enumerates directory entries once and compares their exact, case-preserving names against a literal set. It does not use a prefix glob as its authority. For the closed source-audit phase the exact R8A5 family has eight named entries: bundle, lock, failed verifier, uppercase preregistration, uppercase source audit, uppercase topology diagnosis, uppercase postrun audit and lowercase postrun JSON. The exact R8V1 family has only its uppercase preregistration and lowercase closed lock; its verification output, failure/quarantine/temp paths and source-audit file are absent until a later audited authorization revision.

For diagnosis only, R8V1 separately records the Windows `Path.glob` observation and requires it to equal the expected R8A5 names; this observation cannot override the literal enumeration. A non-writing mutation suite rejects missing paths, lowercase/uppercase extras, arbitrary orphans, `.inprogress` names, failure/quarantine names, case-only collisions and duplicate-casefold names. A later open revision must add exactly its independently audited R8V1 provenance and still require the fresh R8V1 output absent.

## Independent verification

R8V1 independently reparses and hashes all three bundle files, reconstructs the manifest and commit, and requires the exact three-file set. It independently validates the R8A5 authorization extension, exact invocation evidence, R8A5 lock and all frozen historical predicates. It then runs the hash-pinned frozen R7A numerical verifier and requires exactly its 20 named checks, all true.

R8V1 also directly validates the result rather than trusting the first verifier: exact positive kind/status; exact set of 18 physical gates, all true; exact Intel Arc identity and driver/PCI; exact 102 main-ledger rows and their operation cardinalities; 95 ownership rows; 14 host-USM allocations; 18 pointer arguments; 4 launches; one finish; 9 direct reads; 21 releases; exactly one cleanup with 21 attempts, zero live resources and no cleanup errors; 22 passing predevice controls; exact extension counters; six forbidden API counters all zero; all twelve ordered resource samples and resource bounds; and exact five stage hashes.

The inherited R8A4 31-case terminal mutation matrix remains evidence but no new terminal failure path is accepted in R8V1: the only eligible input is the exact immutable committed-positive R8A5 bundle. R8V1 writes one create-new verification JSON only after every topology, provenance, numerical and direct-physical check succeeds. Its claim remains one real expert/input Intel correctness component only.
