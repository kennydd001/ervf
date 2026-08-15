# PH1 NVIDIA full-expert N5 implementation — independent source audit

Date: 2026-08-14  
Mode: strict source-only/read-only audit. No candidate import, preflight, payload read, compiler load, CUDA load or device call was performed.

## Verdict

**NO-GO for the frozen N5 no-device static preflight.**

N5 correctly repairs the Qwen3 path, parent lock-walk payload classification, the concrete child invocation and the numerical-negative equality contradiction. It does not close the production-fixture, structural-verification, ownership or failure-matrix requirements inherited from the N4 audit.

## Frozen integrity and absence

All handed-off hashes match exactly:

| artifact | bytes | SHA-256 |
|---|---:|---|
| common | 20,060 | `05b3af88f6a02a0099a401e6305838a9337402d10721e47db337dcd2f43452e0` |
| CUDA kernels | 6,173 | `9f369ab3621c6d56b2a3597bca59c25be8d15e7ac3a2a150d916d6695623a781` |
| backend | 32,753 | `9797b479cd4f0039d6131905a84486f93172d7604a743d43b3c16c212ff66821` |
| transaction | 6,439 | `6071915af063ce379b3e17879daf962ff15f3151e6c10d1ac26a9c4226f7155b` |
| runner | 13,612 | `6830e9f5caaf831a5e7c0495fbc04d58e8178b2931e87dd4722790d0eb89d68c` |
| verifier | 28,918 | `ed602d1d8f7aa5270c1bd6324c7b6e0609421cf38413f67b50c4b2c20638c1a5` |
| static preflight | 19,196 | `e96a0cc84788b2a9edc223d94db59820b55d91f32b79b3fa23c0fe34ed40bfc8` |
| isolated fixture child | 685 | `c12c70334a414aecdb698c79c5348d8231111dfd84b53986a107f98678695eb4` |
| source preregistration | 888 | `37ecb790ad86e9fd26638fa7260e90678b0fa185e3d9497a68a2e0de37a243b8` |
| source lock | 9,242 | `e042714507f3558e87a41dbe4ad03ab5f6b4acf640742f75166da2db1e09ce04` |
| verifier lock | 8,477 | `1fb2399a41fde3c551cc83f4649b257b774f442bd0392f63d5cd539b0aec5234` |
| preflight lock | 8,731 | `8b5de6818c8643128bf018db9701accc1f24d2efcc34920cbefb5defef299084` |

Small-file rehash plus stat-only payload checks are internally true: source 38/38, verifier 39/39 and preflight 40/40. The four stat-only payload sizes are CPU stages 23,432 bytes, LUT 131,072 bytes, D2 171,696,126 bytes and official shard 3,999,619,288 bytes. Compile/physical outputs, failure roots, quarantine, preflight result, verification result and N5 in-progress paths are absent.

## Blocking findings

### 1. Production-verifier fixtures remain missing-directory tests

`verifier_mutations()` (`preflight_het_next_l0_ph1_nvidia_n5_static.py:146-162`) still creates only a generic `kind="fixture"` bundle. Its only calls to the actual production entrypoints are:

- `verify_compile(root/"missing_compile", injected)`;
- `verify_physical(root/"missing_physical", injected, lambda: None)`.

The asserted result is merely `bundle=false` and `parse=false`. No complete valid N5 compile bundle or physical bundle is accepted, and no protected field is mutated through the production parsers. ABI evidence, one-program compiler ledger, PTX/SASS contract, controls, ownership, resources, schedule, runtime modules, outputs and terminal classification therefore remain untested by the advertised production mutation suite.

The pure snapshot mutations are useful for the small terminal predicate but cannot substitute for production parser fixtures.

### 2. Kernel and schedule validation is unchanged and non-structural

`kernel_contract` (`preflight:64-69`) is still a required-substring gate with one token renamed per mutation. It does not freeze the actual record offsets, row/column mapping, loop bounds, width-8 FMA/reduction tree, LUT indexing or integer BF16 multiply body.

`schedule_contract` (`preflight:98-104`) still checks only a set of constants and presence of several call names. It does not prove exact 9/5/4/9/1/7 order/cardinality, byte counts, buffer operands, stream identity, launch operands, seven meminfo placements, context sequence or no Driver call after primary release. The CUDA/backend implementations are essentially namespace copies of N4, so the structural N4 gate has not been implemented.

### 3. Remaining ownership and failure matrices are not closed

The independent verifier retains the N4 pinned-write improvement but still does not fully adjudicate:

- stream-create `requested_flags` and `registered_owned`;
- module-load exact `numOptions/options/optionValues`, returned handle and registered ownership;
- retain `registered_owned`, exact pushed-context operand and all context-handle crosslinks.

The executable cleanup fixture remains only the 30 ordinary release return codes (`preflight:163-173`). The transaction fixture remains the small positive/stale/create-new/repeat/reject/corrupt/oversize set (`preflight:122-142`). There are still no full NVRTC returned-error/ctypes/null/destroy combinations, acquisition failures, context pop/restore/primary-release failures, meminfo and post-runtime-scan failures, atomic prelink/postlink/fsync/failure-writer failures, or primary/secondary preservation cases.

`atomic_failure` can still leave an attempt directory if final creation fails. Preflight and final-verification publication remain direct non-atomic writes without bounded failure evidence. These are explicit N4 audit gates, not optional later performance checks.

### 4. Payload guarding is local to `fsha`, not an actual file-I/O guard

N5 correctly marks all four scientific payload bindings stat-only in the parent lock walk. However `payload_guard_mutation()` calls only the custom `fsha()` wrapper (`preflight:34-37,52-63`). The counter is not connected to `Path.open`, `Path.read_bytes`, `Path.read_text`, built-in `open` or the common/verifier `read_range` functions. `READ_COUNTER["payload_bytes"]` is never incremented by any I/O primitive.

Consequently a direct payload `read_bytes()` or range read in a fixture would bypass this guard and still report zero. The isolated child also does not run `hash_lock()` or the guard mutation; its `PAYLOAD_PATHS` remains empty and it reports the disconnected zero counter after missing-directory tests.

Moreover, production runner authorization and default verifier `provenance()` still content-hash all source/verifier-lock bindings. A later compile-only phase would therefore read all four scientific payloads before compiling. Payload identity needs a phase-aware manifest/stat verifier shared by runner and independent verifier, plus a real open/read interception test in the static child.

### 5. Child/model mutation contracts are only partially non-vacuous

The actual frozen child invocation is now correct and bounded: one `subprocess.run`, `sys.executable -I -B`, exact child variable, `stdin=DEVNULL`, captured output, 120-second timeout, `check=false`, `shell=false`. This closes the deterministic N4 self-blacklist.

The AST gate checks the argv expression and only the **set of keyword names**, not their values or the assignment of `child`. Mutations to `shell=True`, cwd, timeout, stdin or the child assignment can survive. Likewise `model_path_contract` establishes both current Qwen3 constants but mutates only the common-side occurrence, not the independent-verifier occurrence. Exact field-level mutations are still needed for the stated allowlist/path proof.

## Repairs genuinely present

- Common, independent verifier and locks now agree on the exact Qwen3-Coder-Next shard path.
- CPU stages and LUT join D2/shard as stat-only entries in the parent static lock walk.
- The source uses one concrete safe child subprocess and no longer rejects its own `run` call.
- Numerical-negative terminal handling is materially repaired: `output_integrity` requires exact keys, lengths, finite stage values and ledger digests without forcing equality, while equality remains in the terminal classification. A stages/counters-only negative can now pass the verifier in principle.
- Complete direct lock closure and all prior pinned-write/runtime/evidence improvements remain present.

These repairs are insufficient for static-preflight authorization because the source gate still cannot substantiate its production, structural and failure claims.

## Required N6 gates

Before any no-device static preflight:

1. build complete valid payload-free synthetic compile and physical bundles that pass the actual verifiers, then reject one mutation for every protected production field and both allowed numerical-negative variants;
2. replace kernel token checks with an exact structural arithmetic/DAG contract and mutations of offsets, loops, mapping, reduction and BF16 multiply;
3. replace schedule presence checks with exact ordered control-flow/operand/cardinality validation;
4. close stream/module/context ownership fields in the independent verifier;
5. execute full compiler, acquisition, context, cleanup, atomic publication and failure-writer fault matrices with no orphan state and preserved primary/secondary evidence;
6. instrument real file-open/read primitives in the isolated child and prove direct/range payload attempts are blocked; use phase-aware stat/manifest provenance for compile authorization/verifier;
7. freeze and mutate every child keyword value/child assignment and both independent Qwen path constants.

Compiler and physical execution remain closed. No N5 static-preflight command is authorized.
