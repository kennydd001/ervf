# PH1 NVIDIA full-expert N4 implementation — independent source audit

Date: 2026-08-14  
Mode: strict source-only/read-only audit. No candidate import, preflight, payload read, compiler load, CUDA load or device call was performed.

## Verdict

**NO-GO for the frozen N4 no-device static preflight.**

N4 genuinely fixes the isolated-child bootstrap, direct provenance closure and pinned-write evidence, but it does not close the six N3 audit clusters. Three deterministic static/runtime defects and several non-vacuity/terminal defects remain.

## Frozen integrity and absence

All handed-off hashes match exactly:

| artifact | bytes | SHA-256 |
|---|---:|---|
| common | 20,059 | `d46399a933cfcf9667d781538f5af85825119e4fe93cd3b3e2efb96a2595b238` |
| CUDA kernels | 6,171 | `49ba8c1e6d20a96fd4b20014c7071698226d6d3f3aca3bed11da7b03b13283c4` |
| backend | 32,751 | `b3bcca5591f44f2c6102b261c089c838adf533384b45c23065aeb07cac714847` |
| transaction | 6,437 | `6371819f64224a976f7b0a9da5cdefe3bfa1554ba810db0291a63b05cb07c9fd` |
| runner | 13,610 | `aae8c112a8010c164c863349dc80abbac23099a173a6662ced67a399a5c68d43` |
| verifier | 28,490 | `371555620d3249e185a70bde5092babece82b3d46578addd7c4fa559b5d6b62d` |
| static preflight | 17,519 | `3d9367e4ab4aae055fedbb40b69c31aea8770ab336924bfc1eb18e395fd21e18` |
| isolated fixture child | 683 | `b530c2cf57693f31b527fd16030695d8252b23e968466bb6bbb55d4192d08705` |
| source preregistration | 1,913 | `3a976e9a29b819b0261a624ca99cdb8b293abb4f1c1fe10bdfa84095d1488043` |
| source lock | 8,757 | `81bf412bb58056ddb15978af538bec508040473b1939877c4dfae068eb671207` |
| verifier lock | 7,992 | `5ace26ad9adbb4ea5494e89eda72b196c45dc899501aa35e24cb217832af5f96` |
| preflight lock | 8,247 | `b789b2ae0148dff2430c0d889da4309af9788c90fc2a7dec093a418669575614` |

Small-file rehash plus stat-only large-payload checks are internally true: source 36/36, verifier 37/37, preflight 38/38. D2 is 171,696,126 bytes and the official shard is 3,999,619,288 bytes. Compile/physical outputs, both failure roots, quarantine, preflight result, independent-verification result and in-progress paths are absent. The preflight lock is correctly closed/PENDING.

## Blocking findings

### 1. Candidate and independent verifier use a nonexistent Qwen4 shard path

Both `het_next_l0_ph1_nvidia_n4_common.py:22` and `verify_het_next_l0_ph1_nvidia_n4.py:23` replace the frozen official `models--Qwen--Qwen3-Coder-Next` path with `models--Qwen--Qwen4-Coder-Next`. The locks still correctly bind the Qwen3 shard.

Consequences:

- physical `common.prepare_package()` fails at its first shard range read;
- independent `verify_physical(...).prepare()` fails identically;
- the static source contract does not compare either hardcoded path to the lock-bound official shard, so this deterministic blocker is invisible to the proposed preflight.

This is not a model-quality outcome; it is a source/provenance path error. Both independent constants need exact lock-path equality plus a wrong-model-path mutation.

### 2. The no-payload counter is vacuous and the parent reads 154,504 CPU payload bytes

N4 correctly changes D2 and the 3.999-GB shard to stat-only checks. However `BIG_BINDINGS` contains only `d2_raw` and `official_shard1` (`preflight:19`). `hash_lock()->fsha()` therefore still reads:

- `cpu_stage_freeze.safetensors`: 23,432 bytes;
- `bf16_silu_lut.bin`: 131,072 bytes.

That is 154,504 payload bytes during the claimed no-payload preflight. `READ_COUNTER["payload_bytes"]` is initialized and compared/emitted but is never incremented or connected to `open`, `read_bytes`, `read_text` or `read_range`; these reads are instead counted as `metadata_bytes` (`preflight:34-35,50-56,160-171`). Any future accidental payload read would likewise report zero.

The production runner authorization and default verifier `provenance()` still content-hash every source/verifier-lock binding, so a later compile-only phase would again read D2, shard and CPU tensor payloads merely for authorization. The direct bindings should stay, but every payload binding needs immutable manifest/stat adjudication outside an explicitly authorized payload phase. The preflight requires an actual I/O guard with deliberate forbidden-open mutations, not a disconnected counter.

### 3. The static preflight forbids its own required fixture subprocess

`forbidden_surface()` blacklists the generic AST call name `run` (`preflight:102-104`). The same frozen source calls `subprocess.run` for the dedicated child at line 169. Therefore `checks["forbidden_surface"]` is deterministically false even if the child itself succeeds.

The dedicated 683-byte child is otherwise a genuine repair of the N3 `-I/-B` sibling-import failure: it loads the preflight by exact absolute path, and the preflight bootstraps frozen siblings via absolute `importlib`. The next gate must allow exactly this one frozen subprocess invocation—exact executable, `-I`, `-B`, child path, cwd, capture, timeout—and continue rejecting compiler/device subprocess surfaces.

### 4. Production mutation fixtures are still missing-path tests

The N4 preregistration claims valid production compile/physical bundle fixtures plus mutations. Actual `verifier_mutations()` creates only a generic `kind="fixture"` bundle, corrupts its generic `result.json`, and then invokes `verify_compile`/`verify_physical` only on nonexistent directories (`preflight:129-145`). Injected provenance/prepare functions prevent payload reads, but no valid production compile or physical result is ever accepted.

Thus no mutation reaches the production ABI, compiler ledger, module/function/stream ownership, schedule, resources, outputs or terminal parsers. This leaves the N3 production-nonvacuity blocker open.

### 5. Structural DAG/schedule and complete ownership/failure gates are not implemented

The CUDA kernel is scientifically unchanged from N3 apart from namespace/comment/newline changes. `kernel_contract` remains a required-substring gate with token-removal mutations (`preflight:57-62`). It does not freeze the row mapping, record offsets, loop bounds, exact width-8 reduction or integer BF16 multiply body.

`schedule_contract` remains constant/call-name presence only (`preflight:85-91`); it does not establish exact 9/5/4/9/1/7 order/cardinality, byte counts, pointer/stream operands, seven meminfo placements, context ordering or post-release call ban. Exact ABI-table comparison is retained, but field-level ABI mutations remain partial.

The independent physical verifier now correctly checks all five `pinned_write` rows. It still does not fully check stream-create requested flags/registered ownership, module-load `numOptions/options/optionValues` and registered ownership, or retain registration and exact push operand. These are explicit parts of the inherited ownership contract.

The failure suites are unchanged in substance: ordinary-release error codes 1..30 plus a small transaction happy/corrupt/oversize set. There are no executable NVRTC returned-error/ctypes/null/destroy combinations, acquisition failures, context pop/restore/release failures, meminfo failures, post-runtime-scan failures, atomic prelink/postlink/fsync failures, failure-writer failures or primary/secondary preservation tests. `atomic_failure` can still leave an orphan attempt directory if its final create fails (`transaction:125-135`). Static-preflight and final verification outputs remain direct, non-atomic writes without bounded failure evidence.

### 6. The advertised numerical-negative terminal cannot pass the verifier

`contract_snapshot_valid()` permits a terminal `nvidia_device_numerical_negative` when the false set is confined to `stages_exact` and/or `counters_exact`. But `verify_physical` separately requires `checks["outputs"] = stage_exact and counter_exact` (`verify:202`). Since overall verifier success is `all(checks.values())`, every allowed numerical negative is rejected by the precommit verifier.

The runner then records an infrastructure failure rather than committing the preregistered scientific negative. Raw-output integrity/finiteness/digests must be separated from equality classification: the output evidence check may remain mandatory, while exact equality belongs only to the mutually exclusive positive/numerical-negative terminal adjudicator.

## Repairs genuinely present

- Direct lock closure is complete and all 36/37/38 handed-off bindings are coherent.
- D2/shard are no longer content-read by the parent static lock walk.
- The dedicated absolute-path `-I/-B` child fixes the N3 sibling-import design.
- Injectable verifier provenance/prepare prevents the isolated missing-path calls from hashing payloads.
- Pinned writes are now independently checked against record/input/LUT hashes and pinned pointers.
- N3 evidence preservation, deferred `psutil`, full-path post-execution module scan, exact ABI representation and terminal snapshot restriction remain present.

These repairs do not authorize execution because the frozen preflight is deterministically false and the physical source is path-invalid.

## Required N5 gates

Before any no-device static preflight:

1. restore exact Qwen3 shard paths in common and verifier; compare them structurally to the lock and reject a Qwen4/path mutation;
2. stat/manifest-only every payload binding, including CPU stage tensor and LUT; instrument actual file opens and prove forbidden payload-open mutations fail;
3. replace the generic `run` blacklist with an exact one-child subprocess allowlist;
4. pass complete valid synthetic compile and physical bundles through the real verifiers, then mutate every protected production field;
5. implement exact kernel/schedule AST/control-flow gates and the remaining stream/module/context ownership checks;
6. implement full compiler/context/acquisition/cleanup/atomic-failure matrices and make verifier/preflight publication atomic and bounded;
7. make output-evidence integrity compatible with the explicitly allowed numerical-negative terminal.

Compiler and physical execution remain closed. No N4 static-preflight command is authorized.
