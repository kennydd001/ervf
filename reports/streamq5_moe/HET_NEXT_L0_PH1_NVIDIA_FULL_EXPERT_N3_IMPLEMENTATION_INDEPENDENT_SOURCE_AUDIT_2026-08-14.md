# PH1 NVIDIA full-expert N3 implementation — independent source audit

Date: 2026-08-14  
Mode: strict source-only/read-only audit. No candidate import, preflight, payload read, compiler load, CUDA load or device call was performed.

## Verdict

**NO-GO for the frozen N3 no-device static preflight.**

N3 closes the direct provenance closure and several N2 evidence defects, but the exact static-preflight command cannot pass and is not payload-free. Two deterministic blockers precede the remaining non-vacuity gaps.

## Frozen integrity and absence

All handed-off hashes match exactly:

| artifact | bytes | SHA-256 |
|---|---:|---|
| common | 20,058 | `e2512004e31baee3e78edae87c7cbd956641fd9f8de7392a82d164fb70212053` |
| CUDA kernels | 6,169 | `1070f0f216bc65e87c3e1ee7dabd20498c2b08e0bfabad07e48ed6dfd3be2fb8` |
| backend | 32,750 | `76f8882fed3b4ce42b1a5aadc947d8ad09107e5658eb21f2d227a2b0b2b60523` |
| transaction | 6,436 | `b0506be2567cdf941502b4c96397cd317f317bcbe6622f9524d0c1e09d97a9a5` |
| runner | 13,609 | `4301dfd49b0f6becc484be39ee4e5fb214bc9068469e8253029d6b121f6a6623` |
| verifier | 27,989 | `fb6cdcd058962d393c3b7e00f67faf47b19dd8ebae52e0b07afdf8223767afdc` |
| static preflight | 16,427 | `94c90eba504ae7c8be6746c45d76bdde2badd9a78e62fbf5091a3aafc16ed4b4` |
| source preregistration | 1,923 | `5185aca4c668d68715a9edc7dd960860b18260f316d4a1d96de41bcf0acf02a5` |
| source lock | 7,040 | `07e717185cc3c8816cffd601e476b656bea54cf156c72dde3eda015ebc835f01` |
| verifier lock | 7,358 | `af9f4ce33ac0bdd9bb6613f8d3afe8efd7089638567fb4a52a46631bd1b35660` |
| preflight lock | 7,610 | `8cc5a06ab7bae7395319cc4b6b6946c7eab6df9614120f70d601c6b5714cf0cc` |

Every present binding rehashes true: source 33/33, verifier 34/34 and preflight 35/35. Compile output, physical output, both failure roots, quarantine, static-preflight result, independent-verification result and N3 in-progress paths are absent. The frozen preflight lock is closed/PENDING, as expected before this audit.

## Blocking findings

### 1. The claimed no-payload preflight performs at least 12,513,946,242 bytes of payload hashing

`preflight_het_next_l0_ph1_nvidia_n3_static.py:35,44-48` implements every lock check with `Path.read_bytes()`. The directly complete preflight lock includes the 171,696,126-byte D2 raw tensor and the 3,999,619,288-byte official shard. The parent therefore reads 4,171,315,414 payload bytes before any static check.

The isolated fixture then calls production `verify_compile(missing)` and `verify_physical(missing)` at preflight lines 121-135. Each verifier function evaluates `provenance()` before discovering the missing candidate (`verify...:48-55,163-164,181-182`), and `provenance()` stream-hashes the complete verifier lock, including the same D2 file and shard. The intended command therefore reads those payloads three times: 12,513,946,242 bytes (11.655 GiB), excluding all smaller bindings. The 120-second child timeout makes this an additional practical failure risk.

This directly contradicts the emitted `"no_payload": true` at preflight line 159 and the authorized audit boundary. The same lock-walk also means a later compile authorization and compile verifier would reread payload merely to establish provenance. Direct provenance closure must be retained, but payload identity must be represented by a previously frozen immutable manifest/stat contract during no-payload phases; content rehash is reserved for the explicitly authorized payload phase.

### 2. The isolated fixture subprocess cannot import its sibling transaction module

Preflight line 157 starts the absolute script using `python -I -B ... --isolated-fixtures`. Isolated/safe-path mode does not prepend the script directory or working directory to `sys.path`. The child reaches the sibling import at line 18 before argument handling, so `het_next_l0_ph1_nvidia_n3_transaction` is unavailable unless it is independently installed (no such contract exists). The child consequently exits before lines 153-154, emits no JSON, and makes `isolated_production_fixtures` false at line 158.

This is a deterministic source-level failure independent of CUDA, payload contents or thresholds. A next revision must use a self-contained bootstrap or exact absolute `importlib` loading whose full dependency graph is explicitly bound, while remaining payload-free.

### 3. “Production verifier mutations” do not exercise successful production parsers

The isolated mutation suite tests a small pure snapshot and the generic three-file bundle helper. Its only calls to `verify_compile` and `verify_physical` use nonexistent directories (`preflight:131-136`). They prove only that both parsers reject absence; no valid synthetic compile or physical bundle is accepted and then mutated through the production parser.

Therefore mutations of ABI evidence, module/function/stream ownership, exact schedule, resources, compiler ledger, terminal split or output counters are not demonstrated to fail in the real verifier. The advertised production-fixture claim remains non-vacuous only for missing paths.

### 4. Static schedule/kernel gates and physical ownership evidence remain incomplete

`schedule_contract` (`preflight:77-83`) checks a set of string constants and the presence of five call names. It does not prove exact 9/5/4/9/1/7 ordering/cardinality, buffer sizes, pointer operands, stream identity, context sequence, sample placement or the post-release Driver-call ban. `kernel_contract` (`preflight:49-54`) remains a substring gate; its mutations only rename one required token. Row mapping, record offsets, loop bounds, width-8 reduction shape and the integer BF16 multiply body can change while the required strings remain.

The exact 30-function ABI table comparison is a genuine improvement, but its mutations cover function names, common restype and one width alias—not each frozen argument-vector field or pointer-depth failure.

The production physical verifier is materially stronger, yet `verify...:184-205` still never collects or validates the three `pinned_write` rows (payload name/size/hash/pointer). It also does not fully adjudicate stream-create flags/registered ownership, module-load exact option operands/registered ownership, or the context retain-registration and push operand. Thus the full ownership/evidence contract requested by the N2 audit is not closed.

### 5. Compiler and lifecycle fault matrices are still partial

N3 correctly records ctypes exceptions for an active NVRTC call and destroy, preserves successful backend evidence on post-device runner failures, keeps invalid authorization mutation-free, defers `psutil`, rescans full runtime module paths after execution, and expands ordinary transaction checks. These are real repairs.

However the preflight fault injection still covers only the 30 ordinary releases (`preflight:138-147`). It does not exercise acquisition failures, context pop/restore/primary-release failures, meminfo failures or the final module scan. The transaction suite has no injected prelink/postlink/fsync/failure-writer failures and does not prove primary/secondary exception preservation. `atomic_failure` leaves an attempt directory if its final create fails (`transaction:125-135`). Compiler error evidence likewise has no executable matrix for every returned-error/ctypes/null/destroy combination. These omissions matter because N3 preregisters complete failure matrices and evidence preservation.

## Repairs genuinely present

- All three locks now directly close the requested source/design/CPU/Intel/native-loader/D2/shard provenance chain.
- Exact ABI vectors are represented in both AST preflight and independent verifier; runtime ABI strings are normalized deterministically.
- Full-path runtime-module scanning, a post-execution scan, function-pointer evidence and owner-thread checks are present.
- Post-device protocol/precommit failures retain completed backend evidence and true `device_opened`; invalid authorization is mutation-free; `psutil` is deferred.
- The terminal contract correctly permits only a fully positive result or a device numerical negative whose false set is confined to `stages_exact`/`counters_exact`.
- NVRTC active-call and destroy ctypes exceptions receive explicit attempted rows.

These repairs do not overcome the deterministic preflight failures or residual evidence gaps.

## Required N4 gates

Before any static preflight:

1. separate direct provenance binding from payload-content reads; the no-device preflight must prove zero opens of D2/shard/CPU tensor payloads and must not call a verifier provenance routine that rehashes them;
2. make the `-I -B` child bootstrap executable from a clean environment and assert the exact loaded file/hash graph;
3. accept a complete valid synthetic compile bundle and physical bundle through the actual production verifier, then reject one mutation per protected field;
4. replace token/presence schedule and kernel checks with exact AST/control-flow/arithmetic contracts and field-level mutations;
5. validate pinned writes and all stream/module/context ownership operands in the independent verifier;
6. execute complete compiler, context, acquisition, cleanup and atomic-publication/failure-writer fault matrices, including primary/secondary preservation and no orphan states.

Compiler and physical execution remain closed. No static-preflight command is authorized from this N3 freeze.
