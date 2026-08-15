# PH1 NVIDIA full-expert N1 implementation — independent source audit

Date: 2026-08-14  
Mode: source-only/read-only audit. No candidate import, static preflight, compiler, payload read, NVCUDA/NVRTC load or device call was performed.

## Verdict

**NO-GO for the frozen no-device static preflight.**

The frozen files are intact and the workspace is clean, but the static preflight and independent verifier do not non-vacuously enforce material parts of the frozen N1 contract. The physical runner also collapses infrastructure/resource failures into the valid numerical-negative class. These are protocol blockers, not observed NVIDIA mechanism failures.

## Integrity and absence

All reported candidate hashes match exactly:

| artifact | bytes | SHA-256 |
|---|---:|---|
| common | 18,334 | `b538c3c91cac5f05ab6da6c77e287144e846c56f9db1c5a18319eedfcc9a9d03` |
| CUDA kernels | 6,166 | `c729c6ee63473d55baa6daf706d7089589417ccb9d3ad5895c0e568c4e333123` |
| backend | 27,540 | `c6a732d6a2eeb189ec8c957f8921327e76bf879705a952ccde6077d3bccf75ab` |
| transaction | 6,433 | `3e492767016e4d735d3d73896fde64fb800bbc16d38de773e9f5516d61bb1084` |
| runner | 12,054 | `8fd8cdf9f9e14acb6f620f619b4dbe22b1eb5b112c57ba105d04d1bba76f5ff2` |
| independent verifier | 19,695 | `4f40c892e050e87807f49d0e8cfb2ba0ec783f5ef4f39c810deb02f52da885a4` |
| static preflight | 7,208 | `8acd8a43fb9115225cfeab19c74208f8cd3086c06c885ed7067d134a170b2559` |
| source preregistration | 4,514 | `02ef0d089d53b5b73b7f43803f9a426f3198cff3f9ab6cfa35ef4a92948b05e8` |
| verifier lock | 4,525 | `eeddda7851657ef3800a78529bcf84c09d6551ce947dc614ab56fec9700b791a` |
| source lock | 6,077 | `6266157c5e36de70c90c385ff85dd0069cb6942c64390204f28d9b3acac279a9` |
| preflight lock | 3,656 | `c3db0435ba7bf904c2735da9dce527ae95e3ee0b9ca26940cbc9adcf8184e0d6` |

Independent rehashing found verifier bindings 23/23, source bindings 28/28 and preflight bindings 18/18 valid. Compile output, physical output, compile failures, physical failures, quarantine, static-preflight result and independent-verification result are all absent. No in-progress candidate artifact was found.

## Blocking findings

### 1. Static preflight does not test the frozen ABI/schedule contract

`preflight_het_next_l0_ph1_nvidia_n1_static.py:33-45` reduces the CUDA and backend contracts to substring presence plus a small required-token mutation list. It does not validate or mutate:

- the complete Driver/NVRTC `argtypes` and every `restype`;
- handle and `CUdeviceptr` widths, `_v2` exports, output-pointer levels or calling conventions;
- exact allocation/copy/launch pointer correlation and stable kernel-parameter storage;
- the seven and only seven `cuMemGetInfo_v2` calls at stages 6–12;
- pop/release ordering and the ban on Driver calls after primary release;
- the row mapping, Q5 offsets, virtual counts, reduction distances, BF16 multiply body or counter indexing.

Concrete counterexamples survive the current source gate: changing an unmentioned ABI signature, changing `threadIdx.x / 8`, changing a scale offset, or inserting a Driver call after primary release leaves every tested substring present. This contradicts the N1 capability design’s required complete ABI/operand/source mutation gate. `preflight...:64-69` would nevertheless report the static preflight positive.

The preflight also has no compile-failure-state simulation, no physical ownership/context failure simulation, and no independent-verifier mutation suite. Its transaction fixture (`preflight...:49-59`) covers only a positive bundle, one stale directory, two ordinary failure writes and one corrupt result.

### 2. Compile verifier is not source-bound and does not implement the frozen PTX/SASS gate

`verify_het_next_l0_ph1_nvidia_n1.py:136-141` reads candidate `source.cu`, but checks only the self-described artifact manifest; it never requires its SHA-256 to equal the frozen CUDA source SHA `c729c6ee...`, and it does not check the result’s authorization/observed binding map.

The PTX check is only: two entry-name substrings and absence of `.ftz`/`approx`. The SASS check is only ELF magic plus the two entry-name substrings. Neither parser proves exactly two entries, `sm_120`, the width-8 FMA/add DAG, expected counter atomics, absence of unresolved externals/transcendentals, or source-to-entry/launch correspondence. Therefore a materially changed program can satisfy the present compile verifier.

### 3. Physical verifier omits decisive ABI, pointer, return-code and call-ledger checks

The single compound check at `verify_het_next_l0_ph1_nvidia_n1.py:147-150` does independently rebuild the records/oracle and checks exact outputs, counters, allocation names/sizes and release rows. It does **not** verify:

- `evidence.abi`, the loaded System32 module path/hash, WinDLL convention/winmode, or the driver-load row;
- return code zero and owned-stream identity for every memset, H2D, launch and D2H;
- H2D/D2H source and destination pointers against the 14 pinned/device allocations;
- launch `argument_values` against the frozen argument map and device pointers (it only tests that parameter-slot addresses differ from argument values);
- owner-thread consistency across calls and context rows;
- an exact `cuMemGetInfo_v2` call ledger or the prohibition on other Driver calls after release.

These omissions let forged or mis-correlated protocol evidence pass when its embedded output bytes are correct. This violates the design’s independent exact pointer/copy/launch/ABI/cardinality requirement.

### 4. Forbidden-runtime evidence is asserted, not observed

`het_next_l0_ph1_nvidia_n1_backend.py:289` initializes all forbidden-call counters to zero and `cudart_loaded=false`; there is no loaded-module scan or instrumentation that can change those values. Device identity repeats the same constant at lines 201-204. The verifier (`verify...:150`) trusts those fields. This is vacuous evidence for the frozen “no CUDART/CuPy/Runtime load or call” gate.

### 5. Infrastructure/resource failures become valid numerical negatives

`run_het_next_l0_ph1_nvidia_n1.py:126-130` puts `resources`, `cleanup` and `forbidden` in the same `gates` dictionary as numerical stage/counter equality, then assigns every non-positive combination the status `nvidia_device_numerical_negative` with `terminal_valid=true`.

The frozen contract says predevice, protocol, lifecycle, cleanup and resource failures are infrastructure-invalid, not scientific device negatives. At minimum the terminal classifier must allow a committed numerical negative only for the explicitly frozen numerical fields, while all protected gates must remain true. The independent verifier currently accepts the same overbroad status class.

### 6. Failure evidence is incomplete on deterministic lifecycle branches

Both phase functions call `clean_or_quarantine` outside their `try` blocks (`run...:89-104` and `112-134`). A stale output, existing failure tree or prior quarantine therefore exits without the promised bounded phase failure artifact. Compiler source reading is likewise before the compile `try`.

The compiler failure ledger (`backend:80-131`) records attempted calls only. After create/compile/retrieval failure it does not emit the explicitly required `attempted=false` suffix rows, so the frozen per-failure call-state contract cannot be adjudicated. The static preflight has no fixtures for null/non-null create failure, compile failure with log recovery, each retrieval failure or destroy failure.

### 7. The 22-control evidence omits frozen presented/requested metadata

`het_next_l0_ph1_nvidia_n1_common.py:174-207` performs the checker calls before backend creation, but most retained rows contain only record/control/expected/observed/pass and a hardcoded zero-count map. They omit presented sizes/digests, requested identity/input and mutation location/bytes required by the inherited N0 control evidence. The independent verifier (`verify...:68-78`) reconstructs the same reduced schema and even assigns the wrong-LUT observed class directly rather than calling an independent checker. Thus exact equality between runner and verifier does not restore the missing evidence.

## Non-blocking source observations

The CUDA source itself is internally coherent under static inspection: two entry points, exact width-8 tile mapping, fixed Q5 code/scale layout, `fmaf`/`__fadd_rn` reduction, raw-word LUT lookup, integer BF16 multiply, and one counter increment per output row are present. The direct Driver backend uses the frozen absolute NVCUDA path, `WinDLL(..., winmode=0x800)`, 64-bit `CUdeviceptr`, one nondefault stream, cubin-only module load, 14+14 allocations, 9 memsets, 5 H2D, 4 launches, 9 D2H and 30 reverse ordinary releases. Static inspection cannot establish compile or device success, and this paragraph is not a mechanism result.

The local bound `nvrtc.h` states that `nvrtcGetPTX*` and `nvrtcGetCUBIN*` both retrieve products of the prior compilation and that CUBIN size is zero for a virtual architecture. No source-only deterministic contradiction was found in the one-program API sequence itself.

## Required next revision before any static preflight

1. Replace substring gates with an AST/structured source contract plus exhaustive negative mutations for every frozen ABI, schedule, kernel arithmetic and lifecycle field.
2. Bind candidate `source.cu` byte-for-byte to the frozen kernel; implement real PTX and SASS parsers for the frozen entry/DAG/no-FTZ/no-approx/no-unresolved contract.
3. Expand the physical verifier to exact loader/ABI/op-code/pointer/stream/argument/owner/resource-call correlation and mutate each field.
4. Produce observed loaded-module/forbidden-surface evidence rather than constant zero maps.
5. Split numerical-negative adjudication from protected infrastructure gates.
6. Move recovery/source-read branches under bounded failure handling and retain explicit attempted/not-attempted compiler rows; add the complete failure matrix to static preflight.
7. Retain exact requested/presented metadata for all 22 controls and independently recompute every rejection.

Only a new immutable source/preflight/verifier revision closing these items should be considered for a no-device static-preflight authorization. Compile and physical phases remain closed.
