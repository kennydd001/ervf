# PH1 Intel execution R8A4 — pre-device module-binding failure diagnosis

Date: 2026-08-14  
Scope: read-only diagnosis of the single authorized R8A4 invocation. No retry, source edit, payload read, compiler, OpenCL, or device call was performed during this audit.

## Verdict

The R8A4 invocation is an **infrastructure-negative, pre-device invalid attempt**, not a scientific or mechanism result.

The deterministic cause is an invalid inherited-module attribute chain in the frozen runner. `authorize()` evaluates:

```python
base = prior.prior.frozen
```

For R8A4:

- `prior` is the imported R8A3 module;
- `prior.prior` is the imported R8A2 module;
- R8A2 defines `prior` (R8A1) but defines no attribute named `frozen`.

The expression therefore raises `AttributeError` before historical authorization, `configure()`, the frozen physical authorizer, payload preparation, compiler/OpenCL loading, or any device call. R8A4 `main()` catches every authorization exception and immediately returns `3`, explaining the silent nonzero process result.

## Source proof

- R8A4 imports R8A3 as `prior`: `run_het_next_l0_ph1_intel_execution_r8a4.py:6`.
- R8A4 dereferences `prior.prior.frozen`: line 26.
- R8A3 imports R8A2 as its `prior`: `run_het_next_l0_ph1_intel_execution_r8a3.py:6`.
- R8A2 imports R8A1 as `prior` and has no top-level `frozen` binding: `run_het_next_l0_ph1_intel_execution_r8a2.py:8`.
- R8A4 catches the resulting exception and returns `3`: lines 43–44.
- `configure()` and delegated execution occur only after that catch block: line 45.

R8A3 did not have this failure because its own `prior.prior` resolved one generation deeper to R8A1, which does expose `frozen`. Copying the same textual depth into R8A4 shifted the resolved module from R8A1 to R8A2.

## Filesystem and device boundary

After the one invocation, all six R8A4 paths were absent:

- committed output;
- inherited backend failure root;
- inherited backend quarantine root;
- wrapper failure root;
- wrapper quarantine root;
- independent verifier result.

No matching R8A4 in-progress path existed. This agrees with the source ordering: failure occurs before `configure()` and before any output/failure writer used by the delegated lifecycle. There is consequently no retained runtime telemetry. The zero-device conclusion is a deterministic source-order inference, not a device counter measurement: the failing attribute access precedes every path capable of loading OpenCL or opening the Intel device.

R8A4 must remain immutable. Its attempt is consumed and must not be retried or reclassified as a negative mechanism result.

## Minimal fresh R8A5 repair

Use explicit frozen-module bindings instead of another relative-depth chain. At minimum bind, by exact module name and SHA:

1. the frozen historical/auth contract module (`run_het_next_l0_ph1_intel_execution_r8a.py`);
2. the frozen wrapper lifecycle module (`run_het_next_l0_ph1_intel_execution_r8a1.py`);
3. the frozen physical R7A module, either directly or through a statically asserted identity with the historical module's `physical` binding.

Then:

- call historical gates and `physical.authorize()` through the explicit historical/physical binding;
- redirect fresh R8A5 paths through the explicitly bound lifecycle module;
- call the explicitly bound lifecycle `execute(auth)`;
- prohibit `.prior`, `.frozen`, or similar ancestry-depth traversal in R8A5 authorization, configuration, and execution paths.

Required pre-execution gates for R8A5:

- bind the immutable R8A4 runner, preregistration, lock, R8A4 GO audit, and this diagnosis by SHA-256;
- record R8A4 as exactly one silent pre-device infrastructure failure and require every R8A4 runtime path still absent;
- use a fresh namespace/token and one-attempt flag; never invoke the R8A4 runner again;
- AST/call-graph gate the exact explicit module names, target functions, and absence of ancestry-depth attribute chains;
- run a no-device sentinel that resolves the explicit modules and checks the target functions/physical module identity without authorization, payload, compiler, OpenCL, or device work;
- retain the complete R8A4 terminal adjudicator and 31-case matrix unchanged;
- add a bounded outer infrastructure-failure writer after invocation/clean authorization so an analogous early exception cannot again disappear without canonical evidence;
- freeze and independently audit R8A5 before its single physical attempt.

No scientific claim changes. The latest positive evidence remains the CPU freeze and compile eligibility; R8A4 proves nothing about Intel execution correctness.
