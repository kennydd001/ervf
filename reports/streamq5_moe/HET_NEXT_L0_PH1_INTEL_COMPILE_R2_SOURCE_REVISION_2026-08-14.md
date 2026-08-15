# PH1 Intel compile-only R2 — source diagnosis and revision

Datum: 2026-08-14  
Status: **source/design only; execution closed; no preflight, compiler or device call authorized**.

## Observed R1B outcome

The single authorized R1B compile-only attempt ended as a valid negative with OpenCL build code `-11`. Its immutable failure evidence is:

- `reports/streamq5_moe/het_next_l0_ph1_intel_compile_r1b_failed_attempts/attempt_failure_06df3c72c9c44379a04d39b43d301b53/failure.json`;
- 17,162 bytes;
- SHA-256 `62107b4cee0809fd744bacfe5d6890c7e09ec9002b0b029a6e84c98359f95fbb`.

The raw build log is 564 bytes, SHA-256 `91383f7935630334a5e0d250c01951645a1ce50c4dfe81aaef7f881529d2df2e`. It reports:

1. an unknown-extension warning for `cl_intel_required_subgroup_size` at source line 4;
2. a hard parser error at line 24 because `half` is treated as the OpenCL scalar type keyword, followed by the two expected cascading parse errors.

The attempt retained exact Intel identity, driver `32.0.101.8517`, PCI `0000:00:02.0`, build options, source and extension inventory. Program and context release both returned zero. Payload, queue, kernel, event, memory-object, allocation and launch counts remained zero. This is a source compile failure, not a lifecycle or device-selection failure.

## Frozen R2 byte changes

R2 derives the exact R1 source SHA-256 `06be3a9ba863d5f01d4025dc8d5e5679cdcc9827c13db5663c149227f3254528` and applies only:

1. rename local variable `half` to `halfway`, including its two uses in the round-to-nearest-even predicate;
2. remove `#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable`.

The R2 source is exactly 7,852 UTF-8 bytes, SHA-256
`f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21`.

No entrypoint, signature, integer operation, tie rule, reduction order, buffer layout, attribute, launch geometry or build option changes.

## Why the pragma is removed

The observed device extension list does contain the string `cl_intel_required_subgroup_size`, but the actual Intel OpenCL-C compiler explicitly reports that name as an unknown `OPENCL EXTENSION` pragma and ignores it. The subgroup size is already requested independently at each relevant kernel by the compiler-specific function attribute `__attribute__((intel_reqd_sub_group_size(8)))`. The plain `cl_intel_subgroups` pragma remains and its extension is present in the observed list.

Consequently R2 removes only the warning-producing enable pragma and retains all three required-subgroup-size attributes. A future compile result must still fail closed if those attributes are rejected, ignored, or do not produce a nonempty program binary. R2 does not infer attribute support merely from the R1 parser reaching later lines.

## Next gate

Before any further compiler/device call, an independent static audit must confirm:

- exact R1 failure-evidence and build-log bindings;
- exact two-change derivation and R2 source hash/byte count;
- `half` absent as an identifier, `halfway` declaration/use count exact;
- warning pragma absent, `cl_intel_subgroups` pragma and three subgroup-size attributes retained;
- all previous entrypoint, geometry, BF16 emulator/mutation, compile-only callgraph, nonempty-binary and immutable lifecycle gates remain applicable.

Only after a new closed static preflight and a separate authorization revision may exactly one R2 compile-only attempt be considered. No payload, correctness, timing or performance claim is made.
