# HET-NEXT-L0-PH1-R2 — NVIDIA primary-context lifecycle

Date: 2026-08-14  
State: design-only repair; no implementation, compiler or device action authorized.

This document binds and supersedes only the NVIDIA context-lifecycle ambiguity in PH1-R1 physical contract SHA-256 `7097a304eb6cd082367472cbc4c84ff9792414f3dd67e2590ba55b61dac3e981`. The independent R1 design audit is `HET_NEXT_L0_PH1_R1_PHYSICAL_CONTRACT_INDEPENDENT_DESIGN_AUDIT_2026-08-14.md`, SHA-256 `cb295f83e5a49ebdccce9982342af4442fa4a1fbb3607d547a5a6804eaa97cfe`. No arithmetic, buffer, copy, launch, threshold, control, identity or claim changes.

The NVIDIA backend executes in a fresh dedicated child process and one owner OS thread. Before any context retain, module, stream, allocation or compile/load call:

1. `cuInit(0)` succeeds and exactly one eligible NVIDIA device is selected by the frozen identity policy.
2. `cuCtxGetCurrent(&prior)` succeeds and `prior == NULL`; a non-null prior context is `blocked_capability`, never overwritten.
3. `cuDevicePrimaryCtxGetState(device,&flags,&active)` is retained as diagnostic evidence only. No flag mutation or reset is permitted.
4. `cuDevicePrimaryCtxRetain(&owned,device)` succeeds and returns non-null. Exactly one successful retain is allowed.
5. `cuCtxPushCurrent_v2(owned)` succeeds on the owner thread. `cuCtxGetCurrent(&observed)` must then return the exact same pointer. All subsequent Driver API module/stream/allocation/copy/launch/sync calls occur on this thread while `observed == owned`.

Normal cleanup first attempts the existing 30 reverse releases (14 device allocations, 14 pinned allocations, module, stream), preserving every result even after an earlier release error. Then, regardless of those errors:

6. `cuCtxPopCurrent_v2(&popped)` is attempted exactly once; it must succeed and `popped == owned`.
7. `cuCtxGetCurrent(&restored)` is attempted and must succeed with `restored == prior == NULL`.
8. `cuDevicePrimaryCtxRelease_v2(device)` is attempted exactly once and last. `cuDevicePrimaryCtxReset`, `cuCtxDestroy`, another retain/release, or use of the context after release is forbidden.

If failure occurs after retain but before push, skip pop and attempt exactly one primary release. If failure occurs after push, attempt all owned resource releases, pop/restore, then primary release. If `cuCtxPopCurrent_v2` fails, record the error, query current for evidence, still attempt primary release, and the outcome is negative/invalid. The owner ledger separately contains: prior-current query; primary-state query; retain; push; post-push current query; pop; restored-current query; primary release. These eight context rows do not alter the existing 30 ordinary-release cardinality.

A clean success requires all eight context rows in exact order with expected pointer identities/return codes, the 30 ordinary releases, and zero live resources. The independent verifier must reject mutations of prior-current nullness, retained/pushed/popped pointer identity, row order, missing/duplicate retain/pop/release, restored-current non-null, any reset/destroy call, release-before-pop, and any context use after release.

With an independent GO on this addendum, PH1 standalone source implementation is authorized only. Implemented sources still require full audit and static no-device preflight before a separate one-attempt physical authorization.
