# PORT80B-D4R — native CUDA batch-copy alias repair

D4 produced no correctness or timing result: its first native batch operation
poisoned the CUDA context with `cudaErrorIllegalAddress`. The capability probe
had already established `canUseHostPointerForRegisteredMem = 0` and that every
registered range exposes a distinct nonzero `devicePointer` alias.

D4R repeats the frozen D4 protocol and gates unchanged with one mechanically
required pointer repair: ordinary `cudaMemcpyAsync` retains CPU host pointers,
but native batch descriptor sources use `pointerGetAttributes(...).devicePointer`
plus the in-range expert offset. All aliases are checked nonzero. New result and
report paths are used; D4 artefacts remain preserved.

If this repair still returns an illegal address or any nonzero native status,
the native batch route is closed as incompatible on this local CUDA/WDDM path.
