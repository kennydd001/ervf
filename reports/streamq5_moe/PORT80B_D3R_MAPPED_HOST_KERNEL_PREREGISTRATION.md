# PORT80B-D3R — mapped-host one-kernel replication preregistration

The original D3 attempt produced no correctness or timing result. NVRTC stopped
before kernel execution because the generated source included `<stdint.h>` in a
runtime configuration without standard headers. Its JSON/report remain
preserved as a compile-failure artefact.

D3R repeats the already frozen D3 protocol unchanged. The only source repair is
mechanical: remove `<stdint.h>` and replace the `uintptr_t` cast by a direct
`unsigned long long` to pointer cast. Geometry, registration prefix, schedules,
tokens, sample counts, selection rule, gates and claim boundary are identical
to `PORT80B_D3_MAPPED_HOST_KERNEL_PREREGISTRATION.md`.

No result file may be overwritten and no timing-dependent parameter may be
changed after this repair.
