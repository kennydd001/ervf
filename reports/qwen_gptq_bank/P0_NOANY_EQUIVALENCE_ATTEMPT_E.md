# Pure GPTQ no-`torch.any` equivalence — attempt E lock

Locked after attempt D failed and before attempt-E outputs. This implementation copies the pinned symmetric 2-bit MSE loop operation-for-operation, including the original mask-index assignments. It removes only the host-synchronizing `if torch.any(mask)` guards and executes the identical indexed writes unconditionally; empty-mask writes are no-ops. All other GPTQ operations are unchanged. The same 30 matrices and zero-mismatch rule apply. Failure forces the unmodified official fallback.
