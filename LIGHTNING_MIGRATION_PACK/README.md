# S100 Lightning migration source pack

This directory is a connector-safe transport wrapper for the complete migration tree.

Run from the repository root:

```powershell
pwsh -ExecutionPolicy Bypass -File .\LIGHTNING_MIGRATION_PACK\APPLY_PACK.ps1
```

The script verifies every base64 part and the reconstructed ZIP, then expands the actual 23 source/audit files into the worktree. It does not download model weights and does not execute CUDA.

After extraction, start with `reports/S100_LIGHTNING_MODEL_MIGRATION_AUDIT.md` and `RUN_ALL_S100_LIGHTNING_MIGRATION_AUDIT.ps1`.
