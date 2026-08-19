# S100 Lightning migration toolkit

This directory is the checkpoint-provenance and retest control plane for the
ERVF research repository.

Start with:

```powershell
pwsh -ExecutionPolicy Bypass `
  -File .\RUN_ALL_S100_LIGHTNING_MIGRATION_AUDIT.ps1 `
  -LightningModelDir <path> `
  -NanoModelDir <path> `
  -InspectSafetensors
```

The audit is CPU/file-system work until the model guard is green. Wave 0 is a
separate explicit command.

## Integrity checks

Before model access or CUDA import, the runners execute:

- `validate_registry.py`: 42/42 branch coverage, unique experiment IDs, all dependency IDs resolved;
- `selftest_model_guard.py`: synthetic confirmed-Lightning, Nano, deceptive-path, identity-conflict and mutable-revision cases.

The default acquisition revision is NVIDIA's physically validated public revision `0dcd680e5585c791728c83342b311d0a0026dbeb`. `acquire_lightning.py` still records the resolved immutable revision returned by Hugging Face.
