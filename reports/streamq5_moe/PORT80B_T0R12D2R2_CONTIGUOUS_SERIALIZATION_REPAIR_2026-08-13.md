# T0-R12-D2-R2 contiguous serialization repair

The first D2-R execution completed the full diagnostic compute but stopped before producing an artifact because some retained final-position tensor views were non-contiguous and `safetensors.save_file` rejects such views. This revision changes only serialization input materialization from `save_file(raw, path)` to `save_file({k: v.contiguous() for k, v in raw.items()}, path)`.

No model call, input, metric, schema, threshold, interpretation or claim changes. New runner, lock and output paths are mandatory. Diagnostic-only; no pass, GPU, Q5 or bank claim.
