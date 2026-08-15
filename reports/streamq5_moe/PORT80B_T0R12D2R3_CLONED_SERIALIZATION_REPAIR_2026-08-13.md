# T0-R12-D2-R3 cloned serialization repair

The D2-R2 execution completed all 70 unchanged official CPU forwards and all metrics, but `safetensors` rejected four top-10-ID tensors that shared storage with their top-11 source tensors. `contiguous()` was insufficient because these length-one views were already contiguous.

R3 changes exactly one operation at the serialization boundary: every retained tensor is materialized as `v.detach().clone().contiguous()` immediately before `save_file`. Inputs, official model calls, captures, arithmetic, schema, metrics, interpretation and thresholds are unchanged. The R2 output directory was verified empty and removed; R3 uses a new create-new output path and new provenance locks.

Diagnostic-only: no pass verdict, GPU, Q5, bank, performance or 80B deployment claim.
