# Nemotron N1 — pinned safetensors-header inventory preregistration

Read only the safetensors headers of the five files listed by the official
NVIDIA NVFP4 index at pinned commit
`ce1b118ae66ec705d02c241525192832eb045fd3`. Do not download tensor payloads.

Reconstruct every tensor's dtype, shape, shard and exact stored bytes from data
offsets. Partition names into routed experts (`.experts.<id>.`), shared experts
(`.shared_experts.`), and trunk/other. Require non-overlapping offsets, agreement
with the index keyset, 23 routed-MoE layers × 128 experts, equal routed-record
sizes, and total tensor bytes equal index metadata. Report the exact all-cold
top-6 routed bytes/token and its 26.158915-GB/s transfer floor.

This is metadata/capacity evidence only: no tensor payload, kernel, quality or
performance measurement.
