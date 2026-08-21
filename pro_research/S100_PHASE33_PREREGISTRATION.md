# S100 Phase 33 preregistration — exact warp32 NVFP4 M8 reuse

Frozen after Phase32 adjudication and before any Phase33 compile or timing.

## Hypothesis

Phase32 still launches two direct-L2 M4 passes for every shared-expert matrix
and for the LM head. An eight-row NVFP4 kernel can load and decode every weight
once while preserving the production 256-logical-thread accumulation and
reduction order.

The kernel maps one output row to one physical 32-lane warp. Each lane owns the
eight original logical-thread accumulators `lane + 32*w`. A 32-lane shuffle
reconstructs each original warp sum; the final eight sums use the same
offset-4/2/1 tree as production. This changes physical scheduling only, not the
per-token arithmetic.

## Frozen arms

- `phase32_control`: exact Phase32 `dense_m8`.
- `head_m8`: Phase32 plus one warp32 M8 LM-head pass.
- `shared_m8`: Phase32 plus one warp32 M8 pass for shared UP and DOWN.
- `shared_head_m8`: both changes.
- Production parent: two Phase31 `attention_head_m4` H4 launches.

All routed-UP, sparse DOWN, Mamba, attention, cache and state logic remain
unchanged.

## Gates and protocol

- Compile before runtime. Any nonzero local-memory bytes rejects the affected
  arm.
- Context 1024 screen, fresh process per arm, 4 warm-up + 8 measured windows.
- Fastest arm must have exact eight-token identity and deterministic replay.
- Full state thresholds are identical to Phase32.
- Thermal promotion compares the fastest state-green arm directly with the
  production Phase31×2 parent in four alternating fresh-process rounds.
- Adoption requires median round and paired gains >=5%, >=3/4 positive rounds,
  robust CV <=5%, exact state and zero spills.
- Context 128/4096 only after thermal adoption.

`<=80 ms/H8` is still the zero-draft perfect-acceptance S100 gate. Component
speedups and the non-adopted Phase32 control cannot support a throughput claim.
