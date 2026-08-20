# S100 Lightning Phase 17 — oracle atlas verdict

Date: 2026-08-20
Target: NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4, same frozen quality parent.

## Hard result

The locally completed Phase-17 atlas measured all 20/20 scheduled arms without technical errors. The exact-replay oracle set proves that an architecture-preserving S100 path exists in the measured component set.

Key replay-overhead-corrected ceilings:

- complete MoE `E`: 10.6786 ms/token saving, prompt-clustered 95% interval 10.3763–10.9417 ms;
- complete attention `A`: 1.7167 ms/token, 1.6739–1.7558 ms;
- LM head `L`: 1.3300 ms/token, 1.3175–1.3443 ms;
- combined `M_IO+E+A+L`: 18.7665 ms/token, 18.5339–18.9736 ms.

The combined point leaves about 1.35 ms/token; the conservative bound leaves about 1.58 ms/token. The summary flag is therefore:

`S100_PATH_EXISTS_IN_MEASURED_SET = true`.

All non-LM oracle arms preserve token ids, final-logit hash, hidden state, recurrent state and KV state. LM-head-containing arms are explicitly teacher-forced upper bounds and preserve final probe logits/state.

## Primary conclusion

MoE is the dominant lever. `E` alone removes about 10.68 ms/token from a roughly 19.39 ms reference and therefore covers more than the full 10 ms target gap as an oracle ceiling. This does **not** mean a production MoE implementation can simply reach zero cost; it means the next engineering campaign must explain and attack the MoE critical path before building more small kernels elsewhere.

Attention is a secondary target at ~1.72 ms. LM head is also material at ~1.33 ms but its oracle is teacher-forced and therefore only an upper bound on optimizable free-running cost.

The measured `E+A` combination is super-additive by +0.6487 ms versus the isolated corrected savings. This proves that isolated component timing can hide critical-path overlap.

## Mamba caveat

The three Mamba groups were measured, but the preregistered additivity check failed. Per-group sums for M_IN/M_OUT/M_IO are therefore not valid standalone total-Mamba point estimates. Direct combination arms that actually ran remain valid point measurements, but Mamba rollup must not be inferred by summing the three groups until the additivity problem is resolved.

## Next phase

Phase 18 is MoE surgery. Required first experiment: an exact replay sub-oracle and layer atlas that splits the current 23-layer MoE implementation into routing, shared expert, routed-up/fetch, threshold scan, routed-down/fetch/reduction and accumulation. It must also measure interactions and per-layer ceilings. No production kernel earns implementation work until its exact oracle ceiling is material.

## Publication note

The local Phase-17 atlas implementation and full result JSON/TXT were reported as untracked after completion. They must be staged and pushed to `agent/s100-lightning-phase17-oracle-atlas` without rewriting or dropping raw compact evidence. This verdict file is only the remote summary and is not a substitute for those local artifacts.