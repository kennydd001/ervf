# E100/S100 KIMI Synthesis — 2026-08-16

Branch: `pro-e100-batch` only. No writes to main / pro-research. No thresholds or arithmetic changed.
HEAD at session start: `587db1dcace49c40a191155bf2f43732f3d22953`.
Final HEAD: see `git log` (commits listed below).

## Commands executed

| Fase | Command | Result |
|---|---|---|
| 0 | `pro_research/PUSH_E100_RESULTS.ps1` (smoke backup) | pushed `587db1d` |
| 0 | `git fetch origin && git pull --ff-only origin pro-e100-batch` | up to date |
| 1 | `pro_research/RUN_E100_PREFLIGHT.ps1` | passed=true, 21 files compile, CPU selftest 0/500 mismatches |
| 2 | `pro_research/RUN_E100_NVFP4_TILED_MRHS.ps1 -Mode full` | ran N=4/8/16 + verifier, pushed `22a4536` |
| 3 | `pro_research/RUN_E100_MRHS256.ps1 -Mode full` | ran N=4/8/16 + verifier, pushed `6247ce8` |
| 4 | `pro_research/RUN_S100_METADATA.ps1` | inventory + K0 budget, pushed `02e5980` |
| 5 | `pro_research/RUN_S100_KVERIFY_K1.ps1` | rollback_exact + verifier, pushed `458725d` |

## Raw result paths

- `pro_research/results/e100_nvfp4_tiled_mrhs/PRO_E100_NVFP4_TILED_MRHS.json` (+ `_VERIFICATION.json`, full console log `..._full_20260816T152535Z.console.log`)
- `pro_research/results/e100_mrhs256/PRO_E100_MRHS256.json` (+ `_VERIFICATION.json`, full log `E100_MRHS256_full_20260816T152652Z.console.log`)
- `pro_research/results/s100_mtp_inventory/PRO_S100_MTP_INVENTORY.json`
- `pro_research/results/s100_kverify/PRO_S100_KVERIFY_K0_STATE_BUDGET.json`
- `pro_research/results/s100_kverify/PRO_S100_KVERIFY_K1_MAMBA_ROLLBACK.json` (+ `_VERIFICATION.json`, log `S100_K1_20260816T152757Z.console.log`)

## Results table (component level, no end-to-end claim, no product of speedups)

| mechanism | N | exact | weighted speedup | max ref drift | gate | verdict |
|---|---|---|---|---|---|---|
| MRHS32 | 4 | yes | ~0.943x | — | fail | negative (prior) |
| MRHS256 V3 | 4 | yes | 0.973x | 0.356 | fail | negative |
| MRHS256 V3 | 8 | yes | 1.038x | 0.197 | fail | ~parity |
| MRHS256 V3 | 16 | yes | 1.036x | 0.221 | fail | ~parity, gates fail |
| TILED-MRHS V5 | 4 | yes | 0.568x | 0.144 | fail | negative |
| TILED-MRHS V5 | 8 | yes | 0.653x | 0.322 | fail | negative |
| TILED-MRHS V5 | 16 | yes | 0.687x | 0.181 | fail | negative |
| PairBatch | — | yes | ~1.05x | — | fail (<1.08x) | positive but sub-gate (prior) |

MRHS256 N16 per family: lm_head 0.863x, mamba_in 1.134x, mamba_out 1.270x, min case 0.690x. All 8 families exact; all performance gates false.
TILED-MRHS N16 per family: shared_up 0.652x, shared_down 0.654x, lm_head 0.731x, min case 0.652x. All exact; all performance gates false.
Independent verifiers recomputed identical status (`micro_null`) with `passed=true` for both.

## S100 KVERIFY K1 (correctness, no throughput claim)

- Replay prefix j=0..4: conv and SSM state bitexact at every prefix (0 mismatches).
- Sabotage control (omit transition index 1): diverged as required (conv 12288, ssm 524288 mismatches).
- Gates `all_prefix_states_bit_exact` + `sabotage_diverged` true; independent verifier `passed=true`, status `rollback_exact`.

## S100 metadata (technical feasibility only)

- 270 MTP tensors, 2,670,652,160 bytes (2.487 GiB), all in shard 52.
- Official flat indices 0 and 1, contiguous from zero, matching `mtp.layers.{idx}.*`; enorm/hnorm/eh_proj/final_layernorm all present (1 each).
- config: `num_nextn_predict_layers=1`, `mtp_layers_block_type=["attention","moe"]`; all name-alignment gates true.
- Rollback/state budget: one Mamba snapshot (48.16 MiB) + K stored in_proj outputs: K=2 → 49.96 MiB, K=4 → 51.77 MiB, K=8 → 55.39 MiB (vs naive K snapshots 96–385 MiB). Feasible memory-wise; no speed claim.

## Answers to the preregistered synthesis questions

1. MRHS256 scaling N4→N8→N16: roughly flat (~0.97→1.04→1.04), does not scale up meaningfully.
2. TILED-MRHS scaling N4→N8→N16: improves (0.57→0.65→0.69) but stays far below 1x.
3. No common-weight family reaches ≥2x at N16 (best: mamba_out 1.27x).
4. LM-head at N16 is not >1x (0.86x MRHS256, 0.73x TILED).
5. An integrated batch graph is NOT scientifically justified by these gates.
6. Recommendation: close current MRHS geometries as falsified/null at their preregistered gates.

## Technical failures vs scientific negatives

- Technical failure (previously documented, unchanged): whole-row SMEM MRHS v4 shared-memory design — audit trail preserved, no re-run.
- Scientific negatives (new, verified): MRHS256 N=8/16 and TILED-MRHS N=4/8/16 full scale curves — bitexact but below all performance gates.
- Positive correctness result: S100 K1 rollback exactness proof.

## Not tested / open

- No full-model aggregate E100 integration built (gates failed; awaiting owner decision).
- S100 KVERIFY beyond K1 (K2/K4/K8 rollback proofs) not run.
- MRHS32 N=16 not preregistered/tested; PairBatch left as-is per instruction.

## Next-step recommendation (no implementation)

Owner decision required: either close the MRHS geometry line, or preregister a new architecture experiment as a separate arm. No post-hoc variant was built in this session.
