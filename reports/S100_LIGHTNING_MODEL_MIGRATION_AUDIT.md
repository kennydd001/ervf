# S100 full repository migration audit — Nano to Nemotron 3.5 Lightning

Date: 2026-08-19  
Repository: `kennydd001/ervf`  
Frozen audit base: `agent/s100-phase18-projection-block@a802195`  
Remote branches inspected: **42**  
Experiment registry entries: **98**

## Executive verdict

The repository contains four distinct evidence eras:

1. **DeepSeek-V2-Lite history** — useful infrastructure and methodological
   lessons, never Lightning performance evidence.
2. **mixed PRO evidence** — some results record a `_v35` path, but the old
   campaign did not retain an immutable official model/revision fingerprint.
   These results are preserved under `PROVENANCE_REVALIDATE`.
3. **Nano S100 campaign** — Phase 13 through Phase 18 is confirmed by path,
   campaign lineage or the explicit Phase-18 report to be Nano-based.
4. **hardware-only evidence** — RTX/SM120/PCIe/L2/Arc transfer and synthetic
   Tensor-Core contracts remain valid for the hardware/software snapshot.

The repository must not be reset to zero. It must also not keep model-specific
Nano closures as Lightning closures.

The migration branch therefore:

- freezes all 42 branch heads;
- records 98 experiment-level decisions;
- keeps hardware and generic harness evidence;
- reopens checkpoint-specific quality, routing, cache and economics;
- redesigns standard-MoE experiments for Lightning LatentMoE;
- makes native Lightning MTP the first multi-token baseline;
- refuses every GPU run until the model guard confirms the official
  checkpoint, revision, LatentMoE and MTP markers.

## Critical corrections

### A path is not provenance

The old paths `nemotron_3_5_lightning` and
`nemotron_3_5_lightning_v35` are not accepted as identity. Nano and Lightning
share the same broad 30B-A3B Nemotron-H chassis. Only the official acquisition
record plus architecture/tensor markers can confirm Lightning.

### V3–V6 are not discarded

V3, V4, V5 and V6 store `_v35` in their environment. They remain valuable but
are classified `PROVENANCE_REVALIDATE`, not automatically Lightning and not
Nano by default. Their source hashes and metadata are linked to the new
checkpoint where possible; otherwise those exact arms are rerun.

### Phase 12C reopens

The measured ERVF-M and grouped-MoE negative result remains valid for its old
parent. It does not close a Lightning LatentMoE verifier, especially when the
new target contains native MTP.

### Phase 16E/17 remain high-value

The affine recurrence algebra and CUDA scan implementation are strong generic
results. Their state ABI, numerical parity and complete-layer economics still
require a short true-Lightning rerun.

## Evidence classes

| Class | Treatment |
|---|---|
| Hardware-only | Keep; short sanity only after environment changes |
| Generic runtime/harness | Keep the fix and regression test |
| Claimed Lightning path, no fingerprint | Revalidate; rerun on mismatch |
| Confirmed Nano checkpoint | Preserve as Nano; reopen Lightning verdict |
| Standard-MoE architecture assumption | Redesign for LatentMoE |
| Technical failure/incomplete | Repair; never call it negative |
| Different model | Archive; port only by explicit new hypothesis |
| Lightning-only LatentMoE/MTP | New mandatory baseline |

## All remote branches

| Branch | HEAD | Legacy provenance | Migration action | Experiment family |
|---|---:|---|---|---|
| `agent/s100-phase6-direct-down` | `fc6093dc` | `nano_or_unverified` | `REDESIGN_RETEST` | S100 routed-down direct/static-record lineage |
| `agent/s100-phase6-v2-wave-downflow` | `89f35b86` | `nano_or_unverified` | `REDESIGN_RETEST` | wave/downflow overlap v2 |
| `agent/s100-phase6-wave-downflow` | `e87971f3` | `nano_or_unverified` | `REDESIGN_RETEST` | wave/downflow overlap |
| `agent/s100-phase7-arc140t` | `a5dc1261` | `mixed_hardware_and_model` | `KEEP_HARDWARE_RETEST_MODEL` | Arc 140T heterogeneous lab |
| `agent/s100-phase7-arc140t-plan` | `49164a6c` | `planning_only` | `ADAPT_PLAN` | Arc 140T experiment plan |
| `agent/s100-phase8-arc-downflow` | `2b335989` | `mixed_hardware_and_model` | `KEEP_HARDWARE_REDESIGN_MODEL` | Arc routed-down execution |
| `agent/s100-phase8-overnight` | `2ed9cb96` | `mixed_hardware_and_model` | `KEEP_HARDWARE_RETEST_MODEL` | Arc overnight plan/source |
| `agent/s100-phase8-overnight-hardware` | `a4d8d084` | `mixed_hardware_and_model` | `KEEP_HARDWARE_RETEST_MODEL` | Arc overnight hardware evidence |
| `agent/s100-phase9-cache-miss` | `04b5a6b3` | `nano_or_unverified` | `FULL_RETEST` | route trace/cache/miss campaign |
| `agent/s100-phase9-cache-miss-hardware` | `4b96a818` | `nano_or_unverified` | `FULL_RETEST` | cache/miss hardware run |
| `agent/s100-phase9-cache-miss-repair` | `4b96a818` | `nano_or_unverified` | `FULL_RETEST` | cache/miss repair lineage |
| `agent/s100-phase9-repair-hardware` | `7f2ee8eb` | `nano_or_unverified` | `FULL_RETEST` | repaired route/cache/directhost/Arc suite |
| `agent/s100-phase10-hypotheses` | `ca745801` | `nano_or_unverified` | `RETEST_AND_SANITY` | panel cache + Mamba bandwidth |
| `agent/s100-phase10-mamba-bandwidth` | `ed29d7fb` | `nano_or_unverified` | `SHAPE_SANITY_RETEST` | Mamba bandwidth variant |
| `agent/s100-phase11-dense-byte-compiler` | `4ed7c3d7` | `nano_or_unverified` | `FULL_RETEST` | dense byte compiler / Mamba slimming |
| `agent/s100-phase12-block-ervf` | `a5c1def3` | `nano_or_unverified` | `REDESIGN_HIGH_PRIORITY` | perfect-draft block verifier + route union |
| `agent/s100-phase12c-hardware` | `c2967d13` | `nano_or_unverified` | `REOPEN_REDESIGN` | ERVF-M + grouped MoE |
| `agent/s100-phase13-subspace-entropy` | `e264694e` | `confirmed_nano_path` | `FULL_RETEST_SELECTIVE` | 13A–13L discovery suite |
| `agent/s100-phase14-dflash2-hardware` | `7a62dac1` | `confirmed_nano_campaign` | `REPLACE_WITH_LIGHTNING_SPEC_BASELINE` | DFlash2 economics/correction/selector |
| `agent/s100-phase14-survivors` | `490d0656` | `confirmed_nano_campaign` | `SUPERSEDED_BY_MIGRATION` | Phase13 survivor plan |
| `agent/s100-phase14r-repair` | `f8dc42e5` | `confirmed_nano_campaign` | `KEEP_FIXES_RETEST` | Phase14 repair source |
| `agent/s100-phase14r-repair-hardware` | `751f5461` | `confirmed_nano_campaign` | `RETEST_MODEL_PARTS` | Phase14 repaired hardware results |
| `agent/s100-phase14v2-hardware` | `2caab262` | `confirmed_nano_campaign` | `RETEST_AND_REDESIGN` | native BF16/NVFP4/witness v2 |
| `agent/s100-phase15-bf16-fp32` | `9dd533d8` | `confirmed_nano_campaign` | `SHAPE_SANITY_RETEST` | BF16 FP32-output investigation |
| `agent/s100-phase15-native-bf16-fidelity` | `caf725ba` | `confirmed_nano_campaign` | `FULL_RETEST` | teacher-forced native BF16 fidelity |
| `agent/s100-phase16-localize-horizon-scan` | `2d54fc01` | `confirmed_nano_campaign` | `KEEP_REPAIR_HISTORY` | localization/horizon/scan initial |
| `agent/s100-phase16R-repair` | `5483aee1` | `confirmed_nano_campaign` | `HIGH_PRIORITY_SANITY_RETEST` | repaired local sensitivity + affine scan |
| `agent/s100-phase17-mamba-block-scan` | `7146e8be` | `confirmed_nano_campaign` | `HIGH_PRIORITY_RETEST` | Mamba block scan CUDA microkernel |
| `agent/s100-phase18-projection-block` | `a8021954` | `confirmed_nano_explicit` | `HIGH_PRIORITY_RETEST` | projection blocking |
| `main` | `0072f7ed` | `mixed_historical` | `ARCHIVE_AND_REUSE_INFRA` | historical application + DeepSeek research base |
| `pro-e100-batch` | `485eb0bc` | `nano_or_unverified` | `RETEST_AFTER_MTP` | aggregate/batch architecture |
| `pro-max-v2` | `22c6ba14` | `mixed_unverified` | `PROVENANCE_REVALIDATE` | exact fusions/graph/final-mile campaign |
| `pro-research` | `c839060b` | `mixed_unverified` | `PROVENANCE_REVALIDATE` | cumulative PRO research baseline |
| `pro-s100-dualrhs` | `68b1c8c2` | `nano_or_unverified` | `SHAPE_RETEST` | DualRHS exact multi-position kernels |
| `pro-s100-k2-oracle-v18` | `6ec346e0` | `nano_or_unverified` | `REUSE_SEMANTICS_RETEST_ECONOMICS` | K2 target-verifier oracle |
| `pro-s100-mtp` | `54e766f2` | `nano_or_unverified` | `SUPERSEDE_WITH_NATIVE_LIGHTNING_MTP` | custom MTP/K2 preregistration |
| `pro-s100-nativefp4-c2b` | `898791a7` | `hardware_synthetic_plus_nano_shapes` | `KEEP_HARDWARE_RETEST_SHAPES` | SM120 native FP4 contract/free-M |
| `pro-s100-nativefp4-c2` | `cf3ff3fd` | `hardware_synthetic` | `KEEP_HARDWARE` | native FP4 C2 synthetic hardware |
| `pro-s100-nativefp4-c3-realact` | `6bb00838` | `nano_or_unverified` | `FULL_RETEST` | native FP4 real weights/activations |
| `pro-s100-nativefp4` | `b50a486e` | `nano_checkpoint` | `RERUN_FORMAT_AUDIT` | C0/C1 format and repack |
| `pro-s100-splitk` | `ac4eb013` | `nano_or_unverified` | `SHAPE_RETEST` | attention K/V split-K |
| `pro-v12-async` | `ba132ad7` | `mixed_unverified` | `PROVENANCE_REVALIDATE` | queued/streamed host delivery |

## Retest waves

| Wave | Registry entries | Purpose |
|---:|---:|---|
| 0 | 16 | identity, checkpoint diff, runtime and provenance lock |
| 1 | 23 | true target baseline, LatentMoE, MTP, FP4 and Mamba |
| 2 | 21 | highest-value checkpoint-specific reruns |
| 3 | 15 | block/system/placement experiments |
| 4 | 15 | lower-priority discovery and adaptive mechanisms |
| 5 | 5 | only after profiling justifies them |
| 99 | 3 | archived/superseded evidence |

## Files added by this migration

- `BRANCH_LEDGER.json` — exact remote branch snapshot and branch-level action.
- `S100_LIGHTNING_EXPERIMENT_RETEST_MATRIX.md` — human-readable 98-entry matrix.
- `EXPERIMENT_RETEST_REGISTRY.json` — 98 experiment-level decisions.
- `MODEL_PROVENANCE_POLICY.md` — fail-closed identity policy.
- `model_guard.py` — official repo/revision + MTP + LatentMoE gate.
- `acquire_lightning.py` / `ACQUIRE_S100_LIGHTNING.ps1` — safe acquisition.
- `checkpoint_inventory.py` — config/tensor/shape/dtype/scale inventory.
- `compare_checkpoints.py` — Nano↔Lightning machine diff.
- `audit_all_remote_branches.py` — scans every remote ref without checkout.
- `audit_result_provenance.py` — classifies current result artifacts.
- `build_retest_plan.py` — generates dependency-ordered queue.
- `validate_registry.py` — requires 42/42 branch coverage and valid dependencies.
- `selftest_model_guard.py` — tests confirmed, deceptive, conflicting and mutable identities.
- `lightning_runtime_smoke.py` — fail-closed runtime support test.
- `RUN_ALL_S100_LIGHTNING_MIGRATION_AUDIT.ps1` — one-click audit.
- `RUN_S100_LIGHTNING_WAVE0.ps1` — first runtime smoke only after guard.
- `S100_LIGHTNING_MASTER_AGENT_HANDOFF.md` — agent execution contract.

## Immediate next command

Run the audit against the actual model directories:

```powershell
pwsh -ExecutionPolicy Bypass `
  -File .\RUN_ALL_S100_LIGHTNING_MIGRATION_AUDIT.ps1 `
  -LightningModelDir "C:\path\to\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4" `
  -NanoModelDir "C:\path\to\NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" `
  -InspectSafetensors
```

If the Lightning model is not yet present, add `-Acquire`. Acquisition defaults to the immutable NVIDIA-validated public revision `0dcd680e5585c791728c83342b311d0a0026dbeb`; override it only deliberately.

The runner stops before CUDA when identity is ambiguous. That is intentional.
