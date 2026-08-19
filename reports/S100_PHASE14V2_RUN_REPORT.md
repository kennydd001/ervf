# S100 Phase 14 v2 — run report

Datum: 2026-08-19  
Pack SHA256: `4f50847c1e9fd7a29cc553f186dfd3b3115415525574aebaf1f304da0ba7e684`  
Branch: `agent/s100-phase14v2-hardware`  
Checkpoint: cached five-shard Phase-12C checkpoint

## Execution integrity

The downloaded all-in-one ZIP matched its published SHA256 and every file in
`MANIFEST_SHA256.json` matched after extraction. The first all-in-one pass
fail-soft pushed an incomplete evidence commit because D2/K2 referenced a
missing `diag_s100_d4_weight_only_dense.py` dependency and the worktree had no
Phase-3/Phase-4 quality traces.

The missing dependency was recovered unchanged from the existing S100 reboot
worktree. D2 was additionally isolated from the unrelated mapped expert bank;
the native BF16 component needs resident matrices only. K2 was rerun with a
direct teacher-forced real-logit harness because the supplied trace loader
could not find the preregistered traces.

## D2 — native BF16

The component test measured 37 real BF16 matrices, with a cold cache-scrub
protocol and the independent ERVF kernel as baseline:

| Block | Native speedup | Max case NRMSE | Mean row-argmax agreement | Component gate |
|---:|---:|---:|---:|---:|
| B=1 | 1.307x | 0.00267 | 1.0000 | pass |
| B=2 | 1.894x | 0.00303 | 1.0000 | pass |
| B=4 | 2.814x | 0.00301 | 0.9797 | pass |
| B=8 | 6.000x | 0.00295 | 0.9899 | pass |

The component result is strong. Official validation and heldout quality did
not run: the worktree lacks the required Phase-3 full trace and Phase-4 QFAST
run. Therefore D2 does not open a runtime build gate and does not make a model
quality claim.

## N2 — native NVFP4 / SM120

The authoritative C2D cold M-scaling rerun completed on an NVIDIA RTX PRO 2000
Blackwell Laptop GPU, compute capability 12.0, with 32 MiB L2:

- group-16 packed format: green;
- lossless repack evidence: green;
- native FP4 free-M gate: false;
- real-weight C3 runtime build: closed.

The already-NVFP4 shape screen reported M4 per-token speedups versus ERVF of
about 9.63x for `lm_head`, 3.96x for `shared_up`, 4.61x for `shared_down`, and
3.79x for `routed_up`. These are native FP4 shape microbenchmarks, not a full
model or quality result. N2 correctly refused to authorize real-weight C3.

## K2 — real witness and margin

The repaired direct harness used 10 calibration and 10 validation prompts,
16 teacher-forced tokens per prompt. Exact parent logits generated the prefixes
and targets; native BF16 candidate logits were then produced on the same
prefixes. No synthetic logit noise was used.

Validation results over 160 tokens:

- candidate top-1 agreement with exact parent: `0.00625`;
- K=8 exact-winner inclusion: `0.0625`;
- K=16 exact-winner inclusion: `0.1000`;
- K=32 exact-winner inclusion: `0.1500`;
- K=64 exact-winner inclusion: `0.2000`;
- no calibration-frozen margin gate selected;
- real K=16 witness gate: false;
- real margin gate: false.

This is the decisive correction to synthetic 13K: native BF16 changes upstream
hidden/state values enough that an exact lm-head witness cannot certify the
omitted computation. The witness mechanism itself is valid, but the current
native candidate is not a safe approximation path.

## Combined adjudication

The final combined summary is:

```text
NATIVE_BF16_B1_DIRECT_OPEN: None
NATIVE_BF16_BLOCK_BUILD_OPEN: None
NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN: False
REAL_K16_SHORTLIST_GREEN: False
REAL_MARGIN_GATE_GREEN: False
NEXT_ROUTE: REPAIR_INCOMPLETE_EVIDENCE
S100 SINGLE ACHIEVED: False
```

The correct conclusion is not that native BF16 is useless. It is a strong
component primitive, especially at B=4/B=8. The missing quality traces prevent
opening D2 as a runtime, while K2 shows that direct substitution needs a
state/logit containment strategy before exact witnessing can make it safe.
Native NVFP4 remains technically interesting, but N2 did not authorize a
real-weight C3 build.
