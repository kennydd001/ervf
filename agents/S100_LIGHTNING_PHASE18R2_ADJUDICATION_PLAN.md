# S100 Lightning Phase 18R2 — E reproduction adjudication

Date: 2026-08-20
Target: exact NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 parent.

## Why R2 exists

Phase 18R proved 10/10 exact dual-graph E parity but compared its validation/128/exact-cache-reset measurement directly against Phase 17's calibration/64/persistent-cache measurement. The published Phase-17 source confirms these are different protocols.

Verified Phase-17 facts:
- workload comes from `s100_lightning16r_throughput.frozen_workload()`;
- that function loads `load_trace("calibration")`;
- 64 target positions, first 8 untimed => 560 timed positions;
- `run_pass()` calls `rt.reset()` per prompt, so `_dev_cache` survives;
- pass order A -> recorder -> O1 -> overhead -> B -> optional O2;
- historical E used O2; its estimator uses alpha oracle=.75, overhead=.5.

Phase 18R instead used validation/128, `_reset_exact_state()` and a different five-role estimator. Therefore the historical +/-0.75 ms gate was not a valid same-protocol reproduction gate.

## Frozen R2 tests

1. Exact same-day rerun of the published Phase-17 E source, changing only output directory and forcing the historical O2 estimator.
2. Dual-graph calibration/64 with the Phase-17 table footprint, same pass order and same stats helper; production parent graph always constructs the prompt; `rt.reset()` preserves expert-cache history.
3. Identical dual-graph calibration/64 but `_reset_exact_state()` clears expert cache each prompt.
4. Adjudicate historical drift, same-day legacy-vs-dual agreement and cache-policy effect.

## Release gate

`SURGERY_RELEASED = true` only when:
- dual parity is green;
- same-day dual CAL64 persistent and exact Phase-17 repeat agree within 0.75 ms;
- matched dual E lower95 >= 8.0 ms.

Historical Phase-17 drift is reported separately. No quality gate changes and no surgery runs occur before this gate.
