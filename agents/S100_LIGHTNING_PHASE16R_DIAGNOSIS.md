# Phase 16 selective-native verdict correction

Source: `agent/s100-lightning-phase16-hardware`, commit `3c0418f`.

## Measured facts

Phase-16 calibration files contain:

- `FULL CALIBRATION tc1_k`: strict pass and official pass;
- `FULL CALIBRATION tc1_v`: strict pass and official pass;
- `FULL CALIBRATION tc2_kv`: global top1 is high, but factual-English
  mean CE exceeds both the strict and official domain ceiling;
- `FULL CALIBRATION tc1_greedy`: official pass, strict domain-top1 fail.

`S100_LIGHTNING16_QUALITY_SELECTION.json` nevertheless records `quality:
null` for the canonical candidates and selects none for validation.

## Root cause

The one-click runner used `$Name` for both the candidate and the local
`Run-Step` function parameter. The scriptblock executed in that dynamic
scope, so `--name tc1_k` became `--name "FULL CALIBRATION tc1_k"`.
Quality output filenames therefore included `FULL_CALIBRATION_`, while the
frozen selector constructed filenames from canonical names such as
`TC1_K`.

This is an orchestration bug. It does not invalidate the calibration
numbers, the stream-race fix, the block-verifier measurement, or DFlash2.

## Correct verdict before Phase 16R

- Block-verifier route: measured closed.
- DFlash2 route: measured closed.
- Selective-native route: **not adjudicated beyond calibration**.

Phase 16R repairs only that missing adjudication and keeps every frozen
quality threshold intact.
