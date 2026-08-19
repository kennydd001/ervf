# S100 Lightning Phase 16R — agent handoff
Source evidence: `agent/s100-lightning-phase16-hardware` at commit `3c0418f`.


## Why this recovery exists

Phase 16's stream-race result, block-verifier result and DFlash2 result are
valid. Its selective-native final verdict is not final.

The runner used a helper parameter named `$Name` and also a caller variable
named `$Name`. PowerShell dynamic scope replaced candidate names such as
`tc1_k` with the step label `FULL CALIBRATION tc1_k`. The quality selector then
looked for a different filename and recorded `quality: null`.

Measured Phase-16 calibration evidence already shows:

- TC1 K: strict pass;
- TC1 V: strict pass;
- TC2 K+V: global quality high, but a factual-English domain-CE failure;
- TC1 all-K/V greedy: official pass, but strict domain-top1 failure.

Do not loosen those gates. Recover validation/heldout first.

## Required result

Run `RUN_S100_LIGHTNING_PHASE16R.ps1`. Return:

- `S100_LIGHTNING16R_RECOVERY.json`;
- `S100_LIGHTNING16R_SUBSET_SEARCH.json`;
- all `S100_LIGHTNING16R_QUALITY_*` files;
- `S100_LIGHTNING16R_THROUGHPUT.json`;
- `S100_LIGHTNING16R_SUMMARY.json` and `.txt`.

A candidate is promotable only if:

1. calibration strict pass;
2. validation strict pass;
3. heldout official pass with deterministic repeat;
4. end-to-end teacher-forced throughput is >=1.03x the production graph parent.

No Block-ERVF or DFlash2 rerun is requested.
