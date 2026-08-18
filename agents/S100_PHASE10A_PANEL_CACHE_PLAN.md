# S100 Phase 10A — exact sparse routed-down panel cache

Parent: completed Phase-9 current-map QFAST + alpha=0.0003. ERVF and H-SCALE
remain intact. This tests a distinct hypothesis from the closed routed-up cache:
cache only hot `(layer, expert, panel)` down-projection FP4 code blocks.

Each cached panel stores only 21,504 code bytes. Scales remain in H-SCALE.
Selection is frozen on calibration by avoided sparse PCIe bytes:
`popcount(panel_mask) * 1344`.

Budgets: 8/16/24/32/40/48 MiB. Validation reports byte coverage only.
Promotion requires exact smoke parity, destructive control divergence, then
fresh-process full A/C/C/B with >=765 samples, <=1 ms drift, <=7.8 GiB VRAM
and >=0.15 ms/token end-to-end gain.
