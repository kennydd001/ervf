# S100 Phase 11 one-click pack

Frozen pack: `ervf_s100_phase11_dense_byte_compiler.zip`

SHA256: `7a4dec6a0365770f23fb5918192e50cad73919fe9c5f9dcc71bc28999f506bd8`

Contents:
- Track A: NVFP4 CEIL Mamba main weights + exact FP8 rescue rows at 3.125%, 6.25%, 12.5%, 25%.
- Track B: structural Mamba channel-slimming quality-feasibility at retained fractions 93.75%, 87.5%, 75%, 62.5%.
- Frozen validation/heldout split; Track A fresh A/C/C/B timing only after heldout green.
- ERVF/H-SCALE and the Phase-9 current-map parent remain intact.

Track B masked runtime is explicitly not a performance claim; heldout green only opens the physical shape/state compiler.
