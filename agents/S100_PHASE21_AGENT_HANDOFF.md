# S100 Phase 21 handoff

Frozen evidence:
- Phase20S target math + independent layer oracle green.
- Phase20B full H4 token/state correctness green.
- Phase20B route union median 16 unique experts / 24 slots, repeat rate ~1/3.
- Existing exact Lightning V6: 21.0923 ms/token = 47.41 tok/s.
- Existing exact-snapshot V18: 19.5729 ms/token = 51.09 tok/s.

Hypothesis:
20B regressed because its Python grouped expert scheduler removed device routing,
CUDA graph launch amortization, selective ERVF, and the mature V6/V18 MoE
dataplane. Test that hypothesis before inventing more approximate kernels.

Arms:
- current_grouped: exact Phase20B behavior, cap48.
- selective_grouped: Phase20B grouped path + cap72/nonuniform + frozen
  selective-ERVF.
- v6_device_rows: H4 Mamba block; each MoE layer processes four rows through
  the proven device-resident V6 batched MoE path. No host route readback.
- v18_device_rows: same, plus H-SCALE resident scale planes and double-buffered
  gather overlap if VRAM permits.

Device-row arms deliberately do NOT reuse one expert across multiple H4 rows.
They isolate the regression and create a graphable, host-free H4 parent.
