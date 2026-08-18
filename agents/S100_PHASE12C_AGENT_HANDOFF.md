# S100 Phase 12C agent handoff

## Parent evidence

- Phase 12A proved exact B=2/4/8 block semantics, including token ids, Mamba
  state, convolution state, KV bytes and final logits.
- Ordinary M=1 kernels cost 35.56/70.99/143.90 ms for B=2/4/8.
- Phase 12B found median routed device-weight read reductions of 29.2% at B=4
  and 43.8% at B=8.
- PCIe miss bytes did not fall because the current LRU already deduplicates
  temporal misses.
- Mean rows per unique expert is shallow: 1.43 at B=4 and 1.76 at B=8.

## What this pack decides

### Dense ERVF-M

One block loads each weight element once and updates B independent exact ERVF
reduction trees. It covers all live parent matrices that can be safely
enumerated:

- Mamba FP8 in/out;
- attention Q in QFAST NVFP4;
- attention K/V/O BF16;
- router F32;
- shared-expert NVFP4;
- lm_head NVFP4.

Every output must be bit-identical to B independent current ERVF calls.

### Grouped MoE

The routed-up kernel reuses one expert matrix over M activation rows.

The routed-down kernel reads one panel-major expert record and one H-SCALE plane
for M activation rows. For each row it reconstructs exactly the current sorted
panel list, chunk assignment, mask-bit order, FMA sequence and partial reduction.

The measured M distribution from Phase 12B weights the final B=4/B=8 result.

## Frozen gates

Dense useful-row throughput:

- B=2 >=1.75x;
- B=4 >=3.20x;
- B=8 >=5.50x.

Grouped MoE:

- all tested M values exact;
- each up/down stream rotates >=4x L2;
- weighted B=4 up+down speedup >=1.20x;
- candidate M=1 penalty <=15%.

Integration opens only when dense B=4 and grouped B=4 both pass.

## Claim boundary

This is a microkernel/economics phase. The next phase must build the layer-major
perfect-draft target verifier. No projected cycle time counts as achieved tok/s.
