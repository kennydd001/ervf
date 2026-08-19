# Phase 14D2 preregistration

Target checkpoint: `models/nemotron_3_5_lightning`.

Parent: current QFAST + alpha=0.0003 Phase-9/10 runtime.

Component:
- enumerate all live BF16 Mamba/attention/lm-head projections;
- B = 1, 2, 4, 8;
- baseline = B independent current ERVF BF16 calls;
- candidate = native Torch/cuBLAS BF16 matmul;
- every timing call is preceded by >4x-L2 cache scrub;
- candidate output is compared against current ERVF output.

Component gates:
- B=1 useful speedup >=1.10x;
- B=4 useful speedup >=2.50x;
- max per-case NRMSE <=0.005;
- mean row-argmax agreement >=0.97;
- finite.

Quality:
- replace only BF16 GEMV calls in an eager copy of the same parent runtime;
- validation `_02` first;
- heldout `_03/_04` only if strict validation is green;
- original Phase-3 official gates and deterministic heldout repeat.

Final flags:
- NATIVE_BF16_B1_DIRECT_OPEN
- NATIVE_BF16_BLOCK_BUILD_OPEN

Neither flag is an S100 claim.
