# S100 Phase 13 discovery preregistration

## 13A — checkpoint entropy census

For every resident dense FP8/BF16 matrix and routed expert sample, report:

- symbol entropy globally and per tile sizes 128/256/512/1024 bytes;
- unique symbols per tile;
- ideal palette bit-rate including palette and escape overhead;
- exponent/mantissa split entropy;
- byte-run and delta entropy;
- aggregate encoded-byte estimates by matrix family.

No GPU kernel is built unless aggregate Mamba FP8 estimate is <=6.0 bits/weight or all dense resident traffic is <=70% of current bytes.

## 13B — activation subspace census

Collect calibration activations from `_01` prompts for every Mamba in input, Mamba out input, attention linear input, shared-expert input and final norm input.

For ranks {128,256,384,512,768,1024}, report on disjoint `_02` validation activations:

- activation residual energy;
- output reconstruction NRMSE for `W U U^T x`;
- top-1/top-5 token effects when substituted one family at a time;
- fraction of tokens meeting frozen residual gates;
- measured candidate bytes `W U + U` versus W.

Ranks/gates are frozen before heldout. No heldout access in discovery.

Open SR-ERVF implementation only if one rank gives >=35% projected dense-byte reduction with official validation green or a family-isolated candidate that can remove >=300 MB/token.

## 13C — temporal delta census

On the same activation traces, measure `x_t-x_{t-1}`:

- cosine and norm ratio;
- top-k coordinate energy for k={32,64,128,256,512};
- int8/int4 quantized residual error;
- output error under sparse-column `W delta`.

Open Delta-ERVF only if k<=256 retains >=99% output energy for at least 70% of validation tokens in a family responsible for >=300 MB/token.

## 13D — native tensor-core mini-prefill ceiling

Use real checkpoint matrices and B={2,4,8}. Native BF16/FP8/NVFP4 kernels may differ numerically from ERVF. Report cold-stream useful-row speedup, layer output error, routing agreement and validation fidelity.

Open candidate block integration only if B=4 dense speedup >=2.5x and official validation gates pass.

## 13E — expert shared-basis census

For representative early/middle/late MoE layers, perform joint expert-axis decomposition on deterministic sampled rows/blocks. Report reconstruction error and physical bytes for basis counts {4,8,16,32} and residual budgets {6.25%,12.5%,25%,50%}.

Open full expert compiler only if >=30% routed bytes can be removed at a reconstruction operating point that survives layer-output validation.

## Claim boundary

This phase discovers byte-reduction opportunities. It cannot claim S100. A missing or technical result is `incomplete`, never a no-go.
