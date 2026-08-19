# S100 Phase 14 preregistration — survivor plus DFlash2

## Frozen target

- exact checkpoint: `models/nemotron_3_5_lightning`;
- current quality parent: QFAST + `alpha=0.0003`;
- single-stream decode;
- 8 GiB VRAM, maximum 32 GiB RAM;
- S100 = at least 100 accepted output tokens/s end to end.

No Qwen, Muse or other target may replace the frozen checkpoint. Published
DFlash2 results are prior art, not local measurements.

## Closed tracks

13A entropy and 13C temporal delta are not rerun. Their tested hypotheses remain
closed.

## 14D — native BF16

1. Extend B={2,4,8} native BF16 timing to every live BF16 projection.
2. Require real-weight rotation >4x L2.
3. At B=4 require aggregate useful-row speedup >=2.5x, max case NRMSE <=0.005
   and mean row-argmax agreement >=0.97.
4. Replace eager BF16 GEMVs with native BF16 matmul for numerical validation.
5. Run `_02` strict validation.
6. Read `_03/_04` heldout only after strict validation passes.
7. Require original official quality gates, deterministic repeat and finite
   output.

The eager native run is a fidelity test, not a B=4 speed result.

## 14B2 — output-aware subspace

Capture real calibration and validation X->Y pairs from representative
Mamba layers and every attention layer.

Fit no-intercept reduced-rank regression on calibration only:

```text
Y ~= X T C
```

Ranks: 32/64/128/192/256/384.

Per-case pass:

- physical BF16-factor bytes save >=35%;
- validation output NRMSE <=0.03;
- mean cosine >=0.9995;
- p95 relative row error <=0.08.

A family opens only when >=80% representative cases pass.

## 14E2 — decoded expert basis

Decode actual NVFP4 expert values on early/middle/late MoE layers. Fit
activation-weighted expert-axis ranks 4/8/16/32. Quantize shared bases back to
NVFP4 CEIL and add frozen expert residual blocks of 6.25/12.5/25%.

Open only if every sampled layer has a candidate with:

- byte ratio <=0.70;
- validation sampled-GEMV NRMSE <=0.05;
- cosine >=0.999.

## 14F0 — DFlash2 hard economics

DFlash2 is a trained target-specific sidecar. No full drafter may be trained or
ported until the verifier and memory gates are measured.

For each measured full-block verifier and B in {2,4,8}:

```text
throughput = 1000 * accepted_tokens / (verify_ms + draft_ms)
S100 draft budget = 10 * accepted_tokens - verify_ms
```

Hard gate:

```text
CURRENT_VERIFIER_PERFECT_DRAFT_S100_OPEN
  = any measured full verifier with 1000 * B / verify_ms >= 100
```

Projections are reported but cannot open this measured gate.

Memory screen uses actual free VRAM after building the quality parent. The
reference DFlash2 shape uses block 8, conv kernel 2, conv group 16, selector
rank 256 and top-K 16. Evaluate 2–5 layers, MLP ratios 2.0/3.0/3.75 and
BF16/FP8/NVFP4 storage. Apply an 8% conservative parameter calibration against
the public Qwen3.8 total. Include 12% workspace, 512 MiB reserve and draft-KV
at 4K/32K/128K context.

The final resident-memory gate uses the scaled public reference shape: five
layers, MLP ratio 3.75 and NVFP4 weights, with BF16 draft-KV. Smaller candidates
are reported separately but cannot open the reference gate. FP8 draft-KV is
reported as hypothetical until a compatible non-causal implementation exists.

## 14F1 — DFlash2 transfer proxy

Use the same ten `_01` calibration and ten `_02` validation prompts. Generate
72 greedy target states per prompt and select 32 deterministic anchor windows.
Block size is frozen at 8: one anchor plus seven future slots.

### Base drafter proxy

Fit a calibration-only low-rank map from anchor final hidden to seven future
final-hidden states. Ranks 64/128/192. Select rank on a deterministic internal
calibration split only.

### Suffix-correction proxy

Apply one grouped dynamic predecessor tap after all seven base positions have
been computed. Group sizes 64/128. Dynamic coefficients are predicted from the
anchor latent. Clip coefficient magnitude at 2.0.

Suffix signal passes if:

- first-slot top-16 recall loses no more than 0.01; and
- last-three hidden NRMSE improves >=10%, or last-three top-16 recall improves
  >=0.02.

### Candidate lattice

Run the real target LM head over base and corrected predicted hidden states.
Keep top-16 candidates per slot.

Headroom passes if:

- oracle candidate-lattice acceptance including anchor >=3.0; and
- oracle headroom over independent top-1 drafting >=0.75 token.

### Selector proxy

Use frozen rank-32 projections of token embeddings and predicted hidden states.
Select one transition weight from
`{0,0.25,0.5,1,2,4,8,16}` on calibration and run dynamic programming on
validation. This lower-capacity proxy is diagnostic and does not by itself
close DFlash2.

Transfer signal opens only when suffix correction and candidate-lattice
headroom both pass.

## Final DFlash2 train gate

```text
DFLASH2_TRAINING_BUILD_OPEN
  = CURRENT_VERIFIER_PERFECT_DRAFT_S100_OPEN
    AND RESIDENT_DRAFTER_MEMORY_OPEN_4K_BF16_KV
    AND DFLASH2_NEMOTRON_TRANSFER_SIGNAL_OPEN
```

This authorizes a later training phase only. It is not a runtime or S100 claim.

## Tri-state policy

All final flags use:

- `true`: technically complete evidence passes;
- `false`: technically complete evidence fails;
- `null`: missing or technically failed evidence.

A technical failure never becomes a negative scientific result.
