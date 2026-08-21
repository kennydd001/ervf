# S100 Phase 36 / C4 preregistration — native FP4 H8 LM head

Frozen after C3C passed every gate and before full-verifier timing.

## Candidate

Replace only Phase32's two exact direct-L2 M4 LM-head passes with:

1. C3C fused static NVFP4 activation quantization over all eight rows;
2. one native SM120 `scaled_mm` M8 LM-head;
3. BF16-to-FP32 conversion into the existing argmax buffer.

Checkpoint FP4 code bytes are shared zero-copy with the existing runtime.
Only the native blocked B-scale layout and tiny A/output workspaces are added.
All transformer layers, recurrent state, KV, routing and MoE arithmetic remain
the exact Phase32 path.

## Contract

This is a quality-contract arm, not a bitexact-logit arm. C3B retained 32/32
held-out LM-head top-1 values under the selected static scale, but the native
Tensor-Core accumulation and A quantization may change logits.

## Frozen screen

- Context 1024, 4 warm-up + 16 measured H8 windows.
- Production parent: two Phase31 exact H4 launches in a fresh process.
- Candidate reports top-1 agreement against all 128 canonical target IDs;
  mismatches are recorded, never hidden as runtime failures.
- Candidate must be graph-resident and finite.

Promotion requires >=3% screen gain, >=99% top-1 agreement, no saturation in
the static quantizer's represented range, and no state/KV changes before the
head. Thermal adoption and broader trajectory/corpus quality are separate.

No C4 result may claim exact verification or S100.
