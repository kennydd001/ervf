# PORT80B-T0X CPU/CUDA router-portability preregistration

Date: 2026-08-13

## Question

Does the official Qwen3-Coder-Next layer-0 router select exactly the same top-10 experts and native-BF16 routing weights on the pinned CPU reference backend and this machine's CUDA backend for the already archived R6-D hidden states?

This is a diagnostic portability experiment. It cannot pass T0-R4, T0-R7, Q5 quality, physical 499+13 transport, full-model quality, or throughput.

## Frozen inputs

- Official revision: `a19358a7659bd1f564300250ee189120c49a562f`.
- Official shard 1 SHA-256: `8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a`.
- R6-D raw safetensors SHA-256: `42b9eb25748ce0722f7b3f7c5612069081314eae51b8741a18c39b17abcbdb72`.
- R6-D diagnostic JSON SHA-256: `fd35d86e0bc8679d614ac209dc3d3a679ec735307db7455e670e37947780f797`.
- Tensor `official_gate_input`, shape `[1,16,2048]`, native BF16.
- Tensor `model.layers.0.mlp.gate.weight`, shape `[512,2048]`, native BF16.
- Top-k is 10. Computation is exactly `F.linear`, FP32 softmax, `torch.topk`, FP32 renormalization, then cast selected weights to native BF16.

## Frozen protocol

1. Verify every input/source hash before CUDA initialization.
2. Load only the 2 MiB router weight and archived hidden state; no model forward and no bank build.
3. Recompute the CPU result and require exact equality to archived R6-D logits, IDs, and weights.
4. Execute the same graph twice on one CUDA device with deterministic algorithms enabled and synchronize each call.
5. Save CPU and both CUDA logits/probabilities/IDs/weights as raw safetensors plus a JSON row manifest.
6. Run the independent verifier without rerunning CUDA.
7. No retry, threshold change, alternative precision, expert-ID tie-break, or retuning after outputs open.

## Frozen adjudication

- `exact_cross_backend_pass` requires all 16 rows to have exact ordered CPU/CUDA IDs and bitwise-equal native-BF16 selected weights, both CUDA calls bitwise identical, and all tensors finite.
- Otherwise verdict is `cross_backend_negative`.
- CPU boundary ties are retained and reported but never convert a mismatch into a pass.
- CUDA repeatability is a separate pinned-device observation and is not cross-device portability.

## Resource and safety gates

- At least 1 GiB free VRAM before execution.
- Peak allocated CUDA memory must remain below 256 MiB.
- No host registration, no large allocation, no write larger than 5 MiB.
- No existing synthetic or checkpoint bank is mutated.

## Claim boundary

A pass would establish only exact route portability for 16 archived layer-0 rows on the pinned CPU/CUDA software and hardware. A negative establishes a concrete portability counterexample on this machine. Neither is an industrial-breakthrough claim.
