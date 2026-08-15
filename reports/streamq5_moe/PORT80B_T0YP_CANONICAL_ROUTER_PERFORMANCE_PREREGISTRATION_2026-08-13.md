# PORT80B-T0Y-P canonical router performance preregistration

Date: 2026-08-13

T0Y-R already proved exact CPU/CUDA FP32 logits and ordered IDs on 16 real layer-0 rows. This experiment measures whether its deliberately fixed reduction is a practical GPU primitive or only a correctness oracle.

## Frozen arms

- Reference: resident BF16 PyTorch CUDA `F.linear`, FP32 softmax, top-10, normalize and BF16 weight cast—the official router data path.
- Candidate: resident canonical T0Y-R FP32 logits kernel plus deterministic top-10 selector. The candidate does not compute selected probabilities, so timing is reported both raw and with an analytical warning; it can never claim full official-router parity.

## Protocol

- Same frozen `[16,2048]` hidden states and `[512,2048]` weights as T0Y-R.
- Reuse exact T0Y-R CUDA source; source/hash preflight before CUDA.
- Confirm candidate logits/IDs remain identical to T0Y-R archived tensors before timing.
- 30 warmups, 80 validation AB/BA pairs. Validation opens 240 test AB/BA pairs only if candidate/reference p50 ratio is at most 4.0 and p95 at most 5.0. Test performance pass requires p50 ratio at most 2.0 and p95 at most 2.5.
- Each sample uses CUDA events around only the resident operation and synchronizes the end event. Even pairs run reference then candidate; odd pairs candidate then reference.
- No retuning or alternative block shape after outputs open.

The exactness result remains valid regardless of timing verdict. A performance pass is a narrow resident-router-component result, not model throughput, quality or industrial-breakthrough evidence.
