# S100 Phase55 — Ornith DFlash target-ubatch isolation

## Question

Can single-row or bounded physical target microbatches restore byte-exact greedy
output for the reproducible Phase54 quantized draft-model divergence?

## Frozen setup

- Same exact GGUF pair, llama.cpp build, arithmetic prompt, strict greedy request,
  10 target GPU layers, fully GPU-resident DFlash, fresh process and disabled
  prompt cache as Phase54.
- One target-only default-ubatch reference.
- DFlash K=1 at the minimum valid target microbatch.
- DFlash K=8 across the minimum-valid through default target microbatches.
- Every cell uses a fresh process. Cold throughput is diagnostic only.

## Gates and adjudication

1. All eight cells load and produce non-empty output.
2. The target-only reference matches the stable Phase54 baseline text.
3. Every DFlash cell exposes positive acceptance evidence.
4. Exact output is evaluated independently for each DFlash cell; it is not an
   all-cells pass requirement.

- If the minimum K=8 microbatch matches the baseline and a larger microbatch does not, target
  multi-row verification geometry is isolated and the largest matching ubatch
  is the lossless operating boundary.
- If minimum-ubatch K=1 and K=8 both diverge, physical target batching alone
  cannot explain or mitigate the issue; hidden-state extraction or another
  draft-model graph-path difference remains.

## Technical amendment before valid rerun

The first execution established that build 10549 rejects DFlash `ubatch=1`
before inference with `GGML_ASSERT(n_ubatch > n_keep_tail)`. That result is
archived as a technical failure and carries no model evidence. The valid rerun
therefore freezes `ubatch=2` as the implementation's minimum and uses K=1 at 2
plus K=8 at 2, 3, 4, 8, 16, and 512.

## Second technical amendment before boundary rerun

The attempted `ubatch=2` rerun hit the same pre-inference assertion, disproving
the assumed minimum. It too is archived without model adjudication. The final
boundary sweep tests powers of two 4 through 512. At each size it starts the
target-only control first; the paired K=8 cell runs only if that control is
valid. A cell-level technical failure is recorded instead of aborting later
sizes. This maps both the model/runtime minimum and same-ubatch speculative
exactness without post-hoc deletion of invalid cells.
