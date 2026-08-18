# S100 Phase 13L — persistent compressed-layer pipeline prototype

This prototype executes the intended sequence on GPU tensors: `Uᵀx`, `WU`
projection, residual norm gate, and exact fallback. It measures eager PyTorch
operators on a real BF16 Mamba matrix at B=4 and B=8.

It deliberately does not claim a persistent CUDA kernel. The gated prototype
computes both candidate and fallback paths, so its timing is a safety/overhead
measurement, not a production speed result. Promotion remains closed.

With a rank-256 prototype basis, the subspace-only path was about 6.96x
faster at B=4, but its output NRMSE was 0.95 on the measured random-basis
screen. The gated path computed both branches and was slower than exact
(0.82x at B=4 and 0.85x at B=8). This demonstrates the control-flow shape,
not a viable persistent engine.
