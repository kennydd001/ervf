# S100 Phase 17 preregistration

## Target

Same cached Nemotron 3.5 Lightning checkpoint and production kernel stack.

Sample Mamba layers:
- first
- middle
- last

H = 2, 4, 8.

## Correctness

For every layer/H:

SSM:
- final state NRMSE <= 5e-5
- all y NRMSE <= 5e-5

Conv:
- final conv-state NRMSE <= 5e-6
- all conv outputs NRMSE <= 5e-6

Core:
- final SSM/conv state <= 5e-5
- all gated-norm outputs <= 5e-5

Full layer:
- all Mamba layer outputs <= 1e-4
- final states <= 5e-5

## Timing

CUDA-event timing. All allocations and host/device copies outside the timed
region. State reset is outside both baseline and candidate events.

Gates:

SSM H4:
- candidate speedup >= 1.50x on every sampled layer.

Core H4:
- speedup >= 1.35x on every sampled layer.

Full Mamba layer H4:
- speedup >= 1.10x on every sampled layer.

The faster of prefix and fused-serial SSM candidates is selected independently
per H, but the selected implementation must satisfy correctness.

## Claims

`PHASE18_FULL_BLOCK_VERIFIER_OPEN=true` only if full-layer H4 is correctness
green and >=1.10x on every sampled layer.

This remains a perfect-input layer ceiling, not end-to-end 100 tok/s.
