# Agent prompt — SweepSpec-50 and LightningFlash-50

Open two isolated registries:

```text
LIGHTNINGFLASH_50
SWEEPSPEC_Q38_50
```

Read all historical project reports before execution. Preserve every closed
CRAFT/RSIV/CORETAIL/HERA/STREAMQ5/PORT80B decision.

## Program 1 — LIGHTNINGFLASH_50

Target the existing frozen Nemotron runtime.

Do not alter target weights, routes or quality semantics.

### L0 — baseline

Reproduce AR latency at short, 32K, 128K and 256K context.

### L1 — verifier primitive

Implement Bole-style exact tree verification for the hybrid recurrent and
full-attention stack. Verify state-factor reconstruction and accepted-state
commit.

### L2 — draft baselines

Measure:

- native MTP if present;
- prompt lookup/ngram;
- EAGLE/JetSpec-compatible head;
- DFlash block 16.

### L3 — forest

Combine:

- deep reliable ngram spine;
- causal parallel tree head;
- D²SD recovery at predicted first rejection;
- cost-aware tree budget from the measured verifier curve.

### L4 — overlap

Add Saguaro-style concurrent next-round drafting.

### Gates

```text
lossless target output
4K >=50 tok/s
128K >=40 tok/s
256K >=30 tok/s
mean accepted depth >=4
```

Do not claim 256K/50 until physically measured.

## Program 2 — SWEEPSPEC_Q38_50

Qwen3.8 is dense. No expert-cache mechanism is relevant.

### Q0 — target verification scaling

Build exact hybrid tree verification and measure proposal-node counts:

```text
1,4,8,16,32,64,128,256,512,1024
```

Separate:

- weight sweep;
- Gated DeltaNet tree state;
- full-attention tree mask;
- FFN compute;
- LM head;
- activation memory.

Gate:

```text
256 nodes <=1.5x one-token target pass
```

Close the 50-tok/s line if this fails badly.

### Q1 — native methods

Measure ordinary AR, native MTP, ngram and combinations.

### Q2 — substitute shadow

Build a training-free low-bit substitute draft from offloaded target layers,
following SubSpec/CATS principles. The metric is accepted-branch recall, not
standalone CE.

### Q3 — parallel draft

Train/port:

- DFlash;
- JetSpec or Domino causal parallel head;
- D²SD recovery drafter.

Initialize compatible Qwen3.6 heads where shapes match, but distill only from
the frozen Qwen3.8 target.

### Q4 — offload-aware forest optimization

Fit:

```text
T_verify(n) = T_sweep + n*T_marginal
```

Use a LibraSpec/CaDDTree-style marginal rule with the measured offload cost,
not a server-GPU cost model.

Hard target:

```text
mean accepted tokens >=18
preferred >=22
round latency / accepted token <=20 ms
```

### Q5 — asynchronous pipeline

Overlap draft and verification using Saguaro/SSD. Test CPU, NPU and a separate
GPU stream as draft locations. Do not assume the NPU is useful without a
microbenchmark.

### Q6 — exact LM-head pruning

For greedy mode, implement a cluster-bound exact MIPS verifier. It may skip
LM-head rows only under a mathematical upper-bound certificate. Any ambiguity
falls back to a full scan.

### Q7 — CPU/GPU cooperative target

Benchmark row-partitioned tree GEMM where CPU and GPU operate concurrently and
merge activations. Compare with full GPU streaming.

## Integrity

- Every phase is preregistered.
- Validation selects; test opens once.
- No paper speedups may be multiplied.
- Report acceptance length, tree nodes, draft time, target time and accepted
  tokens per weight sweep separately.
- Output must be lossless relative to the frozen target distribution.
- The 50-tok/s claim requires 512-token rollouts and a one-hour thermal run.
