# N4-R1 — fused NVFP4 expert kernel

Datum: 2026-08-14
Verdict: **F3 NEGATIVE by 2.421 ms. Decode repaired 25.66×; the mechanism is proven, the screen is not met.**
Terminal state: `n4r1_fused_screen_fail`
Independent verification: **28/28**

## Kernresultaat

The N4 diagnosis was correct and the repair works. Decode is no longer the wall.

| arm | p50 | p95 | vs N4 unfused |
|---|---:|---:|---:|
| transport only | 32.273 ms | 32.683 ms | — |
| **fused compute only** | **13.762 ms** | 13.991 ms | **25.66× faster** |
| composed token | 46.954 ms | **47.421 ms** | 8.01× faster |

N4's decode-only arm was 353.133 ms. The fused kernel does the same arithmetic in
13.762 ms, beating the ~15 ms target that N4 set for this phase. The composed
token nevertheless misses the 45 ms screen by **2.421 ms**, because transport
(32.273 ms) and compute (13.762 ms) run strictly serially in this implementation.

## Mechanism

The unfused path decoded a matrix into 9,977,856 float32 weights — about 40 MB
written then 40 MB read — to feed a GEMV that consumes each weight once. The
fused kernel assigns one block per output row, streams that row's packed bytes
and block scales from global memory, decodes in registers and accumulates
through a warp-shuffle tree.

Predicted per-matrix device traffic, recorded before measurement: ~120 MB
unfused versus **2,806,272 B** fused (2,494,464 codes + 311,808 scales). The
measured peak device pool is the strongest confirmation:

```text
peak pool          774,601,728 B
working set        774,533,280 B
difference             68,448 B
```

68,448 B above the bank itself. A single materialised `[1856, 2688]` float32
matrix would be 19,961,856 B, so F5 is settled structurally rather than by
assertion.

## Exactheid

| check | result |
|---|---|
| fused expert output vs N3 CPU reference | `rel_l2 = 1.654e-07`, gate 1e-5 |
| fused intermediate activation vs reference | `rel_l2 = 1.539e-07`, gate 1e-5 |
| all outputs finite | yes |

**Bit-exactness of the fused output is not claimed.** The kernel reduces with a
block-parallel warp-shuffle tree, not the sequential order of the numpy
reference; cross-backend bit identity is explicitly not demanded and claiming it
would be an overreach. The *decode* itself is unchanged — the same integer
unpack and table lookup that N4 proved bit-identical to the CPU float32 decode.

## Meetprotocol

- CuPy 14.1.1, NVRTC-compiled, compute capability 120. No host CUDA toolchain required.
- Working set 138 records × 5,612,560 B = 774,533,280 B, identical to N4 so the comparison is like for like.
- 5 warmup + 30 measured repetitions, fixed before results were opened; all raw arrays retained.
- Route weights from the frozen N3 official capture (synthetic-input routes; timing-neutral).
- Non-interference: 0 foreign CUDA contexts, checked with `nvidia-smi --query-compute-apps`.

Transport measured 32.273 ms here against 29.756 ms in N4. Same bytes, different
copy path: N4 used torch pinned `copy_`, this phase uses CuPy `.set()`. The
~8% difference is recorded, not explained away, and it is not what causes the
gate miss — even at N4's 29.756 ms the serial composed path would land at about
43.5 ms, inside the gate but with under 1.5 ms of margin.

## Onafhankelijke verificatie

A separate verifier recomputed every percentile from the retained arrays,
re-derived the byte accounting, independently recomputed the routed-expert
reference on the CPU, checked the peak-allocation argument against a
materialised-matrix hypothesis, and re-evaluated every gate. It imports nothing
from the runner and opens no GPU.

Result: **28/28 verification checks passed.**

## Gates

| # | gate | threshold | result |
|---|---|---|:--:|
| F1 | expert output vs reference | `rel_l2 ≤ 1e-5` | ✅ 1.654e-07 |
| F2 | activation vs reference | `rel_l2 ≤ 1e-5` | ✅ 1.539e-07 |
| F3 | composed token p95 | ≤ 45 ms | ❌ **47.421 ms** |
| F4 | peak device | ≤ 8.0 GiB | ✅ 0.721 GiB |
| F5 | no materialised matrix | required | ✅ +68,448 B |
| F6 | outputs finite | required | ✅ |
| F7 | protected bytes unchanged | required | ✅ |

The architectural stop is **not** triggered: its threshold is 60 ms and the
measurement is 47.421 ms. Its precondition — a correct fused kernel — is now
satisfied for the first time, so the stop is live from here on.

## Eerlijk verdict

What R1 establishes: the N4 diagnosis was right, the cause was an
implementation, and fusing decode into the GEMV removes 96.1% of the decode
cost while preserving the reference output to 1.65e-07 and adding 68,448 B of
device memory.

What R1 does not establish: the 45 ms screen. It misses by 2.421 ms.

The residual gap has a named, testable cause — transport and compute run
serially — and that is not a hypothesis about the architecture but about the
schedule. It is addressed in `N4-R2` under its own preregistration and its own
gates, with an added requirement R1 did not have: the overlapped path must be
**bit-identical** to the serial one. Widening the 45 ms gate was never an option.

## Artefacten

- Preregistratie: `reports/lightningstream_nemotron/N4R1_FUSED_NVFP4_KERNEL_PREREGISTRATION_2026-08-14.md`
- Kernels: `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py`
- Smoke test: `scripts/lightningstream_nemotron/n4r1_fused_smoke.py`
- Runner: `scripts/lightningstream_nemotron/n4r1_fused_dataplane.py`
- Machine-readable result: `reports/lightningstream_nemotron/n4r1_fused_dataplane.json`
- Independent verifier: `scripts/lightningstream_nemotron/n4r1_independent_verify.py`
- Verification output: `reports/lightningstream_nemotron/n4r1_independent_verification.json`
