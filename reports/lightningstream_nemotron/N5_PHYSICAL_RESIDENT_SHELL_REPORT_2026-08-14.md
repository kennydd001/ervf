# N5 — physical resident shell

Datum: 2026-08-14
Verdict: **PASS. The complete non-routed shell, expert staging, Mamba state and 128K FP8 KV fit in 4.966 GiB with 2.99 GiB free — with embeddings and LM head both on device.**
Terminal state: `n5_resident_shell_fits`
Independent verification: **32/32**

## Kernresultaat

Every prior memory statement in this line was arithmetic. N5 allocates and
touches. Measurement is the `cuMemGetInfo` driver delta, so CUDA context
overhead and allocator fragmentation are inside the number, not outside it.

| variant | embedding | lm_head | uploaded | peak device | free |
|---|---|---|---:|---:|---:|
| **A** | device | device | 2,816,404,992 B | **4.9664 GiB** | **3065.0 MiB** |
| B | host | device | 2,111,761,920 B | 4.9664 GiB | 3737.0 MiB |
| C | host | host | 1,407,118,848 B | 4.9664 GiB | 4409.0 MiB |

All three fit. Held simultaneously in every variant: the shell weights, the
774,533,280-byte expert staging buffer measured in N4-R2, 49,364,992 B of Mamba
state, and 402,653,184 B of FP8 KV at **131,072** context.

**The answer to H4's "embedding/LM head when physically justified" is: both are
justified on device.** There is 2.99 GiB of headroom at full 128K KV with
everything resident, so neither needs to be exiled to the host.

## Partition

| bucket | tensors | bytes | placement |
|---|---:|---:|---|
| routed NVFP4 experts | 23,552 | 16,523,376,640 | **host** |
| shell (trunk/other + shared) | 595 | 2,816,404,992 | device |
| — of which embeddings | 1 | 704,643,072 | |
| — of which `lm_head` | 1 | 704,643,072 | |
| total | 24,147 | 19,339,781,632 | |

The shell figure reconciles exactly with the frozen N2 inventory:
`2,558,227,600` trunk/other + `258,177,392` shared = `2,816,404,992`.

NVFP4 trunk and shared experts are stored **packed**, not dequantised: the
N4-R2 kernel consumes packed codes directly, so dequantising on device would
cost memory for nothing. No tensor was re-quantised or precision-reduced to
achieve the fit (S6).

## Device accounting

```text
device total                8,546,484,224 B
free without a context      7,385,120,768 B
driver reserve                1,161,363,456 B   (total - free, not ours)
CUDA context overhead             2,097,152 B   (measured separately)
variant A peak device       5,332,402,176 B  (4.9664 GiB)
variant A free after all    3,214,082,048 B  (3065.0 MiB)
```

The 1,161,363,456-byte driver reserve is worth naming: it is present before any
allocation of ours and matches the figure the protected D10 audit records for
this GPU. Budgeting against `total` rather than `free` would have overstated
available memory by 1.08 GiB.

## Exactheid

The sharpest consistency test is that moving a tensor to the host must save
**exactly** its byte count:

| delta | measured | expected | match |
|---|---:|---:|:--:|
| A − B uploaded | 704,643,072 B | embeddings | ✅ |
| B − C uploaded | 704,643,072 B | `lm_head` | ✅ |

and `variant A uploaded == full shell bytes`, `variant C host-resident ==
embeddings + lm_head`. All confirmed by the independent verifier against a
fresh read of the checkpoint headers.

Teardown: **leak 0 B**. Free memory returned exactly to its pre-allocation
value. A shell that cannot be released is not a usable shell, so this was gated
rather than assumed.

Process footprint: peak commit **6.0105 GiB** against a 32 GiB gate — the whole
routed bank is *not* host-resident in this phase, only the staging working set,
so this figure is a floor for the eventual runtime, not its final value.

## Meetprotocol

- Every buffer is **written to**, not merely reserved, so a lazily-backed reservation cannot masquerade as a fit.
- `cuMemGetInfo` sampled before the context, after the context, after weights, after runtime buffers, after each KV context, and after teardown.
- KV allocated and touched at both **4,096** (12,582,912 B) and **131,072** (402,653,184 B); the larger is held for the headroom figure.
- Byte counts reconciled against the N2 inventory; a mismatch would be an S1 failure, not a rounding note.
- Non-interference: 0 foreign CUDA contexts. This phase allocates most of an 8 GiB GPU, so the check is load-bearing here rather than a formality.

## Onafhankelijke verificatie

A separate verifier re-derived the routed/shell partition directly from the
checkpoint headers, checked it against the frozen N2 inventory, recomputed the
device accounting from the recorded free-memory samples, tested the variant
deltas against the real embedding and `lm_head` byte counts, and re-evaluated
every gate. It imports nothing from the runner and opens no GPU.

Result: **32/32 verification checks passed.**

## Gates

| # | gate | threshold | result |
|---|---|---|:--:|
| S1 | shell bytes reconcile with N2 | exact | ✅ |
| S2 | peak device | ≤ 8.0 GiB | ✅ 4.9664 |
| S3 | process peak commit | ≤ 32 GiB | ✅ 6.0105 |
| S4 | shell coexists with expert staging | required | ✅ |
| S5 | KV + Mamba allocated at 4K and 128K | required | ✅ |
| S6 | no precision reduction | required | ✅ |
| S7 | free after full allocation | ≥ 256 MiB | ✅ 3065 MiB |
| S8 | teardown clean | ≤ 64 MiB | ✅ 0 B |

Two process corrections, both recorded:

1. The first run reported a **1,558,183,936 B teardown leak**. Cause: clearing
   the `held` list did not drop the loop's local references, so the pool could
   not reclaim the blocks. It was a defect in the runner, not in the shell, and
   the fix brought the leak to exactly 0.
2. `S3` initially read `peak_commit_bytes` as **0** because the psapi entry
   point filled only the base struct, which made the gate vacuously true. It
   now uses `K32GetProcessMemoryInfo`, validates the counters, and **fails
   closed** when the footprint cannot be measured. An unmeasurable gate is not
   a passing gate.

## Eerlijk verdict

What N5 establishes: with real touched allocations on this specific GPU, the
entire non-routed model — trunk, all shared experts, routers, norms, embeddings
and LM head — plus one token's expert staging, the full Mamba state and 128K of
FP8 KV occupy **4.966 GiB**, leaving **2.99 GiB** free. Teardown is clean and no
precision was changed.

What N5 does **not** establish: tokens per second, full-model latency, quality,
or that 128K context is achievable in practice. **KV allocation is necessary for
long context, not sufficient** — nothing here exercises attention over 131,072
positions, and the Mamba state figure is still the N3 projection rather than a
state produced by a real forward.

One judgement worth stating plainly: the runner names **variant C** as "best"
because it minimises device bytes, and that label is misleading. The right
engineering choice is **variant A**. Memory-minimal is not design-optimal when
there is headroom: A keeps the LM head's per-token matvec and the embedding
gather on device, avoiding a host round-trip per token, and it still leaves
2.99 GiB. C would trade 1.34 GiB of unused headroom for two host transfers on
the critical path.

## Wat dit betekent voor de cache

H5 now has a measured budget rather than an estimate: **3,214,082,048 B** free
with everything resident at 128K. At 5,612,560 B per routed expert that is
**572 expert slots**, or 19.4% of the 2,944-record bank — enough for a
meaningful static+dynamic policy, and the first time this line can size a cache
against measured VRAM instead of arithmetic.

That number assumes variant A and 128K KV. At 4K context the KV term drops by
390,070,272 B, buying roughly 69 further slots.

## Artefacten

- Preregistratie: `reports/lightningstream_nemotron/N5_PHYSICAL_RESIDENT_SHELL_PREREGISTRATION_2026-08-14.md`
- Runner: `scripts/lightningstream_nemotron/n5_resident_shell.py`
- Machine-readable result: `reports/lightningstream_nemotron/n5_resident_shell.json`
- Independent verifier: `scripts/lightningstream_nemotron/n5_independent_verify.py`
- Verification output: `reports/lightningstream_nemotron/n5_independent_verification.json`
- Input lock: `reports/lightningstream_nemotron/n5_input_lock.json`
- Protected-80B check: `reports/lightningstream_nemotron/protected_verification_after_n5.json`
