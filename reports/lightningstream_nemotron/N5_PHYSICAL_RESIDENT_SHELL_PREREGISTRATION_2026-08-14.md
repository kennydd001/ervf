# N5 — physical resident shell

Registry: LIGHTNINGSTREAM_NEMOTRON · Phase: `N5_PHYSICAL_RESIDENT_SHELL` (H4)
Datum: 2026-08-14
Status bij schrijven: **design frozen, execution not yet authorized**
Depends on: `N4_R2_CAUSAL_OVERLAP` (PASS, 34/34)
Protected baseline: root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`

## 1. Vraag

Does everything that is **not** a routed expert physically fit on this 8 GiB GPU,
alongside the 774,533,280-byte expert staging buffer that N4-R2 measured, with
real allocations rather than a byte projection?

Every prior memory statement in this line has been arithmetic. N5 allocates.

## 2. Wat resident moet zijn

Per the assignment, GPU-resident: trunk/other weights, all shared experts,
routers and norms, Mamba states, FP8 KV, expert staging, and embedding/LM head
**when physically justified**. Host-resident: only the routed NVFP4 experts.

Frozen input from the N2 inventory:

| item | bytes | note |
|---|---:|---|
| trunk / other | 2,558,227,600 | includes embeddings and `lm_head` |
| shared experts | 258,177,392 | 23 × 11,225,104, stored NVFP4 |
| embeddings (BF16) | 704,643,072 | subset of trunk |
| `lm_head` (BF16) | 704,643,072 | subset of trunk |
| expert staging (N4-R2) | 774,533,280 | one token, 138 records |
| Mamba state (N3 projection) | 49,364,992 | constant in context |
| KV FP8 @ 4,096 | 12,582,912 | 3,072 B/token |
| KV FP8 @ 131,072 | 402,653,184 | |

NVFP4 trunk and shared experts stay **stored**, not dequantised: the N4-R2
kernel consumes packed codes directly, so dequantising them on device would add
memory for no benefit. Norms, routers, `A_log`/`D`/`dt_bias`, `conv1d` and the
six attention layers are BF16/FP32 as stored and are not re-encoded.

## 3. Ablation: embedding and LM head placement

These two are 1,409,286,144 B together, 65% of all BF16 bytes, and they have
opposite runtime profiles:

- **embedding** — one row gathered per token, 5,376 B of the 704,643,072 touched;
- **`lm_head`** — a full 131,072 × 2,688 matvec every token.

Both placements are measured as a declared ablation:

| variant | embedding | lm_head |
|---|---|---|
| A | device | device |
| B | **host** | device |
| C | host | host |

The variant chosen for later phases must be justified by measurement, not
preference. **Precision is not changed in any variant.** If nothing fits, the
result is a reported maximum, not a re-quantised model.

## 4. Frozen gates

| # | gate | threshold |
|---|---|---|
| S1 | every non-routed tensor physically allocated on device; summed bytes equal the N2 inventory exactly | exact |
| S2 | **peak device usage, measured by `cuMemGetInfo` delta** | **<= 8.0 GiB** |
| S3 | process peak working set and commit | <= 32 GiB |
| S4 | shell coexists with the 774,533,280 B expert staging buffer | required |
| S5 | KV + Mamba state allocated and touched at 4,096 and 131,072 context | required |
| S6 | no tensor re-quantised or precision-reduced to achieve fit | required |
| S7 | free device memory after full allocation | **>= 256 MiB** |
| S8 | no protected byte changed | required |

S2 is measured as `free_before - free_after` from the driver, not from an
allocator's own bookkeeping, so CUDA context overhead and allocator
fragmentation are inside the number rather than outside it.

## 5. Meetprotocol

- Allocations are **touched**, not merely reserved: every buffer is written to,
  so a lazily-backed reservation cannot masquerade as a fit.
- `cuMemGetInfo` sampled before the context, after the context, after each
  allocation group, and after teardown.
- CUDA context overhead reported separately, since it is real and is otherwise
  invisible in allocator statistics.
- Process working set, peak working set, commit and page-fault counters sampled
  via the Windows API before and after.
- Every buffer's byte count is reconciled against the N2 tensor inventory; a
  mismatch is a failure of S1, not a rounding note.
- Teardown is verified: free memory must return to within 64 MiB of its
  pre-allocation value, or the phase reports a leak.

## 6. Non-interference

The corrected N4 rule applies unchanged: blocked when another PID holds a CUDA
context per `nvidia-smi --query-compute-apps`, or device memory in use exceeds
256 MiB; fails closed on query error. This phase allocates most of an 8 GiB GPU,
so proceeding while the protected line is on device would be actively harmful —
the check is not a formality here.

## 7. Stop rules

- Shell does not fit at 4K → report the actual maximum context and the actual
  deficit. Do **not** reduce precision, and do not silently drop a component.
- Teardown leaks → report it; a shell that cannot be released is not a usable shell.
- Allocation succeeds but touching fails → report as a failure, not a fit.

## 8. Claim boundary

N5 may claim only: which components physically fit simultaneously on this
specific GPU with real touched allocations, and how much headroom remains. It
may **not** claim tokens per second, full-model latency, quality, that a full
runtime exists, or that any context length is achievable in practice — KV
allocation is necessary for long context, not sufficient.

## 9. Artefacten

| path | kind |
|---|---|
| `scripts/lightningstream_nemotron/n5_resident_shell.py` | runner |
| `scripts/lightningstream_nemotron/n5_independent_verify.py` | independent verifier |
| `reports/lightningstream_nemotron/n5_resident_shell.json` | raw result |
| `reports/lightningstream_nemotron/n5_independent_verification.json` | verifier output |
| `reports/lightningstream_nemotron/N5_PHYSICAL_RESIDENT_SHELL_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n5_input_lock.json` | input lock |
| `reports/lightningstream_nemotron/protected_verification_after_n5.json` | protected check |
