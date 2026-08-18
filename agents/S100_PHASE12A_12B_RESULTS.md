# S100 phase 12A/12B results — block verifier correctness, cost floor, route-union census

Date: 2026-08-18 · Branch: agent/s100-phase12-block-ervf ·
Scripts: `pro_research/s100_phase12a_block_verifier.py`,
`pro_research/s100_phase12b_census.py`,
diagnostic `pro_research/diag_s100_phase12a_divergence.py` ·
Results: `pro_research/results/s100_phase12/`.

## 12A: perfect-draft block verifier

A B-token block graph was built by unrolling B decode bodies with the
existing bit-exact M=1 kernels (no weight sharing — that is 12C scope).
Drafts are the runtime's own greedy continuation, so acceptance is 100% by
construction and verified position-by-position. Cycle time includes draft
upload, one graph launch, harvest copy and sync; the comparator is B
sequential `step_graph` launches with the same sync cadence.

**Correctness (prereg decision step 1): PASS at every B.** Per B in {2,4,8},
2 prompts x 8 cycles: argmax identity 100%, Mamba ssm/conv state
fingerprints equal, KV bytes equal over exactly the written region, final
logits bit-exact. In-place per-body state updates reproduce sequential
decode exactly when acceptance is total — shadow-state machinery is only
needed once real drafters produce partial acceptance (12D).

Harness bug found en route (fixed): the first fingerprint hashed the KV
cache with the wrong layout (`[max_ctx, kv_dim]` instead of the real
`[n_kv, max_ctx, head_dim]`, see `kv_append_fp8_dp`) and included one
unwritten tail row; `reset()` never clears KV, so that row held
nondeterministic warmup garbage. Tokens and Mamba state matched all along —
the "divergence" was measurement-side. This is exactly why the diagnostic
compared every component per cycle before patching anything.

**Cost with ordinary kernels (honest negative, as predicted):**

| B | cycle ms (median) | sequential B tok ms | ratio | useful tok/s | gate | pass |
|---|---:|---:|---:|---:|---:|---|
| 2 | 35.56 (p10 34.00, p90 37.36) | 35.17 | 1.011 | 56.2 | <=18 | no |
| 4 | 70.99 (p10 67.98, p90 74.84) | 71.15 | 0.998 | 56.3 | <=28 | no |
| 8 | 143.90 (p10 138.34, p90 148.49) | 146.72 | 0.981 | 55.6 | <=40 | no |

Block-graph amortisation alone buys ~0-2%: the token loop is weight-read
bound, not launch/sync bound. The preregistered 12A gates cannot be reached
by unrolling — they exist for the ERVF-M/grouped verifier (12C). This
measurement is the clean floor: any 12C candidate must beat these numbers
with identical correctness.

## 12B: route-union census (kill-gate for grouped MoE)

Source: frozen phase-9 routing trace, 8192 counted tokens x 23 MoE layers x
top-6 (`S100_PHASE9_TRACE.npz`; deviation: 8192 < 10,000 prereg tokens, but
the gate-relevant median clears the threshold by >9 points, so no recapture
was triggered).

| B | union/block median | slots | device-read reduction median (mean) | rows/expert mean |
|---|---:|---:|---:|---:|
| 2 | 10 of 12 | 0.833 | 16.7% (16.4%) | 1.19 |
| 4 | 17 of 24 | 0.708 | **29.2%** (30.1%) | 1.43 |
| 8 | 27 of 48 | 0.562 | 43.8% (43.1%) | 1.76 |

**Gate: median routed device-read bytes per token -29.2% at B=4 >= 20% —
grouped MoE OPENS.** Caveats, honestly stated:

- PCIe fetch bytes are unchanged (ratio 1.000): the production LRU already
  deduplicates temporally close repeats. The grouped win is device-side
  weight-read amortisation and fewer, larger kernels — not fewer fetches.
- The M distribution is shallow (1.43 rows/expert at B=4): grouped kernels
  must be efficient at M=1..4, not just at large M.

## Consequence for the decision sequence

1. Block state/KV correctness: proven (this document).
2. Route-union census: grouped MoE justified.
3. **Next build: 12C ERVF-M + grouped MoE microkernels.** The 12A floor
   table above is the target to beat; physical plausibility: at the measured
   ~227-241 GB/s practical stream rate, a B=4 block forward reading dense
   weights once (~1.2 GB) plus the expert union needs far less than the
   28 ms gate — the gates are reachable iff weight sharing is real.
4. Drafter training (12D) stays blocked until the 12C-integrated verifier
   meets a break-even gate.

Kill criteria from the preregistration are unchanged and now measurable:
if a true grouped verifier at B=4 remains >35 ms after 12C, this route
closes and the elastic-derivative path (separate model) remains.
