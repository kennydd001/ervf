# S1–S4 — hypothesis census for the 50 tok/s question (preregistration)

Datum: 2026-08-14
Registry: LIGHTNINGSTREAM_NEMOTRON
Status: PREREGISTERED, before any census measurement has run.

## Context and standing measurements

Reproduced this session, before preregistering anything below, with the frozen
N7-B runner (`n7b_cached_decode.py`, capacity 31, FP8 KV, max-ctx 262144):

| context | N8 report | reproduction |
|---:|---:|---:|
| 0 | 21,722 tok/s | 22,062 |
| 32.768 | 20,141 | 20,147 |
| 131.072 | 16,469 | 16,686 |
| 262.100 | 13,225 | 13,143 |

Correctness gate reproduced: `The capital of France is` → ` Paris`.

N8's measured floor at 262K: MoE-miss PCIe 10,6 ms + device compute 10,4 ms ≈
21 ms → 47,6 tok/s ceiling for the current "what moves" design. 50 tok/s
requires changing what moves. This census measures the four candidate
hypotheses from the assignment, one runner, no timing claims.

## Model facts that shape the hypotheses (from config/index, pre-census)

- MoE expert = **two** matrices: `up_proj` (2688→1856), ReLU², `down_proj`
  (1856→2688). No gate matrix. `mlp_hidden_act: relu2` (config).
- `act[j] == 0` exactly when `up_row_j · x ≤ 0` (ReLU clamps to exact zero).
  Column j of `down_proj` then contributes nothing to the output.
- One routed record = 5,612,560 B, of which down_proj is exactly half the codes
  and half the scales (uniform record layout, `down` then `up`).

## Hypotheses and frozen gates

### S1 — route predictability (can prefetch become causal?)

Measurement: greedy decode, two frozen prompts (A: "The capital of France is",
B: "The history of computing began when"), 256 tokens each, `capture_routes`
hook, all 23 MoE layers, 512 tokens total.

Predictor under test (frozen): per layer L, score each candidate expert e by
`score(e) = Σ_{c ∈ top6(L,t)} N_L(c → e)`, where `N_L(c → e)` counts, in the
TRAIN half (tokens 0–127 of each prompt), how often expert e appeared in
top6(L, t+1) when c was in top6(L, t). Candidates = top-C by score,
ties broken by lower expert id. Evaluation on the TEST half (tokens 128–255):
recall@C = fraction of the 6 actual top6(L, t+1) experts inside the C
candidates, averaged over test rows, reported per layer, mean over layers, and
min over layers, for C ∈ {6, 8, 12, 16, 24}.

Also reported (informational, no gate):
- temporal identity overlap |top6(L,t+1) ∩ top6(L,t)| / 6 (N7-A consistency);
- adjacent-layer within-token overlap |top6(L+1,t) ∩ top6(L,t)| / 6.

Gates:
- **G-S1-PASS**: mean recall@12 ≥ 0.90 AND min-layer recall@12 ≥ 0.75
  → cross-token route prediction justifies a prefetch build phase.
- **G-S1-CLOSE**: mean recall@24 < 0.80 → route prediction is closed as a
  scientific negative.
- Between the two: report the numbers; any further step requires its own
  preregistration. Gates are not widened after seeing results.

### S2 — ReLU² sparsity census (column-selective down_proj)

Measurement: same rollout as S1 (the instrumentation does not alter routing,
arithmetic, or ordering). After every `fused.expert` call, count on device,
accumulated without synchronisation: exact zeros in `act[:1856]`, fully-zero
16-column blocks, fully-zero 64-column blocks. Per-call rows are dumped raw so
the verifier recomputes every statistic.

Rationale for block variants: a dependent column gather at single-column
granularity may cost more in transfer overhead than it saves; the 16/64-column
block statistics bound the viable gather granularity before anything is built.

Gates:
- **G-S2-PASS**: overall mean zero fraction ≥ 0.45 → column-selective transfer
  is projected to cut miss bytes by ≥ 0.45 × 50% ≈ 22% → build phase justified.
- **G-S2-CLOSE**: mean zero fraction < 0.30 → closed (< 15% byte cut).
- Between: report only.

### S3 — NVFP4 code entropy (delta/lossless coding feasibility)

Host-side, read-only, on a frozen sample of the routed bank: all 23 MoE layers,
experts {0, 16, 32, 48, 64, 80, 96, 112}, both matrices, codes and scales
(sample size ≈ 1,3 GB, fixed before measurement).

- E1: per-layer nibble histogram → entropy in bits/code (up and down pooled).
- E2: scale-byte histogram entropy (bits/byte), informational.
- E3: cross-expert delta question — joint histogram of nibbles at identical
  byte positions for sampled expert pairs (0,16), (32,48), (64,80), (96,112)
  per sampled layer → conditional entropy H(code_B | code_A) in bits.

Gates:
- **G-S3-LOSSLESS**: min per-layer nibble entropy ≤ 3,5 bits/code → a lossless
  coder saves ≥ 12,5% of code bytes and stays open; otherwise lossless coding
  is closed.
- **G-S3-DELTA**: mean H(code_B | code_A) ≤ 2,5 bits → expert-delta coding
  worth a design phase; otherwise closed.

### S4 — MTP / speculative-draft weight scan

Measurement: scan `model.safetensors.index.json` keys for
mtp|nextn|eagle|draft|spec (case-insensitive).

Gate: any hit → speculative path open for design; zero hits → publisher-draft
speculative decoding is closed for this checkpoint (no draft weights exist and
training one is out of scope). Batch>1 amortisation is unaffected either way
and is not decided by this census.

## Method rules for this census

- The census produces **statistics and gate verdicts only — no timing claims**,
  no tok/s projections. Component measurements are never promoted to tok/s.
- One rollout serves S1 and S2; the two instrumentations do not interact
  (routes are read out after the step; zero counts accumulate device-side).
- S3 reads bank bytes; it never writes them.
- No gate is changed after execution. A failed gate is recorded, not retried
  with a looser gate.
- The independent verifier recomputes every gated number from the dumped raw
  artifacts (routes JSON, per-call zero-count arrays, bank bytes re-read from
  the safetensors shards) without importing the runner.

## Claim boundary (pre-committed)

This phase may claim: measured route predictability, measured ReLU² sparsity,
measured code entropy, and the presence/absence of draft weights, on this
checkpoint, on this machine, at this sample size. It may NOT claim: any tok/s
figure, any quality result, that any build will succeed, or that a gate-passing
hypothesis will survive contact with the dependent-transfer overheads it
abstracts away.
