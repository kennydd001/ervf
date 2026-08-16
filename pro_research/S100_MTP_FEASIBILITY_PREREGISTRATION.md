# S100-MTP phase 0 preregistration — checkpoint inventory only

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: no local Lightning MTP inventory from this runner exists.

## Why this is allowed to reopen the old speculative-decoding question

The prior PRO line deliberately stopped generic speculation because an external/naive draft path had no measured exact end-to-end case. This phase does **not** reopen arbitrary speculation.

Current NVIDIA Nemotron-v3 documentation describes MTP heads as a native speculative mechanism for members of the family, but public documentation located so far does not establish that the exact local `NVIDIA-Nemotron-3.5-Lightning` checkpoint exposes the same inference-ready MTP structure. Therefore the local checkpoint itself must be inventoried before any implementation claim.

## Phase-0 claim boundary

This is a GPU-free structural inventory. It may establish only:

- whether tensors/config keys explicitly naming MTP/next-token-prediction/speculative components exist;
- their exact tensor names, shapes, dtypes, shards and byte counts;
- config fields that mention those mechanisms;
- coarse name-derived grouping/index structure.

It does **not** establish correct MTP forward semantics, acceptance rate, speedup, quality or exactness.

## Read-only method

Use `ShardIndex` to read only:

- `config.json`;
- `model.safetensors.index.json`;
- safetensors headers needed to enumerate tensor metadata.

Do not read tensor payloads and do not create a CUDA context. Treat the checkpoint directory as immutable.

Match tensor/config names case-insensitively for explicit tokens:

- `mtp`;
- `nextn`, `next_n`, `next_token`;
- `multi_token`, `multitoken`;
- `speculative`.

Store all matches; do not silently infer MTP from unrelated unnamed tensors.

## Frozen outcomes

`named_mtp_present` requires at least one tensor name matched by the explicit vocabulary above.

`structured_mtp_candidate` requires all of:

1. named MTP tensors exist;
2. total bytes >0;
3. at least one matched config field or at least two distinct matched tensor suffix/shape groups indicating a nontrivial module rather than a lone metadata tensor;
4. all safetensors metadata ranges are valid (`end > start`).

If these fail, the single-stream MTP branch remains closed.

If they pass, the **next** phase is still semantics-only: map the local names to an official/open NVIDIA architecture implementation or checkpoint adapter, then reproduce one MTP forward on recorded hidden states. Only after bit/semantic parity may an exact speculative-verification experiment be preregistered.

## Exact-generation rule for any future phase

A draft/MTP proposal is never accepted merely because the draft predicts it. Any future single-stream speed claim must preserve the baseline model's greedy token sequence exactly through target verification. Draft quantization may be considered only if it cannot alter target-verified accepted output. This principle does not itself prove an efficient verifier exists.
