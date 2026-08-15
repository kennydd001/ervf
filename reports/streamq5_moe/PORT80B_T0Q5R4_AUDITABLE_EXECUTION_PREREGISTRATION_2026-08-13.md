# PORT80B-T0Q5-R4 — auditable execution revision

Date: 2026-08-13  
State: source candidate; execution closed pending implementation audit  
Scientific protocol: T0Q5-R3, SHA-256 `e4458eac88f78c5b6608458e34fcd50b80680048c9d37159b3cab17a99e5d9fe`  
R3 failed implementation candidate remains immutable, SHA-256 `ad2558ef6f4aceeb277b602a7249a4355450ed469582183f5119c7d1402f403c`.

R4 changes execution/provenance only; thresholds, inputs, codec semantics and claim boundary are unchanged.

## Required execution graph

- Capture the actual input of `post_attention_layernorm` with a forward pre-hook as the BF16 pre-MLP residual.
- Mirror official experts exactly: one-hot route mask, `.permute(2,1,0)`, ascending nonzero expert IDs, `top_k_pos, token_idx = torch.where(expert_mask[expert])`, fused gate-up, gate-then-up split, official SiLU, down, route-weight multiply, `index_add_`.
- Shared order is exactly `sigmoid_gate * shared_raw`; complete MLP is routed first plus gated shared; layer is residual first plus complete MLP. Source-BF16 arm must be bitwise official at routed/shared-raw/shared-gated/complete/layer.

## Verified-reference handoff

Reference raw/result are written as one create-new transaction and are not Q5 eligibility by themselves. The independent reference verifier reconstructs exact schema/manifest/provenance, prompt replay/disjointness, direct/second router tuple and FP32 top-k semantics, residual add, shared operand order, complete MLP identity, finiteness, source tensor hashes, runtime and resources. It atomically writes a create-new reference-verification artifact. Q5 accepts reference inputs only if that artifact says pass and independently binds the exact reference raw/result hashes, verifier hash and locks.

## Bank and Q5 linkage

Before any bank promotion the runner performs a second full pass over the temporary 1,539 records, independently rereads official source tensors, requantizes without builder functions, and checks codes/scales/header/CRC/padding/offset/source/decoded digests. Bank, canonical manifest and commit marker are promoted as a recoverable transaction. Capture of a final bank without a valid commit is quarantined on failure/recovery.

The independent verifier imports neither runner nor codec. It independently parses and decodes the selected bank records, reruns both the complete full-16 Q5 graph and every unsafe negative-control graph, and requires byte equality with retained raw Q5/control arrays before recomputing metrics. Thus result summaries cannot link arbitrary output to an unrelated bank.

## Controls

Each safe control uses the same independent production record parser as normal execution and records its real rejection exception. Unsafe execution retains the raw BF16 complete-MLP row. Baseline is exactly the retained normal gated complete-MLP row (`routed + sigmoid_gate * shared_raw`). The independent verifier reconstructs safe rejection and unsafe output from bank bytes and compares raw bytes; it never trusts `safe_rejected` or differing-word counts.

## Transactions, resources and failure

Reference bundle, bank/manifest/commit bundle, Q5 raw/result bundle, reference-verification artifact and all failure artifacts use create-new temporary files, file fsync and atomic promotion. A journal declares intended final paths/hashes. On exception, every temp or uncommitted final is moved into a unique `failed_attempts` directory and the create-new failure artifact records dispositions. No overwrite or silent reuse.

Both phases enforce start available RAM >=16 GiB, minimum available RAM >=2 GiB at every recorded sample, Windows peak working set <=12 GiB, start disk free >=4 GiB, CUDA uninitialized, and Q5 total added artifacts <=1.10 GiB before commit. Report all samples and final artifact byte total.

## Source-audit lock

No prompt lock is generated and no preflight/model/shard scan/forward/bank action occurs before independent source GO. The first R4 lock intentionally binds an absent-prompt sentinel and therefore cannot execute. After GO, preserve it and make a new fully bound revision.
