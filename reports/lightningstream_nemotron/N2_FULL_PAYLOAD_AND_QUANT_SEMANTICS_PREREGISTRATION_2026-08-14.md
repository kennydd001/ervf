# N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS — preregistration

**Registry:** LIGHTNINGSTREAM_NEMOTRON · **Phase:** `N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS` (hypothesis H1)
**Date:** 2026-08-14 · **Status at writing:** design frozen, not yet executed.
**Depends on:** `N0R_IDENTITY_REFRESH` (outcome `service_only_unknown_payload`, all gates pass).
**Protected baseline:** root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`.

## 1. Question

Does the official five-shard public NVFP4 checkpoint download byte-exactly, and
is its real on-disk quantization layout exactly the layout derived in N0R §5 —
or is that derivation wrong?

This phase is designed to be able to **falsify** the N0R layout hypothesis. The
hypothesis reproduced four frozen byte buckets exactly, which makes it likely
and therefore dangerous: a likely-but-unverified layout is precisely the kind of
premise that killed GaugePack. It is confirmed here against real tensor entries
or it is discarded.

## 2. Scope

In scope: download, integrity, immutable manifest, tensor-level layout
extraction, quantization semantics, routed/shared/trunk partition, random-access
record layout, and a **bit-exact decoder for a single quantized matrix**.

Out of scope, and explicitly forbidden in this phase: materializing a BF16 model,
decoding the whole bank, any GPU kernel, any timing claim, any quality claim, any
model forward.

## 3. Actions, in order

1. **Download** the five official shards at pinned revision
   `ce1b118ae66ec705d02c241525192832eb045fd3` into
   `models/nemotron_3_5_lightning/`. No other file is fetched to that directory.
2. **Verify** each shard's SHA-256 against the LFS OIDs frozen in N0R:

   | shard | bytes | required SHA-256 |
   |---|---:|---|
   | 1 | 3,998,838,864 | `2fdac76b3e4906ce0fb0dd33ab51f011372a5473e0d6c5bb479b6f10d3f29fdb` |
   | 2 | 4,000,414,120 | `559806ee0cb6edcfc01805e24bac9182cb2611bad3993e0da05487d7a79b4f38` |
   | 3 | 3,999,641,680 | `d820849788701123d041501fb8ac88e4ade24a28a63cd663118797cfae910be2` |
   | 4 | 4,000,413,336 | `f5ccb7cfa7870ab2d099134c3f771ad4a158e0421b3bf7b2a0da53311a09cb14` |
   | 5 | 3,343,488,520 | `c9dd9142839367ad274019a7683bc84993217c8a63e70dd8e18656de0c4050eb` |

   A mismatch on any shard is a **hard stop**, not a retry-until-green loop. One
   redownload of a mismatching shard is permitted and must be recorded as such.
3. **Immutable manifest.** Record path, bytes, SHA-256, and safetensors header
   bytes/hash per shard. Mark the directory read-only in the manifest's intent
   and never write into it again.
4. **Tensor inventory.** Parse all five safetensors headers. For every tensor
   record: name, dtype, shape, byte offsets, byte length. Reconcile against
   `model.safetensors.index.json`.
5. **Layout adjudication.** For a routed expert, extract the *actual* tensor
   names, dtypes and shapes and test the N0R hypothesis field by field:
   - a U8 code tensor of `N/2` bytes per matrix;
   - an F8_E4M3 block-scale tensor of `N/16` elements per matrix;
   - two FP32 global scalars per matrix.
   Record agreement or the exact discrepancy.
6. **Partition.** Recompute routed / shared / trunk byte buckets from real
   tensor entries and compare to N1's frozen buckets.
7. **Excluded modules.** Confirm that the 63 `exclude_modules` entries are
   present as BF16 tensors, and total their bytes — this is the incompressible
   fixed cost H4 must carry.
8. **FP8 KV semantics.** Record what the checkpoint and model code declare about
   FP8 KV; do not infer runtime behavior from the config alone.
9. **Random-access record layout.** For each of the 2,944 routed experts,
   determine whether its tensors are contiguous within one shard, and record the
   contiguous extent and alignment. This decides whether a routed expert can be
   fetched with one range read — which the whole H3 transport design depends on.
10. **Bit-exact one-matrix decoder.** Implement NVFP4 → FP32/BF16 for exactly one
    routed expert matrix and validate it as in §4.

## 4. Decoder validation rule

The decoder is validated **without** a BF16 reference model, because downloading
the BF16 checkpoint is out of scope. Validation is therefore self-consistency
plus published-semantics conformance:

1. **Range invariance.** Every decoded 4-bit code maps into the NVFP4 E2M1 value
   set `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}` scaled by its block scale and global
   scale. Any decoded value outside the representable set is a failure.
2. **Round-trip.** Re-encoding the decoded values with the same block/global
   scales reproduces the original code bytes **bit-exactly** for 100% of a
   preregistered sample of at least 1,048,576 codes.
3. **Independent implementations.** Two decoders written from the published
   format — one table-driven, one bit-arithmetic — must agree on every sampled
   code. Disagreement is a failure of the phase, not a tolerance to be widened.
4. **Structural.** Decoded matrix shape equals the config-implied shape; the
   count of block scales equals `N/16`; no NaN or Inf.

Numerical tolerances are declared **now, before results are opened**: exact
equality for 1–3 (they are integer/bit operations, so any tolerance would be a
bug), and exact shape equality for 4.

## 5. Hard gates

1. all five shards downloaded and SHA-256-equal to the frozen OIDs;
2. all five headers parse; tensor count equals **24,147**;
3. reconstructed tensor bytes equal **19,339,781,632**;
4. routed / shared / trunk buckets equal N1's frozen values exactly;
5. the N0R layout hypothesis is explicitly **confirmed or falsified**, with the
   discrepancy recorded if falsified;
6. decoder passes all four §4 rules;
7. no BF16 materialization of the model, no GPU call, no timing figure;
8. artifacts ≤ 25 GiB;
9. no protected byte changed.

Gates 1–3 failing is a phase failure. Gate 5 resolving to *falsified* is a valid
scientific result, not a failure — and it invalidates N0R §5, which must then be
annotated rather than rewritten.

## 6. Stop rules

- shard hash mismatch twice → stop, report, do not proceed;
- tensor count or byte total mismatch → stop; the pin or the index is wrong;
- decoder disagreement between the two implementations → stop and debug the
  implementations; **make no statement about the checkpoint**;
- a routed expert's tensors turn out to be non-contiguous or split across shards
  → do not treat it as a defect; record it, because it changes the H3 transport
  design from one range read to a gather.

## 7. Claim boundary

N2 may claim only: the local copy is byte-identical to the published checkpoint,
the exact on-disk tensor layout and quantization semantics, the true
routed/shared/trunk partition, and that one matrix decodes correctly under
published NVFP4 semantics. It may not claim model quality, any throughput or
latency figure, that the full model is correct, or that a runtime is feasible.

## 8. Artifacts

| path | kind |
|---|---|
| `models/nemotron_3_5_lightning/` | five shards, treated as immutable after verification |
| `reports/lightningstream_nemotron/n2_payload_manifest.json` | immutable shard manifest |
| `reports/lightningstream_nemotron/n2_tensor_inventory.json` | full tensor inventory and partition |
| `reports/lightningstream_nemotron/n2_layout_adjudication.json` | N0R hypothesis confirmed/falsified |
| `reports/lightningstream_nemotron/n2_decoder_validation.json` | four-rule decoder validation |
| `reports/lightningstream_nemotron/N2_FULL_PAYLOAD_AND_QUANT_SEMANTICS_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n2_input_lock.json` | input lock |
| `reports/lightningstream_nemotron/protected_verification_after_n2.json` | protected check |
| `src/moe_lab/lightningstream_nemotron/nvfp4.py` | the two decoder implementations |

## 9. Non-interference

Before the download starts and before any GPU-adjacent step, check for a live
PORT80B/STREAMQ5 process. This phase is disk- and network-bound and uses no GPU,
so it may proceed while the other agent computes; it must not, however, consume
so much disk that the 80B line's artifacts are endangered. Free space is recorded
before and after. Budget: 18.01 GiB against 272.57 GiB free.
