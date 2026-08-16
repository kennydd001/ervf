# S100 MTP post-Kimi feasibility bound

Date: 2026-08-16
Branch: `pro-s100-mtp`
Base: `pro-e100-batch@485eb0bc37f6162fdd5aa74fa2fab1ed4bd14289`
Status: **analytic bound only; no MTP speed result**.

## Frozen evidence

Kimi's validated phase-0/K1 evidence on the exact Lightning checkpoint established:

- `num_nextn_predict_layers = 1`;
- MTP physical sublayers `mtp.layers.0` and `mtp.layers.1`, pattern `attention -> moe`;
- 270 MTP tensors, 2,670,652,160 B total, BF16 except 512 B F32;
- layer-0 tensor bytes = 75,710,208 B;
- layer-1 tensor bytes = 2,594,941,952 B;
- K1 exact rollback proof for prefixes 0..4, with sabotage divergence;
- one all-Mamba initial snapshot = 48.15625 MiB; snapshot + K=4 stored `in_proj` outputs = 51.77246 MiB.

Official NVIDIA AutoModel commit `5001dd45f051fe137f8bc284f53577f5e0da2fdb` defines `MTPConfig.num_layers` as the number of MTP forward iterations/depths. The common `MTPModule.forward` rolls the future-token input once per depth and returns one hidden state per depth. The exact local config therefore exposes one logical MTP depth, whose two inner sublayers are attention and MoE.

The current exact single-sequence record lives on `pro-research`: V18 candidate p50 = 19.6046 ms/token = 51.0084 tok/s, 765 timed samples, exact token parity.

## Active-weight lower-bound model

This section is a byte-floor calculation, **not measured runtime**.

For one MTP depth, layer 0 is fully active: 75,710,208 B.

Layer 1 contains 128 BF16 routed experts. From the checkpoint shapes, one expert's `up_proj + down_proj` is:

`2 * 2688 * 1856 * 2 = 19,955,712 B`.

The full 128-expert routed bank is therefore 2,554,331,136 B. Non-routed layer-1 weights are:

`2,594,941,952 - 2,554,331,136 = 40,610,816 B`.

Using the main-model top-k=6 only as the minimum active routed set, the active MTP byte set is:

- layer 0: 75,710,208 B;
- fixed layer 1: 40,610,816 B;
- top-6 routed experts: 119,734,272 B;
- total: **236,055,296 B**.

At the independently measured ~330.1 GB/s warm device-memory bandwidth, a physically impossible best case where all these bytes are resident and every other cost is free has a byte floor of about **0.715 ms per MTP draft**.

If the top-6 BF16 routed experts instead miss and traverse the measured ~26.159 GB/s pinned-H2D path, the routed-expert transport alone floors near **4.58 ms**, before attention/fusion/shared-expert compute and routing.

## Consequence for 100 tok/s

With one MTP depth, the optimistic useful-token multiplier is at most about 2 per successful verification iteration. To sustain 100 tok/s, total wall time for the iteration must therefore be below 20.0 ms.

Using V18's best measured one-token target pass as a reference:

- zero-cost draft: `2000 / 19.6046 ~= 102.02 tok/s`;
- plus the **resident-byte floor only** (0.715 ms): `2000 / (19.6046 + 0.715) ~= 98.43 tok/s`;
- plus a PCIe-miss lower bound for top-6 routed MTP experts (~4.58 ms) and resident fixed bytes: roughly 81.5 tok/s.

These are not end-to-end projections because a true K=2 target verifier is not the same execution as one V18 token. The important falsifiable implication is:

> **Native depth-1 MTP can reach S100 only if the K=2 target-verification pass itself becomes materially faster than the current 19.6046 ms single-token pass and/or substantial MTP work overlaps with it.**

For a hypothetical 0.715 ms resident-draft floor, K=2 target verification must be <= **19.285 ms** merely to leave room for 100 tok/s. If top-6 routed MTP experts pay the measured PCIe floor, the K=2 target verifier must be roughly <= **15.07 ms**.

## Decision

Do **not** build an MTP drafter yet. The next experiment is an oracle target-verification test using already-known correct tokens, so draft quality/acceptance cannot hide target cost.

If an exact/target-equivalent K=2 verifier cannot clear the frozen timing gates below, native depth-1 MTP is closed as an S100 mechanism on the current V18 target stack unless the target itself is first accelerated.
