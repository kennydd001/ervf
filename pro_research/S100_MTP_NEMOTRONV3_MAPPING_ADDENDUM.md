# S100-MTP phase 0 addendum — official NemotronV3 mapping fingerprint

Date: 2026-08-16
Branch: `pro-e100-batch`
Status at freeze: no local S100-MTP inventory result exists.

After the generic inventory preregistration, NVIDIA's current open AutoModel source was inspected before any local target output.

Primary implementation reference:

- repository: `NVIDIA-NeMo/Automodel`;
- commit inspected: `5001dd45f051fe137f8bc284f53577f5e0da2fdb`;
- file: `nemo_automodel/components/models/nemotron_v3/mtp.py`;
- common forward: `nemo_automodel/components/models/common/mtp/mtp.py`.

## Frozen official fingerprint

The NemotronV3-specific source states that released Hugging Face checkpoints use flat parameter names:

`mtp.layers.{global_idx}.*`

It resolves MTP structure from:

- `num_nextn_predict_layers`;
- `mtp_hybrid_override_pattern`, or alternatively
- `mtp_layers_block_type`.

For the first sublayer of each physical MTP depth the model may carry fusion modules:

- `enorm`;
- `hnorm`;
- `eh_proj`.

The final sublayer of each physical depth carries `final_layernorm`.

NemotronV3 MTP sublayers otherwise inherit the normal NemotronV3 decoder block, so names below `mtp.layers.N` may include attention/MoE/Mamba block parameters depending on the configured pattern.

The common MTP forward recursively rolls the future token input one position per MTP depth, embeds that future token input, fuses it with the main-model hidden state on the first sublayer, and returns one hidden state per MTP depth.

## New inventory evidence fields

Before reading local output, the inventory is extended to record:

- exact config values for the three official fields above;
- all tensor indices matching `^mtp.layers.(\d+).`;
- whether those indices are contiguous from zero;
- counts/bytes for `enorm`, `hnorm`, `eh_proj`, `final_layernorm` under MTP;
- derived configured pattern length when available;
- configured logical MTP depth count when available;
- an `official_nemotron_v3_name_alignment` gate.

`official_nemotron_v3_name_alignment` requires:

1. at least one `mtp.layers.N.*` tensor;
2. contiguous observed `N` indices from zero;
3. at least one of the official MTP config fields is present;
4. at least one fusion/final-norm marker is present.

This gate establishes only name/layout compatibility with NVIDIA's open implementation. It is **not** forward-semantic parity and not evidence of speculative speedup.

If the local checkpoint aligns, phase 1 may map exact local tensor shapes/storage to the NVIDIA sublayer definition. It must still reproduce the MTP forward semantics before any target-verification benchmark is allowed.
