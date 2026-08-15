# N0R_IDENTITY_REFRESH — preregistration

**Registry:** LIGHTNINGSTREAM_NEMOTRON
**Phase:** `N0R_IDENTITY_REFRESH` (hypothesis H0)
**Date:** 2026-08-14
**Status at writing:** design frozen, not yet executed.
**Protected baseline:** root digest `7c992ce222841f975b349a1e2e3cdecb79606a7372852f67c0dd16dabce946ba`.

## 1. Question

Is the payload served by the NVIDIA NIM endpoint `nvidia/nemotron-3.5-nano-30b-a3b`
the same checkpoint as the public Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`,
a distinct post-training revision, or a serving alias over a payload whose
identity cannot be established from public metadata?

This phase **controls naming only**. Per assignment §4 it must not block local
engineering on the public checkpoint.

## 2. Why this phase is rerun rather than inherited

`NEMOTRON_N0_METADATA_GATE` (2026-08-12, SHA-256
`28d4660af02da40f712fe21fb1f284ad260f76de715461e5cfb95009564e00d7`) already
recorded that the alias gate is open. N0R does not rewrite it. N0R re-pins the
metadata as of today because (a) the pinned HF commit may have moved,
(b) the NIM listing may have changed, and (c) N1's derived quantities assume
top-6 routing while the NIM card states top-5 in at least one place, and that
conflict must be recorded against today's sources rather than a two-day-old
snapshot.

## 3. Actions, in order

1. **HF repository metadata at pinned revision.** Resolve
   `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` at revision
   `ce1b118ae66ec705d02c241525192832eb045fd3`. Record: resolved commit SHA, last
   modified, private/gated flags, tags, license, and the complete sibling file
   list with size and LFS OID (SHA-256) where present.
2. **HF main-branch drift check.** Resolve the same repository at `main` and
   record its commit SHA. If it differs from the pinned commit, record the
   difference; do **not** silently move the pin.
3. **Small text artifacts.** Download only small non-payload files into the
   isolated cache: `config.json`, `generation_config.json`,
   `tokenizer_config.json`, `model.safetensors.index.json`, any
   `hf_quant_config.json` / quantization config, and any `*.py` model code.
   Record byte count and SHA-256 of each.
4. **Architecture extraction.** From the pinned `config.json` record, without
   inference: total layers, layer-type pattern, number of MoE layers, routed
   expert count, **`num_experts_per_tok`**, shared-expert configuration,
   hidden size, intermediate sizes, attention head counts and head dim, KV
   dtype/config, RoPE/state configuration, vocabulary size, and the declared
   maximum position/context.
5. **Routing-arity adjudication.** Compare the extracted `num_experts_per_tok`
   against the NIM card's stated value and against N1's assumption of 6. Record
   agreement or conflict explicitly.
6. **NIM endpoint metadata, best effort.** Attempt public NGC catalog/API
   metadata for `nvidia/nemotron-3.5-nano-30b-a3b`: display name, parameter
   count, context length, container tag/version, publish and modify timestamps,
   license, and deployment profile. No credentials are supplied and none will be
   requested; any authenticated-only field is recorded as `blocked_no_credentials`.
7. **Container manifest inspection, best effort.** Attempt to read a public
   registry manifest for the NIM container without pulling it. If unavailable,
   record `blocked_no_credentials`.
8. **Bind.** Write one machine-readable result plus one report.

No model shard is downloaded in this phase. No prompt is sent to any inference
endpoint in this phase: assignment §H0.4 requires a local reference to exist
first, and it does not.

## 4. Frozen decision rule

Evaluated in this exact order; the first matching branch wins.

| # | condition | outcome |
|---|---|---|
| 1 | A NIM-side manifest or published digest binds the served payload to the five HF shard LFS OIDs | `identity_proven` |
| 2 | No such binding, but NIM metadata declares a different revision, parameter count, quantization or post-training lineage than the pinned HF checkpoint | `distinct_revision` |
| 3 | No such binding, and NIM metadata is obtainable but contains nothing that resolves payload identity | `service_only_unknown_payload` |
| 4 | No such binding and NIM metadata is not obtainable at all | `service_only_unknown_payload` with `nim_metadata_blocked=true` |

`behaviorally_close_identity_unproven` is **not reachable in N0R** because no
prompt suite is run. It is reserved for a later phase once a local reference
exists. Recording that here prevents a post-hoc upgrade of a metadata-only
result into a behavioral one.

**Expected outcome, stated before execution:** branch 3 or 4. NGC container
manifests are normally credential-gated, and NVIDIA does not routinely publish
per-shard digests for optimized NIM payloads. Writing this expectation down in
advance means a `service_only_unknown_payload` result is a confirmed prediction
rather than a disappointment, and an `identity_proven` result would be a genuine
surprise requiring extra scrutiny.

## 5. Hard gates

A pass requires all of:

1. the pinned HF commit resolves and its sibling list is complete;
2. all five shards are present with an LFS SHA-256 OID recorded;
3. `config.json` parses and yields the architecture fields in §3.4;
4. `num_experts_per_tok` is extracted from the pinned config and explicitly
   adjudicated against the NIM card value and against N1's assumption;
5. the outcome is one of the four registered values, selected by §4 and by no
   other reasoning;
6. no protected byte changed.

Failure of 1–3 is a phase failure, not a negative identity result.

## 6. Claim boundary

N0R may claim only: what public metadata says today about the two names, the
exact architecture declared by the pinned public config, and whether payload
identity is bindable from that metadata. It may not claim behavioral
equivalence, quality, throughput, context capability, or that the NIM service
and the local checkpoint are the same model. The declared 1M context is a
metadata statement about the service and is not evidence about anything this
line will run locally.

## 7. Artifacts to be produced

| path | kind |
|---|---|
| `reports/lightningstream_nemotron/n0r_identity_refresh.json` | machine-readable result |
| `reports/lightningstream_nemotron/N0R_IDENTITY_REFRESH_REPORT_2026-08-14.md` | report |
| `reports/lightningstream_nemotron/n0r_input_lock.json` | input/source lock |
| `reports/lightningstream_nemotron/protected_verification_after_n0r.json` | protected-manifest check |

Runner: `scripts/lightningstream_nemotron/n0r_identity_refresh.py`; its SHA-256
is recorded in the input lock and in the result.
