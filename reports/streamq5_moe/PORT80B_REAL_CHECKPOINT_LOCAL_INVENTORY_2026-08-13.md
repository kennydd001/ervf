# PORT80B real-checkpoint local inventory

Date: 2026-08-13  
Method: CPU/read-only filesystem, safetensors-header, index-chain and SHA-256 audit. No download, GPU run, bank mutation or registry edit.

## Verdict

The local machine does **not** contain Qwen3-Coder-Next weight payloads. It contains only the official 6,759,619-byte safetensors index for the pinned target revision. All 40 BF16 shards, `config.json`, tokenizer files and generation config are absent.

There are two complete official BF16 causal-MoE checkpoints locally:

1. `Qwen/Qwen3-30B-A3B-Base`, 16/16 shards and 61,066,575,648 weight-file bytes, independently rehashed against the official LFS SHA-256 metadata in this audit;
2. `deepseek-ai/DeepSeek-V2-Lite`, 4/4 shards and 31,413,626,576 weight-file bytes, likewise rehashed against its official LFS metadata.

Qwen3-30B is therefore the closest immediately usable real-weight rehearsal, but it is **not** a valid substitute for the 80B target: it uses Qwen3-MoE full attention, 128 experts, top-8 and expert width 768. Qwen3-Coder-Next uses the Qwen3-Next hybrid Gated-DeltaNet/full-attention graph, 512 experts, top-10 and expert width 512.

## 1. Exact Qwen3-Coder-Next inventory

Identity and provenance:

- official model ID: `Qwen/Qwen3-Coder-Next`;
- pinned revision: `a19358a7659bd1f564300250ee189120c49a562f`;
- local snapshot directory: `C:\Users\de_do\.cache\huggingface\hub\models--Qwen--Qwen3-Coder-Next\snapshots\a19358a7659bd1f564300250ee189120c49a562f`;
- only local file: `model.safetensors.index.json`;
- file size: **6,759,619 bytes**;
- current SHA-256: `e54c170589a729006db825100b4c69cf1c485ee89d3e8dd30aec9dccbf9cea1b`;
- cache repository size: exactly **6,759,619 bytes**; its `blobs` directory is empty and there is no local `refs/main`.

The index declares:

- **74,391 tensors**;
- **79,674,391,296 BF16 parameters**;
- **159,348,782,592 tensor-payload bytes** (148.405118 GiB);
- **40 shards**, named `model-00001-of-00040.safetensors` through `model-00040-of-00040.safetensors`.

Present/missing chain:

| Artifact | Present | Exact local state |
|---|---:|---|
| safetensors index | yes | 1 file, hash above |
| weight shards | **0/40** | all forty shard names in the index are missing |
| `config.json` | no | no local target config |
| `generation_config.json` | no | absent |
| tokenizer JSON/config/vocab/merges | no | absent |
| official code snapshot | no | no pinned HF Qwen3-Next Python implementation in this cache |

The independently retained N4A evidence, `reports/streamq5_moe/n4a_synthetic_80b_shape_capacity.json`, is metadata-only and records the same model/revision, tensor count and byte total. It is shape evidence, not checkpoint payload.

### Useful shard locality

The target index maps **all 1,550 layer-0 tensors** to `model-00001-of-00040.safetensors`:

- 1,536 routed-expert matrices = 512 experts x gate/up/down;
- three shared-expert matrices;
- eleven router, normalization and Gated-DeltaNet/non-expert tensors.

The same shard also contains the embedding and 17 early layer-1 tensors, for 1,567 indexed tensors in total. In particular, these all map to shard 1:

- `model.layers.0.mlp.experts.0.gate_proj.weight`;
- `model.layers.0.mlp.experts.0.up_proj.weight`;
- `model.layers.0.mlp.experts.0.down_proj.weight`;
- `model.layers.0.mlp.gate.weight`;
- `model.layers.0.mlp.shared_expert_gate.weight` and all three shared-expert projections;
- every layer-0 Gated-DeltaNet tensor.

Therefore a target-valid, complete one-layer truth gate needs only shard 1 plus the small config/reference files, not the full 40-shard payload. The local index does **not** contain per-shard byte sizes or official LFS hashes, so the exact shard-1 download size and SHA must be resolved and preregistered before downloading it. Estimating that size from the 159.35-GB/40-shard average is not acceptable provenance.

## 2. Complete official Qwen3-30B-A3B-Base checkpoint

Identity and provenance:

- official model ID: `Qwen/Qwen3-30B-A3B-Base`;
- pinned revision from every one of the 25 local Hugging Face metadata files: `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`;
- local directory: `C:\Users\de_do\Documents\ChatGPT\New project\models\qwen3-30b-a3b-base`;
- acquisition record: `reports/rsiv_moe/qwen_checkpoint_acquisition.json`, SHA-256 `318980cd6aa634072e97a4c06dfbfd50f7b255cd7f340d00d1f1e6105e6e3daf`;
- status: **16/16 shards present; all 16 current hashes equal the official LFS hashes**.

Index/header audit:

- weight-file bytes: **61,066,575,648** (56.872680 GiB);
- tensor payload from both index and shard offsets: **61,064,245,248 bytes**;
- safetensors header/file overhead: **2,330,400 bytes**;
- top-level model-directory file bytes: **61,079,780,903**;
- tensors: **18,867 index / 18,867 headers**, all BF16, zero missing/unexpected keys;
- config architecture: `Qwen3MoeForCausalLM`, 48 layers, hidden 2,048, 128 experts, top-8, expert intermediate 768, full attention with 32 Q/4 KV heads.

Shard sizes and current/official SHA-256:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| 00001/00016 | 3,999,417,504 | `7fe481b0c3796bee8d4fa63638f4d6d3d0b1c66339ef63a1aa03a4d784c53e73` |
| 00002/00016 | 3,999,974,192 | `3b1e762dd99476a4b7c2d7b331432fe3e32e11e0dd7c90c2822bb84df4881ae2` |
| 00003/00016 | 3,997,360,832 | `c66ff62c6aeb11e085e2215c64cfc5031b50fbcadaead02d0687054b9bf523ff` |
| 00004/00016 | 3,999,975,056 | `534c0b5e5a215d95bbd77f9a034d5d74c3b4258f4d5d3918e56ed63dec1eec49` |
| 00005/00016 | 3,999,975,400 | `8082532f02d2473f51828f4ab5377b2747d5d77b934ba8d93a25d72d4d82078c` |
| 00006/00016 | 3,999,975,400 | `9fab34042ea6ee3348994dbb9b582773bfd51c54defca758beee4521cf54f0bd` |
| 00007/00016 | 3,999,975,472 | `02f0a1c1e62143483d1d0655afee52e22c03d25b6f24c8f6ed3b9d6cf47fbc1b` |
| 00008/00016 | 3,997,362,064 | `b2ccbadc878ec61dc09cec19ae0c4d3ff8f25e9c0e659599a7d26eaa411a8a4d` |
| 00009/00016 | 3,999,975,408 | `0b9dc84d14919b4c65fa56e8b6ffd67c8cf431861673034275c26279a8dd525b` |
| 00010/00016 | 3,999,975,400 | `70cb610487e592d19eea29eadc8785a9e4e4b88ee65b3dbd7ac50d0c54e61a4f` |
| 00011/00016 | 3,999,975,408 | `662f6a0cb5607d3be8d52fac7c1f541769fd8426867cde36c588cbcef9e8bca9` |
| 00012/00016 | 3,987,400,496 | `c8ddc8ffa628697a3618f953a99cfdc51a34da97d0194ccf1ad9a5ba6022178b` |
| 00013/00016 | 3,997,353,632 | `2ea0a94f86a4eba612753a3159d9d06e307556c95873bde88294cb48f8c14f75` |
| 00014/00016 | 3,999,975,400 | `05d8098a9924ca0990db663b934550cb07a6287a6590b13e93b14cf139edc268` |
| 00015/00016 | 3,999,975,400 | `002936e6733de8b73ef36c815013cdd53f2c969ffe19ad62cc47ecab69909cff` |
| 00016/00016 | 1,087,928,584 | `90a2c863affd5c0a6e4bf14ae7e4ac7ff388b6328c3d49aed6ea18c3139c6536` |

Key support files are complete: config, generation config, index, tokenizer JSON/config, vocab, merges, README and license. Current hashes:

- `config.json`: `7e4142150e976c6b4796adf88fce0a2a23e581ed63bcede7e1d69a645e73b362`;
- `model.safetensors.index.json`: `df0d481ec595c55a0ba58426d517390c6214a566ec4ff1c8fc4bbce9f57b3c24`;
- `tokenizer.json`: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`;
- `tokenizer_config.json`: `3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5`.

## 3. Complete official DeepSeek-V2-Lite checkpoint

Identity and provenance:

- official model ID: `deepseek-ai/DeepSeek-V2-Lite` Base;
- pinned revision from all 15 local metadata files: `604d5664dddd88a0433dbae533b7fe9472482de0`;
- local directory: `C:\Users\de_do\Documents\ChatGPT\New project\models\deepseek-v2-lite`;
- status: **4/4 shards present; all four current hashes equal official LFS hashes**.

Index/header audit:

- weight-file bytes: **31,413,626,576** (29.256219 GiB);
- tensor payload from index and shard offsets: **31,412,968,448 bytes**;
- safetensors overhead: **658,128 bytes**;
- top-level model-directory file bytes: **31,418,838,087**;
- tensors: **5,291 index / 5,291 headers**, all BF16, zero missing/unexpected keys;
- config architecture: `DeepseekV2ForCausalLM`, 27 layers, hidden 2,048, 64 routed experts, top-6, two shared experts, expert intermediate 1,408 and MLA.

| Shard | Bytes | Current/official SHA-256 |
|---|---:|---|
| 00001/00004 | 8,594,887,408 | `0d7e9f39bde40111a4c0f390b87497dce4565cf578d916395e6b2c7851f1e8da` |
| 00002/00004 | 8,591,757,448 | `e0656832b0d594b4a64cad40ff8465231de6ed44c854f74f6b217797660aa4bb` |
| 00003/00004 | 8,590,718,520 | `843ec689624f3a520526e040f0326c4dc9865e8172942ca98a084fe136fdb21a` |
| 00004/00004 | 5,636,263,200 | `cfb51658f67cedfbbc4d62ad14187830ceec8ee82c788c5f718feea98905ef31` |

Config, custom model/config/tokenizer Python, generation config, index, tokenizer JSON/config, README and license are present. Current key hashes:

- `config.json`: `f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d`;
- `model.safetensors.index.json`: `d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf`;
- `tokenizer.json`: `41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddb1082f529f669`;
- `tokenizer_config.json`: `31181eaf79394ea26728d95ecb54fe7c8413e6f56085dbabc8b0818134380ec8`.

DeepSeek is complete but less representative of the target than Qwen3-30B: its expert count, top-k, expert width, layer count and MLA graph all differ.

## 4. Derived artifacts that are not official target weights

These are locally useful, but none can close a Qwen3-Coder-Next real-checkpoint gate:

| Artifact | Bytes | Provenance boundary |
|---|---:|---|
| `reports/runs/qwen3-30b-a3b-q5_k_m.gguf` | 21,725,580,864 | locally converted Qwen3-30B Q5_K_M; SHA `ebcc2ac28d8565e8a871016e80428abca8c1f89b3aefea5e782b57070b839e0d`; runnable and benchmarked, but not an upstream artifact and wrong model |
| `reports/runs/qwen_gptq_bank/p0_bank` | 7,700,786,304 | 48-file locally derived pure-GPTQ Qwen3-30B expert bank; verified against source hashes, but wrong architecture/shape |
| `reports/runs/streamq5_moe/p6a_exact_runtime_bank` | 1,565,380,608 | Qwen3-30B-derived exact runtime components; wrong target model |
| `reports/runs/streamq5_moe/port80b_p0/port80b_p0_full_q5_bank.bin` | 49,925,652,480 | synthetic shape-informed PORT80B bank; no official checkpoint payload provenance |
| `data/models/*.safetensors` | 8 small research artifacts | learned/compressed DeepSeek research derivatives, not official checkpoints |

The other Hugging Face cache repositories are TTS, image-generation, Stable Diffusion and Whisper assets. They are not causal-MoE target candidates and were excluded from the real-checkpoint gate.

## 5. Download, disk and RAM requirements

Machine snapshot during this audit:

- physical RAM: **68,103,761,920 bytes** (63.427 GiB);
- available RAM: **56,270,999,552 bytes** (52.406 GiB);
- free C: disk: **307,668,635,648 bytes** (286.539 GiB).

### Full target checkpoint

- exact missing tensor-payload lower bound: **159,348,782,592 bytes** (148.405 GiB);
- actual missing file/download bytes: slightly larger because safetensors headers and small support files are not counted by `metadata.total_size`; the exact total is not recoverable from the local index alone;
- safe operational disk gate: resolve the official file manifest first, then require `exact_missing_file_bytes + 20 GiB` before download. Current free disk is sufficient for the likely full BF16 snapshot, but a simultaneous second converted bank and temporary conversion copies must be separately budgeted;
- direct full BF16 materialization needs at least the 148.405-GiB tensor payload plus allocator/runtime overhead and therefore **cannot fit** in the machine's 63.427-GiB physical RAM;
- full-target work must use shard streaming/mmap plus streamed conversion/offload. A normal `from_pretrained` full CPU load is not a safe plan.

### Minimal target-valid one-layer gate

Required new files:

1. `model-00001-of-00040.safetensors`;
2. pinned `config.json`;
3. a pinned Qwen3-Next reference implementation compatible with that revision;
4. tokenizer files only if the gate uses text rather than frozen token IDs/hidden states.

The exact download/disk value is presently a **metadata blocker** because the local index has neither shard sizes nor LFS SHA-256. A metadata-only manifest resolution must precede authorization. Once shard 1 is present, a selective mmap truth test need not load the shard wholesale into RSS. The selected top-10 plus shared expert BF16 matrices contain 11 x 3 x 1,048,576 x 2 = **69,206,016 bytes**; their aligned Q5 records occupy 11 x 2,027,520 = **22,302,720 bytes**. A conservative start gate of **8 GiB available RAM** and at least **2 GiB after first touch** is ample for this selective test, independent of the mapped shard's virtual size.

### Existing complete checkpoints

- Qwen3-30B truth rehearsal: **0 download bytes and 0 new checkpoint disk bytes**. Full BF16 eager load is unsafe at current 52.406-GiB available RAM, but the established shard-streaming path has a 32-GiB RSS ceiling and the checkpoint has already been used under that constraint.
- DeepSeek-V2-Lite truth rehearsal: **0 download bytes and 0 new checkpoint disk bytes**. Its 29.256-GiB weight files fit only with careful runtime overhead; selective shard/tensor loading remains preferable.

## 6. Reference-runtime blocker

The local Python environment contains `transformers==4.51.3`. It imports Qwen3-MoE but does **not** expose `Qwen3NextConfig` or `Qwen3NextForCausalLM`. The local llama.cpp tree contains Qwen3-Next/Gated-DeltaNet implementation sources, but it is not a drop-in safetensors teacher and its retained build is CPU/Linux-oriented.

Therefore the target gate must pin a compatible official Qwen3-Next reference implementation before execution. It must not silently fall back to the Qwen3-30B model class or treat the synthetic D10 component oracle as checkpoint truth.

## 7. Recommended first real truth-test

### PORT80B-T0: pinned shard-1, real layer-0 truth gate

This is the smallest test that materially advances the 80B claim.

1. Resolve and preregister the official size, LFS SHA-256 and URL for target shard 1, config and chosen reference-code revision. No payload action before those hashes and the current index hash are locked.
2. Download only shard 1 plus config/reference files. Rehash them and prove that all 1,550 layer-0 tensor keys and shapes match the pinned index/N4A contract.
3. Use a frozen token ID or frozen 2,048-wide hidden vector. Execute the official layer-0 Gated-DeltaNet, router, selected top-10 routed experts, shared expert gate/add and residual with the real BF16 weights in an independent CPU reference.
4. Quantize only the selected ten experts plus shared expert into the exact differentiated Q5 record format; retain source-tensor SHA, quantized-record SHA, scales, codes and headers.
5. Compare the GPU candidate against an independent CPU dequantized-Q5 oracle for every intermediate and the composed layer output; preserve raw arrays/digests and wrong-expert/wrong-layer negative controls.
6. Require clean registration/unregistration and the same RAM/VRAM/page telemetry boundaries, but make no throughput or quality claim from this one-layer gate.

A no-download Qwen3-30B expert test can be run first as a loader/quantizer rehearsal, but it must be labeled `QWEN30_REHEARSAL` and may not open the PORT80B checkpoint or breakthrough gates. PORT80B-T0 above is the first direct target truth test.

## Claim boundary

This audit proves local artifact presence, index/header consistency and current hash provenance. It does not download Qwen3-Coder-Next, execute target weights, prove model quality, validate natural target routing or establish an 80B throughput/breakthrough claim.
