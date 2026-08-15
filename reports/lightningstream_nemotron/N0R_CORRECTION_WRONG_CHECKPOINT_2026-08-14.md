# Correctie op N0R — verkeerde checkpoint gebruikt

Datum: 2026-08-14
Status: **materiële correctie. Alle metingen zijn geldig maar op het verkeerde model.**

## Wat er mis is

De hele lijn draait op `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`.
Het genoemde doel van de opdracht is **Nemotron 3.5 Lightning**, en die weights
zijn publiek beschikbaar als:

```
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
```

sha `6dbbd757ea75a8ece6e0702872e3ae53f9987728`,
lastModified **2026-08-13T18:36:59Z** — ongeveer 15 uur vóór deze sessie begon.

## Hoe het kon gebeuren

De opdracht stelde: *"The currently verified public downloadable checkpoint is
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`"* en noemde 3.5 alleen als
NIM-service. N0/N1 (2026-08-12, vorige agent) pinden Nano terecht — toen bestond
de Lightning-repo nog niet.

**`N0R_IDENTITY_REFRESH` was precies de fase die dit had moeten vangen.** Ik heb
de NIM-catalogus bevraagd en de gepinde HF-repo geverifieerd, maar ik heb
**nooit op Hugging Face gezocht naar een Lightning-checkpoint**. De fase heette
"identity refresh" en ik heb de identiteit niet ververst — alleen de bekende
namen gecontroleerd.

## Architectuurvergelijking

| veld | 3 Nano (gebruikt) | **3.5 Lightning (doel)** |
|---|---|---|
| architectures | NemotronHForCausalLM | NemotronHForCausalLM |
| model_type | nemotron_h | nemotron_h |
| num_hidden_layers | 52 | 52 |
| hidden_size | 2688 | 2688 |
| n_routed_experts | 128 | 128 |
| num_experts_per_tok | 6 | 6 |
| moe_intermediate_size | 1856 | 1856 |
| moe_shared_expert_intermediate_size | 3712 | 3712 |
| mlp_hidden_act | relu2 | relu2 |
| num_attention_heads / kv / head_dim | 32 / 2 / 128 | 32 / 2 / 128 |
| mamba_num_heads / head_dim | 64 / 64 | 64 / 64 |
| ssm_state_size / conv_kernel / n_groups | 128 / 4 / 8 | 128 / 4 / 8 |
| chunk_size | 128 | 128 |
| routed_scaling_factor / norm_topk_prob | 2.5 / true | 2.5 / true |
| vocab_size | 131072 | 131072 |
| **max_position_embeddings** | **262 144** | **1 048 576** |

**Identiek op elk gecontroleerd architectuurveld. Eén verschil: de
contextplafond is 4× groter.**

## Gevolgen

### Wat gewoon blijft gelden

Alle bouwstenen zijn shape-identiek en dragen ongewijzigd over: NVFP4-codec en
`uchar4`-loads, de fused decode+GEMV kernel, warp-per-position attention, FP8 KV,
de LRU-expertcache, de bankbouwer, de Mamba/router/norm-kernels, het hele
meet- en verificatieprotocol. Het expert-record is per arithmetiek opnieuw
5.612.560 B — te bevestigen door de headers te lezen, niet aan te nemen.

De gemeten getallen (21,4 / 20,2 / 16,7 tok/s) zullen bij gelijke contextdiepte
vrijwel gelijk zijn, omdat er per token exact evenveel bytes bewegen.

### Wat expliciet fout was

In N0R schreef ik dat het contextplafond **262.144 is en niet 1M**, en dat de
1M-claim service-side marketing was. Dat klopte **voor Nemotron 3 Nano**, maar
ik generaliseerde het naar het doelmodel. In de registry staat daardoor dat
`N13_1M_STRETCH` "geen gewone contextverlenging van dit checkpoint kan zijn" —
**die conclusie is onjuist voor 3.5 Lightning**, dat 1.048.576 declareert.

Beide uitspraken blijven staan zoals geschreven; deze correctie annuleert ze,
conform de append-only regel.

## Wat er moet gebeuren

1. `N0R2_IDENTITY_REFRESH` — pin de Lightning-repo, hash de shards, leg de
   architectuur vast en adjudiceer opnieuw of de NIM-alias eraan te binden is.
2. `N2R` — download de shards, verifieer tegen LFS-hashes, herbevestig de
   NVFP4-layout en het recordformaat tegen echte tensor-entries.
3. `N3R` — hercontroleer één module tegen de officiële code van dít checkpoint.
4. Daarna de meetketen herhalen. Verwacht gelijke tok/s bij gelijke diepte, en
   voor het eerst een **echte** 1M-context-vraag in plaats van een geblokkeerde.

Niets van de kernelwinst hoeft opnieuw uitgevonden te worden; alleen de
provenance-keten moet opnieuw, op het juiste model.
