# FLEQ-MoE bron- en haalbaarheidsaudit

**Datum:** 2026-08-11  
**Project:** `FLEQ_MOE` — Full-rank Low-Entropy Quantization for MoE  
**Codenaam:** `EntropyExperts`

## Oordeel

De hypothese is mechanistisch onafhankelijk van CRAFT en RSIV. Zij behoudt de
volledige matrixrang en probeert niet routes, neurons of activationrichtingen
weg te laten. De discrete gewichtsassignments en groupscales worden binnen een
full-rank scalaire code geoptimaliseerd.

De algemene methode is niet nieuw. De eerste fase is daarom een reproductie- en
haalbaarheidstest, geen novelty- of Eureka-claim. Alleen een later vooraf
geregistreerd expert-only trajectory-QAT-mechanisme kan eventueel een nieuwe
technische bijdrage vormen.

## Primaire bronnen

- GSQ-paper: <https://arxiv.org/abs/2604.18556>, v2 van 2026-05-15.
- Officiële GSQ-code: <https://github.com/IST-DASLab/GSQ>, lokaal gepind op
  commit `03fc16484c369e3127225615d5e03e8d3a6043e3`.
- QMoE-paper: <https://arxiv.org/abs/2310.16795> en de MLSys 2024-versie.
- Qwen-checkpoint: <https://huggingface.co/Qwen/Qwen3-30B-A3B-Base/tree/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9>.

QMoE rapporteert minder dan 0,8 bit per parameter voor een 1,6T
SwitchTransformer, maar dat gebruikt een andere architectuur en bewijst geen
Qwen3-kwaliteit. GSQ beschrijft Gumbel-Softmaxoptimalisatie van scalaire
codeassignments en levert wrappers voor Qwen3-MoE, inclusief 30B-A3B.

## Reproduceerbaarheidsstatus van GSQ HEAD

- De officiële README beveelt voor Qwen3-30B-A3B `4× H200` aan.
- De quantisatietraining is volgens de README vooral op Hopper getest; Ada is
  gedeeltelijk getest. Deze RTX PRO 2000 Blackwell Laptop GPU (`sm_120`) is
  niet als gevalideerd platform genoemd.
- De servingstack vereist onder meer vLLM en Humming-kernels. vLLM is geen
  native Windows-baseline; er volgt dus geen runtimeclaim uit een succesvolle
  PyTorch-quantisatiesmoke.
- De officiële productie-YAML voor Qwen3-30B gebruikt de velden `masks_lr`,
  `signs_lr` en `scales_lr`, terwijl de gepinde parser alleen `lr1` en `lr2`
  accepteert en onbekende velden afwijst. HEAD is daardoor zonder lokale
  configuratiecorrectie niet rechtstreeks reproduceerbaar.
- De single-GPU Qwenwrapper bouwt quantizers voor alle 128 experts van een laag
  tegelijk. Bij 2 bit omvat alleen de BF16 assignment-logitstate vier waarden
  per gewicht. Dit past niet verantwoord binnen 8 GiB wanneer gradients en
  optimizerstate worden meegerekend.

De geldige lokale smoke gebruikt daarom exact de gepinde GSQ-quantizerklassen,
maar streamt één expert tegelijk. Expertbijdragen zijn bij vaste routering
additief; deze factorisatie verandert het discrete codebook niet. Zij bewijst
alleen lokale softwarecompatibiliteit en expertoutputreconstructie.

## Exacte Qwen-boekhouding

De reeds lokaal geverifieerde Base-checkpoint bevat:

- 30.532.122.624 totale parameters;
- 28.991.029.248 routed-expertparameters (94,953%);
- 1.541.093.376 overige parameters.

Met de overige parameters exact op vier bits, en nog zonder scales, metadata,
KV-cache of runtimebuffers:

| Expertcode | Ideale modelomvang | Actieve expertbytes/token | Verkeer bij 10 tok/s |
|---|---:|---:|---:|
| 2 bit | 8,018 GB / 7,468 GiB | 452,985 MB | 4,530 GB/s |
| ternary, log2(3) | 6,514 GB / 6,067 GiB | 358,982 MB | 3,590 GB/s |
| 1 bit | 4,394 GB / 4,093 GiB | 226,492 MB | 2,265 GB/s |
| 0,8 bit | 3,670 GB / 3,418 GiB | 181,194 MB | 1,812 GB/s |

De 2-bitvariant past door praktische overhead niet vanzelf in 8 GiB. Ternary
heeft meer headroom, maar alleen een werkelijk bitpacked artifact en gemeten
runtime kunnen dat bevestigen.

## Bronankers

- Inkomende hypothesetekst SHA-256:
  `921960590eaf1f7487b2b211402162b254d51b5da2b32ef38487894f031875d8`.
- GSQ README SHA-256:
  `236db6631d25ce03e087f537688bd91f1abfc7554f055c88fa08bccf365f8f3f`.
- GSQ 2-bit quantizer SHA-256:
  `81bb1a3ad3b318d4c115a7ba38cd53989d98ea5a3c5943767dac5e6499d7feb0`.
- GSQ ternary quantizer SHA-256:
  `2d4a23117ced5e60359f5432c346187dd38691ab484bfaf725fb7434829cb35e`.
- GSQ Qwen3-MoE-wrapper SHA-256:
  `6052e8e5a25ed4678278141a62af2404ec937b5828f6988e0bd7ae149b8c3e84`.

