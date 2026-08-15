# Next-wave scopegrenzen na de fysieke campagne

Datum: 2026-08-12

Dit document voorkomt dat een verwante negatieve test onuitgevoerde varianten
per ongeluk als getest presenteert.

## N029 — async activation staging, meerdere rijen per lane en TMA

N1A testte gewone coöperatieve shared-memory-activationstaging; N1B testte
uitgelijnde Q5-weightloads; N1C testte virtuele reductiebreedtes. Geen daarvan
is een afzonderlijke `cp.async`/TMA-weightpipeline of een meerdere-outputrijen-
per-lane-kernel.

Een geldige TMA-proef vereist een nieuwe getegelde Q5-opslagindeling, omdat de
huidige 40-bit packs per rij niet als herbruikbare rechthoekige tegel worden
gelezen. Double-buffered activation staging heeft in de huidige GEMV bovendien
geen tweede activatietegel om met de MAC te overlappen. Dit blijft een apart
kernel-/bankformatproject; niet fysiek getest in deze campagne.

## N030 — reductiegraaf over andere kernelfamilies

N1C generaliseert exact over Q8- en Q5-GEMV-vormen. INT4, dense BF16 GEMM,
RMSNorm en softmax hebben andere reductie- en afrondingscontracten. P13 bewijst
alleen de specifieke attentionboom. Een compilertransformatie over deze families
vereist afzonderlijke referentiekernels en evaluators; niet getest.

## N031 — echte GEMM-prefill en fysieke 8K/32K-runtime

N3D sluit de fysieke sequentiële baseline; N3B/N3C sluiten capaciteit. Een echte
prefill-GEMM en 8K/32K-quality wall vereisen gewijzigde attention/KV-layout,
nieuwe kernels en verse lange-contextlabels. Geen geldige randomtensorproef kan
modelkwaliteit vervangen. Niet getest.

## N032 — volledige DeepSeek-V2-Lite fysieke runtime

P14A repliceert de Q5-modelkwaliteit, maar er is geen STREAMQ5-bank, cachepolicy,
trunkbank en decoder voor die architectuur. Het lokale checkpoint alleen maakt
dit nog geen kleine variant. Niet fysiek getest.

## Overige externe blokkades

- GemLite/CUTLASS/QUICK met identieke Q5-BF16-semantiek: afzonderlijke adapters
  en buildcampagne ontbreken.
- Tweede GPU-architectuur: hardware ontbreekt.
- Drafter/MTP-acceptance: passend artifact ontbreekt.
- Volledige Qwen3-Coder-Next-port: N4A/N4B slagen, maar de 46,5-GiB Q5-bank,
  echte routersporen en hybride DeltaNet-runtime zijn nog niet gebouwd.

Deze items zijn dus expliciet **niet getest**, niet stilzwijgend negatief.
