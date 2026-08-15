# Offload-roofline — definitief oordeel

**Oordeel: geen Eureka. Eén conditioneel ondersteunde roofline, twee negatieve
hypothesen en drie door ontbrekende uitvoeringsartifacts begrensde claims.**

## P-A — Qwen wall-clock

De historische blokkade "slechts 16/6.144 experts" is door het latere
CORETAIL-onderzoek gedeeltelijk ingehaald: voor de exacte Q2-GPTQ-route zijn nu
alle 6.144 experts geproduceerd, een fysiek full-bankformaat en fused
microkernel onafhankelijk geverifieerd, en full-depth kwaliteit gemeten.

Dat opent P-A niet alsnog. De geregistreerde P-A vereist een Q4-runtime met
asynchrone expertcache; CORETAIL test een andere exacte Q2-representatie.
Bovendien faalt die route de vooraf vastgelegde kwaliteitsgate met +42,943%
relatieve test-CE. Daardoor is haar geïntegreerde wall-clocktest niet
geautoriseerd. Er blijft dus geen tok/s- of rolloutclaim voor P-A.

## P-B — online residency

Negatief, onafhankelijk 14/14 geverifieerd. Bij 4.700 slots passen expertcache
plus INT4-trunk in 5,702307 GiB en zijn alle domeingemiddelden ≤3 cold calls per
token. De tail-gate faalt echter: math p99=18 en instruction p99=31. Slechts
3/20 gerichte domeinwissels herstellen binnen 200 tokens. De oorspronkelijke
stelling dat een gewone cumulatieve LFU de statische HERA-uitkomst oplost is
daarmee verworpen.

## P-C — K3-roofline

De lokale hardwareleg is nieuw gemeten en 15/15 geverifieerd. De PCIe 5.0 ×8-
link haalt bij 512 MiB een mediaan van 26,158915 GB/s. Als K3 werkelijk 27,28
GB actieve trunkbytes per token vereist, is het conditionele plafond 0,958905
tok/s. Dat ondersteunt `≤1 tok/s`, maar met slechts circa 4,1% marge.

De 27,28-GB-input, actieve K3-trunkbytes en een 64-token K3-decode zijn niet
lokaal gemeten; er is geen K3-checkpoint/runtime. De volledige P-C-claim is dus
niet bewezen.

## P-D — speculative decoding

Geblokkeerd op de primaire acceptatiegate: geen K3-target, geen zelfstandige
H3-drafter en geen klein draftcheckpoint/runtime. Het uitvoerbare subdeel toont
wel sterke tijdcorrelatie in echte Qwen-routes: bij diepte 8 zijn gemiddeld
23,918–30,398 unieke experts per laag in plaats van de naïeve 64.

De formule uit de bronanalyse is ook gecorrigeerd: voor `E=896`, `k=16`, `s=8`
is de uniforme verwachting 120,279427, niet 118,6; de naïef/uniek-factor is
1,0642×, niet ongeveer 1,08×. Zonder geaccepteerde tokens per pass bewijst dit
geen speculative speed-up.

## P-E — neuronpermutatie

Negatief, onafhankelijk 18/18 geverifieerd. De expert-specifieke gebalanceerde
spectrale permutaties zijn uitsluitend op validation-indices 256–1023 geleerd;
validation/test 0–255 bleven hold-out. De historische neuron- en tilemaskers en
metrics zijn exact gereproduceerd.

De gepermuteerde tile-64 verbetert validation van 8,635× naar 5,014× de
neuronoracle-KL, maar test slechts van 8,479× naar 8,322×. Beide missen de
1,20×-gate ruim. De geëiste bit-identieke volledige reconstructie faalt ook:
BF16-down-GEMM-reductieorde geeft NRMSE 0,003901 en maximum absolute fout 4,0.

## Broncorrecties

De aangeleverde HERA-domeinaantallen waren licht verouderd. De officiële
geverifieerde waarden zijn code 4.173, general 4.449, math 4.823,
multilingual 4.317 en instruction 4.951. HERA’s `static_tier_negative` blijft
correct; P-B testte een nieuwe online policy en faalde op tail en adaptatie.

Alle huidige 156 repositorytests slagen.
