# PH1 Intel execution R6P — onafhankelijke frozen source-audit

Datum: 2026-08-14  
Methode: read-only bron-, lock- en schema-audit. Geen preflight, payload, compiler of device uitgevoerd.

## Verdict

**GO voor exact één uitvoering van `preflight_het_next_l0_ph1_intel_execution_r6p.py`.** Dit is uitsluitend een no-device statische preflighttoestemming; payload- en fysieke Intel-uitvoering blijven gesloten.

Beoordeelde freeze:

- R6P-preflight `3e91730cfe51730857825279d4c057283785beac5aa318d1cd8d45bad42b0e69`;
- R6P-preregistratie `84153ea498344b3d66a95943ec07a4e1a8cb7f1fea0368fce911eaa97bca7d5c`;
- R6P-lock `fb87932f7485cefcfc26dfde74348bea8eeae15b99279b4e0fa9669a24cf430e`.

De lock bindt de frozen R6-implementatie, R6-audit `0020a9ee…`, crashdiagnose `ee3bb47d…` en oude gecrashte R6-preflight `363682ed…`. R6-output en R6P-preflightresultaat zijn afwezig; lock staat closed/PENDING.

## Crashreparaties

### Beide width-64 callsites: PASS

- De standalone reductiesentinel gebruikt nu `(1,512)` en `(1,2048)` met uitsluitend de eerste acht woorden `0x3f80`; beide eisen exact `0x4100`.
- De production verifierfunctie `linear()` wordt niet vervangen of gemonkeypatcht.

### Full-shape verifierfixture: PASS

- weights: gate/up `[512,2048]`, down `[2048,512]`;
- input: 2048 BF16 woorden / 4096 bytes;
- gate/up/silu/activation: elk 512 BF16 woorden / 1024 bytes;
- down: 2048 BF16 woorden / 4096 bytes;
- counters: 512/512/512/2048 little-endian `uint32(1)`, exact 2048/2048/2048/8192 bytes;
- records: drie maal 675840 bytes met de bestaande canonical offsets;
- BUFF, ARGS en LAUNCH zijn inhoudelijk exact de frozen productioncontracten.

De fixture bouwt daarna de volledige ledger, 95 ownershiprijen, 12 resourcesamples, outputs en stagehashes en laat de echte onafhankelijke verifier iedere positieve basis en iedere mutation opnieuw beoordelen.

## Provenance en regressies

- R6P verandert geen runner, backend, common, verifier, kernel, codec, buffercontract, launch, threshold, identiteit of claim.
- Alle R6 ownership-, promoted-cleanup-, exact-zero-status-, lifecycle-, transaction-, bundle-, resource-, control- en mutationchecks worden uit de hashgebonden oude R6-preflight hergebruikt; alleen de twee gecrashte fixturefuncties zijn vervangen.
- Tijdelijk overschreven verifierconstanten worden in `finally` hersteld.
- Geen deterministische shape-, byte-, counter-, manifest- of cardinaliteitsblocker gevonden.

## Niet-blokkerende observatie

De full-shape integer-oracle wordt voor de basisfixture en opnieuw voor iedere mutation uitgevoerd. Dat kan CPU-intensief zijn, maar is methodologisch correct en geen devicecall of wetenschappelijke wijziging.

## Toegestane volgende stap

Voer exact de frozen R6P-preflight eenmaal uit. Alleen een volledig PASS-resultaat met onveranderde R6/R6P-hashes mag worden gebruikt voor een afzonderlijke authorization-only revision. Deze audit opent de fysieke runner niet.
