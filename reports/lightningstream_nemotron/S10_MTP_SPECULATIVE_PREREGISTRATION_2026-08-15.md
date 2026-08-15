# S10 — MTP speculatief decoderen: preregistratie

Datum: 2026-08-15
Status: **design bevroren, niet uitgevoerd.** Geen meting mag hieraan voorafgaan.
Model: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, `models/nemotron_3_5_lightning_v35`

## Waarom dit de enige overgebleven hefboom is

Alles in deze runtime is transfer- of leesgebonden. Elke optimalisatie tot nu toe
verlaagde **tijd per byte**. Vier hypotheses die dat verder wilden doen zijn deze
sessie gemeten en weerlegd: minder transcendentals, meer splits, launch-batching
(13% van de MoE-term), kleinere blokken (monotoon slechter).

Speculatief decoderen verlaagt **bytes per token**: één gewichts-sweep van het
grote model verifieert meerdere tokens. Dat is een andere as, en de enige die
nog open ligt.

Kimi's S4 sloot dit af met "0 van 24.147 keys" — correct **voor Nemotron 3 Nano**.
3.5 Lightning heeft `num_nextn_predict_layers: 1`,
`mtp_layers_block_type: ['attention','moe']` en **270 MTP-tensors**.

## Gemeten uitgangspunten (niet geschat)

| grootheid | waarde | bron |
|---|---:|---|
| MTP-blok totaal | 2.670.652.160 B | N2R tensor-inventaris |
| MTP-expert, up én down | BF16 `[1856,2688]` + `[2688,1856]` = 19.955.712 B | N2R |
| MTP-experts, 128 stuks | 2.554.331.136 B (2,38 GiB) | afgeleid |
| per draft-token, top-6 | **119.734.272 B** | 6 × 19.955.712 |
| hoofdmodel-token @262K | **54,28 ms** | S8/n7b |
| huidige cache | 4,328 GiB, hitrate 80,4% | n7b |
| VRAM vrij | **0,000 GiB** | n7b |
| opeenvolgende expert-overlap | 2,011 van 6 (0,335) | N7-A |

## De economie, vooraf uitgerekend

**Kostenkant.** De MTP-experts zijn BF16 en dus **4× groter per expert** dan de
NVFP4-routed experts (19,96 MB vs 4,99 MB). Er is geen VRAM vrij, dus MTP-experts
resident maken kost cache: 2,38 GiB van de 4,328 GiB → capacity ~72 → ~32, en de
hitrate zakt naar schatting 80,4% → ~65%. Dat kost de hoofdforward extra misses
(27 → ~48 records ≈ +2,2 ms).

**Batenkant.** Bij `A` geaccepteerde tokens per sweep en `D` gedrafte tokens:

```
kosten per sweep  = D x MTP_forward + hoofdforward(D+1 tokens)
tokens per sweep  = A + 1
```

Met een geschatte MTP-forward van ~2 ms (119,7 MB device-lees plus één attention-
en één norm-laag) en een hoofdforward die bij verificatie van meerdere tokens
sublineair groeit — N7-A's 0,335 overlap betekent dat 4 tokens ~15–18 unieke
experts per laag nodig hebben in plaats van 24:

| A geaccepteerd (D=4) | ms/token | tok/s |
|---:|---:|---:|
| 1 | ~34 | ~29 |
| 2 | ~23 | ~43 |
| 3 | ~17 | ~58 |

**De hele zaak hangt op `A`, en `A` is ongemeten.** Daarom is stap 1 hieronder
een meting en geen bouw.

## Stap 1 — acceptatiegraad meten vóór er iets gebouwd wordt

Draai het MTP-blok en het hoofdmodel los van elkaar op dezelfde echte prompts.
Voor elke stap: laat MTP `D=4` tokens voorstellen, laat het hoofdmodel greedy de
werkelijke tokens produceren, en tel hoeveel voorstellen matchen tot het eerste
verschil. Geen speculatieve runtime, geen gedeelde staat — alleen twee forwards
en een vergelijking.

Uitkomst: de verdeling van `A` over ≥ 200 stappen en 3 prompts.

**Poort G-S10-1: gemiddelde `A` ≥ 1,5.** Daaronder is de rekensom hierboven
negatief en wordt S10 gesloten zonder bouw.

## Stap 2 — alleen bij een geslaagde stap 1

Speculatieve lus met bevroren ontwerp: MTP-blok resident (2,38 GiB), cache
teruggebracht tot wat overblijft, verificatie van `D+1` tokens in één sweep met
de unie van hun routes, greedy acceptatie tot het eerste verschil.

Poorten:
- **G-S10-C1:** de geaccepteerde tokenreeks is **identiek** aan de niet-speculatieve
  greedy generatie over 2 × 64 tokens. Speculatief decoderen dat de uitvoer
  verandert is een fout, geen afweging.
- **G-S10-P1:** ctx 262100 p50 ≥ 25 tok/s (huidige stand 18,424).
- **G-S10-P2:** ctx 262100 p50 ≥ 35 tok/s.
- **G-S10-P3:** ctx 0 geen regressie onder 25 tok/s.

## Risico's die de meting moet uitwijzen

1. **MTP is BF16 en 4× duurder per expert.** Als de MTP-forward duurder uitvalt
   dan ~2 ms, kantelt de rekensom snel.
2. **De cache krimpt met 2,38 GiB.** De extra misses in de hoofdforward moeten
   uit de winst betaald worden.
3. **Verificatie van D+1 tokens vraagt de unie van hun routes.** Bij 0,335
   overlap groeit dat sublineair, maar niet gratis — dat moet gemeten, niet
   aangenomen.
4. Bij lange context is de attention-term per geverifieerd token vrijwel
   constant, dus de winst kan bij 262K kleiner zijn dan bij ctx 0.

## Claim boundary

Deze preregistratie bevat **geen meting**. De tabel met tok/s is een
rekenvoorbeeld uit gemeten byte- en tijdgetallen, expliciet geen voorspelling en
zeker geen resultaat. Niets hiervan mag geciteerd worden als prestatie.

## Artefacten die stap 1 moet opleveren

`scripts/lightningstream_nemotron/s10a_mtp_acceptance.py` ·
`s10a_mtp_acceptance.json` · onafhankelijke verifier ·
`protected_verification_after_s10a.json`
