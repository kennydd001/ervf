# S10-A — MTP-acceptatiegraad: gemeten

Datum: 2026-08-15
Verdict: **G-S10-1 gehaald. Gemiddelde `A` = 2,114 over 360 stappen en 3 prompts (poort 1,5). S10 blijft open. Maar de kostenkant die deze fase en passant mat, is 2,4× duurder dan de bovenliggende preregistratie aannam, en de beslissende term voor stap 2 is nog steeds ongemeten.**
Terminal state: `s10a_acceptance_gate_passed_cost_side_worse_than_assumed`
Preregistratie: `S10A_MTP_ACCEPTANCE_PREREGISTRATION_2026-08-15.md` (bevroren vóór uitvoering)

## 0. Reproductie van de uitgangsmeting

Vóór er iets nieuws gemeten werd, is de bestaande meting opnieuw gedraaid met
dezelfde runner en configuratie (`n7b_cached_decode.py --capacity 72
--embed-on-host --max-ctx 262144`):

| context | bevroren | gereproduceerd | Δ |
|---:|---:|---:|---:|
| 0 | 27,743 | 27,574 | −0,6% |
| 32.768 | 26,200 | 25,523 | −2,6% |
| 131.072 | 21,699 | 21,794 | +0,4% |
| 262.100 | 18,424 | 18,358 | −0,4% |

Shell 2,521 GiB, cache 4,328 GiB, vrij 0,000 GiB — identiek. Generatie
byte-identiek. De omgeving is in orde. De vorige meting is bewaard als
`n7b_cached_decode_prior_20260815T0610Z.json`.

## 1. A1 — de wiring is empirisch vastgesteld, en dat was nodig

De MTP-bedrading stond niet vast. `transformers` 5.15.0 negeert de MTP-tensors
expliciet (`_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`) en de modelmap
bevat geen `modeling_*.py`, dus er is geen referentie-implementatie om tegen af
te lezen. Twee dingen waren daardoor onbepaald: de volgorde in de concat naar
`eh_proj`, en welke backbone-hidden `hnorm` voedt.

Alle vier de combinaties zijn gedraaid op bevroren, teacher-forced WikiText-2
(rijen 50/55/56/61, 632 posities per variant, disjunct van alles wat A2 gebruikt).
Beslisregel vooraf: laagste gemiddelde NLL wint, met een afbreekplafond van
7,0 nats.

| variant | gem. NLL | mediaan | top-1 vs. echte tekst |
|---|---:|---:|---:|
| **eh_post** | **3,2625** | 2,0972 | **0,4241** |
| eh_pre | 4,0180 | 3,0140 | 0,3465 |
| he_pre | 10,5343 | 10,0855 | 0,0237 |
| he_post | 10,4575 | 10,3695 | 0,0095 |

De concat-volgorde is niet marginaal maar categorisch: `[hnorm(h) ; enorm(emb)]`
zit op 10,5 nats, vlakbij uniform over 131.072 tokens (11,78). De juiste wiring is

```
h_mtp = eh_proj( concat( enorm(embed[τ]), hnorm(norm_f(h)) ) )
```

— dus de **genormaliseerde** backbone-hidden, ná `norm_f`, niet de ruwe
residual. Dat verschil is subtiel (3,26 vs 4,02 nats) maar consistent in beide
metrieken.

Schaalanker uit dezelfde run: de backbone zelf haalt **2,473 nats** op de
next-token-taak op deze tekst. Het MTP-blok haalt 3,262 nats op de veel
moeilijkere next-**next**-token-taak. Dat is de verhouding die je bij een
werkend MTP-blok verwacht, en het is het sterkste onafhankelijke bewijs dat de
wiring klopt.

## 2. A2 — de poort

`D = 4` voorstellen per stap, geketend op de MTP's eigen hidden state, greedy
vergeleken met de tokens die de backbone zelf produceert.

| prompt | stappen | gem. `A` | A=0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| expository | 120 | 1,850 | 27 | 28 | 25 | 16 | 24 |
| narrative | 120 | 1,208 | 43 | 36 | 21 | 13 | 7 |
| code | 120 | 3,283 | 7 | 6 | 12 | 16 | 79 |
| **gepoold** | **360** | **2,114** | 77 | 70 | 58 | 45 | 110 |

| poort | vereist | gemeten | |
|---|---:|---:|:--:|
| **G-S10-1** | gem. `A` ≥ 1,5 | **2,114** | ✅ |
| | ≥ 200 stappen | 360 | ✅ |
| | ≥ 3 prompts | 3 | ✅ |

**Twee dingen die de gepoolde 2,114 verbergt.** Ten eerste: de spreiding over
domeinen is enorm. `narrative` haalt 1,208 en zou de poort **alleen niet halen**;
`code` haalt 3,283. Wie S10 op één domein beoordeelt krijgt een ander antwoord.
De poort was vooraf gedefinieerd als het gepoolde gemiddelde en blijft dat.

Ten tweede: de acceptatieladder daalt nauwelijks.

| positie | P(accept | vorige geaccepteerd) |
|---|---:|
| u₁ | 0,7861 |
| u₂ | 0,7527 |
| u₃ | 0,7277 |
| u₄ | 0,7097 |

Dat is ongewoon vlak. Normaal stort de acceptatiekans in bij het derde of vierde
voorstel omdat het draftmodel zonder verse hidden state wegdrijft. Hier niet, en
`A = 4` is met 110 van 360 stappen de **grootste** klasse. Als er ooit gebouwd
wordt, is `D > 4` daarmee een serieuze kandidaat — die 110 stappen zijn
afgekapt door `D`, niet door het model.

Generaties waren coherent in alle drie de domeinen, bijvoorbeeld:
`" the technology was widely adopted. James Watt's improvements in the 1760s
made the steam engine practical for industrial use…"`

## 3. Secundaire arm — acceptatie bij diepere context (geen poort)

4.096 tokens natuurlijke tekst als prompt, 60 meetstappen: gemiddelde `A` =
**2,083** (histogram 18/6/9/7/20). Dat is binnen ruis gelijk aan de gepoolde
2,114 bij korte context. Het is één observatie op 4K en dekt 262K niet, maar het
weerlegt wel de zorg dat de acceptatie met diepte instort.

## 4. De kostenkant — wat deze fase en passant mat

Dit zijn **componentmetingen** en worden hieronder niet naar tokens per seconde
omgerekend.

| grootheid | gemeten |
|---|---:|
| MTP-keten van 4 drafts, p50 @ctx ~200 | **19,10 ms** |
| idem @ctx 4.096 | 21,01 ms |
| per MTP-forward | ≈ 4,78 ms |
| aanname in de S10-preregistratie | ~2 ms |

De MTP-forward is dus **2,4× duurder dan aangenomen**, en dat is met alle 128
MTP-experts **device-resident** — de gunstigst mogelijke geheugenopstelling.
Streamen kan alleen duurder zijn.

Het geheugenbeeld is wél beter dan de preregistratie vreesde:

| | baseline (n7b) | deze meting |
|---|---:|---:|
| shell | 2,521 GiB (incl. 262K KV) | 1,795 GiB (8K KV) |
| backbone-cache | 4,328 GiB (capacity 72) | 1,924 GiB (capacity 32) |
| MTP-experts | — | 2,379 GiB |
| vrij | 0,000 GiB | 0,577 GiB |

Bij 262K zou dat neerkomen op shell 2,521 + MTP 2,379 → een cachebudget van
ruwweg 1,95 GiB, oftewel **capacity ≈ 30**. Residente MTP-experts zijn dus
betaalbaar, maar precies door de cache te halveren, en N7-A's simulatie geeft bij
capacity 32 een hitrate van 0,650 tegenover de 0,804 die nu bij 72 gemeten is.

## 5. Wat dit wel en niet beslist

De poort is gehaald, dus **S10 gaat niet dicht** — dat is de afspraak en die
wordt niet achteraf herschreven omdat de kostenkant tegenvalt.

Maar de poort beslist alleen de batenkant. Voor stap 2 is dit de rekensom, en
het is een **rekensom, geen meting**:

- tokens per sweep = `A + 1` = **3,114**
- MTP-kosten per geaccepteerd token = 19,10 / 3,114 = **6,13 ms**
- het gebruikersdoel van 50 tok/s vraagt 20 ms/token; de MTP-keten alleen eet
  daar al ruim 30% van op, vóór de backbone iets doet

En dan de term die alles beslist en die **niemand gemeten heeft**: wat kost een
backbone-verificatie-sweep over `D+1 = 5` tokens? Speculatief decoderen wint
normaal omdat één gewichts-sweep meerdere tokens verifieert. In dit hybride model
geldt dat maar voor een deel van de lagen:

| term | S8-meting @262K | gedrag bij 5 tokens in één sweep |
|---|---:|---|
| attention (6 lagen) | 18,634 ms | KV wordt één keer gelezen voor 5 queries → amortiseert goed |
| Mamba (23 lagen) | 8,309 ms | recurrentie is sequentieel, maar `in_proj`/`out_proj` worden één keer gelezen → amortiseert deels |
| `lm_head` | 2,106 ms | amortiseert |
| **MoE (23 lagen)** | **39,523 ms** | schaalt met de **unie** van 5 routes, niet met 1 |

De MoE-term is 39,5 van de 54,3 ms per token, en dat is precies de term die
níét meeschaalt. Uit N7-A's gemeten opeenvolgende overlap (2,011 van 6) volgt
een **bovengrens** van 6 + 4 × (6 − 2,011) = 21,96 unieke experts per laag bij 5
tokens, tegen 6 nu: een factor **≤ 3,66**. De echte unie ligt lager, want ook
niet-aangrenzende tokens delen experts — maar hoeveel lager is ongemeten, en op
dat getal staat of valt de hele zaak. Bij 3,66 gaat de rekensom negatief; bij
een unie die dicht bij 6 blijft, positief.

**Aanbeveling voor S10 stap 2, mét poort vóór er iets gebouwd wordt:** meet die
unie eerst. Het kost geen bouw — `step(capture_routes=...)` bestaat al, en de
gegenereerde reeksen uit deze fase liggen op schijf. Tel het aantal unieke
experts per laag over elk venster van 5 opeenvolgende tokens en over de echte
routes, niet over een aanname. Poortvoorstel: als de gemiddelde unie over 5
tokens groter is dan ~12 van de 128 per laag (een verdubbeling t.o.v. 6), dan
verdubbelen de MoE-bytes per sweep terwijl er maar 3,1 token uitkomt, en is stap
2 negatief vóór er een regel kernel geschreven is.

Verder, als er ooit gebouwd wordt, blijft **G-S10-C1** staan: de geaccepteerde
tokenreeks moet identiek zijn aan de niet-speculatieve generatie.

## 6. Bedreigingen voor de geldigheid

Alle vooraf benoemd in §8 van de preregistratie; hier de stand na de meting.

1. **De wiring is empirisch, niet uit een referentie.** Gekozen is de van vier
   kandidaten voor MTP gunstigste. Een poortfalen zou daardoor conservatief zijn
   geweest; een poort-**slagen** is dat niet, en is dus niet definitief. Het
   NLL-anker (3,26 vs backbone 2,47) maakt het onwaarschijnlijk dat er een veel
   betere wiring bestaat, maar sluit het niet uit.
2. **262K is niet gemeten.** De diepste arm is 4.096 tokens.
3. **De MTP-experts waren resident.** In een gebouwd systeem kost dat de halve
   backbone-cache; §4 becijfert dat, maar meet het niet.
4. **De backbone draaide met FP8-KV**, zoals hij draait. `A` ten opzichte van een
   FP32-KV-model is niet gemeten.
5. **Greedy.** Onder sampling verandert de acceptatielogica.
6. **Domeinspreiding.** 1,208 tot 3,283 tussen drie prompts. Drie prompts is wat
   de preregistratie eiste; het is weinig voor een spreiding van die orde.

## 7. Onafhankelijke verificatie

`s10a_independent_verify.py` importeert niets uit de runner, herberekent alle
`A`-waarden uit de ruwe token-lijsten met een eigen matchtelling, hertokeniseert
de bevroren prompts met de tokenizer van het model, hercontroleert de
corpus-hashes tot aan het bron-parquet, en leidt de A1-winnaar opnieuw af uit de
vier NLL's. **48 van 48 checks, verdict `VERIFIED`.**

Protected manifest na deze fase: **0 modified / 0 removed**.

## 8. Claim boundary

Gemeten is het aantal geaccepteerde tokens `A` van vier geketende MTP-drafts ten
opzichte van de greedy-tokens die **deze runtime zelf** produceert, op deze GPU,
bij korte tot middellange context. Er is **geen speculatieve lus gebouwd** en er
is **geen doorvoer gemeten**. De 19,10 ms MTP-keten is een componentmeting met
alle experts resident; die mag niet naar tokens per seconde worden omgerekend en
is dat hier ook niet. De rekensom in §5 is aritmetiek op gemeten componenten,
uitdrukkelijk geen voorspelling en zeker geen resultaat. Geen kwaliteitsclaim,
geen benchmarkscore, geen uitspraak over andere hardware, batchgroottes of
prompts.

## 9. Artefacten

`S10A_MTP_ACCEPTANCE_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/s10a_extract_corpus.py` · `s10a_corpus.json` ·
`src/moe_lab/lightningstream_nemotron/mtp.py` ·
`scripts/lightningstream_nemotron/s10a_mtp_smoke.py` ·
`scripts/lightningstream_nemotron/s10a_mtp_acceptance.py` ·
`s10a_wiring_resolution.json` · `s10a_mtp_acceptance.json` ·
`scripts/lightningstream_nemotron/s10a_independent_verify.py` ·
`s10a_independent_verification.json` · `n7b_cached_decode.json` (reproductie) ·
`n7b_cached_decode_prior_20260815T0610Z.json` (bevroren voorganger) ·
`protected_verification_after_s10a.json`
