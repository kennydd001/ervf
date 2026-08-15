# S13 — expert-unie over speculatieve vensters: de bouwpoort valt negatief uit

Datum: 2026-08-15
Verdict: **G-S13-C1 en G-S13-S1 gehaald, G-S13-U1 gefaald. De gemiddelde expert-unie over een venster van 5 opeenvolgende tokens is 19,512 van de 128 per MoE-laag — ruim boven de vooraf bevroren grens van 12,0. Een speculatieve lus bouwen op deze runtime is daarmee weerlegd vóór er een regel kernel geschreven is: de dominante MoE-term amortiseert niet over de gemeten acceptatie.**
Terminal state: `s13_expert_union_gate_failed_no_build`
Preregistratie: `S13_EXPERT_UNION_PREREGISTRATION_2026-08-15.md` (bevroren vóór uitvoering)

## 1. Waarom deze meting bestaat

S10A mat de batenkant van MTP-speculatie: gemiddeld `A` = 2,114 geaccepteerde
drafts per stap (poort gehaald). De kostenkant hangt op de vraag hoeveel unieke
experts een verificatie-sweep over `W` tokens per laag raakt, want de MoE-term
(39,5 van 54,3 ms/token bij 262K, S8) schaalt met die unie en niet met het
aantal tokens. S10A §5 zette de poort vooraf: unie > ~12 van 128 bij `W = 5` →
niet bouwen. Deze fase voert die meting uit en bouwt niets.

De uitkomst geldt ook het LIGHTNINGFLASH_50-programma uit
`info/SWEEPSPEC_50_PACK_2026-08-14`: elke denkbare draftbron (MTP, ngram-spine,
parallelle kop) deelt dezelfde verificatie-sweep. De unie per geverifieerd
token is een eigenschap van de target-routes, niet van de drafter.

## 2. De meting

Vier armen, één modelload, productieconfiguratie (v35-checkpoint, capacity 72,
`embed_on_host`, FP8-KV, greedy). Routes vastgelegd met het bestaande
`step(capture_routes=...)`; `runtime.py` onaangeraakt (input lock + verifier
bevestigen de hash). Vensters niet-overlappend, zoals speculatieve ronden.

| arm | stappen | unie @W=5 gemiddelde | p95 |
|---|---:|---:|---:|
| A-expository (ctx ~200) | 124 | 19,457 | 26 |
| A-narrative (ctx ~200) | 124 | 19,518 | 26 |
| A-code (ctx ~200) | 124 | 20,862 | 27 |
| B-4K (ctx 4.096) | 64 | 18,210 | 25 |
| **gepoold** | 436 | **19,512** | — |

De volledige venstercurve (gepooled gemiddelde, van 128 mogelijk):

| W | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| unie | 6,000 | 9,895 | 13,421 | 16,659 | 19,512 | 22,180 | 27,204 |

Elke extra token in het venster voegt ~2,5–3,9 experts toe die de vorige token
nog niet raakte. Dat is veel meer overlap dan de worst case (30 bij W=5) en
veel minder dan de beslisgrens eist.

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-S13-C1** | generatie bit-identiek aan de S10A-sequenties (3×124 tokens) | identiek, alle drie | ✅ |
| **G-S13-S1** | W=1-unie exact 6,0 overal; gepoold gemiddelde monotoon in W | 6,0 overal; monotoon | ✅ |
| **G-S13-U1** | gepoolde gemiddelde unie @W=5 ≤ 12,0 | **19,512** | ❌ |

G-S13-C1 is niet triviaal: de routes zijn vastgelegd in een generatie die
token-voor-token gelijk is aan de generatie waarin S10A de acceptatie mat. De
unie en de acceptatie komen dus uit dezelfde toestanden.

## 3. Wat de getallen betekenen (rekenwerk op gemeten componenten, geen meting)

Per **geverifieerde** token daalt de MoE-bytekost wél: 19,512/5 = 3,90
expert-records per token tegen 6,00 bij gewoon decoderen (−35%). Maar een sweep
van 5 posities levert bij de gemeten acceptatie slechts `A + 1` = 3,114
**gecommitteerde** tokens op. Per gecommitteerde token:

```
MoE-records per token:  19,512 / 3,114 = 6,27   versus 6,00 zonder speculatie
```

De speculatieve sweep beweegt dus **meer** MoE-bytes per opgeleverde token, niet
minder — en daar komt de MTP-draftketen van 19,10 ms per ronde bovenop (S10A,
componentmeting: 6,13 ms per geaccepteerd token). De enige termen die wél
amortiseren (attention-KV, `lm_head`, deels Mamba-gewichten) zijn bij 262K
samen kleiner dan de MoE-term die dat niet doet.

De grens ligt er scherp in: bij een hypothetische perfecte drafter
(`A = 4`, alles geaccepteerd) zou de MoE-kost per token 3,90/6,00 = 0,65×
worden. De gemeten drafter haalt 2,114, en juist het domein met de hoogste
acceptatie (code, `A` = 3,283) heeft óók de hoogste unie (20,862) — coherent
gedrag kost meer experts. De twee parameters die speculatie zouden redden
werken hier tegen elkaar.

## 4. Beslissing

G-S13-U1 is vooraf bevroren en is niet gehaald: **er wordt geen speculatieve
lus gebouwd.** Dat geldt voor MTP-D4 en evenzeer voor ngram- of boom-varianten
met vensters van deze orde — grotere vensters vergroten de unie alleen maar
(W=8: 27,2). De poort beslist zoals beloofd alleen de bouw; zij zegt niets over
andere assen (cachebeleid, sync-plaatsing, batch).

Dit sluit tevens het LIGHTNINGFLASH_50-spoor uit het SWEEPSPEC-pack voor dit
model in deze vorm: de poorten daar (`mean accepted depth >= 4`, 50 tok/s bij
4K) veronderstellen een verificatie-sweep die over tokens amortiseert; op dit
MoE-zware hybride model doet de dominante term dat bij de gemeten acceptatie
niet. De pack-hypothesen 1, 2 en 5 (offload-amortized verification, causal
diffusion forest, speculatieve overlap) erven allemaal deze voorwaarde.

## 5. Wat deze fase niet doet

Geen bouw, geen doorvoermeting, geen kwaliteitsclaim. 262K is niet gemeten
(een echte prefill kost ~2,6 uur sequentiële stappen; de unie is een
routeringsstatistiek en S10A vond `A` stabiel tussen ctx ~200 en 4.096, dus de
contextafhankelijkheid is secundair — dit staat hier expliciet, niet
stilzwijgend). De rekensommen in §3 zijn aritmetiek op eerder gemeten
componenten, uitdrukkelijk geen voorspelling van een eindgetal.

## 6. Onafhankelijke verificatie

`s13_independent_verify.py` importeert niets uit de runner. Hij hertokent de
gate-prompts uit het bevroren corpus, herhaalt de C1-vergelijking tegen
`s10a_mtp_acceptance.json`, herberekent elke unie-statistiek uit de ruwe routes
met een eigen telling (43.000+ vensters), controleert dat runner, runtime en
corpora nog op de input-lock-hashes staan, en evalueert de drie poorten
opnieuw. **43 van 43 checks, verdict `VERIFIED`.**

Protected manifest na deze fase: zie `protected_verification_after_s13.json`.

## 7. Claim boundary

Een routeringsstatistiek op echte greedy generatie bij korte en 4K-context, op
dit checkpoint, deze runtime en deze GPU. De beslissing "niet bouwen" volgt uit
een vooraf bevroren poort op die statistiek. Niets hiervan is een
tokens-per-seconde-meting, een uitspraak over 262K, andere domeinen, andere
drafters met wezenlijk hogere acceptatie dan gemeten, of andere hardware.

## 8. Artefacten

`S13_EXPERT_UNION_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/s13_expert_union.py` · `s13_expert_union.json` ·
`scripts/lightningstream_nemotron/s13_independent_verify.py` ·
`s13_independent_verification.json` · `s13_input_lock.json` ·
`protected_verification_after_s13.json`
