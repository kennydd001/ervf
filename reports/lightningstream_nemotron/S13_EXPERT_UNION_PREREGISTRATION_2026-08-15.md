# S13 — expert-unie over speculatieve vensters: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.** Geschreven ná S10A/S11/S12, die volledig
en ongewijzigd blijven staan.

## 1. Vraag

S10A mat de batenkant van MTP-speculatief decoderen: gemiddelde `A` = 2,114
geaccepteerde drafts per stap (poort G-S10-1 gehaald). De kostenkant hangt op
één ongemeten getal: **hoeveel unieke experts raakt een backbone-verificatie-
sweep over `W` opeenvolgende tokens per MoE-laag?** De MoE-term is 39,5 van de
54,3 ms per token bij 262K (S8) en schaalt met de unie van de routes in de
sweep, niet met 1 token. S10A §5 stelde daarom voor: meet de unie vóór er iets
gebouwd wordt; bij een gemiddelde unie > ~12 van de 128 per laag bij `W = 5`
verdubbelen de MoE-bytes per sweep terwijl er maar ~3,1 tokens uitkomen, en is
de bouw negatief vóór er een regel kernel geschreven is.

Deze fase meet dat. Zij bouwt **niets**: geen speculatieve lus, geen kernel,
geen wijziging aan `runtime.py`.

Deze meting is tevens de make-or-break-voormeting voor het LIGHTNINGFLASH_50-
programma uit `info/SWEEPSPEC_50_PACK_2026-08-14`: elke draftbron (MTP, ngram,
parallelle kop) deelt dezelfde verificatie-sweep, dus de unie per geverifieerd
token is een eigenschap van de routes van het target, niet van de drafter.

## 2. Metriek

Voor elke MoE-laag (23 stuks, top-k = 6 uit 128) en elke venstergrootte
`W ∈ {2, 3, 4, 5, 6, 8}`:

```
unie(laag, venster) = | ⋃_{t ∈ venster} routes(laag, t) |
```

met **niet-overlappende** vensters over de as van gegenereerde tokens — dat is
de geometrie van speculatieve ronden. Gerapporteerd worden per arm en gepoold:
gemiddelde, p50, p95 en maximum over lagen × vensters, plus per laag het
gemiddelde bij `W = 5`.

Beslismetriek (uit S10A §5): de gepoolde gemiddelde unie bij **W = 5**.
Secundair, zonder poort: `W = 4` (de werkelijke sweep omvat de 4 draft-posities;
de huidige positie is al berekend) en de volledige W-curve.

## 3. Armen

Eén modelload, één proces, configuratie identiek aan de huidige productierun
(`LS_MODEL_DIR=nemotron_3_5_lightning_v35`, capacity 72, `embed_on_host`,
FP8-KV, greedy argmax):

- **A-kort** — de drie bevroren S10A-gate-prompts uit `s10a_corpus.json`
  (expository / narrative / code), 124 decode-stappen per prompt, routes
  vastgelegd met het bestaande `step(capture_routes=...)`.
- **B-4K** — de eerste 4.096 tokens van `long_ctx_text` uit hetzelfde corpus als
  prompt, 64 decode-stappen, routes vastgelegd.

262K wordt **niet** gemeten: een echte prefill van 262.100 tokens kost op deze
decode-only runtime ~2,6 uur sequentiële stappen, en de unie is een
routeringsstatistiek waarvan de contextafhankelijkheid secundair is (S10A vond
`A` stabiel tussen ctx ~200 en 4.096: 2,114 vs 2,083). Dit staat als
begrenzing in de claim boundary, niet stilzwijgend.

## 4. Poorten

- **G-S13-C1 — de routes komen uit dezelfde generatie als de A-meting.** Voor
  elk van de drie gate-prompts is de gegenereerde tokenreeks bit-identiek aan
  posities `[prompt_tokens : prompt_tokens + 124]` van `sequence` in
  `s10a_mtp_acceptance.json`. Faalt dit voor een prompt, dan gelden de routes
  van die arm niet en wordt alleen over de overige armen gerapporteerd — de
  poort wordt niet verruimd.
- **G-S13-S1 — telcode-sanity.** Bij `W = 1` is elke unie exact 6,0 in elke arm
  en elke laag, en de gepoolde gemiddelde unie is monotoon niet-dalend in `W`.
  Faalt dit, dan is de telling stuk en is er geen resultaat.
- **G-S13-U1 — beslispoort.** Gepoolde gemiddelde unie bij `W = 5` ≤ 12,0 →
  S10 stap 2 (bouw van een speculatieve lus) is **niet weerlegd**; > 12,0 →
  stap 2 is **negatief** en er wordt niet gebouwd. De poort beslist alleen over
  de bouw, niet over de uiteindelijke haalbaarheid van 50 tok/s.

## 5. Verificatie

De onafhankelijke verifier importeert de runner niet. Hij leest de ruwe routes
uit `s13_expert_union.json`, herberekent alle unie-statistieken met een eigen
telling, hertokeniseert de gate-prompts uit `s10a_corpus.json`, herhaalt de
C1-vergelijking tegen `s10a_mtp_acceptance.json` en evalueert de drie poorten
opnieuw.

## 6. Claim boundary (vooraf)

Een routeringsstatistiek op echte greedy generatie, korte en 4K-context, op dit
checkpoint en deze runtime. Geen doorvoermeting, geen bouw, geen
kwaliteitsclaim, geen uitspraak over 262K of andere domeinen dan de drie
gemeten. De beslispoort zegt iets over MoE-bytes per verificatie-sweep en niets
over acceptatiegraad, draftkosten of eind-tok/s.

## 7. Artefacten (te produceren)

`scripts/lightningstream_nemotron/s13_expert_union.py` ·
`s13_expert_union.json` ·
`scripts/lightningstream_nemotron/s13_independent_verify.py` ·
`s13_independent_verification.json` ·
`S13_EXPERT_UNION_REPORT_2026-08-15.md` · `s13_input_lock.json`
