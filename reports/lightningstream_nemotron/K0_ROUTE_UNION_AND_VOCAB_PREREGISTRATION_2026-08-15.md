# K0/K2 — route-unie-census en actief vocabulaire: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Aanleiding: `LIGHTNINGSPEC_50_FINAL_PASS_2026-08-15` (Kimi), fasen P0 en P2.
Model: `models/nemotron_3_5_lightning_v35`

## 0. Kimi's correctie wordt overgenomen

Mijn S10-A-rapport stelde voor een vijf-token-route-unie boven ~12 experts per
laag als negatief te behandelen. **Dat was fout.** Bij een unie van 12 en 3,114
uitgestoten tokens per ronde is dat 12/3,114 = 3,85 expert-records per
uitgestoten token, tegen 6 nu — een verbetering van 36%, geen falen. De juiste
pariteitsgrens is die van Kimi:

```
U* = top_k × (A+1) = 6 × 3,114 = 18,684 unieke experts per laag
```

Er wordt hieronder **geen** drempel verzonnen. De AR-basislijn (6 records per
uitgestoten token) ís de drempel, en de meting produceert de curve.

## 1. Wat hier gemeten wordt, en wat niet

Meetbaar zonder één nieuwe kernel, en dus hier: **P0** (route-unie- en
miss-census) en het vocabulaire-deel van **P2**.

Niet hier, met reden: **P1** (exacte B-token verifier: ReplaySSM/STree-Mamba,
één KV-sweep voor B queries, expert-major MoE-GEMM) is een kernelproject van
meerdere dagen, en **P3–P8** hangen daar allemaal aan of vragen training. Die
worden niet "getest" maar benoemd met wat ze nodig hebben. Een preregistratie
die belooft wat ze niet meet is erger dan geen.

## 2. Fase K0 — route-unie en cache-miss-census

Geen nieuwe kernel. `step(capture_routes=…)` bestaat al en levert de **officiële**
route-ids per laag per token — dezelfde die de runtime gebruikt, niet een
herberekende top-k.

Bron: dezelfde bevroren prompts als S10-A (`s10a_corpus.json`), 120 greedy
gegenereerde tokens per prompt, plus de 4.096-token-arm met 60 tokens.

Gemeten per laag en per venster van `B` opeenvolgende tokens, voor
`B ∈ {2,3,5,7,9,13}` (dus `D ∈ {1,2,4,6,8,12}`):

- `U_B` = aantal unieke experts in het venster (mean/p50/p95/p99, per domein)
- expert-token-multipliciteit = `B × top_k / U_B`
- cache-replay van de bestaande per-laag-LRU bij capacity 32/48/56/60/64/72,
  in twee ordes: (a) AR, één token tegelijk zoals nu, en (b) rondegebaseerd,
  waarbij een ronde de unie in één keer opvraagt en alleen het geaccepteerde pad
  gecommitteerd wordt
- misses per **uitgestoten** token in beide ordes

Uitgestoten tokens per ronde volgen uit de gemeten S10-A-verdeling:
`emitted(D) = 1 + Σ_{k=1..D} P(A ≥ k)`, dus 1,786 / 2,378 / 2,808 / **3,114**
voor D = 1/2/3/4. Voor `D > 4` is `A` **niet gemeten** — daarvoor wordt alleen de
unie gerapporteerd plus de acceptatiegraad die pariteit zou vragen. Er wordt niet
geëxtrapoleerd.

**Poorten (de AR-basislijn is de drempel, niet een verzonnen getal):**

- **G-K0-1:** `U₅ / 3,114` expert-records per uitgestoten token, vergeleken met
  **6** (AR). Is het ≥ 6, dan is de rauwe-belasting-premisse onder H3 weerlegd.
- **G-K0-2:** misses per uitgestoten token, rondegebaseerd bij capacity 72,
  vergeleken met AR bij capacity 72. Is het ≥ AR, dan faalt H2's
  "cache-aware expert bytes/emitted token < autoregressive baseline".
- **G-K0-3:** de census moet ≥ 300 vensters per `B` per domein dekken.

## 3. Fase K2 — actief vocabulaire (MicroSpec), één variabele

De draft-keten kost 19,10 ms voor vier drafts. Elke draft doet één `lm_head` over
**131.072** rijen; S8 mat die projectie op 2,106 ms per aanroep. Als dat klopt,
zit ~44% van de keten in de LM-kop en niet in de experts.

Interventie, en verder niets: bouw per commit-positie een **context-lokaal**
vocabulaire uit de top-`N` rijen van de logits die de **backbone zelf** al
berekend heeft, en laat de vier drafts alleen daarover projecteren. `N ∈
{1024, 2048, 4096}`. De backbone-verificatie blijft ongewijzigd, dus dit kan de
uitvoer alleen via acceptatie raken.

Gemeten: recall van het echte doel-token binnen `V`, `A` per domein en gepoold,
en de keten-p50.

**Poorten:**

- **G-K2-R1:** recall van het doel-token binnen het actieve vocabulaire
  **≥ 99,5%** (Kimi's eigen poort, ongewijzigd overgenomen).
- **G-K2-A1:** gepoolde `A` daalt **niet meer dan 0,05 absoluut** onder de
  gemeten 2,114. Een vocabulaire-restrictie kan acceptatie alleen verliezen.
- **G-K2-T1:** keten-p50 daalt **≥ 30%** onder 19,10 ms. Onderbouwing vooraf:
  4 × 2,106 ms = 8,42 ms is 44% van de keten, en `N ≤ 4096` snijdt die term met
  ≥ 32×.

Kimi's volledige P2-poorten (keten ≤ 10 ms, gepoolde `A` ≥ 1,90, narrative
≥ 1,60, code ≥ 3,0, cache-capacity ≥ 56) gelden voor de **hele** compacte stapel
inclusief kwantisatie en windowing. Eén interventie kan die niet beslissen en er
wordt hier dus niet tegen afgerekend; ze worden gerapporteerd als open.

## 4. Fase K1 — waar de draft-keten zijn tijd laat (in de lus)

Zelfde methode als S12-R1, want dezelfde valkuil geldt: elke probe-arm ligt
tussen twee basislijn-armen en wordt tegen hun gemiddelde gemeten, met een eigen
lokale ruisvloer. Gerepliceerd worden `lm_head`, de zes experts, het
attention-blok, en `eh_proj`.

- **G-K1-D1:** een marginale waarde telt alleen als zij haar eigen lokale drift
  overschrijdt.
- Elke marginale waarde is een **ondergrens** (warme L2), wordt niet omgerekend
  naar een aandeel en niet naar tok/s.

## 5. Wat er niet gebeurt

Geen nieuwe kernel, geen speculatieve lus, geen training, geen wijziging aan
`runtime.py`. Geen enkele paper-versnelling wordt met een andere vermenigvuldigd.
`G-S10-C1` blijft staan voor als er ooit gebouwd wordt.

## 6. Namespace

Kimi's prompt vraagt om `reports/lightningspec_50/` e.d. De schrijfregels die ik
van de gebruiker heb, staan alleen de `lightningstream_nemotron`-paden toe. Ik
schrijf daarom dáár, met `K`-prefix. Het verplaatsen naar een eigen namespace is
een kopieeractie die de gebruiker kan vragen.

## 7. Artefacten

`scripts/lightningstream_nemotron/k0_route_union_census.py` ·
`k0_route_union_census.json` · `k1k2_draft_vocab_and_decomposition.json` ·
`scripts/lightningstream_nemotron/k0_independent_verify.py` ·
`k0_independent_verification.json` · rapport met claim boundary.

## 8. Claim boundary van dit document

Geen meting, geen resultaat.
