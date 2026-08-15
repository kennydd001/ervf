# S10-A — MTP-acceptatiegraad: preregistratie

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.** Geen enkele meting uit deze fase bestond
toen dit document werd geschreven.
Model: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, `models/nemotron_3_5_lightning_v35`
Voorafgaand: `S10_MTP_SPECULATIVE_PREREGISTRATION_2026-08-15.md` (stap 1),
`s10a0_mtp_structure.json` (wiring-inventaris).

## 1. Wat hier gemeten wordt

Eén grootheid: **`A`**, het aantal MTP-voorgestelde tokens dat overeenkomt met de
tokens die het hoofdmodel greedy zélf produceert, geteld tot het eerste verschil,
met `D = 4` voorstellen per stap. `A ∈ {0,1,2,3,4}`.

Er wordt **geen speculatieve lus gebouwd**. Twee losse forwards, geen gedeelde
staat tussen hoofdmodel en MTP behalve de expliciet doorgegeven hidden state.
`runtime.py` wordt niet gewijzigd; de meting leest alleen `rt.h` uit na `step()`.

## 2. Wat hier níét gemeten wordt

- Geen tok/s. Geen doorvoer. Geen snelheidswinst. De MTP-forwardtijd die deze
  fase incidenteel oplevert is een **componentmeting** en wordt niet omgerekend
  naar tokens per seconde (registry: `forbidden_hypotheses`, laatste regel).
- Geen acceptatiegraad bij 262K context. Zie §8.
- Geen uitspraak over sampling; alles is greedy (argmax, gelijkspel naar de
  laagste token-id).

## 3. De wiring: wat vaststaat en wat niet

Vast uit `s10a0_mtp_structure.json` (270 tensors, 2.670.652.160 B):

```
x  = eh_proj( concat( enorm(embed[τ]), hnorm(ρ) ) )        # [2688,5376] BF16
x += o_proj( GQA-attn( q,k,v van rmsnorm(x, layers.0.norm) ) )   # 32/2/2 × 128, eigen KV
x += moe( rmsnorm(x, layers.1.norm) )                      # gate top-6 van 128 BF16 + shared
y  = rmsnorm(x, layers.1.final_layernorm)
logits = BACKBONE lm_head (NVFP4) · y
```

Niet-ambigue keuzes, overgenomen van de backbone omdat het dezelfde architectuur
is (`config.mlp_hidden_act = relu2`, `norm_topk_prob = true`, `n_group = 1`):
router = sigmoid(logits) + `e_score_correction_bias`, top-6 op die som, gewichten
= genormaliseerde sigmoid-scores × `routed_scaling_factor` (2,5); expert =
`down( relu(up(x))² )`, geen gate-projectie; shared expert met gewicht 1,0;
attention zonder RoPE (Nemotron-H hybride: positie zit in Mamba, en de
backbone-attentie in deze runtime gebruikt evenmin RoPE).

**Twee dingen staan niet vast** en zijn uit geen enkel artefact af te leiden.
De HF-referentie helpt niet: `transformers` 5.15.0 negeert de MTP-tensors
expliciet (`_keys_to_ignore_on_load_unexpected = [r"mtp.*"]`), en de
modelmap bevat geen `modeling_*.py`.

| # | ambiguïteit | varianten |
|---|---|---|
| W1 | volgorde in de concat naar `eh_proj` | `[emb ; h]` of `[h ; emb]` |
| W2 | bron van `ρ` | backbone-hidden ná het laatste blok, vóór `norm_f` — of ná `norm_f` |

## 4. Fase A1 — wiring-resolutie (geen hypothesetoets)

Dit is een *bepaling*, geen experiment, in dezelfde geest als
`n3_nibble_order_resolution.py`. Alle vier de combinaties W1×W2 worden gedraaid.

Meetopzet: teacher-forced natuurlijke tekst (WikiText-2 raw, validation, vaste
rij-indices, vastgelegd in `s10a_corpus.json` mét sha256 van het bronbestand).
Voor elke positie `i`: backbone-forward op de echte tekst, dan één MTP-forward
met `τ = S[i+1]`, `ρ = h_i`, en de NLL van de MTP-draftlogits t.o.v. het
**echte** token `S[i+2]`. Geen ketening; dit isoleert de wiring.

- ≥ 256 gescoorde posities.
- **Beslisregel, vooraf vastgelegd:** kies de variant met de laagste gemiddelde
  NLL. Dit is bewust de voor MTP **gunstigste** variant, zodat een latere
  poortfalen conservatief is.
- **Afbreekregel:** als de winnende variant een gemiddelde NLL > 7,0 nats heeft,
  is geen enkele variant een plausibele wiring. Dan wordt A2 **niet** gedraaid en
  wordt S10 **niet gesloten**; het rapport meldt "wiring niet vastgesteld, poort
  niet evalueerbaar". (Referentiepunten: uniform over 131.072 tokens = 11,78
  nats; de backbone zelf haalt op deze tekst een next-token-NLL die apart wordt
  gerapporteerd als schaalanker.)
- Alle vier de NLL's worden gerapporteerd, ook de verliezers.

De corpusrijen van A1 zijn **disjunct** van alles wat A2 gebruikt.

## 5. Fase A2 — de acceptatiemeting (de poort)

Realisatie van de tokenreeks `S`: prompt-tokens gevolgd door greedy generatie van
het hoofdmodel. `h_i` is de backbone-hidden ná het consumeren van `S[i]`.

Per meetstap `i`:

1. MTP-commit op positie `i`: forward(`τ = S[i+1]`, `ρ = h_i`) → `x₁`, draft `u₁`.
2. Ketening `j = 2..4`: forward(`τ = u_{j-1}`, `ρ = x_{j-1}`) op positie `i+j-1`
   → `x_j`, `u_j`. (Eén MTP-module, dus meerstaps-drafting draait recursief op
   zijn eigen hidden state — er is geen tweede module.)
3. `A_i` = aantal leidende overeenkomsten van `(u₁,u₂,u₃,u₄)` met
   `(S[i+2],S[i+3],S[i+4],S[i+5])`.

De MTP-KV op positie `i` is gecommitteerd (echt token, echte `h_i`); posities
`i+1..i+3` zijn speculatief en worden bij de volgende stap overschreven. Zij
worden nooit gelezen voorbij `t = pos+1`, dus er lekt geen stale staat.

**Configuratie, bevroren vóór uitvoering:**

| parameter | waarde | reden |
|---|---|---|
| `D` | 4 | uit S10-preregistratie |
| prompts | 3, verschillende domeinen | uit S10-preregistratie |
| meetstappen | ≥ 120 per prompt (≥ 360 totaal) | poort eist ≥ 200 |
| backbone | `capacity 32`, `--embed-on-host`, FP8-KV, `max_ctx 8192` | capacity raakt alleen residentie, niet de waarden: de cache bevat exacte kopieën. Kleinere capacity maakt VRAM vrij voor de MTP-experts |
| MTP-experts | BF16, alle 128 device-resident (2,38 GiB) | meting, geen prestatieclaim |
| MTP-KV | FP32 | meest getrouwe MTP; een gebouwde versie met FP8-KV moet apart geverifieerd worden |
| decodering | greedy argmax, gelijkspel → laagste id | determinisme |

**Poort G-S10-1: gemiddelde `A` ≥ 1,5**, gepoold over alle meetstappen van de
drie prompts. Deze poort wordt na het zien van het resultaat niet verruimd,
niet herwogen en niet per prompt heronderhandeld. Onder 1,5 gaat S10 dicht
zonder bouw.

Gerapporteerd: de volledige verdeling van `A` over `{0,1,2,3,4}`, het gemiddelde
per prompt, het gepoolde gemiddelde, en de acceptatiekans per draftpositie
(`P(u_j` correct `| u_{1..j-1}` correct`)`).

## 6. Secundaire observatie (uitdrukkelijk geen poort)

Eén extra passage van ~4.096 tokens natuurlijke tekst, 60 meetstappen, om te
zien of `A` met contextdiepte verschuift. Dit telt **niet** mee in G-S10-1 en
kan de poort niet redden of laten vallen. Reden om het toch te doen: de hele
S10-rekensom in de bovenliggende preregistratie staat op 262K, en `A` bij diepe
context is ongemeten.

## 7. Verifier

Een aparte `s10a_independent_verify.py` die `s10a_mtp_acceptance.json` inleest en
**zonder de runner te importeren** herberekent: de verdeling van `A` uit de ruwe
per-stap draft- en referentietokens, het gepoolde gemiddelde, de poortuitslag,
de corpus-sha256, en of het aantal meetstappen ≥ 200 is. De verifier
herimplementeert de matchtelling zelfstandig uit de opgeslagen tokenreeksen.

## 8. Bedreigingen voor de geldigheid, vooraf benoemd

1. **Wiring is empirisch vastgesteld, niet uit een referentie-implementatie.**
   Een gunstiger onbekende variant zou `A` kunnen verhogen. De gekozen variant
   is de best scorende van vier; dat maakt een *falen* conservatief maar een
   *slagen* niet definitief.
2. **Contextdiepte.** `A` wordt gemeten bij korte tot middellange context. De
   S10-economie staat op 262K. §6 is een indicatie, geen dekking.
3. **De MTP-experts zijn hier device-resident.** In een gebouwd systeem is er
   0,000 GiB VRAM vrij en moeten ze gestreamd of ingeruild worden tegen
   cache-slots. Dat raakt de kosten, niet `A`.
4. **FP8-KV in de backbone.** Het hoofdmodel is de referentie zoals het draait,
   inclusief zijn FP8-KV. Dat is precies goed voor G-S10-C1 (identieke uitvoer
   t.o.v. déze runtime), maar `A` t.o.v. een FP32-KV-model is niet gemeten.
5. **Greedy.** Onder sampling verandert de acceptatielogica; niet gemeten.

## 9. Artefacten

`scripts/lightningstream_nemotron/s10a_extract_corpus.py` ·
`s10a_corpus.json` · `scripts/lightningstream_nemotron/s10a_mtp_acceptance.py` ·
`src/moe_lab/lightningstream_nemotron/mtp.py` · `s10a_wiring_resolution.json` ·
`s10a_mtp_acceptance.json` ·
`scripts/lightningstream_nemotron/s10a_independent_verify.py` ·
`s10a_independent_verification.json` · `protected_verification_after_s10a.json` ·
rapport met claim boundary.

## 10. Claim boundary van dit document

Dit document bevat geen meting en geen resultaat. Het legt alleen vast wat
gemeten gaat worden, met welke vaste keuzes, en welke uitkomst de poort haalt of
niet haalt.
