# A1 — de bewezen stack is geadopteerd als default, en het anker is opnieuw bevroren

Datum: 2026-08-15 · Registry `TREESWEEP200`
Preregistratie: `A1_ADOPTION_PREREGISTRATION_2026-08-15.md` (vóór uitvoering geschreven)
Verdict: **Adoptie doorgezet. De harde poort haalde het, de controle-arm faalde zoals vereist — de test hád onderscheidend vermogen — en een runtime zonder enige vlag reproduceert nu exact wat A1 onder de expliciete stack mat. Het oude anker wordt niet gereproduceerd; dat was vooraf voorzien en betekent iets specifieks.**
Terminal state: `a1_adoption_complete_v36_anchor_frozen`

## 1. Waarom E6 alleen niet genoeg was

E6 vergeleek twee armen binnen één proces met dezelfde cachegeschiedenis. Dat is
exact het regime waarin de exactheid vóór D1 óók léék te kloppen: NERVF-3,
NERVF-4, E4 en S11 haalden allemaal hun pariteitscheck over 2×64 tokens terwijl
de runtime in werkelijkheid niet run-to-run deterministisch was. Een adoptie mag
niet rusten op een test die deze fout niet kán zien.

## 2. De test met vermogen: verander de cachecapaciteit

Capacity 72 tegen 56. Dat verandert welke experts hit en welke miss zijn op
vrijwel elke laag van elke token — dus de hit-dan-miss-optelvolgorde verandert
radicaal — zonder één gewicht, route of kernel aan te raken. Correcte uitvoer kan
per definitie niet van de cachegrootte afhangen.

| poort | vereist | gemeten | uitslag |
|---|:--|:--|:--|
| **G-A1-CAP** (hard) | identiek met D1 aan | identiek, 2 × 256 tokens | ✅ |
| **G-A1-CTL** (controle) | **moet verschillen** zonder D1 | divergeert (expository, token 224) | ✅ gefaald zoals vereist |
| **G-A2** anker (informatief) | — | reproduceert V35 **niet**, divergeert bij token 1 | zie §4 |

De controle verdient een eerlijke kanttekening: hij divergeerde in **één van de
twee** prompts, en pas bij token 224. De test heeft dus vermogen, maar geen ruim
vermogen — de ordeafhankelijkheid is echt en zichtbaar, maar ze slaat traag toe.
Dat is precies waarom ze eerder vier fasen lang onopgemerkt bleef.

## 3. Wat er is omgezet

| vlag | was | is | grond |
|---|:--|:--|:--|
| `fused.use_ervf` | False | **True** | NERVF-2 bitexact 0/72 op vier breedtes, 1,936× |
| `runtime.attn` | v1 aan de call-site | **`attention_fp8_gqa4`** | E4 bitexact, −17,8% |
| `deterministic_accum` | False | **True** | D1 + G-A1-CAP hierboven |
| `fused.gatherless_down` | False | False | E2/NERVF-4 weerlegd |

De attentiekernel wordt nu via `rt.attn` gekozen in plaats van op de call-site.
**Let op voor scripts van vóór de adoptie:** die wisselen de kernel door
`rt.k.attention_fp8_gqa` te overschrijven, en dat pad wordt niet meer aangeroepen.
Zo'n A/B klapt dicht tot een nul-resultaat in plaats van stil verkeerd te meten.
Wissel voortaan `rt.attn`.

**G-A1B**: een runtime die met *geen enkele* vlag geconstrueerd wordt produceert
bit-identiek wat A1 onder de expliciete stack mat, en de waargenomen defaults
zijn geassert, niet aangenomen. Het defaultpad **is** het gevalideerde pad.

## 4. Het oude anker wordt niet gereproduceerd — en dat hoort zo

V35 divergeert al bij token 1. Dat is geen regressie en er wordt niets voor
verruimd. E4 toonde eerder dat de v4-kernel het anker wél reproduceert, dus de
divergentie komt van **D1**: het anker is bevroren onder de hit-dan-miss-volgorde
en legde daarmee een ordeafhankelijk artefact vast. De vooraf vastgelegde regel
was: nieuw anker bevriezen, oud anker bewaren, niet-vergelijkbaarheid opschrijven.

- `V35_GENERATION_ANCHOR.json` — **blijft staan**, ongewijzigd. Referentie voor
  elke meting van vóór de adoptie.
- `V36_DETERMINISTIC_ANCHOR.json` — **nieuw**. Referentie voor alle werk hierna.
- De twee zijn **niet bit-vergelijkbaar**. Vergelijk nooit een run tegen het
  verkeerde anker; kies op datum van de meting.

## 5. Claim boundary

2 prompts × 256 causale tokens per arm bij capacity 72 tegen 56, plus 2 × 64
tokens voor de ankerverificatie, `contexts_max=4096`, één GPU, één modelload per
capaciteit. Dit toetst **ordeonafhankelijkheid van de uitvoer**, niet snelheid —
er wordt hier geen enkele latency- of tok/s-claim gedaan; die staan in E6. Het
toont determinisme aan over de twee geteste capaciteiten, niet over alle
mogelijke cachestaten.

## 6. Artefacten

`scripts/treesweep200/a1_adoption_precondition.py` · `A1_ADOPTION_PRECONDITION.json`
`scripts/treesweep200/a1_adopt_verify_and_freeze.py` · `A1B_ADOPTION_VERIFY.json`
`V36_DETERMINISTIC_ANCHOR.json`
