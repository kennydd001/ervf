# A1 — preregistratie: precondities voor adoptie van de bewezen stack

Datum: 2026-08-15 · Registry `TREESWEEP200`
**Geschreven vóór uitvoering. Poorten worden hierna niet verruimd.**

## Aanleiding

E6 heeft fysiek gemeten dat ERVF (breedte 16) + de v4-attentiekernel + D1
samen 41,980 → 37,490 ms per token halen met bit-identieke uitvoer. Kimi's
handoff legde de adoptiebeslissing expliciet bij E6 ("adoptie van v4 als default
bewust NIET omgezet — hoort bij E6"). Die beslissing staat nu open.

Ik zet de defaults **niet** om op grond van E6 alleen. E6 vergeleek twee armen
binnen één proces met dezelfde cachegeschiedenis. Dat is precies het regime
waarin de exactheid vóór D1 óók leek te kloppen (NERVF-3, NERVF-4, E4 en S11
haalden allemaal hun pariteitscheck over 2×64 tokens, terwijl de runtime in
werkelijkheid niet deterministisch was). Een adoptie mag niet op een test rusten
die deze fout niet kán zien.

## De test die wél onderscheidend vermogen heeft

Verander de **cachecapaciteit**. Dat verandert welke experts hit en welke miss
zijn over vrijwel elke laag en elke token, dus het verandert de hit-dan-miss
volgorde radicaal — zonder ook maar één gewicht, route of kernel aan te raken.
De wiskundig juiste uitvoer is per definitie onafhankelijk van de cachegrootte.

- **G-A1-CAP** (hard, beslist adoptie): met `deterministic_accum=True` is de
  gegenereerde tokenreeks bij capacity 72 **identiek** aan die bij capacity 56,
  over 2 prompts × 256 causale tokens.
- **G-A1-CTL** (controle, moet FALEN): dezelfde vergelijking met
  `deterministic_accum=False`. Als deze arm óók identiek uitkomt, heeft de test
  geen onderscheidend vermogen en bewijst G-A1-CAP niets — dan wordt de adoptie
  **niet** doorgezet en wordt dat zo gerapporteerd.
- **G-A2-ANCHOR** (informatief, geen adoptiepoort): reproduceert de volledige
  adoptiestack het bevroren `V35_GENERATION_ANCHOR.json` bitexact?

## Wat elk uitkomstpad betekent, vooraf vastgelegd

| G-A1-CAP | G-A1-CTL | conclusie |
|:--|:--|:--|
| PASS | FAIL | test heeft vermogen én D1 lost het op → **adoptie doorzetten** |
| PASS | PASS | test blind → **geen adoptie**, ontwerp een scherpere test |
| FAIL | — | D1 lost de ordeafhankelijkheid niet volledig op → **geen adoptie**, onderzoeken |

Over G-A2-ANCHOR, óók vooraf: het anker is bevroren onder de
hit-dan-miss-volgorde. Als de D1-stack het anker niet reproduceert, is dat geen
fout van D1 en geen reden om iets te verruimen — het betekent dat het anker zelf
een ordeafhankelijk artefact vastlegde. In dat geval wordt een **nieuw anker**
bevroren onder de deterministische volgorde, wordt het oude anker bewaard, en
wordt in het rapport vastgelegd dat de twee niet bit-vergelijkbaar zijn en
waarom. Het oude anker wordt niet overschreven.

## Wat er bij adoptie omgaat

Alle drie tegelijk, want los van elkaar zijn ze niet zinvol: `use_ervf = True`,
`attention_fp8_gqa = attention_fp8_gqa4`, `deterministic_accum = True`. De
`gatherless_down`-vlag blijft **uit** (E2/NERVF-4 weerlegd).

## Claim boundary vooraf

2 prompts × 256 causale tokens per arm, capacity 72 tegen 56, één modelload per
capaciteit, `contexts_max=4096`. Dit toetst **ordeonafhankelijkheid van de
uitvoer**, niet snelheid — er worden in dit experiment geen latencyclaims
gedaan. Het toont determinisme aan over de twee geteste capaciteiten, niet over
alle mogelijke cachestaten.
