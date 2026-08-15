# X1 — SweepSpec: exact gebouwd, en het levert niets op

Datum: 2026-08-15
Verdict: **De expert-major blockverifier is gebouwd en is bit-identiek aan het bestaande pad — 0 mismatches, rel_l2 exact 0,0 over 460 laag-blokken. Maar hij is niet sneller. De verhouding sweep/sequentieel loopt van 1,147 bij B=1 naar 1,0017 bij B=5: de MoE-term is lineair in het aantal geverifieerde posities, niet in het aantal unieke expert-records. Poort G-X1-P1 vroeg 0,6228 en meet 1,0017.**
Terminal state: `x1_sweepspec_exact_but_moe_linear_in_positions_not_experts`
Preregistratie: `X1_SWEEPSPEC_MOE_ORACLE_PREREGISTRATION_2026-08-15.md`

## 1. Wat er gebouwd is

Twee nieuwe kernels in `sweepspec.py`, buiten `runtime.py`:

- `gemm_nvfp4_rows_b` — de bestaande `gemv_nvfp4_rows` met `B`
  activatievectoren. Het gewicht wordt één keer gelezen en tegen alle `B`
  vectoren vermenigvuldigd; decode, de acht FMA's per `uchar4` en hun volgorde
  zijn ongewijzigd.
- `gemm_down_masked_b` — de masked down-GEMV met `B` vectoren over de
  **unie**-panelmasker. Node-selectie en resultaat-scatter lopen via
  index-arrays, nooit via kopieën per (node, expert)-paar; die zouden meer
  kosten dan de groepering bespaart en zouden de harnas meten in plaats van het
  idee.

Bijdragen worden per `(node, slot)` opgeslagen en pas daarna in **route-volgorde**
gesommeerd, zodat de optelvolgorde per node identiek blijft.

## 2. Exactheid: volledig

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-X1-C1** | batched met `B=1` bit-identiek aan `gemv_nvfp4_rows` | 8/8 rijen identiek, ook los getest voor `B` t/m 8 | ✅ |
| **G-X1-C2** | sweep bit-identiek aan sequentieel | **460 laag-blokken, 0 mismatches, worst rel_l2 0,000e+00** | ✅ |

Dat is geen tolerantie maar gelijkheid. De redenering die het mogelijk maakt:
de unie-mask voegt alleen kolommen toe waar de activatie exact nul is, en
`fmaf(w, 0, acc) = acc`; de panelwandeling blijft oplopend, dus de volgorde van
de wél bijdragende termen verandert niet. Bij `nchunks = 1` is er ook geen
herverdeling over chunks. Exacte targetsemantiek is daarmee bewezen, niet
aangenomen.

## 3. Prestatie: de hypothese is weerlegd

Alle 23 MoE-lagen, echte hidden states, echte officiële routes, elke expert van
het blok resident, gebracketeerde basislijnen.

| B | unie/laag | sequentieel p50 | sweep p50 | verhouding | seq / seq(B=1) |
|---:|---:|---:|---:|---:|---:|
| 1 | 6,00 | 22,454 ms | 25,744 ms | 1,1465 | 1,000 |
| 2 | 9,77 | 43,993 | 47,304 | 1,0753 | 1,959 |
| 3 | 13,18 | 66,580 | 68,861 | 1,0343 | 2,965 |
| 4 | 16,16 | 88,799 | 90,501 | 1,0192 | 3,955 |
| **5** | **18,85** | **111,359** | **111,546** | **1,0017** | **4,959** |

**G-X1-P1 gefaald:** vereist < 0,6228 (want een ronde stoot 3,1139 tokens uit
tegen vijf geverifieerde posities), gemeten 1,0017. Verifier 35/35, `VERIFIED`.

### Het mechanisme, in één regel

De laatste kolom is de hele uitleg: het sequentiële pad schaalt **4,959× bij
B=5** — vrijwel perfect lineair in het aantal posities. En de sweep, die elk
record maar 18,85 keer per laag laadt in plaats van 30, komt op **dezelfde
tijd** uit.

Dus: **de MoE-term is lineair in het aantal (node, expert)-toewijzingen, niet in
het aantal unieke expert-records.** Elk record 1,59× minder vaak lezen levert
niets op, omdat het lezen van records niet is waar de tijd zit. Dat is precies
wat S12 in de lus mat (per-expert marginalen 12,23 van de 39,5 ms) en wat S11
langs de byte-kant liet zien (2,9× meer PCIe kost 4,8%). Drie onafhankelijke
metingen, dezelfde conclusie.

De sweep is bij kleine `B` zelfs *trager* (1,147 bij B=1) omdat de gebatchte
kernel meer shared memory stageert en daardoor minder occupancy heeft. Dat nadeel
verdwijnt naarmate `B` groeit, maar er komt nooit een voordeel voor terug.

## 4. Wat dat betekent voor de rondebudgetten

Beide packs leiden dezelfde budgetten af: 62,280 ms voor 50 tok/s kort, 77,850
voor 40 bij 128K, 103,800 voor 30 bij 262K.

Gemeten is dat **alleen het routed-expert-deel** van een B=5-verificatie
**111,5 ms** kost — bij korte context, met alles resident, zonder router, zonder
shared expert, zonder Mamba, zonder attention, zonder LM-kop en zonder de
draft-keten. Dat is **1,8× het volledige rondebudget voor 50 tok/s** voordat er
ook maar iets anders is meegeteld.

En het helpt niet om ondieper te gaan. Per uitgestoten token kost het
routed-expert-deel:

| B | D | uitgestoten | ms/uitgestoten token | tegen AR (B=1) |
|---:|---:|---:|---:|---:|
| 1 | — | 1,000 | 22,45 | — |
| 2 | 1 | 1,786 | 24,63 | +9,7% |
| 3 | 2 | 2,378 | 27,99 | +24,7% |
| 4 | 3 | 2,808 | 31,62 | +40,8% |
| 5 | 4 | 3,114 | **35,76** | **+59,3%** |

Speculatieve verificatie maakt de dominante term bij **elke** blokgrootte
duurder per uitgestoten token, want je rekent `B` posities om er `1 + A` te
krijgen en de kosten volgen `B`. Bij B=5 is dat exact de 5/3,114 = 1,606 die K0
langs de cache-kant ook al mat (1,62× misses per uitgestoten token).

## 5. Wat hiermee dicht gaat, en wat niet

**Dicht.** SweepSpec als route naar 50 tok/s op deze runtime. Niet omdat de
verifier niet exact te bouwen is — dat is hij, bit-identiek — maar omdat de
gewichts-sweep die hij bespaart niet is waar de tijd zit. Daarmee vervalt ook de
premisse onder ExactFlow's hypothese E, onder LightningSpec's H2/H3, en onder de
optimistische lezing die ik zelf in het K0-rapport openliet: die lezing is nu
gemeten en het is de pessimistische die klopt.

**Niet dicht.** De vraag waar de MoE-tijd dán zit. S9 verklaarde ~9,0 ms met
GEMV-microbenchmarks, S12 dekte 15,53 ms met in-lus marginalen, en ~24 ms van de
39,5 is nog steeds door niemand gelokaliseerd. Elke verdere versnelling van dit
model moet daaruit komen. Dat is één term, in één plaats, en het is nu de enige
open vraag die er nog toe doet.

Ook niet dicht: de andere hypothesen uit beide packs die de **target** goedkoper
maken in plaats van hem vaker te gebruiken — OrbitANS (exacte hercodering),
CertiPlane (bewijsdragende precisie), PathQ (gemengde precisie met een
kwaliteitsgrens). Die vallen de 39,5 ms zelf aan en zijn door dit resultaat niet
geraakt. Wel geldt voor alle drie dat S11 en S12 al hebben laten zien dat
**bytes besparen op deze runtime geen tijd bespaart**, dus een bytegerichte
hypothese moet eerst uitleggen waarom zij anders zou uitpakken.

## 6. Meetnotitie: een kernelbug, en hoe hij gevonden is

De eerste versie faalde met `CUDA_ERROR_ILLEGAL_ADDRESS` zodra `B ≥ 4`. Drie
probes — over `B`, over `rows`/`cols`, en over shared-memory-grootte — lieten
zien dat het puur van `B` afhing en niet van een resourcelimiet: `B=4` faalde bij
16 KB shared terwijl `B=3` bij 32 KB slaagde. De oorzaak was de lokale array
`float acc[MAX_B]` met een runtime-index; de b-lussen zijn nu compile-time
ontrold met een `if (b >= B) break;`, waarna `local_size_bytes` van 32 naar **0**
ging en `B` t/m 8 werkt. De exactheidspoorten zijn na die wijziging opnieuw
gedraaid, niet ervoor.

## 7. Claim boundary

Gemeten kosten van het **routed-expert-deel** van alle 23 MoE-lagen voor `B`
nodes, op echte hidden states en echte officiële routes uit een echte greedy
generatie, met elke expert van het blok al resident, zodat de enige variabele
token-major tegen expert-major is. Het is **geen tokentijd en geen doorvoer**:
router, shared expert, Mamba, attention en de LM-kop zitten in geen van beide
armen, en er bestaat geen speculatieve lus. De ms-per-uitgestoten-token-tabel in
§4 deelt een gemeten componenttijd door een gemeten acceptatie; het is een
componentverhouding en uitdrukkelijk geen tok/s. Exactheid is bewezen bij
`nchunks = 1`; de getimede configuratie gebruikt de productie-`nchunks` en
verschilt alleen door float-herassociatie.

## 8. Artefacten

`X1_SWEEPSPEC_MOE_ORACLE_PREREGISTRATION_2026-08-15.md` ·
`src/moe_lab/lightningstream_nemotron/sweepspec.py` ·
`scripts/lightningstream_nemotron/x1_kernel_smoke.py` ·
`scripts/lightningstream_nemotron/x1_sweepspec_moe_oracle.py` ·
`x1_sweepspec_moe_oracle.json` ·
`scripts/lightningstream_nemotron/x1_independent_verify.py` ·
`x1_independent_verification.json` · `protected_verification_after_x1.json`
