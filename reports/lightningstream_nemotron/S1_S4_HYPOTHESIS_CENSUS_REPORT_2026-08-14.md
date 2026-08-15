# S1–S4 — hypothesis census: rapport

Datum: 2026-08-14
Verdict: **Drie van vier hypothesen weerlegd, één overtuigend geopend.**
Terminal state: `s1s4_census_relu2_sparsity_opened`

Preregistratie: `S1_S4_HYPOTHESIS_CENSUS_PREREGISTRATION_2026-08-14.md`
Runner: `s1_s4_hypothesis_census.py` · Verifier: `s1_s4_independent_verify.py`
**Onafhankelijke verificatie: 20/20** — elke gegate metriek herberekend uit de
ruwe dumps resp. rechtstreeks uit de safetensors-shards, zonder de runner te
importeren.

## Uitkomsten per hypothese

| fase | hypothese | gemeten | poort | uitspraak |
|---|---|---|---|---|
| S1 | route-voorspelling (temporele bigram) | recall@12 gem 0,611 / min 0,511; recall@24 0,724 | CLOSE: recall@24 < 0,80 | **weerlegd** |
| S2 | ReLU²-sparsity | **90,69% exacte nullen** | PASS: ≥ 0,45 | **geopend** |
| S3a | lossless codering NVFP4-codes | min nibble-entropie 3,9671 bits | open ≤ 3,5 | **weerlegd** |
| S3b | expert-delta-codering | H(B\|A) 3,9663 bits | open ≤ 2,5 | **weerlegd** |
| S4 | MTP/speculatieve gewichten | 0 van 24.147 keys | — | **gesloten** (geen draft-gewichten) |

### S1 — route-voorspelling is dood (voor deze predictor)

Temporele bigram-voorspelling haalt recall@12 = 0,611 en zelfs met 24
kandidaten (4× overfetch) 0,724 — onder de gesloten-drempel 0,80. Temporele
identiteit (overlap t→t+1) is 0,338, consistent met N7-A's 2,011/6 = 0,335.
Adjacent-layer overlap binnen een token: 0,050 — lagen correleren vrijwel niet
onderling. **Cross-layer/cross-token prefetch op routevoorspelling is met deze
predictor-klasse causaal onhaalbaar** en gaat de lijst van weerlegde
hypothesen in. Een geleerde neuraal voorspeller is een trainingsvraagstuk en
valt buiten deze lijn.

### S2 — ReLU²-sparsity is de winnaar

De up-projectie produceert over 72.174 expert-calls (2 prompts × ~259 stappen ×
138 experts) **90,69% exacte nullen**. Per laag uiteenlopend maar overal hoog
(zie `s1_s4_hypothesis_census.json`).

Cruciaal detail voor het ontwerp: de nullen **clusteren niet**. Volledig-nul
16-kolomsblokken: 30,6%; 64-kolomsblokken: 4,9%. Wie op 16-kolomsgranulariteit
gaart, houdt maar ~31% van de besparing; de volle 90,7% vraagt
**kolom-nauwkeurige selectie**.

Rekenkundige betekenis (component-niveau, géén tok/s-claim): een miss transferert
nu 5,6 MB (up 50% + down 50%). Kolom-selectief: up + 9,3% van down ≈ 54,7% van
het record → ~45% minder miss-bytes. Op N8's gemeten vloer bij 262K is de
PCIe-term 10,6 ms → ~5,8 ms; de som der vloeren ~21 ms → ~16 ms. Of de
afhankelijke/selectieve transfer die winst opeet, is exact wat de bouwfase moet
meten.

### S3 — codering is dood

NVFP4-nibbles zijn vrijwel uniform (3,9671 van 4 bits entropie); nibbles van
verschillende experts op dezelfde positie zijn vrijwel onafhankelijk
(H(B|A) = 3,9663). Er valt lossless niets te comprimeren en delta tussen
experts bestaat niet. Definitief gesloten.

### S4 — speculatief decoden via uitgever-gewichten is dood

De checkpoint bevat geen MTP/eagle/nextn/draft-tensors. Een drafter trainen is
buiten scope. Batch>1-amortisatie blijft als engineering-optie bestaan maar is
geen onderdeel van deze campagne.

## Claim boundary

Dit zijn statistieken over routes, activaties en code-bytes van het publieke
Nemotron 3 Nano NVFP4-checkpoint, gemeten op één rollout van 2 prompts × 256
greedy tokens plus bank-bytes, op deze machine. **Geen tok/s-claims, geen
timing-claims, geen kwaliteitsclaims.** De S2-projectie (~45% minder
miss-bytes) is een componentberekening, geen meting; ze motiveert een bouwfase
met eigen preregistratie, poorten en verifier.

## Artefacten

- `s1_s4_hypothesis_census.json` — resultaat met poorten
- `s1_routes_raw.json` / `s2_sparsity_raw.json` — ruwe data voor de verifier
- `s1_s4_independent_verification.json` — 20/20
- `s1_s4_input_lock.json` — bron-hashes
- `protected_verification_after_s1s4.json` — PROTECTED_80B_INTACT (0 modified / 0 removed)
