# W1-R2 / C1-R1 / O1 — derde meetopzet, scherpere sound bound, OrbitANS

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**

## 1. W1-R2 — de derde opzet, en waarom deze wél kan werken

W1 mat bij 262100 +0,511 ms tegen 4,520 ms arm-drift. W1-R1 probeerde tripletten
op opeenvolgende stappen en werd **slechter**: 6,8 ms triplet-drift, want pairing
verwijdert thermische drift maar geen per-stap-ruis, en elk triplet gebruikte één
sample per arm.

De twee opzetten deden elk de helft van wat nodig is. W1 middelde per arm (32
samples) maar liet de armen minuten uit elkaar liggen; W1-R1 zette de armen naast
elkaar maar middelde niet. Deze opzet doet allebei: **blokken van 8 samples per
arm, direct afgewisseld**, `base · fast · base · fast · …`, 12 blokken per arm.

Per aangrenzend paar `(base_k, fast_k)`: `effect_k = median(base_k) −
median(fast_k)`. Per aangrenzend base-paar: `drift_k = |median(base_k) −
median(base_{k+1})|`.

- **G-W1R2-C1** — generatie bit-identiek.
- **G-W1R2-R1 — resolutie.** Mediane blok-tot-blok-drift **< 1,0 ms**. Haalt de
  opzet dat niet, dan wordt er opnieuw niets over de grootte geconcludeerd en
  stop ik met deze vraag.
- **G-W1-P1 blijft ongewijzigd op ≥ 1,0 ms bij 262100** voor adoptie.
- **G-W1R2-E1 — teken.** Mediaan positief én ≥ 60% van de paren positief.

## 2. C1-R1 — een scherpere bound, en waarom hij sound blijft

C1's grens was 9 tot 31× ruimer dan de preactivatie. De oorzaak is dat
Cauchy-Schwarz per groep de bijdragen als worst-case optelt alsof elke residual
met `x` meebeweegt.

Er is een strikt scherpere sound bound, en hij is gratis in informatie:
**de core bevat het tekenbit**. Voor bit-truncatie `core = code & mask` geldt
bovendien dat de magnitude alleen kan stijgen, dus per element:

```
Δw_k = s_k · δ_k        s_k = teken uit de core (bekend)
                        δ_k ∈ [0, δmax(core_k)]  (δmax volgt uit de coretabel)
```

Daarmee is de richting van élke term bekend zonder de staart te lezen, en geldt

```
δy_j ≤ Σ_k δmax(core_jk) · max(s_jk · x_k, 0)
```

Alleen de termen die de verkeerde kant op duwen tellen mee — ruwweg de helft —
en met hun **exacte** maximale magnitude in plaats van een L2-aggregaat. De
grens blijft mechanisch: elke overgeslagen residual heeft nog steeds een bewijs.

Zelfde meetopzet als C1, zelfde poorten, zelfde paren:

- **G-C1R1-S1** — nul valse certificaten.
- **G-C1R1-R1** — ≥ 30% van de staartbytes certificeerbaar (het pack's poort).
- **G-C1R1-B1** — ≥ 30% van de werkelijke nullen.
- Extra gerapporteerd: de verhouding `grens / |y0|` naast C1's 9–31×, zodat
  zichtbaar is hoeveel de scherpere bound wint.

De poorten zijn **niet** verlaagd omdat C1 faalde.

## 3. O1 — OrbitANS, gemeten in plaats van geërfd

ExactFlow A stelt exacte hercodering van NVFP4-codes en FP8-scales voor. Kimi's
S3 mat de code-entropie op 3,967 van 4 bits en dat staat in
`forbidden_hypotheses` — maar dat ging over de codes alleen, met een marginaal
model. Het pack stelt méér voor: conditionele modellen op matrixtype, laag,
schaal-exponent en expertcluster, plus **aparte codering van de scales** via
exponent/mantissa-delta's.

Gemeten op echte records uit deze checkpoint, over meerdere lagen en experts:

- marginale entropie van de codes;
- **conditionele** entropie van de codes gegeven de schaal-exponent van hun groep
  (het model dat het pack expliciet noemt);
- entropie van de scale-bytes, marginaal en als delta ten opzichte van hun
  buur;
- de daaruit volgende ondergrens op de recordgrootte, en de bijbehorende
  packreductie.

Poorten, letterlijk uit het pack:

- **G-O1-1** — ≥ **12%** echte packreductie voor doorgang;
- **G-O1-2** — ≥ **20%** voor de sterke poort;
- Gerapporteerd: wat die reductie via Y2-R1's gemeten helling (68% byte-gebonden)
  hooguit aan GEMV-tijd oplevert.

Dit is een entropie-ondergrens, geen codec: een echte ANS-codec haalt de
entropie niet exact en betaalt decode-overhead. De gemeten reductie is dus een
**bovengrens** op wat OrbitANS kan.

## 4. Wat hier niet gebeurt

Geen codec gebouwd, geen certificaatkernel, geen wijziging aan `runtime.py`
behalve wat er al staat. Niets wordt naar tok/s omgerekend.

## 5. Artefacten

`scripts/lightningstream_nemotron/w1r2_block_paired.py` · `w1r2_block_paired.json` ·
`scripts/lightningstream_nemotron/c1r1_o1_bound_and_entropy.py` ·
`c1r1_o1_bound_and_entropy.json` ·
`scripts/lightningstream_nemotron/w1r2_c1r1_o1_independent_verify.py` ·
`w1r2_c1r1_o1_independent_verification.json` · rapport met claim boundary.

## 6. Claim boundary van dit document

Geen meting, geen resultaat.
