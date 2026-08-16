# Onderzoekslogboek

Eén blok per fase, **nieuwste bovenaan**. Schrijf hier ook wat er *niet* werkte
en waarom — dat is meestal het bruikbaarste deel. Formaat:

```
## <datum> — <fase> — <verdict in één zin>
**Vraag** · **Opzet** (armen, één variabele) · **Uitkomst** (getallen) ·
**Poorten** · **Wat dit sluit of opent** · **Artefacten**
```

---

## 2026-08-16 — ⚠️ **INTREKKING van mijn eigen 99 tok/s-projectie.** Herrekend met de al gemeten route-unie komt native FP4 + MTP op **52-53 tok/s**, niet 99 — en dat sluit de speculatieve route opnieuw

**Wat ik fout deed.** Eén blok hieronder projecteerde ik ~99 tok/s door "al het
deelbare gewichtswerk" door M te delen, en ik zette **`up_proj` (2,253 ms) in de
deelbare kolom**. Dat is verkeerd, en de reden stond in mijn eigen bericht —
ik noemde de groeiende routing-unie als voorwaarde 3 en paste hem vervolgens
niet toe. **`up_proj` is routed.** De kosten daarvan delen niet door M; ze
volgen de **expert-unie**, en die is al gemeten: **19,88 van 128 experts over 5
posities tegen 6 voor één token = 3,313×** (`diag_mtp_route_union.json`).
Hetzelfde geldt voor de gather, down_masked en panel_scan/reduce/accumulate.

**De juiste driedeling van het 19,60 ms-token.**

| klasse | ms | aandeel | schaalt met D+1 posities als |
|---|---:|---:|---|
| **M-vrij** (Mamba-proj, attention-proj, shared expert, lm_head) | **7,928** | 40% | ×1 — één gewichtspas voor alle posities |
| **routed** (gather, up_proj, down_masked, panel/reduce/accum) | **7,930** | 40% | **×3,313 bij 5 posities** (gemeten unie) |
| **strikt per positie** (ssm_step, conv/dt, gated_norm, norms/adds, attention-KV) | 2,701 | 14% | ×D+1 — `ssm_step` is een recurrentie |
| niet-geattribueerd | 1,042 | 5% | ×D+1 (conservatief) |

**Herrekening, met de gemeten acceptatiegraad A+1 = 3,114 (S10-A, 360 stappen, poort ≥1,5 gehaald):**

| scenario | ronde | per token | tok/s | vs V18 |
|---|---:|---:|---:|---:|
| D+1=5 (D=4 drafts) | 59,81 ms / 3,114 | 19,21 ms | **52,1** | 1,020× |
| D+1=2 (checkpoint-MTP, `num_nextn_predict_layers=1`) | 30,34 ms / 1,6 | 18,96 ms | **52,7** | 1,034× |

**Dus 2-3,4% winst, niet 94%.** Dat is binnen de ruis van wat deze machine
tussen runs drift.

**Waarom de speculatieve route hier structureel niet werkt.** Slechts **40% van
het token is M-vrij**. De andere 60% bestaat uit twee dingen die géén van beide
over posities delen:
1. de **routed MoE (40%)** — bij 128 experts en top-6 routeren opeenvolgende
   tokens naar grotendeels verschillende sets, dus een gezamenlijke sweep
   beweegt bijna evenveel expert-bytes als losse tokens genereren (3,313× voor
   5 posities); dit was al de kern van de oorspronkelijke sluiting en het
   **overleeft native FP4 volledig**;
2. de **Mamba-recurrentie (`ssm_step`, 1,010 ms/token)** — die is per definitie
   sequentieel in positie en kan door geen enkele kernel gedeeld worden.

C2d's vrije-M-eigenschap is echt, maar hij bijt alleen op die 40%. De
oorspronkelijke MTP-sluiting kwam op −6,0%; met vrije M wordt dat +2 à +3,4%.
**De conclusie kantelt niet.**

**Wat wél overeind blijft van C2b/C2c/C2d.** Native FP4 op M=1 is een echte,
formaatbehoudende winst op lm_head (2,52×) en shared_down (1,68×): **−1,275
ms/token → 51,0 → 54,6 tok/s**. Dat is gemeten, vraagt geen quantisatiewijziging
van Mamba of attention, en blijft de beste openstaande kandidaat. `routed_up`
(0,96×) hoort er niet bij.

**De les.** Dit is dezelfde fout die dit project al twee keer eerder heeft
gemaakt en zelf als werkregel heeft opgeschreven: *een component-eigenschap
extrapoleren naar het geheel zonder te controleren welke termen er werkelijk
onder vallen.* De vrije-M-meting was correct; de toewijzing eromheen niet. Ik had
`up_proj` als routed moeten herkennen — het staat letterlijk in de naam van de
kolom "MoE" in de tokenkaart.

**Artefacten.** `pro_research/diag_mtp_native_fp4_economics.py` + `.json`
(reproduceerbare rekensom over uitsluitend al gemeten grootheden; de aanname
attention-projectie/KV = 0,60 is expliciet gemarkeerd als aanname).

---

## 2026-08-16 — **C2d: M is GRATIS tot M=8 op native FP4 — acht posities voor de prijs van één gewichtslezing.** Dit heropent wat de batch-analyse gesloten had

**Vraag.** C2c wees M=2 aan als de echte hefboom, niet FP4 op zich. Hoe ver gaat
die as? Als M=4 of M=8 óók gratis is, verandert het hele plafond.

**Opzet.** M ∈ {1,2,4,8,16} op dezelfde vier al-NVFP4-shapes, zelfde koude
rotatie (≥4× L2), zelfde eventtiming. Eén variabele: M.

| shape | M=1 | M2/M1 | M4/M1 | **M8/M1** | M16/M1 |
|---|---:|---:|---:|---:|---:|
| lm_head | 600,70 µs | 0,984 | 0,986 | **0,989** | 1,435 |
| shared_up | 39,82 | 0,864 | 0,810 | **0,814** | 0,729 |
| shared_down | 24,96 | 0,973 | 0,968 | **1,003** | 0,895 |
| routed_up | 24,06 | 0,933 | 0,938 | **0,891** | 0,960 |

**Acht posities kosten hetzelfde als één, op élke shape.** M=16 is nog steeds
gratis op drie van de vier; alleen lm_head (198 MB) zakt door naar 1,435×.

Per token over die vier shapes (6,302 van de 19,60 ms van het V18-record):

| | ms/token | projectie |
|---|---:|---:|
| ERVF (nu) | 6,302 | 51,0 tok/s |
| native M=1 | 5,411 | 53,5 |
| native M=2 | 2,519 | 63,2 |
| native M=4 | 1,251 | 68,7 |
| native M=8 | **0,609** | **71,9** |
| native M=16 | 0,327 | 73,4 |

Die projectie **verzadigt rond 73** puur omdat ze niets buiten die vier shapes
aanraakt — de overige 13,3 ms van het token blijft per constructie ongemoeid.

**Waarom dit belangrijker is dan de FP4-conversie zelf.** Ons eigen gebatchte
ERVF-plafond was gemeten op **×1,64 bij N=4**, en de verklaring daarvoor stond
al in `DECISION_SINGLE_STREAM_VS_BATCH.md`: ERVF draait al op 77% van de
apparaatbandbreedte, en batching betaalt alleen waar je bandbreedte verspilde.
Native FP4 haalt ~8× op precies díe as omdat het op **Tensor Cores** draait: de
GEMM is gewichtsbandbreedte-gebonden en de tensor cores hebben rekenruimte
over, dus extra kolommen zijn gratis. **Dat is het mechanisme dat onze
CUDA-core-kernels niet kunnen reproduceren, en daarom heropent de M-as wat de
batch-analyse had afgesloten.**

**De rekening als de M-deling zich uitstrekt tot al het gewichtsgebonden werk.**
Uit de tokenkaart zijn deelbaar: Mamba in/out GEMV 4,187 + attention-projecties
(~1,5 van 2,479) + up_proj 2,253 + shared_expert 1,810 + lm_head 1,107 ≈
**10,86 ms**. Bij M=8 wordt dat ~1,36 ms → **−9,5 ms** → 19,60 − 9,5 ≈
**10,1 ms ≈ 99 tok/s**. Niet deelbaar en dus overblijvend: `ssm_step` 1,095 (die
is per-positie state), de PCIe-gather 3,849 (de routing-unie groeit met M),
down_masked 1,372, panel_scan/reduce/accumulate 1,119, norms/adds 0,840, en het
KV-deel van attention.

**Dat is de eerste route in deze hele sessie die met gemeten invoer op ~100
uitkomt.** Maar de voorwaarden zijn zwaar en moeten expliciet blijven:
1. Mamba (FP8) en attention (BF16) naar FP4 is een **echte
   quantisatiewijziging** met een ongemeten kwaliteitsprijs — geen gratis winst;
2. er moet een **M-weg verificatiepad** zijn (speculatief/MTP) met een hoge
   acceptatiegraad, anders betaal je M× rekenwerk voor <M geaccepteerde tokens;
3. de **MoE-routing-unie groeit met M** — bij N=16 gemeten op 63,9 van 128
   experts per laag, dus de PCIe-post schaalt mee;
4. alles hierboven is **synthetisch** gemeten, op de GEMM alleen.

**Wat het wél hard maakt.** De M-as is geen hypothese meer: 8 posities voor de
prijs van 1 is gemeten, koud, op de echte shapes, op deze GPU. Dat was de
ontbrekende schakel onder K2 (dat 1,012× haalde omdat het posities herschikte
zonder hun gewichten fysiek te delen).

**Artefacten.** `pro_research/diag_native_nvfp4_c2d_mscaling.py` +
`results/native_nvfp4/C2D_M_SCALING.json` (branch `pro-s100-nativefp4-c2b`).

---

## 2026-08-16 — **C2c: de eerlijke head-to-head. Native FP4 is NIET uniform sneller — 0,96× op de meest aangeroepen shape — maar M=2 is gratis op élke shape**

**Vraag.** C2b bewees dat native FP4 draait en dat M=2 gratis is. Het bewees
**niet** dat het sneller is dan onze ERVF-kernel. Dat is de vraag die telt.

**Twee redenen waarom C2b die vraag niet kon beantwoorden — allebei eerst gefixt.**
1. **Verkeerde shapes.** C2b timede Q-like (BF16 in het checkpoint) en Mamba-in
   (FP8). Die naar FP4 brengen is een **quantisatiewijziging** met een ongemeten
   kwaliteitsprijs — een andere claim. De **formaatbehoudende** shapes zijn
   `lm_head`, `shared_up`, `shared_down`, `routed_up`: die zijn al NVFP4, dus
   alleen de accumulatievolgorde verandert.
2. **L2.** C2b herleest **één** matrix per shape. De kleine NVFP4-shapes zijn
   2,8-5,6 MB tegen een L2 van 32 MiB — dat waren dus L2-residente snelheden.
   Precies het artefact dat vandaag al één keer een spookwinst van 1,46×
   opleverde.

**Opzet.** Zelfde koude-rotatieprotocol als
`diag_nvfp4_ervf_reference_rates.py` (werkset ≥4× L2, CUDA-events, p50 over 7
rondes), M=1 én M=2.

| shape | ERVF | native M=1 | M=1 speedup | M2/M1 | per token @M=2 |
|---|---:|---:|---:|---:|---:|
| lm_head 131072×2688 | 1512,26 µs | 599,95 (293,6 GB/s) | **2,52×** | 0,983 | **5,13×** |
| shared_down 2688×3712 | 43,62 | 25,96 (192,2) | 1,68× | 0,953 | 3,53× |
| shared_up 3712×2688 | 39,14 | 35,26 (141,5) | 1,11× | 0,976 | 2,27× |
| **routed_up 1856×2688** | 20,92 | 21,88 (114,0) | **0,96×** | 1,095 | 1,75× |

**Zelfcorrectie — mijn eigen projectie van twintig minuten eerder was fout.**
Ik nam aan dat lm_head's 2,52× overdraagbaar was naar de andere drie en kwam op
63,6 tok/s bij M=1. Dat klopt niet. Native FP4 is sterk **shape-afhankelijk**:
293,6 GB/s op de matrix van 198 MB, maar 114-192 GB/s op de kleine — en op
**`routed_up`, met 138 aanroepen per token veruit de meest aangeroepen shape van
het model, is het 0,96×, dus lichtjes TRAGER dan onze eigen ERVF-kernel.**

**Per token over alle vier de al-NVFP4-shapes:**

| | ms/token | token | tok/s |
|---|---:|---:|---:|
| ERVF (huidig) | 6,302 | 19,60 | 51,0 |
| native M=1 | 5,027 (−1,275) | 18,33 | **54,6** |
| native M=2, per token | 2,629 (−3,674) | 15,93 | **62,8** |

**Wat dit betekent.** Native FP4 op M=1 is ongeveer **+3,6 tok/s** waard, niet
+12. De echte hefboom is **M=2 — gratis op élke shape (0,953-1,095) op koude
data**, wat C2b's bevinding bevestigt op precies de shapes die ertoe doen. Dat
is exact wat layer-major K2-scheduling niet kon leveren (1,012×), en het is nu
**gemeten** in plaats van gehoopt. De volgorde is daarmee duidelijk: niet "zet
alles op native FP4", maar **native FP4 als drager voor een M=2-verificatiepad**,
en selectief — lm_head en shared_down eerst, `routed_up` voorlopig niet.

**Voorbehouden.** Synthetische waarden; niet bitexact tegen de ERVF-reductieboom
(dus een **kwaliteits**poort, geen exactheidspoort); `routed_up`'s rotatie haalde
maar 1,8× L2 tegen het doel van 4× doordat het matrixaantal op 24 gecapt is.

**Artefacten.** `pro_research/diag_native_nvfp4_c2c_cold.py` +
`results/native_nvfp4/C2C_COLD_NVFP4_SHAPES.json` (branch
`pro-s100-nativefp4-c2b`), `pro_research/diag_nvfp4_ervf_reference_rates.py` +
`.json` (de ERVF-referentiekant, branch `pro-research`).

---

## 2026-08-16 — **C2b: native Blackwell FP4 DRAAIT. Alle 17 poorten groen, en M=2 is gratis.** De eerste hefboom van deze hele sessie die groot genoeg is voor 100

**Vraag.** C2 (branch `pro-s100-nativefp4-c2`) was een *software*-negatief: de
Nemotron-venv draait Torch 2.9.1+cu128, waar `F.scaled_mm`, `BlockWise1x16`,
`SWIZZLE_32_4_4` en `_scaled_mm_v2` niet bestaan. De Tensor Cores zijn dus nooit
bereikt. C2b bouwt een geïsoleerde Torch 2.12.1+cu132-venv (`.venv-fp4-c2b`,
`.venv-nemotron` blijft onaangeroerd) en stelt de vraag opnieuw: **kan deze
SM120-laptop native FP4-GEMM's fysiek uitvoeren, en wat kost M=2 tegenover M=1?**

**Eerste run: alle vier known-value-cases faalden — mét het API-contract volledig
aanwezig** (G1-G7 groen). Eén foutmelding, en die was precies genoeg:

    ValueError: For Blockwise scaling both scales should be contiguous

**Oorzaak, uit de opgeslagen strides.** `scale_b` werd gebouwd als
`torch.ones((bsh[1], bsh[0])).t()` → een getransponeerde **view** met stride
`(1, sfp)`. De fout was B's transpositie spiegelen op B's **schaal**: `b` staat
getransponeerd omdat een GEMM K-major operanden wil, maar het blok-schaaltensor
is een aparte buffer en het blockwise-pad eist dat die contigu is in zijn
logische `(sfp, ceil(n,128))`-vorm. Gefixt door hem contigu te alloceren; de
waarden zijn synthetische enen, dus er verandert verder niets. De contiguïteit
van beide schalen wordt nu per case weggeschreven, zodat dit niet stilletjes kan
terugvallen in een misleidend "API rejected"-resultaat.

**Met die ene fix: alle 17 poorten groen.**

Exactheid — M=1/2/16/128 draaien allemaal, `expected_bf16 = 256.0`,
`max_abs_error = 0.0`, deterministisch, eindig.

| shape | M=1 | M=2 | M2/M1 | GB/s (FP4-gewichtsbytes) |
|---|---:|---:|---:|---:|
| Q-like (4096×2688) | 0,03015 ms | 0,02996 | **0,993** | 182,6 / 183,8 |
| Mamba-in (10304×2688) | 0,04736 | 0,04736 | **1,0001** | 292,4 / 292,4 |
| LM-head (131072×2688) | 0,58202 | 0,58417 | **1,0037** | 302,7 / 301,6 |

**Twee dingen volgen hieruit.**

1. **Native FP4 haalt 292-303 GB/s** op de grote shapes, tegen de **230-261
   GB/s** die onze beste koude ERVF-kernel haalt — en dat op **de helft van de
   bytes** overal waar de bron FP8 is. (Q-like blijft op 182,6 GB/s steken: met
   5,51 MB is die matrix te klein, launch-gebonden.)
2. **M=2 kost hetzelfde als M=1, tot op 0,3% na.** Dat is precies de smoking gun
   die het K2-resultaat miste: layer-major K2-scheduling leverde **1,012×** op,
   omdat het twee posities herschikte zónder hun gedeelde gewichten fysiek te
   hergebruiken. Blackwell trekt twee targetposities **gratis** door één
   gewichtsstroom.

**Grenzen, ongewijzigd t.o.v. de preregistratie — dit is geen tok/s-claim.**
Synthetische +1-waarden en +1-BlockWise1x16-schalen. Het is een executie- en
timingcontract, geen Lightning-kwaliteit, geen echte schalen, geen
activatiequantisatie. **En belangrijk: de accumulatievolgorde van de Tensor Core
is niet die van onze ERVF-boom, dus C3 heeft een KWALITEITSpoort nodig, geen
bitexactheidspoort.** Dat is een ander soort bewijs dan alles wat deze sessie tot
nu toe gebruikt heeft, en dat moet expliciet blijven.

**Waarom dit de eerste hefboom is die groot genoeg is.** De sessie sloot
single-stream af op ~94 tok/s theoretisch maximum en batch op 70-85, allebei
onder het doel, omdat ERVF al op 77% van het apparaatplafond zit en je dezelfde
inefficiëntie niet twee keer kunt opeten. Native FP4 verandert de vergelijking
op een andere as: **minder bytes** (FP4 i.p.v. FP8/BF16 waar de kwaliteit het
toelaat) én **meer bandbreedte** (292-303 vs 230-261) én **M=2 gratis**. Dat zijn
drie onafhankelijke factoren, geen derde poging op dezelfde.

**Wat het níet zegt.** De routed experts, shared expert en lm_head zijn al
NVFP4 — daar is native FP4 formaatbehoudend en alleen de kernel verandert. Maar
**Mamba (892 MB/token, de grootste post) is FP8** en attention BF16; die naar FP4
brengen is een echte quantisatiewijziging met een kwaliteitsprijs die nog
volledig ongemeten is. De winst is dus niet uniform over de 2048 MB/token.

**Artefacten.** `pro_research/results/native_nvfp4/C2B_TORCH212_CONTRACT.json`
(+ `_VERIFICATION.json`, onafhankelijke verifier `passed: true`),
`pro_research/diag_native_nvfp4_c2b_torch212.py` (branch
`pro-s100-nativefp4-c2b`, HEAD na de ABI-fix).

**Volgende stap: C3.** Echte Lightning NVFP4-gewichten (C1 bewees al dat de
repack naar Blackwell-layout verliesloos is, 10/10 poorten, 1,678%
scale-padding) + activatiequantisatie + een kwaliteitspoort. Pas dán mag er een
tok/s-getal aan hangen.

---

## 2026-08-16 — V19 (V18 + ssm-block): **geen winst** — en dat verklaart V18's super-additiviteit scherper dan de winst zelf deed

**Vraag.** V18 liet zien dat twee bitexacte mechanismen, elk onder hun eigen
poort, samen ~2× hun rekenkundige som opleverden. Is dat een algemene
eigenschap van stapelen, of iets specifieks? Test: voeg een **derde** bitexact
mechanisme toe dat bewust **disjunct** is — de blok-per-(h,p) `ssm_step` zit in
Mamba, niet in het down_proj-pad, dus er is geen gedeelde resource om over te
vechten. Alleen ×1,031 (−0,024 ms) in isolatie.

**Vooraf opgeschreven verwachting:** hooguit additief, ~−0,02 ms, "want
disjunctheid snijdt twee kanten op — geen contentie, maar ook geen gedeelde
bottleneck om twee keer te verlichten".

**Uitkomst.** Alle poorten groen, bitexact, drift 0,0889 ms.
CAND **19,7418 ms = 50,654 tok/s**, delta −1,6200 ms.

**Naast de twee V18-runs gelegd:**

| run | CAND p50 | delta | midden |
|---|---:|---:|---:|
| V18 run 2 | **19,6046** | −1,1823 | 20,787 |
| V18 run 1 | **19,6897** | −1,5594 | 21,249 |
| V19 | **19,7418** | −1,6200 | 21,362 |

**De delta van V19 is de grootste (−1,62), maar zijn absolute CAND is de
traagste van de drie.** Dat verschil komt volledig uit de baselines: die
verschoven thermisch tussen de runs (midden 20,79 → 21,36). De drie
CAND-waarden liggen 0,137 ms uit elkaar — de ruisbreedte van deze harness.

**Conclusie: V19 ≈ V18 binnen de ruis. De ssm-blokvariant levert in de
geïntegreerde stack niets op**, ondanks ×1,031 in isolatie. Zijn 0,024 ms zit
onder de ruisvloer van de integratiemeting. **Niet adopteren** — extra
complexiteit zonder meetbare winst.

**Waarom dit het V18-resultaat scherper maakt.** Super-additiviteit is dus
**geen algemene eigenschap van stapelen**. V18's ×2 kwam doordat H-SCALE en B3
**dezelfde bottleneck** (het down_proj-PCIe-pad) op complementaire manieren
aanvallen: H-SCALE verkleint de transfer, B3 verbergt wat overblijft, en B3's
stroomstructuur vangt bovendien H-SCALE's eigen plane-fetch-kost op. Twee
mechanismen op verschillende bottlenecks tellen gewoon op — en als één van de
twee onder de ruis ligt, telt hij niet eens zichtbaar op.

**Praktische regel die hieruit volgt en die ik in TODO zet:** zoek combinaties
van mechanismen die **dezelfde** bottleneck van verschillende kanten aanvallen;
disjuncte mechanismen stapelen hooguit additief en zijn dus alleen de
complexiteit waard als ze elk apart al boven de ruis uitkomen.

**Het record blijft V18: 19,60-19,69 ms = 50,8-51,0 tok/s.**

**Artefacten.** `pro_research/ssm_block_install.py`,
`pro_research/combined_v19.py`,
`pro_research/results/v19_combined_ssm/PRO_V19_COMBINED_SSM.json`.

---

## 2026-08-16 — 🎉 **NIEUW RECORD: 51,0 tok/s (19,60 ms). E50 gehaald.** H-SCALE + B3 samen zijn **super-additief**: −1,18 tot −1,56 ms tegen −0,79 als ze onafhankelijk waren

**Vraag.** Twee mechanismen van vandaag zijn elk bitexact en vielen elk net
onder hun eigen poort: **V13 H-SCALE** (−0,374 ms, poort ≥0,5 — schaalvlakken
residentie in VRAM, 52% minder gather-bytes) en **V14G B3** (−0,416 ms, poort
≥0,8 — dubbelgebufferde mirror plus gather-stream, zodat slot s+1's PCIe loopt
terwijl slot s rekent). Ze vallen dezelfde weg van twee kanten aan. De werkregel
van dit project verbiedt componentgetallen op te tellen — dus samen meten.

**Verwachting vooraf.** Ergens tussen −0,42 en −0,79 ms, met een reële kans op
**anti-compositie**: een kleinere gather laat B3 minder te verbergen.

**Uitkomst — twee onafhankelijke volledige runs, beide alle poorten groen.**

| | run 1 | run 2 |
|---|---:|---:|
| BASE_A | 21,2252 ms (47,11 tok/s) | 20,6810 ms (48,35 tok/s) |
| **CAND** | **19,6897 ms (50,79 tok/s)** | **19,6046 ms (51,01 tok/s)** |
| BASE_B | 21,2730 ms | 20,8926 ms |
| midden | 21,2491 | 20,7869 |
| drift | **0,0478** | **0,2114** |
| **CAND − midden** | **−1,5594 ms** | **−1,1823 ms** |

Poorten beide runs: C1 bitexact **PASS** (3 prompts × 765 tokens), C2 **PASS**,
V1 VRAM **PASS**, D1 drift **PASS**, **P1 (≥0,5 ms) PASS**.

**De absolute CAND-waarde is opvallend stabiel: 19,6046 en 19,6897 ms — 0,085 ms
uit elkaar.** De deltas verschillen (−1,56 vs −1,18) puur omdat de baselines
tussen de runs thermisch verschoven; de kandidaat zelf niet.

### Dit is een nieuw record en het haalt E50

| | ms/token | tok/s |
|---|---:|---:|
| vorig record (V6 + capaciteitstuning) | 21,0923 | 47,41 |
| **V18 = V6 + H-SCALE + B3** | **19,60-19,69** | **50,8-51,0** |

Zelfde SYNC-semantiek (één replay + één ring-harvest per token) als waarin het
21,0923 ms-record gemeten is, dus direct vergelijkbaar. **E50 — de mijlpaal die
de hele PRO-MAX V2-campagne niet haalde en die daar op de driftpoort strandde —
is hiermee gehaald, met drift 0,048 ms.**

### Waarom super-additief

−1,18 tot −1,56 ms tegen een rekenkundige som van −0,790: **ongeveer twee keer
zo goed samen als apart.** Werkhypothese, expliciet **niet gemeten**: V13's
grootste kostenpost was de plane-fetch (+0,327 ms, apart gemeten met een
marginale probe), en B3's stroomstructuur verplaatst juist dat soort verkeer
naar een aparte stream waar het achter compute verdwijnt. V13 verkleint de
gather, B3 verbergt wat overblijft, en V13's eigen extra kost wordt door B3
opgevangen. Dat is een verklaring die past bij alle drie de metingen, maar het
is een hypothese en verdient een eigen probe voordat iemand erop bouwt.

**Wat dit ook bevestigt:** de werkregel "nooit componentmetingen optellen" is
hier geen formaliteit geweest. Wie had opgeteld had −0,79 verwacht en was
mogelijk gestopt omdat dat nog steeds onder de gecombineerde poort lag. De
werkelijkheid was twee keer beter.

**Kosten.** VRAM: 492,4 MiB residente schaalvlakken (cap 72 × 23 lagen) plus één
extra globale mirror van 2,81 MB. Poort gecontroleerd vóór allocatie. Geen
rekenkundige wijziging: zelfde expert, paneel, rij, schaalbyte,
`e4m3_lut[byte] * global_scale`, zelfde fmaf-volgorde — alleen wáár bytes staan
en wannéér ze bewegen.

**Artefacten.** `pro_research/moe_dev_combined.py`,
`pro_research/combined_v18.py`,
`pro_research/results/v18_combined/PRO_V18_COMBINED.json` (run 2) en
`PRO_V18_COMBINED_run1.json`.

---

## 2026-08-16 — Vlaggen-audit over álle kandidaatkernels: **precies één bestand heeft zowel een mismatch als een gevoelige operatie — en dat is exact de kandidaat die faalde.** De geadopteerde V6-stack is vrijgepleit

**Vraag.** Als PV2-10 sneuvelde op een compileervlag, welke andere kandidaten
lopen dan hetzelfde risico? En — belangrijker — staat er iets mis in de
**geadopteerde** V6-stack?

**Opzet.** Elke `RawModule(...)` in `pro_research/` en `pro_max_v2/` opgezocht,
de vlaggen naast de module gelegd die de kernel vervangt, en per bestand geteld
hoeveel **fast-math-gevoelige** operaties erin staan (`rsqrtf`, `__expf`,
`__logf`, deling door een float, `__fdividef`). Alleen waar béíde fout zijn —
mismatch én een gevoelige operatie — kan het verschil maken.

| bestand | eigen vlaggen | vervangt (vlaggen) | gevoelige ops | verdict |
|---|---|---|---:|---|
| `ervf_dense.py` | fast-math | gpu_kernels (fast-math) | 0 | ✅ match |
| `up_proj_batch_kernels.py` | fast-math | fused_nvfp4 (géén) | **0** | ⚠️ mismatch, **onschadelijk** |
| `down_proj_batch_kernels.py` | géén | fused_nvfp4 (géén) | 0 | ✅ match |
| `scale_resident_kernels.py` | géén | fused_nvfp4 (géén) | 0 | ✅ match |
| `gather_small_grid.py`, `moe_dev_overlap` | géén | fused_nvfp4 (géén) | 0 | ✅ match |
| `pro_max_v2/qkv_v8.py` | **fast-math** | gpu_kernels (fast-math) | 0 | ✅ match |
| **`pro_max_v2/addnorm_v7.py`** | **géén** | gpu_kernels `rmsnorm` (**fast-math**) | **1** | ❌ **mismatch + gevoelig** |

**Precies één bestand in de hele codebase heeft beide problemen, en dat is
`addnorm_v7.py` — PV2-10, de enige kandidaat die op causale pariteit faalde.**
Alle andere matchen óf hebben nul gevoelige operaties, waardoor de vlag daar
geen verschil maakt.

**Een natuurlijk experiment binnen Kimi's eigen pakket.** `qkv_v8.py` gebruikt
wél `--use_fast_math` en haalde causale pariteit op alle drie de prompts;
`addnorm_v7.py` gebruikt hem niet en faalde bij token 124. Zelfde campagne,
zelfde harness, zelfde prompts — het enige verschil is de vlag. Dat is zo dicht
bij een gecontroleerd experiment als je achteraf kunt krijgen.

**De geadopteerde V6-stack is vrijgepleit.** `up_proj_batch_kernels.py` heeft
technisch een mismatch (fast-math terwijl `fused_nvfp4` het niet gebruikt), maar
**nul gevoelige operaties** — alleen `fmaf`, LUT-lookups en shuffles — dus de
vlag is daar een no-op. Dat verklaart ook waarom zijn bitexacte poorten tijdens
de V6-bouw gewoon slaagden: het was geen geluk, er viel niets te verschillen.
Wel opgeruimd zodra iemand die kernel aanraakt.

**Eén kanttekening bij mijn eigen diagnostiek van vandaag.**
`ssm_decode_step` bevat **twee** `__expf`-aanroepen en is dus zeer
fast-math-gevoelig, terwijl mijn `diag_ssm_*`-scripts zonder de vlag compileren.
Binnen elk script delen alle armen dezelfde vlag, dus de **A/B-vergelijkingen**
(×0,685, ×0,945, ×1,031) blijven geldig. Maar de **absolute** ssm-tijden daarin
wijken licht af van productie, en wie een van die varianten ooit integreert moet
`--use_fast_math` toevoegen — anders herhaalt hij PV2-10 letterlijk.

**Wat dit opent.** **PV2-10 is weer een geldige kandidaat**: zijn
correctheidsfalen is verklaard en met één vlag opgelost. Of hij ook snelheid
oplevert is een aparte vraag — mijn eigen versie van dezelfde fusie werd
+0,127 ms trager omdat de add zijn parallellisme verliest, dus de verwachting is
laag. Maar hij hoort niet langer op de lijst "eerst debuggen".

**Artefacten.** Audit reproduceerbaar met een grep op `RawModule(` in
`pro_research/` en `pro_research/pro_max_v2/`.

---

## 2026-08-16 — **PV2-10's onopgeloste bug gevonden en opgelost: een compileervlag.** De add+norm-fusie is nu bitexact — maar in de graph netto trager, dus alsnog niet adopteren

**Aanleiding.** De complete tokenkaart wees kernel-fusie aan als goedkoopste
resterende winst: 3,53 µs per in-graph launch, 105 norm/add-launches per token.
`add_` schrijft precies de buffer die de volgende `norm` leest, dus die twee
horen één kernel te zijn: 105 → 53 launches.

**Geïsoleerd (eager): bitexact op `h` én `out`, ×1,745, −0,354 ms/token.**
Ik heb dat getal meteen gehalveerd vóór ik het geloofde — de geïmpliceerde
6,80 µs/launch matcht de **eager** 7,75 µs, niet de in-graph 3,53 — dus de
verwachting in de graph was ~0,18 ms.

**Eerste graph-run: correctheidspoort FAALT.** CAND wijkt af bij **gegenereerd
token 124**, op één van de drie prompts, de andere twee schoon.

**Dat is letterlijk PV2-10's handtekening.** Kimi's campagne rapporteerde voor
exact dezelfde fusie (add + next-RMSNorm): "full causal parity: **FAIL** — one
prompt first diverges at generated token 124", met als conclusie "eerst
debuggen vóór ooit herindienen". Zelfde fusie, zelfde prompt, zelfde token.

**De oorzaak: een compileervlag.**

    gpu_kernels.py:1525   options=("-std=c++14", "--use_fast_math")
    mijn RawModule        options=("-std=c++14",)

`--use_fast_math` verandert `rsqrtf` naar de benaderende hardware-instructie en
zet denormal-flush aan. Mijn fusie was dus **arithmetisch net niet identiek** aan
de productie-`rmsnorm_bf16w` die hij verving — een verschil van enkele ulps in
de normalisatieschaal, dat via de MoE-routing over 124 tokens opblaast tot een
andere expertkeuze en daarmee een ander token.

Met `--use_fast_math` toegevoegd: **G-V17-C1 bitexact PASS** op alle drie de
prompts, 765 tokens per arm. **De bug is daarmee opgelost, en het verklaart
PV2-10 volledig** — inclusief waarom PV2's micro-tests bitexact waren (die
vergeleken tegen een referentie die dezelfde vlag deelde) terwijl alleen de
causale run het blootlegde.

**Maar de fusie is alsnog geen winst: +0,127 ms/token in de graph** (drift
0,1212, dus goed opgelost). Reden: productie's `add_inplace` draait op
**11 blokken** en mijn fusie doet de add binnen het **ene blok** van de norm —
11× minder parallellisme voor dat deel. In de graph, waar een launch maar
3,53 µs kost, is die ruil negatief. Poort P1 niet gehaald; niet adopteren.

**De les die generaliseert, en die is de echte opbrengst.**
**Elke nieuwe kernel moet met dezelfde compileervlaggen gebouwd worden als de
module die hij vervangt of waarmee hij samenwerkt.** En dat is in dit project
níet uniform:

| module | vlaggen |
|---|---|
| `gpu_kernels.py` | `-std=c++14`, **`--use_fast_math`** |
| `fused_nvfp4.py` | `-std=c++14` (géén fast-math) |

Een kernel die `fused_nvfp4`-werk vervangt mag dus juist **géén** fast-math
gebruiken, en een die `gpu_kernels`-werk vervangt moet het wél. Dit staat
nergens gedocumenteerd en heeft nu aantoonbaar één campagne een onverklaarde
faalarm gekost. Toegevoegd aan de meetregels in TODO.

**Artefacten.** `pro_research/diag_add_norm_fusion.py` + `.json`,
`pro_research/fused_norm_v17.py`,
`pro_research/results/v17_fused_norm/PRO_V17_FUSED_NORM.json`.

---

## 2026-08-16 — **De tokenkaart is compleet.** lm_head 1,107 (69% efficiënt) · norms+adds 0,370 voor 105 launches = **3,53 µs per kernel-launch, ín een gevangen graph**

**Vraag.** Na alle eerdere attributie bleef ~2,6 ms van de 21,24 ms
ongeattribueerd: `lm_head`, 53 `norm`-launches en 52 `add_`-launches. Dat is het
enige deel van `_step_body_graph` dat nooit geprobed is.

**Opzet.** Probes wikkelen `_step_body_graph` in plaats van een component, want
deze kernels staan buiten de laag-bodies. Twee armen, niet vier.
`argmax` en `pos_increment` zijn **bewust niet** geprobed — die schrijven
`_tok_dev` respectievelijk de positie, dus een tweede aanroep zou de reeks
verschuiven. Alles wat wel geprobed wordt schrijft naar kladbuffers.

**Uitkomst (poorten groen: alle armen bitexact, drift 0,1612 ms).**

| | ms/token | detail |
|---|---:|---|
| **`lm_head`** | **1,107** | 179,0 GB/s tegen een vloer van 0,762 → **0,345 ms hoofdruimte, 69% efficiënt** |
| **norms + adds** | **0,370** | 105 launches → **3,53 µs per launch** |

**Het getal dat het meest zegt: 3,53 µs per kernel-launch, binnen een gevangen
CUDA-graph.** `rmsnorm_bf16w` draait als **één blok** voor een 2688-elements
reductie — 10,75 KB, oftewel 0,03 µs aan echt verkeer bij 345,9 GB/s. Vrijwel
de hele 3,53 µs is dus vaste kosten per kernel, niet werk. Dat is een
**fusie**-doel, niet een bandbreedte-doel, en het is de eerste keer dat dit
project een in-graph per-kernel-ondergrens heeft gemeten.

Praktisch: de 52 `add_`-launches wegfuseren in de voorafgaande kernel scheelt
~0,18 ms; hetzelfde voor de norms nog eens ~0,19. Klein maar concreet, en
goedkoper dan alles wat vandaag aan kernelherschrijving geprobeerd is.

### De volledige tokenkaart — alles in-graph, elke regel apart gemeten, elke arm bitexact

| post | ms | % |
|---|---:|---:|
| Mamba in_proj + out_proj (GEMV) | 4,187 | 19,7% |
| MoE gather (PCIe) | 3,849 | 18,1% |
| attention (q/o + flash_decode + kv_write) | 2,479 | 11,7% |
| MoE up_proj | 2,253 | 10,6% |
| MoE shared_expert | 1,810 | 8,5% |
| MoE down_masked | 1,372 | 6,5% |
| MoE panel_scan + reduce + accumulate | 1,119 | 5,3% |
| lm_head | 1,107 | 5,2% |
| Mamba ssm_step | 1,095 | 5,2% |
| norms + adds (105 launches) | 0,370 | 1,7% |
| Mamba gated_norm | 0,273 | 1,3% |
| Mamba conv_step + dt_activate | 0,197 | 0,9% |
| **som** | **20,11** | **94,7%** |
| gemeten token | 21,24 | |

**Voor het eerst is de hele token verklaard** — 94,7% expliciet toegerekend, de
rest binnen de gemeten drift. Elke regel is een eigen marginale meting in de
gevangen graph met een bitexacte poort, geen enkele is afgeleid door aftrekken.

**Wat de kaart zegt over 100 tok/s.** De vijf grootste posten zijn Mamba-GEMV,
PCIe-gather, attention, up_proj en shared_expert — samen 14,6 ms van de 21,24.
Vier daarvan zijn ERVF-GEMV's die al op 69-90% van het kernelplafond draaien;
de vijfde is PCIe-verkeer met een harde ondergrens van 2,47 ms. **Er is geen
enkele post waar meer dan ~0,7 ms uit te halen valt zonder de rekenkunde of de
numerieke precisie te veranderen** — en dat is precies waarom de negen
kernelingrepen van vandaag samen 0,42 ms opleverden.

**Artefacten.** `pro_research/diag_glue_marginals.py` + `.json`.

---

## 2026-08-16 — `ssm_step` afgesloten: **drie bitexacte varianten gebouwd, alle drie gemeten, geen enkele een winst.** Layout ×0,685 · twee-fasen ×0,945 · blok-per-(h,p) ×1,031 — en dat sluit occupancy óók uit

**Vraag.** `ssm_step` draait op 34% van het kernel-tempo, de slechtste van het
model. Na de weerlegde layout-hypothese bleef **occupancy** over als enige
verklaring: 64 blokken × 64 threads = 128 warps op 26 SM's ≈ 5 warps/SM.

**Variant 2 — twee fasen.** Fase 1 volledig parallel over (h, p, n): 524.288
elementen in plaats van 4.096, werkt de state bij met exact dezelfde `fmaf`.
Fase 2 één thread per (h, p) die de sequentiële `acc` doet over wat fase 1 net
schreef. De optelvolgorde blijft daarmee ongemoeid — dat is het hele punt.
Kosten: fase 2 herleest de state, dus 6,29 MB per laag in plaats van 4,19
(**+50% verkeer**).

Uitkomst: **bitexact** (y én state), **×0,945 — 5,5% trager**. Maar de
efficiëntie per byte ging fors omhoog: **123,8 → 167-176 GB/s (+42%)**. Het
parallellisme hélpt dus wel; het extra verkeer eet de winst op.

**Variant 3 — één blok per (h, p), N threads.** Dit haalt het parallellisme
zónder extra verkeer: de state-update is volledig parallel en gecoalesceerd
binnen het blok (128 aaneengesloten floats), en thread 0 doet daarna de
sequentiële `acc` over shared memory. Verkeer identiek aan de productie-kernel.
Parallellisme: 4.096 blokken × 128 threads = **16.384 warps** tegen 128.
(Detail dat ertoe doet: thread 0 moet zelf `fmaf(s, C, acc)` doen — `s*C`
parallel voorberekenen en daarna sommeren rondt twee keer in plaats van één.)

Uitkomst: **bitexact** (y én state), **×1,031 — 3,1%**, 0,024 ms/token.

**Wat dit uitsluit, en dat is de waarde.** Een factor **128× meer parallellisme
bij identiek verkeer levert 3%.** `ssm_step` is dus **niet occupancy-gebonden**
— net zomin als layout-gebonden. Alle drie de bitexacte remedies zijn gebouwd,
geverifieerd en gemeten:

| variant | bitexact | resultaat |
|---|---|---:|
| layout `[h][n][p]` | ✅ y + state | ×0,685 |
| twee fasen | ✅ y + state | ×0,945 |
| blok-per-(h,p) | ✅ y + state | **×1,031** |

**Wat er dan overblijft — en het valt buiten de bitexacte discipline.** De
SSM-state is **float32**: 2,10 MB per laag, 48,3 MB per token, en dat is het
enige buffer in het hele model dat níet gekwantiseerd is (gewichten zijn NVFP4/
FP8/BF16, de KV-cache is FP8). In bf16 zou het verkeer **halveren** → ~0,39 ms
in plaats van 0,78. Maar de SSM-state **accumuleert over tijdstappen**, dus dat
is een echte numerieke wijziging met een kwaliteitspoort, geen dataplaatsing.
Dat is een andere soort beslissing dan alles wat deze sessie gedaan heeft en
hoort expliciet aan de gebruiker voorgelegd te worden in plaats van
binnengeslopen — genoteerd in TODO, niet gebouwd.

**Stand van `ssm_step`:** 1,095 ms in-graph, 0,724 ms hoofdruimte, en na drie
gebouwde en gemeten varianten is daar **0,024 ms** van gerealiseerd. De rest
zit achter een numerieke keuze, niet achter een implementatiekeuze.

**Artefacten.** `pro_research/diag_ssm_twophase.py` + `.json` (bevat beide
varianten), `pro_research/diag_ssm_layout.py` + `.json`.

---

## 2026-08-16 — De `ssm_step`-layouttranspositie: **WEERLEGD, en de warme-L2-versie van dezelfde meting had een regressie verkocht** (×1,48 warm → ×0,685 koud)

**Hypothese.** `ssm_step` heeft state `[h][p][n]` terwijl thread `p` een hele
n-rij bezit, dus bij elke innerlijke stap liggen aangrenzende threads
`N·4 = 512 B` uit elkaar. Een warp-instructie zou dan 32 sectoren ophalen en er
4 bytes van elk gebruiken — ~12,5% benutting. Transponeren naar `[h][n][p]` maakt
de 64 threads van een stap 64 aaneengesloten floats lezen. Rekenkundig
identiek, dus bitexact — anders dan de acc-reductie parallelliseren, wat de
optelvolgorde zou breken.

**Poorten: beide groen.** `y` bitexact **en** de state bitexact modulo de
transpositie. De transformatie zelf klopt dus.

**Eerste meting (één state-buffer, 23× hergebruikt): ×1,484 vóór transponeren.**
0,713 → 0,480 ms/token, 135 → 201 GB/s. Overtuigend.

**Maar die opzet was fout, en ik ving het net op tijd.** Eén state van 2,10 MB
past ruim in 32 MiB L2; de échte lus raakt **23 verschillende** states
(48,3 MB) en heeft ze dus allemaal koud. Precies het artefact dat vanochtend een
GEMV-meting op 336 GB/s zette. Met 23 losse buffers:

| layout | ms/token | GB/s |
|---|---:|---:|
| **`[h][p][n]` (productie)** | **0,799** | **120,8** |
| `[h][n][p]` (transponeerd) | 1,165 | 82,8 |
| | **×0,685 — 46% TRAGER** | |

**Het teken klapt volledig om. De hypothese is weerlegd en de productielayout is
de betere.**

**Waarom mijn analyse fout was.** Ik keek naar coalescing *per instructie* en
vergat dat in `[h][p][n]` elke thread een **aaneengesloten 512 B-rij** streamt.
Per blok leest dat 64 threads × 512 B = 32 KB aaneengesloten; de
prefetcher/L1 bedient dat prima. Instructie-niveau-coalescing is niet hetzelfde
als geheugensysteem-efficiëntie, en bij een per-thread streaming-patroon wint de
tweede.

**Wat er dan wél overblijft voor `ssm_step`'s 34%.** De launch is
`(H,) × min(256, P)` = **64 blokken × 64 threads = 4096 threads = 128 warps op
26 SM's ≈ 5 warps/SM**. Dat is bij lange na niet genoeg om DRAM-latentie te
verbergen, en het is nu de enige overgebleven verklaring. Het lastige: de
n-lus is elementgewijs onafhankelijk (`s[n]` hangt alleen van `s[n]` en `Bv[n]`
af) **behalve de `acc`-reductie**, en juist die parallelliseren breekt de
bitexactheid.

Eén bitexacte uitweg die nog niet geprobeerd is: **twee fasen** — fase 1
volledig parallel over (p, n) die de state bijwerkt en `s` wegschrijft, fase 2
één thread per p die de sequentiële `acc` over de zojuist geschreven `s` doet.
Fase 1 krijgt dan 524.288 parallelle elementen in plaats van 4096, en fase 2
leest data die net geschreven is (dus L1/L2-warm). Grotere ingreep, maar de
enige die de occupancy aanpakt zonder de optelvolgorde te raken.

**De methodische les, en het is dezelfde als vanochtend maar nu duurder
voorkomen.** Een geïsoleerd kernelbenchmark met één hergebruikte buffer meet L2,
niet DRAM. Als ik dit niet had nagemeten had ik een **regressie van 46%** als
winst van 48% gerapporteerd — een verschil van bijna een factor 2 in de
verkeerde richting. **Regel: elke kernelmeting rouleert over evenveel losse
buffers als de echte lus aanraakt, of hij telt niet.**

**Artefacten.** `pro_research/diag_ssm_layout.py` + `.json`.

---

## 2026-08-16 — `ssm_step` is de dader: **1,095 ms tegen `gated_norm`'s 0,273** — en met 88 GB/s draait hij op **34%** van het kernel-tempo, de slechtste efficiëntie in het hele model

**Vraag.** De gemeten stage-splitsing zette `ssm_step + gated_norm` op 1,011 ms
(4,9% van het token) — de grootste ongemeten enkele post die overbleef. Welke
van de twee?

**Opzet.** Vier armen (BASE_A / ssm_step / gated_norm / BASE_B), conform de
eigen regel dat kleine kandidaten niet in een sweep horen. De `ssm_step`-probe
krijgt **eigen kladrecurrentie** want hij schrijft `ssm[i]`; beide probes
schrijven naar kladbuffers.

**Uitkomst (poorten groen: alle armen bitexact, drift 0,1648 ms).**

| | ms/token | % van token |
|---|---:|---:|
| **`ssm_step`** | **1,095** | 5,2% |
| `gated_norm` | 0,273 | 1,3% |
| som | 1,368 | (gegroepeerd gemeten: 1,011) |

**Eerlijkheid over de onzekerheid:** som 1,368 tegen de eerder gegroepeerd
gemeten 1,011 scheelt **0,357 ms** — precies het ruisniveau dat ik vanmiddag
zelf vastlegde voor deze magnitude. De **absolute** waarden dragen dus ±0,36 ms.
De **verhouding** 4:1 ligt daar ruim buiten en is wél hard.

**De byteboekhouding maakt het scherp.** De SSM-state is
64 heads × 64 hdim × 128 state × 4 B = **2,10 MB per laag**. `ssm_step` is een
read-modify-write van die hele state: 4,19 MB per laag, **96,5 MB per token**
over 23 lagen.

| | |
|---|---:|
| vloer bij 260 GB/s (het gemeten kernel-tempo) | **0,371 ms** |
| gemeten | **1,095 ms** |
| behaald | **88,1 GB/s = 34%** |
| **hoofdruimte** | **0,724 ms (3,4% van het token)** |

**34% is de slechtste efficiëntie van elke component die vandaag gemeten is** —
slechter dan attention (45,5%), down_masked (60%), Mamba's GEMV's (80-86%) en
shared_expert (90%). En het is een **pure VRAM read-modify-write**: geen PCIe,
geen sparsity, geen data-afhankelijke grid, geen LRU. Dat maakt het de best
afgebakende kerneloefening die er nog ligt.

**Waarom het traag kán zijn (hypotheses, niet gemeten).** De scan werkt per head
op een [64, 128]-tegel; met 64 heads × 23 lagen zijn dat 1472 kleine
onafhankelijke updates per token. Kandidaten: te weinig parallellisme per launch
(zoals bij de K/V-projecties), een niet-gecoalesceerd state-layout
(`[head][hdim][state]` versus de leesvolgorde), of een afhankelijke keten per
head. Alle drie zijn met dezelfde ablatie-techniek te scheiden die
`down_masked` ontleedde.

**Wat dit betekent voor het totaalbeeld.** Het verandert de conclusie over 100
tok/s niet — 0,72 ms op een token van 21,07 is 3,4% — maar het is wel de
grootste enkele hoofdruimte die nog **in één kernel** zit, en het is er een
zonder de complicaties (PCIe, sparsity, statefulness over sequenties) die elke
andere post vandaag onaantrekkelijk maakten.

**Artefacten.** `pro_research/diag_ssm_vs_gatednorm.py` + `.json`.

---

## 2026-08-16 — Mamba per stage **gemeten** in plaats van afgeleid: GEMV's 4,187 ms · ssm+gated_norm 1,011 · conv+dt 0,197. De in-lus-GEMV draait op **213 GB/s** — tussen mijn twee eerdere claims in

**Vraag.** Ik heb Mamba's 5,168 ms vandaag twee keer verschillend gesplitst,
beide keren door **aftrekken** van een geïsoleerd getal:
- eerst: alles aan de GEMV's → 172,6 GB/s in de lus, "een gat van een half token";
- toen gecorrigeerd: GEMV's 3,448 ms (geïsoleerd tempo) → 258,7 GB/s, "geen gat".

Een geïsoleerd getal van een in-lus getal aftrekken is precies de beweging die
vandaag al één fantoomprioriteit opleverde. Dus: meten.

**Opzet.** Marginale probes in de gevangen graph, **bewust gegroepeerd in drie
armen** in plaats van zes — mijn eigen regel van vanmiddag na een 7-armige
sweep met 0,39 ms drift tussen tussenliggende armen. `conv_step` en `ssm_step`
muteren `conv[i]`/`ssm[i]`, dus die probes krijgen **eigen kladrecurrentie**;
alle probes schrijven naar kladbuffers zodat niets stroomafwaarts verandert.

**Uitkomst (alle poorten groen: G1 alle armen bitexact, G2 drift 0,3969 ms;
basis-midden 20,774 ms).**

| groep | ms/token | % van token |
|---|---:|---:|
| **in_proj + out_proj (GEMV)** | **4,187** | 20,2% |
| **ssm_step + gated_norm** | **1,011** | 4,9% |
| conv_step + dt_activate | 0,197 | 0,9% |
| som | 5,394 | (Mamba-marginaal was 5,168 — sluit binnen de drift) |

**Beide eerdere splitsingen waren fout, en de waarheid ligt ertussenin.**

| bewering | GEMV-deel | in-lus GB/s |
|---|---:|---:|
| eerste claim (alles = GEMV) | 5,168 | 172,6 |
| gecorrigeerde claim (geïsoleerd tempo) | 3,448 | 258,7 |
| **gemeten** | **4,187** | **213,0** |

**De in-lus GEMV haalt 213 GB/s tegen 248-267 geïsoleerd = 80-86%.** Er ís dus
een in-lus-boete, maar hij is ~15-20%, niet de 36% die ik eerst opschreef en
niet nul zoals mijn correctie beweerde. En dat is nog een **bovengrens op de
efficiëntie**: de probe draait direct na de echte aanroep, dus in_proj (27,7 MB)
zit deels nog in L2 (32 MiB) — de échte eerste aanroep is dus mogelijk trager
dan 213 GB/s, wat de boete alleen maar groter maakt.

**De niet-GEMV-post is 1,208 ms (5,8% van het token), en zit vrijwel volledig in
`ssm_step + gated_norm` (1,011).** `conv_step + dt_activate` is met 0,197 ms
verwaarloosbaar. De SSM-scan is dus de enige echte post daar — en die is niet
bandbreedtegebonden (de state is 48 MB over 23 lagen, maar per laag maar
2,1 MB), dus het is een reken-/latentiepost.

**Wat dit sluit of opent.** Sluit: het afleiden-door-aftrekken van Mamba's
splitsing, in beide richtingen. Opent: (a) `ssm_step` vs `gated_norm` uitsplitsen
in een 3-armige A/B — 1 ms is de moeite waard en het is de grootste ongemeten
post die overblijft; (b) de in-lus-boete van 15-20% op de GEMV's is reëel maar
klein, en de twee voor de hand liggende oorzaken (L2-verdringing,
`copy_stream`-contentie) zijn vandaag al apart gemeten en **weerlegd** — dus de
oorzaak daarvan is nog open, maar de post is nu ~1,1 ms in plaats van ~11 en
verdient navenante prioriteit.

**Metaregel die ik hieraan overhoud.** Vandaag zijn er vier kopgetallen
gesneuveld en drie daarvan kwamen uit een aftreksom tussen twee meetregimes.
**Een marginaal is alleen te vertrouwen als hij zélf gemeten is; een verschil
tussen een geïsoleerde en een in-lus meting is een hypothese, geen getal.**

**Artefacten.** `pro_research/diag_mamba_stage_marginals.py` + `.json`.

---

## 2026-08-16 — ⚠️ **VIERDE ZELFCORRECTIE, en deze trekt het "gat van een half token" weer in: de GEMV's draaien in de lus wél op 259 GB/s. Ik had 33% van Mamba's kost aan de verkeerde kernels toegeschreven**

**Hoe het boven kwam.** Ik had het gat (geïsoleerd 248-267 GB/s vs "in de lus"
172,6) als hoogste prioriteit weggeschreven. De twee toetsbare hypotheses zijn
gemeten en **beide weerlegd** (`diag_inloop_gap.json`): dezelfde ERVF-kernel,
alleen de omgeving veranderd, koude rotatie, klokken per arm geregistreerd.

| arm | GB/s |
|---|---:|
| baseline_a | 264,8 |
| L2 leeggeveegd tussen calls (64 MiB) | 261,2 |
| PCIe-contentie op tweede stream | 251,6 |
| beide | 280,9 |
| baseline_b | 274,3 |

baseline-drift 9,5 GB/s. **Geen enkele arm komt in de buurt van 172,6.** L2-
verdringing en `copy_stream`-contentie zijn dus geen van beide de verklaring.

**Waarom niet: het gat bestond niet.** De Mamba-marginaal van 5,168 ms omvat
**niet alleen** de twee GEMV's maar ook `conv_step`, `ssm_step`, `gated_norm` en
`dt_activate`. Ik had de volle 892 MB tegen de volle 5,168 ms gezet. Reken het
met de vandaag gemeten geïsoleerde tijden:

    in_proj  105,2 µs + out_proj 44,7 µs = 149,9 µs per laag
    × 23 lagen                            =  3,448 ms  GEMV-werk
    gemeten Mamba-marginaal               =  5,168 ms
    → rest (conv/ssm/gated_norm/dt)       =  1,720 ms  (33% van Mamba)
    → impliciete GEMV-snelheid in de lus  =  892 MB / 3,448 ms = **258,7 GB/s**

**258,7 tegen 248-267 geïsoleerd — de GEMV's draaien in de lus gewoon op hun
volle tempo.** Er is geen 1,55×-gat. Wat er wél is: **1,72 ms aan
Mamba-toestandswerk (33% van Mamba, 8% van het token) dat nooit apart gemeten
of benoemd is** — de SSM-pijplijn is niet bandbreedtegebonden (de state is
klein) en viel daardoor buiten elke byte-gebaseerde analyse.

**Herziene tokenboekhouding.** Met alle GEMV's op ~260 GB/s:

| post | ms |
|---|---:|
| alle GEMV's (2048 MB bij ~260 GB/s) | ~7,9 |
| PCIe-gather (gemeten in de lus) | 3,85 |
| Mamba SSM/conv-pijplijn | 1,72 |
| attention flash_decode + kv_write | 1,14 |
| MoE panel_scan + reduce + accumulate | 1,12 |
| norms, embed, argmax, lijm | ~1 |
| **verklaard** | **~16,7** |
| **gemeten** | **21,24** |
| **onverklaard** | **~4,5** |

Het restgat is dus **~4,5 ms, niet ~11 ms**. Nog steeds de moeite waard, maar
een kwart van wat ik een blok geleden opschreef, en **niet meer de grootste
post**.

**De les, en die is anders dan de vorige drie.** De eerdere correcties gingen
over een verkeerde baseline. Deze gaat over **een marginaal toeschrijven aan
minder dan wat hij meet**: `_mamba` is niet "twee GEMV's", het is twee GEMV's
plus een recurrentiepijplijn. Regel erbij: **voor je een marginaal door bytes
deelt, som eerst op wat die component allemaal uitvoert.** Anders krijg je een
plausibel ogende GB/s die een kwart te laag is en die vervolgens een
fantoomprioriteit wordt.

**Wat dit opent.** De 1,72 ms SSM/conv-pijplijn is nu een genoemde, ongemeten
post van 8% van het token — vergelijkbaar met attention (2,48 ms) en groter dan
het hele down_masked-pad. Die verdient een eigen marginale ontleding
(`conv_step` / `ssm_step` / `gated_norm` / `dt_activate` apart), met dezelfde
voorzichtigheid: `_mamba` is stateful, dus elke probe heeft eigen kladstate
nodig — dat is vandaag al één keer misgegaan.

**Artefacten.** `pro_research/diag_inloop_gap.py` + `.json`; afleiding uit
`diag_ervf_batched_tiled.json` (geïsoleerde tijden) en
`diag_component_marginals_graph.json` (marginaal).

---

## 2026-08-16 — K-tiling WEERLEGD (×1,14 tegen ×1,64), het batchplafond staat daarmee vast — **maar de vergelijking legt een nieuw gat bloot: in de lus haalt dezelfde kernel maar 64% van zijn geïsoleerde snelheid**

**Vraag.** De ×1,64 batchwinst op ERVF-paden was een ondergrens omdat mijn
kernel X uit global las (N kopieën in shared > 48 KB). Een K-getegelde variant
die X per tegel in shared zet was de laatste manier om die bovengrens te
verleggen — en dat ene getal besliste of batch richting 100 kon.

**Opzet.** Tegels van QT = 256 uchar4-groepen (1024 elementen). Omdat QT gelijk
is aan de tid-stride krijgt elke virtuele tid **precies één q per tegel** en
loopt hij q = tid, 256+tid, 512+tid … — dezelfde reeks in dezelfde volgorde als
de ongetegelde lus. Dát is het bitexactheidsargument, en het is als poort
gecontroleerd. Shared: N·1024·4 B + 1 KB LUT = 17 KB bij N=4, 33 KB bij N=8.
De motivatie was geometrisch, niet hoopvol: in ERVF hangt de elementindex
`tid = lane + 16·vi` **niet** van `sub` af, dus alle 16 rijen in een blok lopen
dezelfde k-set — elk X-element wordt door 16 threads gebruikt.

**Uitkomst — weerlegd, en duidelijk.** N=1 bitexact tegen productie-ERVF, alle
outputs eindig.

| | N=1 | N=2 | N=4 | N=8 |
|---|---:|---:|---:|---:|
| getegeld (X in shared) | 0,804 | 1,197 | **1,135** | 1,037 |
| eerder: X uit global | 0,971 | 1,377 | **1,640** | — |

**Getegeld is slechter, niet beter**, en verslechtert richting N=8. Verklaring
die uit de opzet volgt: de 16× hergebruik van X werd al door L1 geleverd, dus
stageren verwijdert geen verkeer maar voegt wel twee `__syncthreads()` per tegel
en een kopieerlus toe; bij N=8 kost 33 KB shared per blok bovendien occupancy.

**Daarmee staat het batchplafond vast: ×1,64 bij N=4 op de ERVF-paden**, en de
projectie van **~71 tok/s bij N=4 / ~83 bij N=8** blijft staan. De laatste
openstaande manier om die bovengrens te verleggen is dichtgemeten.

### Maar de cijfers naast elkaar leggen levert iets nieuws op

| | GB/s |
|---|---:|
| apparaat, puur streamen | 345,9 |
| ERVF **geïsoleerd**, koud, N=1 | **248-267** (72-77%) |
| Mamba **in de lus** (in-graph marginaal, 892 MB / 5,168 ms) | **172,6** (50%) |

**Dezelfde kernel, dezelfde shape, dezelfde dtype — en in de lus haalt hij maar
64% van zijn geïsoleerde snelheid.** Dat is een gat van ~1,55× dat noch aan de
kernel, noch aan batching ligt, en het is nooit eerder apart benoemd.

Reken het door: als élke GEMV in de lus zijn geïsoleerde tempo haalde, zou het
token 2048 MB / 267 GB/s = 7,67 ms VRAM + 2,47 ms PCIe = **10,1 ms ≈ 99 tok/s**
kosten. We meten 21,24 ms. **Bijna de helft van het token gaat verloren tussen
"wat de kernel kan" en "wat de kernel in de lus doet".**

Kandidaat-oorzaken, geen van alle gemeten (dus expliciet hypotheses):
L2-verdringing tussen lagen (de werkset per token is 2 GB, dus niets blijft
staan tussen twee aanroepen van dezelfde kernel), bandbreedteconcurrentie met de
`copy_stream` die tegelijk expert-misses binnenhaalt, en thermische throttling
onder aanhoudende belasting (795 MHz gezien tegen 1777 in één run).

**Dat is de eerstvolgende meting, en het is een grotere post dan alles wat
vandaag geprobeerd is.** Hij is bovendien orthogonaal aan de single-stream/batch
keuze: sluit je dit gat, dan profiteren beide routes.

**Artefacten.** `pro_research/diag_ervf_batched_tiled.py` + `.json`.

---

## 2026-08-16 — **De reikwijdte van de correctie: ESSENTIEEL ALLE grote GEMV's zijn al ERVF.** Daarmee geldt het ×1,64-plafond voor de hele dense stroom — en batch landt naar verwachting op 70-85 tok/s, niet 100+

**Wat ik in mijn vorige bericht nog fout had.** Ik schreef dat `shared_up` en
`routed_up` "níet ERVF" zijn en dat daar de ~3,6× waarschijnlijk overeind bleef.
Dat is onjuist. Nagekeken in de bron:

- `fused.gemv_into` (fused_nvfp4.py:944) doet `if self.use_ervf:` en gaat dan
  naar `gemv_ervf` — **ERVF is de standaard**, niet een opt-in. Dat pad bedient
  de **shared expert up én down** en de **lm_head**.
- `up_kernels.run_batched` (up_proj_batch_kernels.py:265) gebruikt
  `rpb = 256 // WIDTH` met WIDTH 16 — **dezelfde ERVF-geometrie**, voor de
  routed up-proj.

**Volledige inventaris van wat ERVF bedient:**

| pad | shape | dtype | ERVF? | MB/token |
|---|---|---|---|---:|
| Mamba in_proj | (10304, 2688) | FP8 | ✅ whitelist | 637,4 |
| Mamba out_proj | (2688, 4096) | FP8 | ✅ whitelist | 253,2 |
| q_proj | (4096, 2688) | BF16 | ✅ whitelist | 132,1 |
| o_proj | (2688, 4096) | BF16 | ✅ whitelist | 132,1 |
| shared up/down | — | NVFP4 | ✅ `gemv_into` default | 290,0 |
| routed up | (1856, 2688) | NVFP4 | ✅ `run_batched` WIDTH 16 | 387,3 |
| lm_head | — | NVFP4 | ✅ `gemv_into` default | 198,2 |
| **k/v_proj** | (256, 2688) | BF16 | ❌ niet whitelisted | 16,5 |

**Van de 2048 MB/token VRAM-verkeer gaat 2031 MB (99,2%) door een
ERVF-kernel.** Alleen de K/V-projecties (16,5 MB) niet — en die zijn eerder
vandaag gemeten als 0,75× onder ERVF, dus terecht uitgesloten.

**Waarom dat de batch-verwachting halveert.** De gemeten ERVF-eigenschap is:
**247-266 GB/s bij N=1 = 77% van het apparaatplafond**, en batching daarop geeft
**×1,64 bij N=4**. Dat gold voor Mamba, en het geldt nu dus voor vrijwel álles.
Mijn vorige projectie ging er nog van uit dat 677 MB/token (shared + routed) de
volle ~3,6× zou halen; dat vervalt.

**Herziene projectie (alle aannames expliciet, dit is géén meting).** Bij N=4,
met ×1,64 op de ERVF-paden en de gemeten unie-factor ~2,5/4 op de
routed/PCIe-posten:

| post | nu | bij N=4 | bespaart |
|---|---:|---:|---:|
| Mamba | 5,168 | 3,152 | 2,02 |
| attention | 2,479 | ~1,88 | ~0,60 |
| shared_expert | 1,810 | 1,104 | 0,71 |
| routed up_proj | 2,253 | ~1,25 | ~1,00 |
| gather (PCIe, unie) | 3,849 | 2,406 | 1,44 |
| down_masked + scan + reduce + accum | 2,491 | 1,557 | 0,93 |
| rest (lm_head ERVF, norms, embed) | ~3,7 | ~3,2 | ~0,50 |
| **totaal** | **21,24** | **~14,0** | **~7,2** |

→ **~14,0 ms ≈ 71 tok/s bij N=4**; bij N=8 met ×~1,9 en unie 4/8 ruwweg
**~12 ms ≈ 83 tok/s**.

**Dus: batch landt naar verwachting op 70-85 tok/s, niet boven de 100.**
Samen met single-stream's harde plafond van ~94 tok/s betekent dat: **met de
huidige kerneltechnologie haalt géén van beide routes de 100.**

**De diepere les, en die is opbouwend.** *Het succes van ERVF is precies wat de
bovenkant van batching wegneemt.* Batching's klassieke winst is het amortiseren
van gewichtslezingen — dat werkt alleen als je bandbreedte verspilde. V4-V6
hebben dat verspillen al grotendeels gestopt (77% van het apparaatplafond). Je
kunt niet twee keer dezelfde inefficiëntie opeten. Dat is geen tegenslag maar
een consistent beeld: alle metingen van vandaag wijzen dezelfde kant op.

**Wat het nog wél kan worden — één onbeantwoorde vraag die het verschil maakt.**
Mijn batched ERVF-kernel leest X uit **global** omdat N kopieën in shared de
48 KB-limiet overschrijden. Bij N=4 is dat 4 × 4 B aan X-verkeer per
gewichtsbyte. **De ×1,64 is daarmee een ondergrens, en mogelijk een forse.** Een
K-getegelde variant die X in shared houdt is de enige nog openstaande manier om
de bovenkant te verleggen. **Dat getal beslist of batch richting 100 kan.** Het
is niet gemeten en mag niet aangenomen worden.

**Artefacten.** Broninventaris uit `fused_nvfp4.py:944`,
`up_proj_batch_kernels.py:265`, `selective_ervf_v3.py:37-38`;
metingen in `diag_ervf_batched_fp8.json` en `diag_batched_vs_ervf_baseline.json`.

---

## 2026-08-16 — ⚠️⚠️ **TWEEDE, ZWAARDERE CORRECTIE: de batch-versnelling was gemeten tegen een baseline die productie niet draait. Tegen de échte ERVF-baseline is het ×1,64 bij N=4, niet ×3,5**

**Wat er mis was.** `diag_batched_gemv_scaling` en `diag_batched_gemv_fp8`
vergeleken hun batched kernel met de **one-block-per-row**-geometrie. Maar
productie draait die niet voor deze shapes: `(10304, 2688)` en `(2688, 4096)`
staan in `FP8_ERVF_SHAPES`, dus `_install_selective` routeert ze naar
**ERVF-16**. Gemeten verschil:

| shape | row-block | ERVF-16 | ERVF sneller |
|---|---:|---:|---:|
| mamba_in_proj | 77,9 GB/s | **266,3 GB/s** | **3,38×** |
| mamba_out_proj | 66,8 GB/s | **247,5 GB/s** | **3,69×** |

**Vrijwel de hele "batch-winst" die ik rapporteerde was het terugwinnen van
terrein dat ERVF al had.** Het waarschuwingssignaal stond al in de data en ik
had het moeten zien: de geïsoleerde FP8 N=1-meting gaf 93 GB/s terwijl de
in-lus Mamba-marginaal ~170 GB/s impliceerde — de lus kán niet sneller zijn dan
de geïsoleerde kernel tenzij het een ándere kernel is. Dat was precies het geval.

**De eerlijke meting.** Batching gebouwd **op** de ERVF-16-geometrie (N
accumulatorsets), vergeleken met wat productie draait. N=1 bitexact tegen
productie-ERVF:

| | N=1 | N=2 | N=4 |
|---|---:|---:|---:|
| mamba_in_proj | 1,049 | 1,303 | **1,597** |
| mamba_out_proj | 0,777 | 1,562 | **1,746** |
| **MB-gewogen** | 0,971 | 1,377 | **1,640** |

**×1,64 bij N=4 op 890 MB/token — niet ×3,5.**

**Waarom, en dit is de kern.** ERVF haalt bij N=1 al **247-266 GB/s = 77% van
het apparaatplafond (345,9)**. Er is dus nauwelijks bandbreedte-hoofdruimte
over om met batching terug te winnen. Batching helpt wanneer je
bandbreedte-gebonden bent mét een matige kernel; ERVF is dat niet. Bij N=4 doet
de kernel 4× zoveel FMA's op dezelfde bytes en wordt **rekengebonden**: de
batched stap haalt nog maar ~105 GB/s aan gewichtsverkeer.

**Een eerlijke beperking van deze meting zelf.** Mijn batched ERVF-kernel leest
X uit **global** in plaats van shared memory, omdat N kopieën van X stageren de
48 KB dynamische-shared-limiet overschrijdt bij N=4, cols=4096. Dat betekent
N × 4 B aan X-verkeer per gewichtsbyte. **De 1,64× is daarmee een ondergrens**;
een echte implementatie zou de K-dimensie tegelen zodat X in shared blijft. Hoe
veel dat scheelt is niet gemeten en mag niet aangenomen worden.

**Herziene projectie, en dit verandert het advies niet maar wel de verwachting.**
De componenten splitsen nu in twee groepen:
- **al ERVF, weinig batchwinst (~1,6×)**: Mamba 892 MB + q/o_proj 264 MB
- **niet ERVF, wél veel batchwinst (~3,6×)**: shared_up 290 MB, routed_up 387 MB,
  plus de MoE-gather/down-pijplijn die daarnaast nog unie-deling krijgt

Doorgerekend op de in-graph componenttijden komt N=4 daarmee op **~13,0 ms ≈
77 tok/s** in plaats van de eerder geprojecteerde 88. **De 100 wordt bij N=4 dus
niet gehaald, en bij N=8 is de marge onduidelijk.**

**Wat dit betekent voor de beslissing.** Het advies "ga naar batch" blijft staan
— single-stream zit hard op ~94 tok/s theoretisch maximum en batch is het enige
pad met structurele hoofdruimte — maar **de verwachte opbrengst is fors lager
dan ik twee blokken geleden schreef, en het is niet meer vanzelfsprekend dat
batch de 100 haalt.** Wat er nu eerst moet gebeuren, vóór B1:
1. een **getegelde** batched ERVF-kernel die X in shared houdt, om de echte
   bovengrens van de dense-batchwinst te bepalen (de 1,64× is een ondergrens);
2. de niet-ERVF-shapes (shared_up, routed_up) apart meten tegen hún
   productiekernel — die zijn níet ERVF, dus daar geldt de row-block-baseline
   wél en staat de ~3,6× waarschijnlijk overeind.

**De les, en die is duur betaald.** Drie keer op rij heb ik in dit blok een
kopgetal moeten corrigeren: verkeerde dtype, verkeerde baseline, en een
kernel-handicap in mijn eigen kandidaat. Telkens was het signaal al aanwezig in
eerder gemeten data. **Regel voor de rest van dit project: vergelijk een
kandidaat altijd met wat de runtime daadwerkelijk uitvoert voor díe shape —
niet met een referentie-implementatie die ernaast staat.** Dat betekent
concreet: check `BF16_ERVF_SHAPES` / `FP8_ERVF_SHAPES` vóór je een baseline kiest.

**Artefacten.** `pro_research/diag_batched_vs_ervf_baseline.py` + `.json`,
`pro_research/diag_ervf_batched_fp8.json`.

---

## 2026-08-16 — ⚠️ **Correctie op mijn eigen batch-meting: de twee Mamba-shapes zijn FP8, niet BF16.** Gecorrigeerd ×3,52 bij N=4 en ×4,64 bij N=8 — en FP8 verzadigt eerder, wat een echte waarschuwing is

**Wat er mis was.** `diag_batched_gemv_scaling` mat alle zes shapes als **BF16**.
Voor de twee grootste klopt dat niet: het Mamba-laagbestand is 38.782.608 B voor
38,7M parameters = **1 byte per parameter, dus FP8** (en `quantization_config`
richt zich expliciet op `mixer.in_proj`/`out_proj` met 8 bits). Die twee dragen
**890,6 van de 1685,7 MB/token** die het gewogen gemiddelde dekte — dus het
kopgetal moest opnieuw.

**Opzet.** Zelfde methode, juiste dtype: FP8 e4m3 met tensor-scale, koude
rotatie, poort G1 = N=1 bitexact tegen de productie-`gemv_fp8_tensor`-geometrie
**plus** een eindigheidscontrole.

**Uitkomst (beide shapes bitexact en eindig).**

| shape | µs/token N=1 | µs/token N=8 | ×N=4 | ×N=8 |
|---|---:|---:|---:|---:|
| mamba_in_proj (FP8) | 297,6 | 72,5 | 3,49 | **4,11** |
| mamba_out_proj (FP8) | 106,3 | 25,5 | 3,42 | **4,17** |

**Gecorrigeerd MB-gewogen over alle zes shapes:**

| | N=2 | N=4 | N=8 |
|---|---:|---:|---:|
| eerder (alles als BF16) | 1,94 | 3,61 | 5,10 |
| **gecorrigeerd** | ~1,92 | **3,52** | **4,64** |

**Bij N=4 verandert er weinig (3,61 → 3,52), bij N=8 wel (5,10 → 4,64).**
Herziene projectie: N=4 ≈ 11,3 ms ≈ **88 tok/s**, N=8 ≈ 9,5 ms ≈ **105 tok/s**.
Nog steeds boven de 100 bij N=8, maar met minder marge.

**De onderliggende vondst is belangrijker dan de correctie zelf.** FP8 heeft de
**helft** van de bytes van BF16, maar de per-token tijd bij N=1 is vrijwel
identiek: **297,6 µs (FP8) tegen 296,2 µs (BF16)** op dezelfde shape. Halveer de
bytes en er verandert niets → **deze kernel is bij N=1 niet
bandbreedte-gebonden maar decode/reken-gebonden**. En precies daarom verzadigt
hij eerder onder batching: bij N=8 haalt FP8 4,1× waar BF16 5,1× haalt.

Dat is een concrete waarschuwing voor het batchprogramma: **Mamba's
hoofdruimte onder batching is kleiner dan de byte-boekhouding suggereert**,
want Mamba's 892 MB/token wordt door een kernel gelezen die al niet
bandbreedte-gebonden is. De 79%-dense-redenering uit de beslisnota blijft
kloppen in richting, maar de winst op het grootste blok is ~4× en niet ~8× bij
N=8. Dat hoort in B0's ontwerpkeuze voor `N_MAX` mee te wegen: **N=4 levert 90%
van perfecte schaling, N=8 nog maar 58%** — de extra VRAM (~485 MB) en
complexiteit van N=8 kopen steeds minder.

**Artefacten.** `pro_research/diag_batched_gemv_fp8.py` + `.json`.

---

## 2026-08-16 — **De batch-hypothese is GEMETEN, niet meer beredeneerd: per-token ×3,61 bij N=4 en ×5,10 bij N=8, bitexact bij N=1** — dit is het eerste resultaat van vandaag dat richting 100 wijst

**Vraag.** De beslisnota (`DECISION_SINGLE_STREAM_VS_BATCH.md`) adviseert
single-stream los te laten, en steunt op één fysieke claim: *bij N>1 wordt een
GEMV een GEMM met kleine N, de gewichtsmatrix wordt één keer gelezen voor N
tokens, en de kernels die nu op 157-172 GB/s vastzitten houden op
bandbreedte-gebonden te zijn.* Dat was **rekenwerk, geen meting**. Vóór iemand
weken in een runtime-herschrijving stopt (elke buffer is 1D en single-sequence)
moet dat op déze hardware met déze shapes gecontroleerd worden.

**Opzet.** Y[N, rows] = W[rows, cols] · X[N, cols] voor N ∈ {1,2,4,8} op de zes
echte shapes, koude rotatie van 6 matrices per shape (geen L2-artefact). De
kernel is de productie-`gemv_bf16`-geometrie met N accumulatoren per thread:
`w[k]` wordt **één keer** geladen en N keer gebruikt — precies de deling waar
het batchprogramma op rust. Per-token tijd is altijd ms_per_batch_step / N.

**Twee eigen bugs eerst gevonden en gefixt.**
1. `float acc[MAXN]` met N als **runtime**-argument dwingt dynamische indexering
   van een lokale array; dat crashte met `cudaErrorIllegalAddress` bij N≥4.
   Opgelost door per N een aparte kernel te genereren met N als
   **compile-time**-constante (zoals een echte implementatie het ook zou doen
   met een vaste `N_MAX`).
2. Willekeurige `uint16` als bf16 levert **NaN/Inf**-exponentpatronen. Een
   bitsgewijze outputvergelijking slaagt daar triviaal op (NaN-bits == NaN-bits).
   Nu echte bf16 (getrunceerde float32) en een expliciete `finite`-controle.
   ⚠️ **Diezelfde zwakte zit in `diag_kv_proj_ervf.py` van vandaag** — die
   gebruikte ook willekeurige uint16. De bitexactheid daar is nog steeds geldig
   (identieke bits), maar de conclusie "ERVF is 0,75× op de K/V-shape" verdient
   een hermeting met echte waarden vóór hij zwaar geciteerd wordt.

**Uitkomst — poort G1: N=1 bitexact tegen de productiekernel op alle zes de
shapes, alle outputs eindig.**

| shape | µs/token N=1 | µs/token N=8 | ×N=4 | ×N=8 |
|---|---:|---:|---:|---:|
| mamba_in_proj | 296,2 | 58,4 | 3,68 | 5,08 |
| mamba_out_proj | 105,6 | 22,1 | 3,53 | 4,77 |
| q_proj | 109,4 | 22,1 | 3,48 | 4,96 |
| o_proj | 115,7 | 21,9 | 3,66 | 5,28 |
| shared_up | 106,2 | 19,8 | 3,64 | 5,36 |
| routed_up | 51,2 | 9,8 | 3,58 | 5,25 |
| **MB-gewogen (1686 MB/token gedekt)** | **1,00** | | **3,61** | **5,10** |

**De claim houdt stand, en opvallend consistent over alle zes de shapes.** Bij
N=4 is 3,61 van de ideale 4 = **90% van perfecte schaling**; bij N=8 is 5,10 van
8 = 64% — daar begint iets anders te knellen (x-verkeer of rekenwerk), maar het
blijft flink stijgen.

**Wat dat voor het doel betekent — projectie, uitdrukkelijk geen meting.**
Toegepast op de in-graph componenttijden (21,24 ms/token), met de gemeten
unie-factor voor het routed deel:
- N=4: dense (Mamba 5,168 + attention 2,479 + shared 1,810 = 9,457) → ~2,63;
  routed-posten via unie ~2,5/4 → totaal ≈ **11,2 ms/token ≈ 89 tok/s**
- N=8: dense → ~1,85; routed-unie ~4/8 → totaal ≈ **9,3 ms/token ≈ 107 tok/s**

**Bij N=8 komt de projectie voor het eerst boven de 100 uit.** Dat is een
projectie met stapelende aannames (kernel-schaling toegepast op
componenttijden, geen orkestratiekost, unie-factor geëxtrapoleerd) en géén
resultaat — maar het is de eerste keer deze sessie dat een pad naar 100 niet
door de rekensom wordt uitgesloten. Single-stream werd op ~94 tok/s
theoretisch maximum vastgepind; dit ligt erboven met marge over.

**VRAM-controle, want dat is de eerste plek waar dit stukloopt.** Per sequentie:
Mamba-state 23 × 64 × 64 × 128 × 4 B ≈ 48 MB, KV bij ctx 4096 ≈ 12,6 MB. Bij
N=8 dus ~485 MB extra tegen ~605 MiB gemeten vrij. **Het past, maar krap** — en
dat is een harde randvoorwaarde voor de B0-ontwerpkeuze `N_MAX`.

**Wat dit sluit of opent.** Sluit: de twijfel of het batchprogramma op een
denkfout rust. Opent: B1 (dense shell) mag gebouwd worden op een gemeten
fundament in plaats van op een redenering — en de volgorde uit de beslisnota
(dense eerst, 79% van het verkeer en het makkelijkste deel) wordt door deze
cijfers bevestigd: de dense shapes schalen 3,5-3,7× bij N=4, net zo goed als de
routed shape.

**Artefacten.** `pro_research/diag_batched_gemv_scaling.py` + `.json`.

---

## 2026-08-16 — Attention per stage ontleed: **de K/V-projecties draaien op ~35 GB/s, 3,3× slechter per byte dan Q** — plus een eerlijke meetgrens die ik hier moet vastleggen

**Vraag.** Attention is in de graph 2,479 ms tegen een vloer van 1,128 (45,5%,
het minst efficiënte pad). PV2-11's fusie van Q/K/V werd 2,628 ms trager, dus
het zit niet in het aantal launches. Welke van de vijf stages dan wel?

**Opzet.** Marginale probes per stage in de gevangen graph, zelfde methode die
`down_masked` lokaliseerde. Alle stages zijn idempotent (projecties
overschrijven `qv`/`kv_`/`vv`, de KV-writes adresseren dezelfde device-positie
met dezelfde bytes, flash-decode overschrijft `ctx`, O-proj overschrijft `out`)
— en dat is als poort gecontroleerd, niet aangenomen.

**Uitkomst (twee runs, beide alle poorten groen).**

| stage | run 1 (5 armen) | run 2 (7 armen) | bytes/token | GB/s (run 2) |
|---|---:|---:|---:|---:|
| qkv_proj | 1,008 | 1,386 | 148,6 MB | 107,2 |
| — q_proj alleen | — | 1,170 | 132,1 MB | **112,9** |
| — kv_proj alleen | — | 0,477 | 16,5 MB | **34,6** |
| flash_decode | 0,616 | 0,819 | (KV, ~0,6 MB) | — |
| kv_write | 0,131 | 0,322 | — | — |
| o_proj | 0,394 | 0,005 | 132,1 MB | — |
| basis-midden | 20,786 | 21,268 | | |
| drift | 0,336 | 0,423 | | |

**⚠️ Meetgrens die ik hier expliciet vastleg.** De twee runs verschillen op
`o_proj` met **0,39 ms** op dezelfde probe, en `q_proj + kv_proj` (1,647) telt
niet op tot `qkv_proj` (1,386) — een gat van 0,26 ms. De driftpoort vergelijkt
alleen de **eerste en laatste** arm; tussenliggende armen kunnen op een ander
thermisch punt zitten, en bij 7 armen duurt de run twee keer zo lang. **Stage-
marginalen van deze orde (~0,1-0,5 ms) zijn in een run met veel armen dus niet
betrouwbaar opgelost.** Dat is geen reden om de meting weg te gooien, wel om er
alleen conclusies uit te trekken die groter zijn dan die ruis — en om kleine
kandidaten voortaan in een **3-armige** A/B te meten, niet in een sweep.

**Wat wél ruim boven de ruis uitkomt.** `kv_proj` doet **16,5 MB in 0,477 ms =
34,6 GB/s**, terwijl `q_proj` **132,1 MB in 1,170 ms = 112,9 GB/s** doet. Dat is
**3,3× slechter per byte**, veel groter dan het 0,3 ms ruisniveau, en het heeft
een voor de hand liggende verklaring: K en V hebben elk maar **256 rijen**, dus
bij de ERVF-16-geometrie (16 rijen per blok) leveren ze **16 blokken** op een
GPU met **26 SM's**. Ze vullen het apparaat niet eens half. Q heeft 4096 rijen =
256 blokken en draait daarom 3,3× efficiënter per byte.

**Wat dit opent.** De K/V-projecties zijn een **occupancy**-probleem, niet een
bandbreedte- of latentieprobleem — een derde soort dan alles wat vandaag langskwam.
De voor de hand liggende ingreep is split-K: de reductie over de 2688 kolommen
verdelen over meerdere blokken (16 rijen × 4 splits = 64 blokken). Dat verandert
de optelvolgorde en is dus **niet vanzelf bitexact** — precies waar de
ERVF-techniek voor bestaat, dus het is te doen maar het is echt werk, geen
parameterwissel. Verwachte winst is klein in absolute zin (0,3-0,4 ms) en moet
in een 3-armige A/B gemeten worden gezien de ruis hierboven.

**Artefacten.** `pro_research/diag_attention_stage_marginals.py` + `.json`.

---

## 2026-08-16 — PRO V16: **PV2-11 (Q/K/V one-launch) is nu beslist — en het is negatief.** Bitexact, maar **+2,628 ms/token**, bij een drift van 0,0416 ms

**Vraag.** PV2-11 was de enige exacte kandidaat die de PRO-MAX V2-campagne
expliciet als *onbeslist* achterliet: micro bitexact, causale pariteit PASS op
alle drie prompts, kandidaat 0,2387 ms ónder het baseline-midden — en uitsluitend
gesneuveld op de driftpoort (1,8577 ms tegen een grens van 1,0). Kimi's eigen
verdict: "correctheid overleeft; performance is **onopgelost, niet negatief**.
Alleen hermeten onder een steady-state/interleaved harness." Bovendien zit hij
op precies het pad dat de in-graph-attributie zojuist als **minst efficiënt**
aanwees (attention, 45,5%). Dubbel gemotiveerd dus.

**Opzet.** Kimi's `pro_max_v2/qkv_v8.py` ongewijzigd geïnstalleerd in de
drift-stabiele graph-harness (V12-recept: preheat naar steady state, één
gevangen runtime, hercapture per arm zonder heralloceren, armen kort na elkaar
in één proces). SYNC-semantiek, dus vergelijkbaar met het 21,0923 ms-record.

**Uitkomst.**

| arm | p50 ms | tok/s |
|---|---:|---:|
| BASE_A | 23,6136 | 42,348 |
| **CAND (PV2-11)** | **26,2623** | **38,077** |
| BASE_B | 23,6552 | 42,274 |

midden 23,6344 · **drift 0,0416 ms** · **CAND − midden = +2,6279 ms**.
Poorten: C1 bitexact **PASS** · C2 **PASS** · D1 **PASS** · P1 **FAIL**.

**Wat dit beslist.** De 0,2387 ms "winst" die PV2 rapporteerde was
**driftruis**. Bij een drift van 0,0416 ms — 45× kleiner dan PV2's 1,8577 —
is de kandidaat **2,6 ms trager**, wat ruim buiten elke ruismarge valt.
**PV2-11 is hiermee gesloten, negatief.** Dat is precies waarvoor de kandidaat
openstond, en het antwoord is er nu.

**Waarom hij verliest, en dat is de bruikbare les.** De kandidaat vervangt
`_attention` in zijn geheel en berekent Q/K/V met een eigen **BF16**-kernel in
één launch. Maar de V6-stack draait Q en O al via **selectieve ERVF** — dat wás
V3-G1B's hele winst (−3,3841 ms). De fusie gooit die ERVF-winst op Q dus weg om
twee kernel-launches te besparen, en twee launches zijn ~15 µs waard terwijl de
weggegooide ERVF-winst een veelvoud daarvan is. **Fusie die een snellere kernel
door een langzamere vervangt is netto verlies, hoe elegant de fusie ook is.**
Dat generaliseert naar de andere final-mile-fusiekandidaten: check eerst of het
pad al geoptimaliseerd is vóór je het fuseert.

**Wat dit ook bevestigt.** De harness detecteert een effect van 2,6 ms bij
0,04 ms drift, en gaf eerder dezelfde dag betrouwbaar −0,42 (B3) en +0,04 (V15).
Het meetprobleem dat de hele PRO-MAX V2-campagne onbeslisbaar maakte is
daadwerkelijk opgelost.

**Artefacten.** `pro_research/qkv_v16.py`,
`pro_research/results/v16_qkv/PRO_V16_QKV_GRAPH.json`.

---

## 2026-08-16 — Componentattributie **in de gevangen graph**: MoE 9,41 · Mamba 5,17 · attention 2,48 ms — en attention blijkt met **45,5%** het minst efficiënte pad, niet down_masked

**Vraag.** De eager marginalen bevatten ~7,75 µs uitgiftetijd per kernel-launch,
die de productiegraph niet betaalt. Hoe ziet de hoofdruimtetabel eruit in het
regime waarin productie écht draait?

**Opzet.** Dezelfde marginale probes, maar elke arm **hercapture't de graph** na
het installeren van zijn probe (`_recapture` doet alleen de capture opnieuw en
hergebruikt alle buffers van `setup_graph()`; `setup_graph()` zelf zou
early-returnen én per aanroep 0,656 GiB pinned embedding heralloceren).
SYNC-semantiek: één replay plus één ring-harvest per token — hetzelfde regime
als het 21,0923 ms V6-record. Full: 3 prompts × 192 tokens per arm.

**Uitkomst (alle poorten groen: G1 alle armen bitexact, G2 drift 0,3777 ms).**

Basis-midden **20,7722 ms = 48,14 tok/s**. Dat is de V6-stack, gemeten in een
drift-gecontroleerde harness met preheat — geen nieuw mechanisme en dus geen
nieuw record, maar wel de best gecontroleerde meting van dezelfde stack tot nu
toe (het record staat op 21,0923 ms; het verschil is meetregime en thermische
toestand, niet code).

| component | **graph** | eager | verschil | voorspelde launch-overhead | launches |
|---|---:|---:|---:|---:|---:|
| MoE | **9,408** | 11,004 | 1,596 | 3,209 | 414 |
| Mamba | **5,168** | 5,662 | 0,494 | 0,357 | 46 |
| attention | **2,479** | 1,917 | **−0,562** | 0,186 | 24 |
| som | 17,056 (82%) | | | | |

**Wat dit over de launch-overhead-correctie zegt.** Mamba klopt netjes
(0,494 gemeten tegen 0,357 voorspeld). MoE realiseert maar de **helft** van de
voorspelde 3,209 ms — bevestiging dat die 7,75 µs *CPU-uitgiftetijd* is die in
eager mode deels overlapt met GPU-werk zolang er genoeg GPU-werk is. De
correctie van vanmiddag was dus terecht maar de naïeve vermenigvuldiging
overschatte hem ~2×. Attention wordt in de graph juist **duurder**; dat is nog
onverklaard en verdient een eigen meting vóór er iets op gebouwd wordt.

**De eerlijke hoofdruimtetabel (graph, vloer bij 249 GB/s VRAM / 25,9 GB/s PCIe).**

| component | gemeten | vloer | **hoofdruimte** | efficiëntie |
|---|---:|---:|---:|---:|
| **MoE** | 9,408 | 5,19 (2,72 VRAM + 2,47 PCIe) | **4,22** | — |
| rest (lm_head, norms, embed, argmax) | ~3,72 | ~0,9 | ~2,8 | — |
| **Mamba** | 5,168 | 3,582 | **1,586** | 69,3% |
| **attention** | 2,479 | 1,128 | **1,352** | **45,5%** |

**Twee dingen die dit omgooit.**
1. **Attention is het minst efficiënte pad (45,5%), niet down_masked.** Mamba
   zit op 69,3%, shared_expert (eerder gemeten) op 90%. Attention is bovendien
   de kleinste van de drie in absolute zin, dus het is een goede, kleine
   testcase voor een kernelvraag — en PV2-11 (Q/K/V one-launch) is precies een
   exacte kandidaat op dat pad die alleen op de driftpoort sneuvelde. Die is nu
   dubbel gemotiveerd.
2. **De totale hoofdruimte is ~10 ms van de 20,77 ms**, dus een vloer rond
   **10,7 ms ≈ 93 tok/s** bij seriële PCIe. Dat is consistent met de eerdere
   plafondrekening en bevestigt hem langs een tweede, onafhankelijke weg.

**Artefacten.** `pro_research/diag_component_marginals_graph.py` + `.json`.

---

## 2026-08-16 — ⚠️ **ZELFCORRECTIE: de eager sub-kernelmarginalen bevatten ~7,75 µs launch-overhead per launch. `down_masked` doet in werkelijkheid 0,431 ms GPU-werk, geen 1,655 — de "1,40 ms hoofdruimte" bestond niet**

**Hoe dit boven kwam.** Na vijf weerlegde hypotheses voor `down_masked` heb ik
de binnenkant van de kernel geableerd (`diag_down_masked_ablate`, timing-only,
elke arm consumeert nog steeds wat hij laadt zodat geen load als dode code
verdwijnt):

| arm | ms/token | verwijderd |
|---|---:|---:|
| full | 1,530 | — |
| no_code_load | 1,578 | −0,048 (trager!) |
| no_scale_load | 1,551 | −0,021 |
| no_luts | 1,524 | 0,005 |
| no_act | 1,507 | 0,023 |
| **loop_only** (álle dataloads en rekenwerk weg) | **1,316** | **0,213 = 13,9%** |

Met álle geheugentoegang en alle rekenwerk eruit blijft **86%** van de kernel
over. Eerste verdenking was de pointer-chase `p = panel_list[pi];
m = panel_masks[p]` — de tweede load kan pas starten als de eerste terug is,
~11,5 keer per thread, en elke thread doet dezelfde chase opnieuw. Metadata in
shared memory stagen (`diag_down_masked_smem_meta`, 928 B extra SMEM) gaf
**0,98-1,01×**: ook weerlegd.

**De echte oorzaak, gemeten met een lege controle-kernel.** 138 launches van een
**no-op** kernel in dezelfde harness:

| launches | grid | ms | µs/launch |
|---:|---|---:|---:|
| 138 | (21, 8) | 1,0698 | **7,75** |
| 138 | (1, 1) | 1,0805 | 7,83 |
| 414 | (1, 1) | 3,0737 | 7,42 |

**~7,75 µs per eager kernel-launch, onafhankelijk van de gridgrootte.** De
geïsoleerde `down_masked`-harness doet 138 launches van een kernel die ~3,1 µs
GPU-werk doet — hij is dus **CPU-uitgifte-gebonden**, niet kernel-gebonden:

    1,501 ms gemeten − 1,070 ms pure uitgifte = **0,431 ms echt GPU-werk**

Tegen de bandbreedtevloer van 0,257 ms is dat **60% efficiëntie** — gewoon
netjes. **De 15% en de 1,40 ms hoofdruimte waren een meetartefact.**

**Wat hierdoor allemaal klopt dat eerst raadselachtig was.**
- Waarom vijf structurele varianten niets deden: ik mat uitgiftetijd, niet de
  kernel.
- Waarom V15 (batching) **in de graph** neutraal was: daar was de
  launch-overhead al weg, dus er viel niets te winnen.
- Waarom `diag_event_op_cost`'s baseline 414 launches op 3,2775 ms zette: MoE
  doet per token ~414 kernel-launches (138 gather + 138 down_masked + 23 up +
  46 shared + 23 scan + 23 reduce + 23 accumulate) — **exact hetzelfde getal**.

**Reikwijdte van de correctie.** `diag_component_marginals_v6` én
`diag_moe_subkernel_marginals` zijn **eager** gemeten (`rt.step()`), dus elke
marginaal daar bevat ~7,75 µs × zijn eigen launch-aantal. Dat raakt de zware
posten het hardst: gather en down_masked (138 launches elk, ~1,07 ms elk),
shared_expert (46, ~0,36), en up_proj/panel_scan/reduce/accumulate (23 elk,
~0,18). **De productiestack draait in een gevangen graph en betaalt dit niet.**
De hoofdruimte-tabellen in `STATE_OF_THE_WORK.md` en `TODO.md` overschatten
daarmee de winst die er in de kernels zelf te halen valt; ze zijn hierbij
gemarkeerd als eager-getallen en moeten in-graph opnieuw gemeten worden vóór ze
nog richting geven.

**Bijvangst — een latente kwetsbaarheid in productie.**
`gemv_down_masked_partial_ind` doet `if (threadIdx.x < 256) s_e4m3[threadIdx.x]
= e4m3_lut[threadIdx.x];` maar wordt met **128 threads** gelanceerd, dus
`s_e4m3[128..255]` blijft **ongeïnitialiseerd**. Dat is nu correct omdat
down_proj-blokschalen in de praktijk positieve E4M3-waarden zijn (byte < 128),
maar dat is een **ongedocumenteerde en ongecontroleerde aanname**. Mijn
benchmark vulde de mirror met willekeurige bytes 0-255 en liep er meteen tegenaan
(de bitexactheidsvergelijking faalde op ongeïnitialiseerde SMEM, niet op de
kandidaat). Een `for (i = threadIdx.x; i < 256; i += blockDim.x)` kost niets en
haalt de aanname weg — apart genoteerd in TODO.

**Wat dit opent.** De echte resterende inefficiëntie is **kleiner** dan de eager
cijfers suggereerden, en zit dus niet in `down_masked`. Volgende stap: de
componentattributie **in de graph** herhalen, zodat de hoofdruimte-tabel klopt
vóór er nog een kernel voor herschreven wordt.

**Artefacten.** `pro_research/diag_down_masked_ablate.py` + `.json`,
`pro_research/diag_down_masked_smem_meta.py` + `.json`.

---

## 2026-08-16 — `down_masked` ingesloten: **vijf hypotheses getest, alle vijf weerlegd**, harde vloer op ~1,5 ms die geen enkele gebruikelijke as verklaart — verdere voortgang vraagt een profiler

**Vraag.** `down_masked` kost 1,655 ms/token tegen een bandbreedtevloer van
0,257 ms (15% efficiëntie, 1,40 ms hoofdruimte) — de slechtste kernel in het
model. Waardoor?

**Meting 1 — `nchunks`-sweep (kettinglengte).** `nchunks` is `gridDim.y` en de
paneellus stride't ermee, dus het verandert de afhankelijke-load-keten per
thread zónder het totale werk te veranderen. Geïsoleerd, één tokenlading
(138 launches):

| nchunks | blokken | afh. loads/thread | ms/token |
|---:|---:|---:|---:|
| 2 | 42 | 124,5 | 4,004 |
| 4 | 84 | 62,2 | 2,535 |
| **8** | **168** | **31,1** | **1,549** ← productie |
| 16 | 336 | 15,6 | 1,772 |
| 32 | 672 | 7,8 | 1,603 |
| 64 | 1344 | 3,9 | 1,613 |

Onder 8 is hij ketting-gebonden (elke halvering van de keten halveert de tijd);
**vanaf 8 is hij vlak**. De productiewaarde zit precies op de knik. Meer
parallellisme — van 168 naar 1344 blokken — levert **niets**.

**Meting 2 — één thread per BYTE in plaats van per rij.** In de referentie
berekenen rij 2k en 2k+1 allebei `hb = k`, dus **twee threads laden dezelfde
byte** en houden elk één nibble; 32 threads van een warp raken maar 16 unieke
adressen. De kandidaat geeft één thread de byte plus beide rijen (twee
accumulatoren), wat het aantal global loads halveert en de bruikbare bytes per
request verdubbelt. **Bitexact op elke `nchunks`** — de herindeling is
aantoonbaar correct, alleen de thread→rij-afbeelding verandert.

Resultaat: **geen winst.** Beste kandidaat 1,589 ms tegen productie 1,492 ms =
**0,939×**. Het halveren van de load-instructies levert niets op.

**Wat daarmee is uitgesloten voor `down_masked` — vijf assen, alle vijf
gemeten en weerlegd:**
1. bandbreedte — 63,2 MB nuttig / 1,492 ms = 42,4 GB/s; zelfs mét
   sectorverlies meegerekend (92,4 MB opgehaald) is het 61,9 GB/s tegen een
   kernelplafond van 249;
2. instructiedoorvoer — ~60M FMA's/token × ~8 ondersteunende instructies ≈
   0,08 ms, een orde eronder;
3. launch-overhead / gridgrootte / occupancy — V15 gaf 6× de blokken in één
   launch, neutraal;
4. afhankelijke-ketenlengte — vlak vanaf `nchunks = 8` tot 64;
5. overbodige load-instructies en sectorverlies — byte-per-thread halveert
   beide, bitexact, en levert niets.

**Eerlijke stand.** Er is een harde vloer rond **~1,5 ms** die geen van de
gebruikelijke assen verklaart. Ik kan dit niet verder dichtnagelen met
end-to-end timing alleen; wat nu nodig is, is een **profiler**. De
capability-census van PV2-21 stelde vast dat `nsys`, `ncu` en
`compute-sanitizer` **niet op PATH staan** op deze machine. **Het beschikbaar
maken van `ncu` is daarmee de hoogst renderende deblokkerende actie die er nu
is** — niet nóg een blinde kernelvariant. Dat is een eerlijke grens, geen
conclusie dat er niets te halen valt: de 1,40 ms hoofdruimte staat gewoon nog
open.

**Artefacten.** `pro_research/diag_down_masked_chain.py` + `.json`,
`pro_research/diag_down_masked_byterow.py` + `.json`.

---

## 2026-08-16 — PRO V15: gebatchte gather + down_masked (VRAM-blokkade weggenomen) is bitexact maar **neutraal (+0,035 ms)** — en dat sluit occupancy/launch-overhead uit als verklaring voor down_masked's 15%

**Vraag.** `down_masked` draait op 1,655 ms tegen een vloer van 0,257 ms en de
gather op 3,479 ms. Beide worden **zes keer per laag** gelanceerd met kleine
grids — `down_masked` draait `(hidden/128, nchunks) = (21, 8) = 168` blokken van
128 threads, dus ~6,5 blokken per SM gedurende ~12 µs, 138 keer per token. Is
dat het probleem?

**De blokkade die eerst weg moest.** De gebatchte kernels bestáán al
(`down_gather_batch_kernels.py`) en zijn destijds bitexact geverifieerd op echte
gevangen activaties. Ze zijn nooit geadopteerd omdat ze `top_k` onafhankelijke
mirrors nodig hebben en de eerste implementatie die **per laag** alloceerde:
23 × 6 × 2,806 MB = **387 MB**, terecht afgewezen door de VRAM-poort. Maar die
mirror is transiënte kladruimte die binnen dezelfde laag geconsumeerd wordt,
precies zoals de runtime's eigen `mstate["mirror"]` — één globale kopie volstaat:
**16,8 MB**. Gefixt in `moe_dev_batched.py`. Dit is dezelfde bugklasse die de
VRAM-poort tijdens de V6-bouw al één keer ving (een overbodige per-laag mirror
van 61,6 MB). Er was dus geen nieuwe rekenkunde nodig — alleen een
sizing-fout die een al geverifieerd kernelpad buiten de stack hield.

**Uitkomst (graph, SYNC-semantiek).** BASE_A 23,4471 · CAND 23,5807 ·
BASE_B 23,6434 ms; midden 23,5453, drift 0,1963. **CAND − midden = +0,0355 ms.**
Alle correctheidspoorten groen (bitexact, BASE_A == BASE_B), P1 FAIL.
**Neutraal — geen winst, geen verlies.**

**Wat dit uitsluit, en dat is de waarde.** Eén launch van
`(21, 8, 6) = 1008` blokken in plaats van zes launches van 168 verandert
**niets**. Dus down_masked's 15%-efficiëntie is **niet** te verklaren door
launch-overhead, gridgrootte of occupancy — die zijn met een factor 6 verbeterd
zonder effect. Hetzelfde geldt voor de gather (consistent met de eerdere
bevinding dat gather-batching "maar bescheiden" hielp omdat hij
PCIe-gebonden is).

**Wat er dan wél overblijft voor down_masked (nog niet gemeten).** De rekening
sluit langs geen van de gebruikelijke assen: bandbreedte zou 64 MB / 249 GB/s =
0,26 ms zijn, instructies ~60M FMA's/token met ~8 ondersteunende instructies =
~0,08 ms. Beide een orde onder de gemeten 1,655 ms. Wat overblijft is
**geheugenlatentie in een afhankelijke keten** (elke thread doorloopt ~11
panelen × ~1,8 kolommen = ~20 afhankelijke byte-loads, elk 1344 B verderop) en
**shared-memory bankconflicten** op `s_e4m3[pbase[row]]`, waar de index een
gewichtsbyte is en dus willekeurig over 256 entries loopt. Dat laatste is
verdacht: de LUT-vrije test van vanochtend liet zien dat de LUT in de *dense*
GEMV goedkoop is, maar die kernel was bandbreedtegebonden met ALU over — hier is
er geen bandbreedtedruk, dus de SMEM-doorvoer kan wél de rem zijn. Dat is de
eerstvolgende meting, en die is goedkoop.

**Artefacten.** `pro_research/batched_gather_v15.py`,
`pro_research/results/v15_batched_gather/PRO_V15_BATCHED_GATHER_GRAPH.json`,
mirror-fix in `pro_research/moe_dev_batched.py`.

---

## 2026-08-16 — MoE's 11 ms ontleed per sub-kernel (alle poorten groen): **gather 3,48 · up_proj 2,16 · down_masked 1,66 · shared 1,30** — en `down_masked` draait op 15% van zijn vloer

**Vraag.** MoE is 11,004 ms van een 23,141 ms-token tegen een vloer van 5,19 ms:
5,81 ms hoofdruimte, het grootste blok dat er nog is. Enkelvoudige
kernelbandbreedte verklaart het níet (de dense GEMV haalt koud 230-261 GB/s).
Welke van de zes stappen soupeert het op?

**Opzet.** Dezelfde marginale methode, één niveau dieper: de échte lus met
exact één sub-kernel één keer extra aangeroepen. `accumulate` schrijft naar een
kladbuffer omdat hij **accumuleert** (`dst[i] = fmaf(src[i], w, dst[i])`) en dus
niet idempotent is; `cache_assign` is uitgesloten omdat hij de LRU-tick
opschuift — dezelfde soort val die de naïeve `_mamba`-probe liet divergeren.
Elke arm is alsnog op bitexacte token-ids gepoort in plaats van op die
redenering vertrouwd.

**Uitkomst (full, alle poorten groen: G1 alle armen bitexact, G2 drift
0,1792 ms; basis-midden 23,1157 ms).**

| sub-kernel | ms/token | % van token | behaald | vloer | **hoofdruimte** |
|---|---:|---:|---:|---:|---:|
| **gather** | **3,479** | 15,1% | 18,4 GB/s (PCIe) | 2,47 | **1,01** |
| **up_proj** | **2,162** | 9,4% | 179,1 GB/s | 1,56 | **0,61** |
| **down_masked** | **1,655** | 7,2% | 38,7 GB/s | 0,26 | **1,40** |
| shared_expert | 1,300 | 5,6% | 223,0 GB/s | 1,17 | 0,14 |
| accumulate | 0,315 | 1,4% | — | — | ~0,3 |
| panel_scan | 0,264 | 1,1% | — | — | ~0,26 |
| reduce | 0,218 | 0,9% | — | — | ~0,22 |
| **som** | **9,393** | 40,6% | | | |

De 11,004 ms MoE-totaal min 9,393 ms laat **~1,6 ms** over voor
router-GEMV + `route_topk` + `cache_assign` + `cache_fetch`, die bewust niet
geprobed zijn.

**Drie dingen die dit vaststelt.**
1. **`down_masked` is de slechtste van allemaal: 1,655 ms tegen een vloer van
   0,257 ms — 15% efficiëntie, 1,40 ms hoofdruimte.** Dat is de grootste
   enkelvoudige inefficiëntie in het hele model. Oorzaak zit in het
   toegangspatroon: elke thread doet één uitvoerrij en leest
   `pcodes[c*rowhalf + hb]`, dus opeenvolgende kolommen liggen **1344 bytes uit
   elkaar** terwijl 128 threads samen maar 64 aaneengesloten bytes per kolom
   ophalen. Halve-sector-reads op een gescatterde stride.
2. **`shared_expert` draait op 223 GB/s = 90% van het eerlijke kernelplafond
   (249).** Daar is niets meer te halen — en dat is meteen het bewijs dat de
   GEMV-kernel op zich prima is; het probleem zit in de MoE-specifieke paden.
3. **De gather kost in de lus 3,479 ms**, niet de 2,47 ms die de pure
   byte/bandbreedte-rekening voorspelde: 18,4 GB/s tegen het gemeten
   PCIe-plafond van 25,9. Dat sluit netjes aan op de geïsoleerde meting
   (3,855 ms voor 63,96 MB) — de twee metingen bevestigen elkaar nu, in
   tegenstelling tot eerder vandaag.

**Wat dit opent.** De prioriteitsvolgorde binnen MoE ligt nu vast op gemeten
getallen in plaats van op vermoedens: **down_masked (1,40) > gather-boven-de-
PCIe-vloer (1,01) > router/assign/fetch (~1,6, nog niet uitgesplitst) >
up_proj (0,61)**. `down_masked` is bovendien puur een VRAM-kernel — geen PCIe,
geen sparsity-afhankelijke grid — dus daar is een herschreven toegangspatroon
een gewone, afgebakende kerneloefening met een bitexacte poort.

**Artefacten.** `pro_research/diag_moe_subkernel_marginals.py` + `.json`.

---

## 2026-08-16 — PRO V14 (B3, PCIe-gather overlappen): eager **+3,65 ms** (weerlegd door de scheduler, niet door het idee), in de graph **−0,416 ms** en bitexact — maar dat is pas **16,8%** van de 2,47 ms, en de reden daarvoor is structureel

**Vraag.** De plafondrekening zegt: serieel 10,69 ms = 93,6 tok/s → 100
onbereikbaar; overlappend 8,22 ms = 122 tok/s → 100 haalbaar. Kan de
down_proj-PCIe-gather onder het VRAM-werk verstopt worden?

**Opzet.** `moe_dev_overlap.py`: twee mirrors in ping-pong plus een eigen
gather-stream, zodat de gather van slot s+1 loopt terwijl slot s rekent. De
edge die makkelijk vergeten wordt en stil zou corrumperen — `gather_stream`
moet op `m_done[s-1]` wachten voordat hij `mirror[(s+1)&1]` overschrijft, want
slot s-1 leest daar nog uit — zit erin. Kost: één extra mirror van 2,81 MB
(`mstate["mirror"]` is globale scratch, niet per laag, dus 5,6 MB totaal).
Niets aan de rekenkunde verandert; alleen wannéér de gather draait.

**Ronde 1 — eager: +3,6518 ms/token. Bitexact, maar veel trager.**
Alle correctheidspoorten groen (inclusief de ping-pong-hazard), drift
0,0525 ms. Maar de kandidaat is 3,65 ms **langzamer**.

**De diagnose, gemeten in plaats van geraden** (`diag_event_op_cost.py`): een
kale `Event.record` kost **0,285 µs** — gratis. Maar het volledige
cross-stream `wait_event`/`record`-heen-en-weer-patroon kost **~183 µs per
iteratie** in eager mode. V14 doet 23 lagen × 6 slots = **138** van die
fork/join-hops per token. Dat is de hele regressie. **De eager-uitslag prijst
de scheduler, niet het idee.**

**Ronde 2 — in de graph: −0,4158 ms/token, bitexact.**
Binnen een gevangen CUDA-graph zijn fork/join statische graafranden, één keer
opgelost bij capture; tijdens replay wordt er geen dependency-API aangeroepen.
Capture accepteerde de multi-stream-topologie zonder klacht (`capture_error:
null`) — de fork/join-vorm was al capture-legaal.

| arm (graph, SYNC-semantiek) | p50 ms | tok/s |
|---|---:|---:|
| BASE_A (V6) | 22,3962 | 44,650 |
| **CAND (B3 overlap)** | **22,1405** | **45,166** |
| BASE_B (V6) | 22,7164 | 44,021 |

midden 22,5563 · drift 0,3202 · **CAND − midden = −0,4158 ms**. Poorten:
C1 bitexact **PASS** · C2 **PASS** · D1 **PASS** · **P1 (≥0,8 ms) FAIL**.
Poort niet verruimd. **Het teken is wel omgeklapt: van +3,65 naar −0,42.**

**Waarom maar 16,8% van de 2,47 ms?** Structureel, en het is een echte les:
`gather_down_sparse_ind` is een **SM-side zero-copy kernel**, geen DMA-kopie.
Hij moet dat zijn, want de selectie is data-afhankelijk (alleen de nonzero
kolommen) en dat kan een copy-engine niet. Gevolg: gather en `down_masked`
overlappen wel in de tijd, maar **vechten om dezelfde SM's**. Een warp die op
PCIe staat te wachten geeft zijn issue-slots vrij maar houdt zijn warp-slot en
registers bezet, dus hij verdringt reken-warps. **PCIe-tijd verstoppen achter
compute lukt alleen echt als de transfer op de copy-engine ligt.**

**Ronde 3 — de voor de hand liggende vervolgstap, en die is WEERLEGD.**
Hypothese: de gather-launch is op de worst case gesized (`blocks =
(inter + npanel)·32/256 = 247`, dus 1976 warps) terwijl er bij de gemeten
sparsity maar ~255 warps werk hebben; een kleine statische grid met een
grid-stride-lus zou veel SM vrijlaten voor `down_masked`. Gebouwd
(`gather_small_grid.py`, identieke per-warp-body, alleen andere
werkverdeling), bitexact, en gesweept in dezelfde graph-harness:

| gather-grid | delta ms | PCIe verstopt |
|---|---:|---:|
| 32 blokken | −0,0512 | 2,1% |
| 64 blokken | −0,1878 | 7,6% |
| **247 (productie)** | **−0,4158** | **16,8%** |

**Precies andersom dan de hypothese.** Minder blokken verstopt mínder, niet
meer. Verklaring die daaruit volgt: het aantal *werkende* warps ligt met ~255
vast door de data, maar bij 247 blokken liggen die verspreid over alle 26 SM's
en bij 32 blokken opeengepakt op ~32. De gather is PCIe-**latency**-gebonden, en
zijn doorvoer schaalt met het aantal SM's waarover hij uitstaande requests kan
verdelen. Verspreiden levert meer op dan het aan contentie kost. De
productie-geometrie zat dus al aan de goede kant van die ruil.

**Wat dit sluit of opent.** Sluit: gather-grid-krimp als hefboom. Wat overblijft
voor de resterende 83% PCIe-tijd is de transfer naar de **copy-engine** krijgen,
en dat vereist een niet-data-afhankelijk kopieerpatroon — hele panelen DMA'en is
4,4× zoveel bytes (1,94 MB/expert tegen 440 KB) = 10,4 ms/token bij 25,9 GB/s en
dus slechter, en `full`-cachemode is door S11 al weerlegd bij gelijke bytes.
**Belangrijker inzicht uit de rekening:** met 16,8% overlap zakt de seriële
vloer van 10,69 naar ~10,27 ms = 97,4 tok/s — nog steeds net onder 100, maar
de échte afstand zit niet daar. We staan op 22,14 ms tegen een vloer van
~10,3 ms: **~12 ms is pure implementatie-inefficiëntie**, verdeeld als MoE
11,00 tegen 5,19 vloer, Mamba 5,66 tegen 3,58, attention 1,92 tegen 1,13. Dát
is waar de volgende 12 ms ligt, niet in nog meer overlap.

**Artefacten.** `pro_research/moe_dev_overlap.py`, `pro_research/overlap_v14.py`,
`pro_research/overlap_v14_graph.py`, `pro_research/diag_event_op_cost.py` +
`.json`, `pro_research/results/v14_overlap/PRO_V14_OVERLAP.json` en
`PRO_V14G_OVERLAP_GRAPH.json`.

---

## 2026-08-16 — Eerste **onbevangen** componentattributie van het hele project (alle poorten groen): MoE 47,6%, Mamba 24,5%, attention 8,3% van een token

**Vraag.** Waar gaat een token echt heen? De enige bestaande attributie
(`diag_v6_component_breakdown`) stubt componenten naar nul en is daarmee
onbruikbaar: dat verandert de residual stream → de MoE-routing → welke experts
de LRU missen → het PCIe-verkeer. Andere werklast, geen attributie. (Die arm gaf
Mamba een NEGATIEVE bovengrens van −0,429 ms, "gratis".)

**Opzet.** S12's marginale methode: de échte lus, met exact één extra aanroep
van één component naar een weggegooide kladbuffer. Residual stream, routing,
cache en geproduceerde tokens blijven bit-identiek — en dát is de poort.
Armen BASE_A → MARGINAL_MAMBA → MARGINAL_ATTN → MARGINAL_MOE → BASE_B, eager,
3 prompts × 192 tokens per arm, preheat 128.

**Een echte bug die de eerste run ving.** MARGINAL_MAMBA divergeerde (bij
gegenereerd token 0 / 3 / 7), ATTN en MOE niet. Oorzaak: `_mamba` is
**stateful** — `conv_step` schrijft `self.conv[i]`, `ssm_step` schrijft
`self.ssm[i]` — dus een tweede aanroep laat de recurrentie twee keer
voortlopen. De probe krijgt nu zijn eigen kladrecurrentie-state (`conv`/`ssm`
zijn per-laag dicts, geen gestapelde array — ook dat kostte een correctie).
`_attention` en `_moe_dev` bleken wél idempotent binnen een token (de KV-append
is positie-geadresseerd; een herhaalde `cache_assign` her-hit de ids die hij net
installeerde) — **geverifieerd, niet aangenomen**: hun ids matchten al in de
eerste run.

**Uitkomst (alle poorten groen: G1 alle armen bitexact, G2 drift 0,0397 ms).**

| component | marginaal ms/token | % van token | bytes/token | GB/s behaald |
|---|---:|---:|---:|---:|
| **MoE** | **11,004** | **47,6%** | 741 MB (677 VRAM + 64 PCIe) | 67,4 |
| **Mamba** | **5,662** | **24,5%** | 892 MB | 157,5 |
| **attention** | **1,917** | **8,3%** | 281 MB | 146,5 |
| som | 18,58 | 80,3% | | |
| rest (lm_head, norms, embed, argmax, lijm) | ~4,56 | 19,7% | | |

basis-midden 23,141 ms. **Alle drie zijn ondergrenzen**: de probe draait direct
na de echte aanroep, dus tot 32 MB (L2) van de zojuist gelezen gewichten is nog
warm. De echte kosten liggen hoger.

**Hoofdruimte, met het eerlijke kernelplafond van 249 GB/s en 25,9 GB/s PCIe:**

| component | vloer | gemeten | **hoofdruimte** |
|---|---:|---:|---:|
| MoE | 2,72 VRAM + 2,47 PCIe = 5,19 | 11,00 | **5,81 ms** |
| Mamba | 3,58 | 5,66 | **2,08 ms** |
| attention | 1,13 | 1,92 | **0,79 ms** |

**Wat dit sluit of opent.** Sluit: "Mamba is gratis" (het is 24,5%) en alle vier
de stub-bovengrenzen uit `diag_v6_component_breakdown`. Opent: **MoE blijft met
afstand het grootste doel (5,8 ms hoofdruimte)**, en van die 5,8 is 2,47 ms
zuivere PCIe-tijd die alleen door **overlap** verdwijnt, niet door een snellere
kernel. Dat sluit precies aan op de plafondrekening hieronder.

**Artefacten.** `pro_research/diag_component_marginals_v6.py` + `.json`.

---

## 2026-08-16 — De dense-GEMV-bandbreedte eindelijk eerlijk vastgepind: **209-229 GB/s koud**, niet 336 (L2-artefact) en niet 128 (E5). Het echte plafond van deze machine is daarmee ~88-117 tok/s, niet 165 — en de LUT-hypothese is weerlegd

**Vraag.** De byte-boekhouding wees 1661 van de 2048 MB/token toe aan dense
GEMV's, en E5 mat die suite op 127,9 GB/s tegen een apparaat dat 345,9 haalt.
Als dat klopt is dát de grootste hefboom in het systeem. Klopt het?

### Deel 1 — de LUT-hypothese, weerlegd

**Hypothese.** `pro_gemv_fp8_tensor_ervf16` doet per gewichtsbyte één
shared-memory-lookup `lut[q.x]` met een **data-afhankelijke** index. Zestien
lanes × vier willekeurige lookups in een 256-entry SMEM-tabel is een
bank-conflict-generator op de binnenste regel van de heetste lus van het model
(~892M lookups/token alleen al voor Mamba). E4M3 heeft die tabel niet nodig:
voor E≠0 valt de IEEE-754-layout er direct uit (exponentveld E+120, mantisse
m<<20), en de subnormale tak is één int-naar-float plus een vermenigvuldiging.

**Uitkomst.** P0: alle 256 bytewaarden **bitidentiek** tussen LUT en
rekenkundige decode. P1 op de echte shapes: de LUT-vrije kernel is bitexact
**maar 25-27% LANGZAMER** (speedup 0,72-0,75 op alle vier de shapes).
**Hypothese weerlegd.** De kernel is bandbreedtegebonden met ALU over; een
SMEM-lookup is daar goedkoper dan zes extra ALU-ops per byte. Deur dicht.

### Deel 2 — en meteen een veel groter probleem in de meting zelf

Diezelfde run mat de referentiekernel op **295-357 GB/s** — 85-103% van de
345,9 GB/s die dit apparaat levert. Dat is onverenigbaar met de in-lus-marginaal
van 152-154 GB/s uit `diag_component_marginals_v6`. Twee metingen van dezelfde
kernel die 2,2× verschillen; die moeten verzoend worden, niet uitgekozen.

**Opzet.** Eén variabele: kan de werkset in L2 blijven? Zelfde kernel, zelfde
shape, zelfde launch-geometrie, zelfde rondes — alleen het aantal verschillende
matrices in de rotatie verandert. Apparaat: **L2 = 32,0 MiB, 26 SM's**.

| arm | werkset | ×L2 | µs/call | GB/s |
|---|---:|---:|---:|---:|
| mamba_in_proj, 1 matrix | 27,7 MB | 0,83 | 82,4 | **335,9** |
| mamba_in_proj, 4 matrices | 110,8 MB | 3,30 | 119,9 | **231,0** |
| mamba_in_proj, 12 matrices | 332,4 MB | 9,91 | 120,7 | **229,4** |
| mamba_out_proj, 1 matrix | 11,0 MB | 0,33 | 34,0 | **324,2** |
| mamba_out_proj, 4 matrices | 44,0 MB | 1,31 | 52,7 | **208,8** |
| mamba_out_proj, 12 matrices | 132,1 MB | 3,94 | 52,7 | **208,9** |

**Verdict: `isolated_number_was_an_L2_artifact`.** Mamba's in_proj is 27,7 MB
en past dus in een L2 van 32 MB; 200 keer dezelfde matrix lezen meet deels
L2-bandbreedte. Bij 3,3× L2 zakt het naar 231 GB/s en tussen 3,3× en 9,9× is
het **vlak** — dat is de echte koude-DRAM-snelheid van deze kernel.

**Wat hiermee vastligt (drie getallen die niet door elkaar mogen):**
- **345,9 GB/s** — wat het geheugensysteem levert bij puur streamen
  (`diag_vram_bandwidth_check`, 512 MiB, byte-geverifieerd; copy 316,1, triad
  330,1 — het projectgetal 338,4 houdt stand);
- **~230-261 GB/s** — wat déze GEMV-kernel haalt op een koude werkset:
  **67-76% van het apparaat**. Dit is het eerlijke kernelplafond.
  ⚠️ **Zelfcorrectie:** dit blok schreef eerst 209-229 GB/s op basis van
  `diag_gemv_l2_vs_dram` alleen. Die run liep deels op een **zwaar
  gethrottelde SM-klok** — de eigen klokregistratie in dat bestand toont
  795 MHz bij twee armen tegen 1732-1777 MHz bij de andere. De herhaling in
  `diag_gemv_width32` (koud, 4-9,9× L2, koelere klok) geeft voor dezelfde
  w16-kernel 230,5 / 234,8 / 248,2 / 261,2 GB/s. De hogere reeks is de
  eerlijke; 209-229 is throttle-besmet en moet niet geciteerd worden.
- **146-158 GB/s** — wat er in de échte lus uitkomt, en dat is nog een
  **onderschatting van de kost**: de marginale probe roept de component direct
  na de echte aanroep aan, dus tot 32 MB van de zojuist gelezen gewichten zit
  nog in L2. De werkelijke in-lus-kost ligt dus **boven** de gemeten waarden.

**Waarom dit het 100-doel raakt.** Met het eerlijke kernelplafond van
249 GB/s wordt de VRAM-vloer 2048/249 = **8,22 ms**; plus 2,47 ms PCIe als die
serieel is → **10,69 ms = 93,6 tok/s**, bij volledige overlap **8,22 ms =
122 tok/s**. **Dat is de scherpste uitspraak die dit project tot nu toe heeft:
100 tok/s is onbereikbaar zolang de down_proj-PCIe-gather serieel achter het
VRAM-werk aan loopt, en bereikbaar zodra een substantieel deel ervan eronder
verstopt wordt.** Overlap is daarmee geen "nice to have" meer maar de
poortvoorwaarde — precies B3 (double-buffer expert fetch) uit
`POST_V6_100TPS_PLAN.md`.

**En de kernelgeometrie-hypothese is óók weerlegd.** Werkhypothese was dat
`PRO_WIDTH = 16` zorgt voor 16 lanes × 4 B = 64 B per instructie (een halve
cacheline) en 16 gelijktijdige rij-streams per blok. Een 32-lane-variant
(volle 128 B, 8 rijen per blok) is gebouwd en is **bitexact op alle vier de
echte shapes** — de ERVF-afbeelding is bij width 32 zelfs directer dan bij 16,
want virtuele tid `t = lane + 32·vj` valt exact in referentiewarp `vj` op
positie `lane`, dus `acc[vj]` ís die warp en een gewone 32-brede
shuffle-reductie reproduceert de 16/8/4/2/1-boom letterlijk. Maar de snelheid is
**neutraal**: 0,954 / 1,033 / 1,068 / 0,982× — ruis rond 1,0. De cacheline-breedte
was niet de rem. `diag_gemv_width32.json`.

**Artefacten.** `pro_research/diag_fp8_lutfree_gemv.py` + `.json`,
`pro_research/diag_gemv_l2_vs_dram.py` + `.json`,
`pro_research/diag_vram_bandwidth_check.py` + `.json`.

---

## 2026-08-16 — PRO V13 H-SCALE gebouwd en gedraaid: bitexact, past in VRAM, **−0,374 ms/token** — maar de eigen adoptiepoort (≥0,5 ms) is NIET gehaald, en de geïsoleerde 1,380 ms bleek de lus-winst 2× te overschatten

**Vraag.** `diag_gather_pcie_ceiling` mat dat 52,2% van al het
down_proj-PCIe-verkeer FP8-blokschalen zijn en prijsde het weghalen daarvan op
−1,380 ms/token. Levert dat in de échte lus ook op?

**Opzet.** Nieuwe kernels (`scale_resident_kernels.py`) en een H-SCALE-variant
van het V6-`_moe_dev` (`moe_dev_scale_resident.py`, zelfde niet-invasieve
`types.MethodType`-patroon; runtime.py en fused_nvfp4.py onaangeraakt).
Drie wijzigingen, alle drie **dataplaatsing, geen rekenkunde**: per laag een
`planes`-buffer van `cap × 311.808 B`; de copy-stream stageert bij een miss óók
het schaalvlak (zelfde `need[]/slots[]/ids[]`-contract, zelfde wide-uint4
patroon als `cache_fetch`, dus graph-capturebaar); de gather laat zijn
paneelschaal-tak vallen en de masked GEMV leest de schaalbyte uit het residente
vlak. Dezelfde expert, hetzelfde paneel, dezelfde rij, dezelfde byte, dezelfde
`e4m3_lut[byte] * global_scale`, dezelfde fmaf-volgorde. Armen
BASE_A → CAND → CAND_PROBE → BASE_B, eager, 3 prompts × 256 tokens per arm.

**Uitkomst (full, 765 tokens per arm; twee onafhankelijke full-runs).**

| arm | p50 ms | tok/s |
|---|---:|---:|
| BASE_A (V6) | 22,6799 | 44,092 |
| **CAND (H-SCALE)** | **22,4256** | **44,592** |
| CAND_PROBE (H-SCALE + 1× extra plane-fetch per laag) | 22,7530 | 43,954 |
| BASE_B (V6) | 22,9193 | 43,631 |

midden 22,7996 · drift **0,2394 ms** · **CAND − midden = −0,3740 ms/token**
(eerste full-run onafhankelijk: −0,3757 ms — repliceert).

**Poorten.** G-V13-C1 CAND bitexact vs BASE_A **PASS** · G-V13-C2
BASE_A == BASE_B **PASS** · G-V13-C3 probe bitexact **PASS** · G-V13-V1 VRAM
**PASS** (492,4 MiB gepland, 605 MiB vrij vóór, 207 MiB over na) · G-V13-D1
drift ≤1 ms **PASS** (0,239) · **G-V13-P1 winst ≥0,5 ms FAIL** (−0,374).
Status: `gate_failed`. De poort wordt **niet** verruimd.

**De marginale ontleding, en waarom die het waard was.** CAND_PROBE roept
`fetch_planes` één extra keer per MoE-laag aan. Die aanroep is idempotent
(dezelfde bytes van dezelfde experts naar dezelfde slots), dus de tokens blijven
bit-identiek — gecontroleerd als poort — en het verschil is de eigen kost van de
fetch:

- plane-fetch: **+0,3274 ms/token**
- bruto gather-besparing: **0,7014 ms/token**
- netto: 0,7014 − 0,3274 = **0,3740** ✓ (sluit exact)

**Twee lessen, allebei bruikbaar.**
1. **De geïsoleerde 1,380 ms overschat de lus-winst met een factor 2.** In het
   losse benchmark liepen 138 gathers rug-aan-rug, puur PCIe-gebonden; in de
   echte lus is de helft daarvan al verstopt achter ander werk. Dit is S8's les
   voor de derde keer: nooit een component apart timen en de winst overzetten.
   Het getal 1,380 blijft geldig als *byte*-uitspraak, niet als winstvoorspelling.
2. **De plane-fetch is duurder dan nodig** (0,327 ms tegen ~0,24 ms verwacht),
   omdat hij een *strided* gather doet: 116 blokjes van 2688 B met bronstride
   24192. Eén keer bij het laden een contiguë schaalvlak-bank op de host
   repacken maakt er één aaneengesloten kopie van.

**Wat dit sluit of opent.** Het mechanisme werkt en is exact — dat is nu bewezen
en herbruikbaar. Twee wegen om alsnog boven de poort te komen, beide gemeten in
plaats van geschat: (a) contiguë host-repack van de schaalvlakken (~−0,09 ms),
(b) schaalvlakken voor **alle 128** experts residentie maken zodat de fetch
verdwijnt (875 MiB nodig tegen 605 vrij → up-capaciteit 72→68, kost ~+0,16 ms
aan extra up-missers) → samen ongeveer −0,54 ms. Dat haalt de poort net, en is
het aanleggen waard, maar het is geen doorbraak. **De veel grotere hefboom die
de byte-boekhouding blootlegt (dense-GEMV op 37,8% van de bandbreedte, over
1661 van de 2048 MB/token) heeft voorrang.**

**Artefacten.** `pro_research/scale_resident_kernels.py`,
`pro_research/moe_dev_scale_resident.py`, `pro_research/scale_resident_v13.py`,
`pro_research/results/v13_scale_resident/PRO_V13_SCALE_RESIDENT.json`.

---

## 2026-08-16 — Exacte byte-boekhouding per token uit de safetensors-headers: het 165 tok/s-roofline gereproduceerd, en **Mamba blijkt 43,6% van alle bytes** — de enige component die nooit gemeten of geoptimaliseerd is

**Vraag.** Waar gaan de 6,05 ms van het roofline heen? Het project citeert
"165 tok/s (ctx0)" al sinds de Y-lijn, maar nergens staat de **verdeling** over
componenten. Zonder die verdeling is niet te zeggen welke component nog winst
kán opleveren.

**Opzet.** Geen meting, een telling: per blocktype één representatieve laag uit
`model.safetensors.index.json` + de safetensors-headers (`data_offsets`, dus
echte bytes op schijf inclusief scales, niet geschat), maal het aantal lagen uit
`config.layers_block_type` (23 mamba, 23 moe, 6 attention, 52 totaal). Voor MoE
alleen de **actieve** experts (top_k=6 van 128) plus shared+gate.

**Uitkomst.**

| component | MB/token | roofline-ms bij 338,4 GB/s | aandeel |
|---|---:|---:|---:|
| **Mamba** (23 × 38,78 MB) | **892,0** | **2,64** | **43,6%** |
| routed up (23 × 6 × 2,806 MB, uit VRAM-cache) | 387,3 | 1,14 | 18,9% |
| shared expert + gate (23 × 12,61 MB) | 290,0 | 0,86 | 14,2% |
| attention (6 × 46,80 MB) | 280,8 | 0,83 | 13,7% |
| lm_head (NVFP4) | 198,2 | 0,59 | 9,7% |
| **subtotaal VRAM** | **2048** | **6,05** | 100% |
| routed down, sparse over PCIe | ~64 | (2,47 ms op de PCIe-bus) | apart |

**Validatie.** 2048 MB / 338,4 GB/s = **6,05 ms = 165,3 tok/s** — dat
reproduceert het roofline-getal van de Y-lijn exact. De boekhouding klopt dus
met een onafhankelijk eerder resultaat, en mag als basis dienen.

**Wat dit voor het eerst zichtbaar maakt.** Er zijn **twee bussen**, niet één.
De VRAM-vloer is 6,05 ms; de sparse down_proj rijdt over PCIe (25,9 GB/s
gemeten) en kost daar ~2,47 ms. Als die volledig serieel is, is de echte vloer
**8,52 ms = 117 tok/s**; bij perfecte overlap tussen beide bussen **6,05 ms =
165 tok/s**. **100 tok/s = 10,0 ms/token ligt dus binnen de fysica** — het
vraagt 85% van de seriële vloer, of 60% van het VRAM-roofline mét overlap. Zwaar,
maar niet onmogelijk. Dat is een scherpere en optimistischere uitspraak dan
`PATH_TO_100_TOKS.md` nu doet, en die moet daar bijgewerkt worden.

**De vondst die eruit springt.** **Mamba is met 43,6% veruit de grootste
byteconsument van het model** — groter dan alle routed experts samen — en is de
enige component waar geen enkele optimalisatie van deze sessie naartoe ging
(ERVF/V4/V5/V6 zitten allemaal op MoE en de dense GEMV's; `_install_selective`
raakt Mamba's projecties zijdelings via `mv_bf16`/`mv_fp8_tensor`).

**Waarom niemand dat zag.** `diag_v6_component_breakdown.json` mat de
Mamba-arm op **21,288 ms tegen 20,859 ms real** — een NEGATIEVE bovengrens van
−0,429 ms, gelezen als "Mamba is gratis". Die arm is structureel onbruikbaar:
hij stubt `rt._mamba → out.fill(0)`, en dat verandert de residual stream →
verandert de MoE-routing → verandert welke experts de LRU missen → verandert het
PCIe-verkeer. Het is niet dezelfde werklast. Het bestand waarschuwt zelf dat de
STUB-armen verkeerde tokens geven "by design", maar rekent niet af met het feit
dat ze óók een andere *cache*-werklast geven. **Alle vier de stub-bovengrenzen in
dat bestand staan onder dezelfde twijfel** en moeten niet meer als attributie
geciteerd worden.

**Wat dit opent.** Een onbevangen meting van Mamba's echte in-lus-kost via
S12's marginale methode (de échte lus, één extra aanroep van één component naar
een weggegooide kladbuffer — residual stream, routing, cache en geproduceerde
tokens blijven bit-identiek, en dát wordt als poort gecontroleerd).
Gebouwd als `pro_research/diag_component_marginals_v6.py`.

**Artefacten.** Telling reproduceerbaar uit `models/.../model.safetensors.index.json`
+ headers; runner `pro_research/diag_component_marginals_v6.py`.

---

## 2026-08-16 — Gather-PCIe-plafond gemeten: de gather zit op 64% van het linkplafond (niet 17%), concurrency-varianten helpen NIET — maar **52% van de down-bytes is schaal-metadata**, en die weghalen is gemeten −1,380 ms/token

**Vraag.** N2 rapporteerde `gather_down_sparse` op **~4,3 GB/s effectief in de
lus** tegen **25,05 GB/s** die S5 geïsoleerd mat — "6× slechter in de lus". Het
down-pad is 6,5058 ms van een 22,53 ms-token (28,9%) en de STUB-arm zonder
down_proj draait 16,02 ms = 62,4 tok/s. Als dat 6×-gat echt is, is dit de
grootste onaangepakte inefficiëntie in het hele systeem. Klopt het nog?

**Waarom S8/S11 dit niet beantwoorden.** Beide concludeerden "de MoE-term is
niet transfergebonden", maar beide maten het **DMA**-pad (bulk
`cudaMemcpyAsync` van up_proj-missers). De gather is een ánder mechanisme:
SM-side zero-copy reads over PCIe. Die twee kunnen uiteenlopen en niets in het
logboek scheidde ze. Dit doet dat.

**Opzet.** Geïsoleerde bandbreedtemeting, géén modelload. Eén tokenlading
gather-werk (23 lagen × 6 experts = 138 calls) op een pinned panel-major bank
van echte afmeting (128 experts × 2.806.272 B = 359 MiB, zodat de
"verspreid-over-een-groot-gebied"-eigenschap behouden blijft). Sparsity
synthetisch maar gekalibreerd op de S2-census (9% nonzero, niet-geclusterd).
**De bytesverzameling ligt vast over alle armen** en elke niet-referentiearm
wordt byte-voor-byte vergeleken met de productie-body. Link gemeten:
`nvidia-smi` → PCIe **Gen5 ×8**.

**Eerst een eigen fout, gevonden vóór publicatie.** De eerste run rapporteerde
de referentiearm op **97,9 GB/s** — fysiek onmogelijk over een ~31,5 GB/s-link.
Guard-script `diag_gather_ceiling_check.py` toonde `sm_copy_bytes_correct:
false` én een device→device-arm op 1630 GB/s: **`uchar4` is VIER bytes, geen
zestien**, dus mijn referentiearm kopieerde een kwart van wat hij claimde
(97,9/4 = 24,5 — precies het echte plafond). De productiekernel had het altijd
goed (`rowhalf/4` = 336 uchar4 = 1344 B ✓). Beide referentiearmen worden nu
byte-geverifieerd vóór hun getal telt.

**Uitkomst (gecorrigeerd, alle armen byte-geverifieerd).**

| arm | ms/token | GB/s | bytes == v0? |
|---|---:|---:|---|
| v0 productie-body | 3,855 | **16,591** | ✓ |
| v1 unroll×4 (4 loads in flight) | 4,089 | 15,643 | ✓ |
| v2 split×2 / ×4 / ×8 / ×16 warps per kolom | 4,17 / 4,06 / 4,70 / 4,35 | 15,3 / 15,8 / 13,6 / 14,7 | ✓ |
| *ref: contigue SM-read, zelfde bytes-aantal* | *2,592* | *24,676* | *geverifieerd* |
| *ref: `cudaMemcpyAsync`, zelfde bytes-aantal* | *2,469* | ***25,908*** | *geverifieerd* |

**Bevinding 1 — N2's 6×-gat bestaat niet meer.** De gather draait op
**16,591 GB/s = 64,0% van het linkplafond** (25,908 GB/s), niet op 17%. De
V4-V6-lijn (gebatchte gather, graph-residentie) heeft dat gat grotendeels al
gedicht. De 4,3 GB/s uit N2 is daarmee **verouderd en moet niet meer geciteerd
worden**.

**Bevinding 2 — de resterende 36% is géén concurrency-probleem.** Vier
onafhankelijke manieren om meer PCIe-reads tegelijk in de lucht te hebben
(unroll naar 4 loads per warp; 2, 4, 8 en 16 warps per kolom) maken het
**allemaal langzamer**, niet sneller. De hypothese "de warps hebben te weinig
uitstaande requests" is dus weerlegd. Wat overblijft is het verspreide
toegangspatroon zelf, en dat is niet met een kernelvariant te repareren. Eerlijk
negatief resultaat: hier zit geen goedkope winst.

**Bevinding 3 — de echte hefboom is het bytesaantal, en die is groot.**
Byte-boekhouding per call: 164,7 nonzero kolommen × 1344 B = 221,3 KB aan
gewichten, plus 90,1 actieve panelen × 2688 B = 242,2 KB aan **FP8-blokschalen**.
Dus **52,2% van al het down-PCIe-verkeer is schaal-metadata, geen gewichten**
(63,96 MB/token totaal). Dat komt doordat een paneel zijn 2688 schaalbytes deelt
over 16 kolommen, terwijl er bij 9% sparsity gemiddeld maar **1,8 van die 16
kolommen** nonzero is — de schalen zijn per uitvoerrij nodig en dus onverkleinbaar
per actief paneel, maar ze hoeven niet elke keer opnieuw over PCIe.

**Hypothesearm v3 (geen kandidaat, prijskaartje).** Dezelfde kernel maar zonder
de schaalpanelen op te halen — een strikte deelverzameling van v0's bytes,
geverifieerd: **2,475 ms vs 3,855 ms = −1,380 ms/token**, 30,5 MB in plaats van
64,0 MB.

**Wat dit opent — H-SCALE.** Houd de down_proj-blokschaalvlakken van
cache-residente experts in VRAM en haal alleen de nonzero gewichtkolommen nog
over PCIe. Het is een **pure data-plaatsing**, geen rekenkundige wijziging: het
zijn dezelfde schaalbytes, alleen van elders gelezen, dus bit-identiek per
constructie (net als de bestaande `up_only`-cache en de panel-major repack).
- kosten: 116 × 2688 = 311.808 B per expert; 72 slots × 23 lagen = **492,4 MiB**;
  gemeten vrije VRAM tijdens de V12-full-run: 8151 − 7512 = **639 MiB** → past.
- baten: −1,380 ms/token gemeten, minus +0,244 ms extra missverkeer
  (20,24 missers/token × 311.808 B bij 25,9 GB/s) = **netto ≈ −1,14 ms/token**.
- tegen het V6-record van 21,0923 ms → **19,95 ms ≈ 50,1 tok/s**; tegen de
  V12-harness-baseline van 22,17 ms → 21,03 ms ≈ 47,6 tok/s.

Dat is geen 100, maar het is de eerste kandidaat sinds V6 met een **gemeten**
(niet geschatte) winst die het E50-gat alleen al overbrugt, en hij is exact.
Bouwstappen staan in `agents/TODO.md` onder "Open — eerstvolgend".

**Artefacten.** `pro_research/diag_gather_pcie_ceiling.py` + `.json`,
`pro_research/diag_gather_ceiling_check.py` + `.json` (de guard die de
uchar4-fout ving).

---

## 2026-08-16 — PRO V12 async-harvest (Kimi's prereg, door Claude gedraaid): hypothese WEERLEGD — er is geen queue-starvation; maar de harness haalt als eerste in dit project de driftpoort (0,108 ms)

**Vraag.** PV2-20's controle-arm mat losse gequeude child-replays op
18,758 ms (K=2) / 19,066 ms (K=4) — beide onder de 20 ms E50-drempel —
terwijl de V6-runner na élk token `ring_harvest()` roept, wat de graph-stream
synchroniseert. Is het resterende E50-gat dus vooral host-synchronisatie die
causaal niet nodig is (de graph schrijft argmax zelf naar `_tok_dev`)?

**Opzet.** Kimi's bevroren `V12_ASYNC_HARVEST_PREREGISTRATION.md`, ongewijzigd
gedraaid op de V6-stack (device-routing + graph-safe + selectieve ERVF +
gebatchte panel_scan/reduce_partials/accumulate/up-proj + per-laag capaciteit).
Geen enkele rekenkundige wijziging. Drie metrieken die nooit door elkaar mogen:
SYNC (blokkerende host-round-trip per token), QUEUED-K (K replays queuen, één
harvest), EVENT-STREAM-K (K replays queuen, per token een non-timing event
pollen zodat elk token individueel bij de host aankomt). Full: 3 prompts ×
256 tokens per arm, K ∈ {2,4,8,16,32} queued en {4,8,16} event-stream,
128 tokens preheat, SYNC_A vóór en SYNC_B ná alle kandidaten.

**Uitkomst (full, 765 tokens per arm).**

| arm | tok/s | opmerking |
|---|---:|---|
| SYNC_A | 45,110 | 22,1682 ms p50 |
| SYNC_B | 44,891 | 22,2764 ms p50 |
| QUEUED K=2 / 4 / 8 / 16 / 32 | 44,654 / 44,502 / 44,481 / 43,130 / 44,359 | allemaal bitexact |
| EVENT-STREAM K=4 / 8 / 16 | 44,459 / 44,567 / 44,570 | leveringsgat p50 22,6-22,8 ms |

**De beslissende meting is niet de tok/s maar de issue-kost.** Pure
Python-uitgiftetijd per token: **0,0537 ms (K=2) tot 0,0128 ms (K=32)** — dat
is **0,06-0,24% van een token**. De host geeft een replay uit in 13-54 µs
terwijl de GPU er 22.700 µs over doet. Er is niets te starven: de queue was
nooit leeg.

**Poorten.** `sync_a_b_token_parity` PASS · `all_queued_exact` PASS (alle K,
alle 3 prompts, geen enkele divergentie) · `full_tokens_ge_500` PASS ·
`baseline_drift_le_1ms` **PASS met 0,108 ms** · `queued_E50_any` FAIL ·
`event_stream_E50_any` FAIL. Status: `gate_failed`.

**Wat dit sluit.** Host-scheduling is *volledig* van de lijst E50-hefbomen af.
Niet "klein" — meetbaar nul. Elke variant hierop (V12B rolling credit-window,
V12C blocking-event credit) optimaliseert per constructie hetzelfde 0,05 ms en
is daarmee overbodig geworden vóórdat hij gedraaid is; dat is de goedkoopste
manier waarop dit resultaat waarde oplevert. Tegelijk **weerlegt dit PV2-20's
eigen controle-arm**: de 18,8-19,1 ms/token reproduceert niet binnen de
volledige V6-rekenkunde. Wat die arm ook mat, het was niet "dezelfde
tokengraph, bevrijd van host-sync". Dat blijft een open, eerlijk gemelde
discrepantie — géén nieuw record en geen bruikbaar getal.

**Wat dit opent — en dit is het echte resultaat.** De harness haalt
`drift = 0,108 ms`, waar PV2's hele campagne op 1,86-3,24 ms strandde en
daardoor géén enkele ±0,3 ms-kandidaat kon beslissen. Het recept dat het doet:
(1) 128 tokens preheat naar thermische steady-state, (2) één gevangen runtime
in plaats van hercompileren tussen armen, (3) `_reset_exact_state()` dat model-
en LRU-state wist zónder graph-gebonden pointers opnieuw te alloceren, (4) armen
kort na elkaar in één proces. **PV2-11 (Q/K/V one-launch) was exact op alle drie
prompts en 0,2387 ms sneller dan het baseline-midden, en sneuvelde ALLEEN op de
driftpoort.** Die kandidaat is nu meetbaar geworden en verdient een hermeting in
deze harness — dat is de eerstvolgende exacte kandidaat.

**Artefacten.** `pro_research/results/v12_async/PRO_V12_ASYNC_HARVEST.json`
(full, overschrijft de smoke), `pro_research/results/v12_async_full_console.log`,
`pro_research/results/v12_async_smoke_console.log`. Runner en preregistratie
komen van `origin/pro-v12-async` (Kimi), ongewijzigd overgenomen.

---

## 2026-08-16 — PRO-MAX V2 (branch pro-max-v2) full-campagne: geen E50; QKV exact maar drift-gedood; add+norm divergeert pas bij 256 tokens; thermische drift > het hele E50-gat

**Vraag.** Kan de exacte final-mile (add+next-RMSNorm, Q/K/V one-launch,
LM-head ERVF + hiërarchische argmax, daarna fysieke compositie V10) de
laatste 1,0923 ms/token naar 50 tok/s wegnemen? (ChatGPT's gepreregistreerde
pakket, bevroren poorten, vaste volgorde, eigen verifier.)

**Integriteit vóór uitvoering.** Payload gereconstrueerd uit 7 base64-delen;
zip-SHA256 matchte `PAYLOAD_INFO.json`, 19/19 bronbestanden matchten
`SOURCE_MANIFEST_SHA256.json`. `-Mode install` PASS, toen smoke, toen full.

**Uitkomst full (definitief; ≥500 samples/arm — samples_ge_500 overal true).**
- **PV2-10 add+norm**: micro bitexact + micro-speedup groen, maar
  **causal_parity FALSE bij 256 tokens** (smoke: true). Een echte, zeldzame
  divergentie in de fusiekandidaat — terecht niet geadopteerd. NB: dit is
  precies het patroon uit de D1-les: korte runs zien de fout niet.
- **PV2-11 Q/K/V**: alle exactheidspoorten groen, kandidaat 21,49 ms vs
  baseline-midden 21,73 ms (**+0,24 ms**, regressiegrens OK) — afgewezen
  ALLEEN op de drift-poort (base A/B-drift 1,86 ms > 1,0 ms).
- **PV2-12 LM-head+argmax**: exact maar echt trager (22,37 ms; micro-
  speedup-poort én regressiepoort falen) — terecht weg.
- **PV2-13 finale**: selectie leeg → V10 = kale V6: 21,52 ms = 46,48 tok/s;
  drift 3,24 ms → gate_failed. **E50/E75/E100: false.**
- **PV2-20 child-graph epochs** (K=2,4): parent-graph bit-identiek aan losse
  launches — het `cudaGraphAddChildGraphNode`-mechanisme WERKT op deze stack
  — maar speedup 0,96×/0,99×: geen winst boven al-gequeude launches. Wel
  gemeten: pure gequeue-de replay-rate ~19,2-19,5 ms/token (~51 tok/s
  queued) — de productie-21,09 ms bevat dus ~1,6 ms/token Python/launch-
  overhead die binnen een graph geamortiseerd kan worden.
- **PV2-21 capability census**: child-graph API + conditionele-graph-symbolen
  + TMA-architectuurvoorwaarde aanwezig; mapped-host→SMEM TMA NIET bewezen
  (eist eigen byte-exacte bandbreedte-microbenchmark).

**Verificatie.** De onafhankelijke verifier (importeert de runners niet)
herberekende elke poort en bevestigde elke status — inclusief de failures.
Methodologie van het pakket deugt; de uitslagen zijn het antwoord.

**De systematische vondst die alles overhangt.** base_a ligt bij
20,57-20,80 ms (~48,2-48,6 tok/s — sneller dan het 47,41-record!) en base_b
bij 22,65-23,98 ms: **de baseline zelf drift 1,9-3,2 ms binnen één arm**
(thermische throttling onder volhouden load; GPU zakt tussen runs naar
P2/low-clock). Drift > het E50-gat → geen enkele ±0,3 ms-kandidaat is
beslisbaar en de bevroren drift-poort ≤1,0 ms is onhaalbaar in de huidige
meetomgeving. De poort mag niet verruimd (werkregel); de OMGEVING mag wel
verbeterd: vaste clocks (admin), of meten in verzadigde thermische
steady-state, of A/B-arms korter interleaven zodat drift beide armen gelijk
raakt.

**Wat dit sluit of opent.** Sluit: LM-head+argmax-fusie (te traag) en
add+norm in zijn huidige vorm (divergentie bij 256 tokens — eerst debuggen
vóór ooit herindienen). Opent: Q/K/V one-launch is exact en ~+0,24 ms en
verdient een hermeting onder thermisch stabiele condities; child-graphs zijn
een werkend exact mechanisme voor Path B. Pad naar 100 blijft de gebatchte
graph (Path B uit POST_V6_100TPS_PLAN.md).

**Artefacten.** `pro_research/pro_max_v2/` (brongeverifieerd),
`pro_research/results/pro_max_v2/PV2_*.json`, `PV2_FINAL_REPORT.md`,
`PV2_VERIFICATION.json`.

---

## 2026-08-16 — Waarom shared < private bij N=2-graphs: thrash-hypothese WEERLEGD, delen werkt juist — de enkele stream is de boosdoener

**Vraag.** De gedeelde-cache graph-pair (cap 64, één stream, 33,52 tok/s) was
trager dan private caches (cap 24×2, 36,86 tok/s) ondanks de grotere cache.
Werkhypothese was LRU-thrash door afwisselende werkingssets. Klopt dat?

**Opzet.** `pro_research/diag_n2_graph_cache_hitrates.py`: de device-tellers
van `cache_assign` (`stats2[0]`=hits, `[1]`=misses, per laag, gesommeerd)
uitgelezen voor vijf configuraties, zelfde twee prompts, 20 decode-stappen.
Integriteitsassert: beide N=2-armen reproduceerden de gated-prototype-tokens
exact, en solo cap-72/64/24 reproduceerden de seq0-tokens (capaciteits-
invariantie E1F21-INV houdt stand, ook onder graphs).

**Uitkomst (hitrate).**
- solo cap 72: 63,4% · cap 64: 62,9% · cap 24: 55,1% (cap 64→72 is nagenoeg
  verzadigd; 24→64 levert +7,8pp)
- private N=2: seq0 55,1% (identiek aan solo cap24 — determinisme-check),
  seq1 49,5%
- **shared N=2: 71,1%** — HOGER dan solo cap 64. Delen heft de hitrate op:
  de twee sequenties treffen elkaars experts (cross-sequentie-lokaliteit is
  reëel, consistent met de eerder gemeten unie van 63,9/128 bij N=16).
  **Thrash is hiermee weerlegd.**

**Interpretatie.** Shared had 2873 missers tegen private's 4809 — ~1936
missers × ~2,8 MB / 26 GB/s ≈ ~209 ms bespaard over de run — maar was toch
108 ms LANGZAMER. De enige resterende verklaring is de enkele gedeelde
stream: twee graph-launches op één stream serialiseren volledig, terwijl de
private variant seq0's PCIe-miss-fetch (copy_stream in graph 0) laat
overlappen met seq1's compute (graph 1 op eigen stream). Gerekend: ~317 ms
overlap verloren tegen ~209 ms miss-winst → netto −2,7 ms/token. **De
winnaar zou "gedeelde cache + twee streams" zijn, maar dat racet op de
gedeelde LRU-tabellen** (slot_of/expert_of zijn read-modify-write; een race
kan een slot aan de verkeerde expert toewijzen → verkeerde data → geen
bitexactheid meer). Dat is structureel niet op te lossen met twee losse
per-sequentie-graphs — het is precies het argument voor Path B uit
`agents/POST_V6_100TPS_PLAN.md`: één gebatchte graph die N sequenties per
replay voortschrijdt, met de expert-unie ín de graph.

**Wat dit opent.** Voor de huidige twee-graphs-opbouw: private caches zijn
de betere arm (overlap > hitrate). De échte stap blijft de gebatchte graph
(B0-B3). Een tussenstap die wél kan: N=3/N=4 met private mini-caches meten
of aggregate meeschaalt met overlap (cap ~16-20 per sequentie, VRAM-kritisch).

**Artefacten.** `pro_research/diag_n2_graph_cache_hitrates.py`,
`pro_research/diag_n2_graph_cache_hitrates.json`.

---

## 2026-08-16 — Multi-seq graph N=2 HERMETEN na twee echte bugfixes: bitexact én boven naïef — private 36,86 / shared 33,52 tok/s — CORRIGEERT het blok hieronder

**Waarom dit blok boven het eerdere N=2-blok staat.** Het blok hieronder
("23,59 tok/s, fase 2 PASS") rapporteerde een **ongeldige** meting: een
staging-race corrupteerde de prompt-staging, dus zowel ground truth als de
geswapte arm produceerden degenerate output en de "bitexacte PASS" vergeleek
garbage met garbage. De timing (23,59) mat een corrupte executie. Dat blok
blijft staan als les; dit blok is de geldige meting.

**Bug 1 — staging-race in `step_graph` (runtime.py, al eerder deze sessie
gefixt).** Eén gedeelde 4-byte pinned staging-slot werd host-zijdig
overschreven vóór de async H2D-copy hem las bij back-to-back prompt-tokens.
V4's driver ontliep dit met een sync per prompt-token — dat bleek
load-bearing, geen stijl. Fix: 256-slot pinned ring in `setup_graph`
(`_stage_mem`/`_stage_np`/`_stage_i`). Les: **een async H2D-copy leest zijn
pinned bron op GPU-executietijd, niet op enqueue-tijd.**

**Bug 2 — CuPy-pool-aliasing tussen de twee graphs (de phase-2-faalt na
bug 1).** `cache` en `_dev_cache` zaten NIET in de state-snapshot van
`proto_multi_seq_graph_n2.py`. Replay voert geen Python uit — de graph bindt
die buffers bij capture op ruwe pointer — dus de enige referentie was
`rt._dev_cache`/`rt.cache`. Bij het bouwen van sequentie 1 vielen die refs
weg, de pool gaf sequentie 1's cache-allocaties (zelfde groottes, zelfde
volgorde) **exact dezelfde adressen**, en beide graphs schreven daarna in
hetzelfde geheugen. Faalpatroon klopte precies: seq0 divergeerde op
decode-token 2 (meteen na seq1's eerste replay), seq1 pas bij token 8.
Fix: `CACHE_ATTRS = ["cache", "_dev_cache", "cache_mode", "cache_stats"]`
in de snapshot — referenties vasthouden is hier geen boekhouding maar
correctheid. Les: **bij CUDA-graphs die buffers by-pointer binden moet elke
gebonden buffer een levenslange Python-referentie hebben; "replay raakt het
niet via Python" is niet genoeg.**

**Bug 3 — `setup_graph()` early-return (shared variant).** `setup_graph`
returnt stil als `self._graph is not None`; `build_graph_state_shared`
riep het voor sequentie 1 zonder `_graph = None` → sequentie 1 kreeg nooit
een eigen graph; haar snapshot wees naar sequentie 0's graph/tok_dev/ring.
Fix: `rt._graph = None` vóór de tweede capture (graph 0 blijft leven via
state[0]).

**Opzet.** Beide prototypes opnieuw gedraaid met echte, coherente prompts
(staging-ring) + bugfixes 2/3. Fase 2: interleaved graph-decode bitexact vs
onafhankelijke solo-graph ground truth (20 tokens × 2 sequenties). Fase 3:
aggregate timing over 40 echte tokens. Ook fase-3-herbouw beschermd: oude
states expliciet gedropt + pool vrijgegeven vóór rebuild (anders dubbelt de
cache-VRAM-piek nu refs bewust levend gehouden worden).

**Uitkomst (geldig).**
- **Private caches (cap 24×2): fase 2 PASS bitexact; 36,86 tok/s aggregate
  (27,13 ms/token, 1085,2 ms / 40 tokens).** Boven naïef-eager N=2 (31,66,
  +16,4%) en boven expliciete-deling (11,23). Ver onder solo V6 (47,41) en
  ver onder 2× solo.
- **Gedeelde cache (cap 64, één stream): fase 2 PASS bitexact; 33,52 tok/s
  (29,84 ms/token).** Boven naïef (+5,9%), maar **onder private ondanks de
  grotere cache** — tegen de verwachting in. Inmiddels opgelost (zie het
  hitrate-blok hierboven): thrash weerlegd — shared hitrate 71,1% is HOGER
  dan solo cap-64; de enkele gedeelde stream kost de cross-sequentie-
  PCIe/compute-overlap die de private variant wél heeft.

**Wat dit sluit of opent.** Graph-residentie voor multi-seq WERKT nu
correct en slaat de naïeve eager-aanpak — de route uit
`PATH_TO_100_TOKS.md` is uitvoerbaar. Maar de scaling is nog slecht:
36,86/47,41 = 0,78× solo voor 2× het werk. De resterende kloof zit in
cache-druk (cap 24/64 i.p.v. 72) en het ontbreken van de expert-unie-deling
IN de graph (die is elders al bitexact bewezen: `proto_batch_moe_*`,
1,71× fetch-winst over 23 lagen). Volgende logische stap: unie-fetch in de
graph-georkestreerde multi-seq-loop, of N=4 met hetzelfde patroon om te
zien of aggregate meeschaalt.

**Artefacten.** `pro_research/proto_multi_seq_graph_n2.py` (+`.json`),
`pro_research/proto_multi_seq_graph_n2_shared.py` (+`.json`),
`src/moe_lab/lightningstream_nemotron/runtime.py` (staging-ring).

---

## 2026-08-16 — Multi-sequentie graph N=2, per-sequentie caches: bitexact, maar 23,59 tok/s — GESPLITSTE cache domineert de launch-winst

> ⚠️ **ONGELDIG — zie het correctieblok hierboven.** Deze run had een
> staging-race in `step_graph` (prompts corrupt) én pool-aliasing tussen de
> graphs; "fase 2 PASS" vergeleek garbage met garbage en de 23,59 tok/s mat
> een corrupte executie. Geldige hermeting: private 36,86 / shared 33,52
> tok/s, beide écht bitexact.

**Vraag.** Herstelt CUDA-graph-residentie alleen (zonder expliciete deling) de
Python/launch-overhead die de naive multi-sequentie-aanpak bij grotere N liet
instorten? (Vervolg op Claude's root-cause: N=8-instorting = launch-overhead,
niet caching.)

**Opzet.** `pro_research/proto_multi_seq_graph_n2.py` (Claude's ongedraaide
prototype, afgemaakt deze sessie): één CUDA-graph per sequentie via de
bestaande, door V4-V6 bewezen `setup_graph()`/`step_graph()`-machinerie —
het "NOT YET RUN"-commentaar in runtime.py was inderdaad stale, dat was zijn
open vraag. **VRAM-fix vóór de eerste run nodig**: de bedoelde 2× cap-72
cache (2×4,33 GiB) past niet in 8151 MiB; numeriek invariant (E1F21-INV),
dus cap 24 per sequentie (2×1,45 GiB). Fase 2: bitexact vs onafhankelijke
solo-graph-ground-truth. Fase 3: aggregate timing, 20 stappen × 2 sequenties.

**Uitkomst.** Fase 2: **PASS, bitexact, 20/20 tokens × 2 sequenties** — het
per-sequentie-graph-wisselmechanisme (STATE_ATTRS + GRAPH_ATTRS) is correct.
Fase 3: **23,59 tok/s aggregate** (42,39 ms/token, 1695 ms / 40 tokens) —
**onder** de naive N=2-baseline (31,66 robuust) en onder solo (29,8).

**Interpretatie.** De gesplitste cache (2×24 slots i.p.v. gedeelde 72)
veroorzaakt zoveel extra PCIe-missers per token (~2× de tokentijd van solo-
graph's 21,1 ms) dat het wegvallen van de launch-overhead dat niet
compenseert. De vergelijking is hiermee conservatief-maar-verward: "graph
alleen" is getest, maar met een veel kleinere cache dan de baseline had.
**Conclusie: private caches per sequentie zijn op 8 GiB geen optie (VRAM) én
niet wenselijk (missers) — de multi-seq-graph moet de cache DELEN.**

**Wat dit opent.** De gedeelde-cache-variant: één cache (cap ~64, VRAM-
begrensd) + één `_dev_cache` gebonden in beide graphs, maar dan **geserialiseerd
op één stream** — anders racen de twee graphs op de gedeelde LRU-tabellen
(ids/w/slots worden door beide graphs beschreven). Verwachting: per-token ≈
solo-graph (~21-23 ms, hoge hitrate), N=2 aggregate ~42-45 tok/s.

**Artefacten.** `pro_research/proto_multi_seq_graph_n2.py`,
`pro_research/proto_multi_seq_graph_n2.json`.

---

## 2026-08-16 — Routekaartstap 1 begonnen: device-only unie-berekening geverifieerd met bestaande kernels, geen nieuwe CUDA-code nodig

**Vraag.** `PATH_TO_100_TOKS.md` se routekaart-item 1 noemt "device-only
routing-unie-berekening" als eerste concrete stap om
`proto_multi_seq_moe_shared.py`'s host-sync (`cp.asnumpy` per sequentie per
laag, puur om de Python-unie te bouwen) weg te nemen. Is dat mogelijk met
de **al bestaande, ongewijzigde** `cache_assign`/`cache_fetch`-kernels, of
is er echt nieuwe CUDA-code nodig?

**Bevinding uit de kernelbroncode zelf (niet aangenomen, nagelezen).**
`cache_assign` dedupliceert al binnen één aanroep: bij een HERHAALDE id in
dezelfde (ongededupliceerde) lijst vindt de tweede+ occurrence
`slot_of[e]` al gezet (door de eerste, in dezelfde sequentiële lus) en zet
`need[s]=0`, met `slots[s]` toch correct gevuld. `cache_fetch` heeft zelf
`if (!need[s]) return;` en schrijft naar `cache_c + slots[s]*code_bytes` —
meerdere posities die naar hetzelfde fysieke slot wijzen is dus al precies
wat deze kernel ondersteunt, zonder enige aanpassing.

**Opzet.** `pro_research/diag_device_only_union.py`. N=2 echte sequenties,
echte `route_topk`-output per sequentie op één echte MoE-laag. Vergelijkt:
(a) de HUIDIGE aanpak (`cp.asnumpy` + Python `set()` + `dict`, zoals
`proto_multi_seq_moe_shared.py` nu doet) tegen (b) een NIEUWE aanpak: de
RUWE, ongededupliceerde N×top_k-idlijst rechtstreeks in `cache_assign`
gooien (cap=top_k=N×top_k, dus nooit eviction binnen deze ene aanroep),
zonder enige host-side Python-unieberekening.

**Een echte bug gevonden en gefixt vóór een geldige meting.** Eerste
versie gaf **12/12 byte-mismatches** — `cache_assign` leest de `ids`-
PARAMETER die je meegeeft, maar `cache_fetch` leest daarna specifiek
`dev["ids"]` — als die twee niet hetzelfde array zijn, blijft `dev["ids"]`
op nul staan (van de allocatie) en haalt `cache_fetch` voor elke positie
**expert 0** op in plaats van de juiste expert. Productie se `_moe_dev`
vermijdt dit door `route_topk` DIRECT in `dev["ids"]` te laten schrijven;
deze test miste dat éérst. Gefixt door expliciet `dev_union["ids"][:] =
...` te zetten vóórdat `cache_assign` ermee wordt aangeroepen, exact het
productiepatroon volgend.

**Uitkomst na de fix: bitexact, 0/12 mismatches, dedup-patroon exact
zoals verwacht** (`need`==1 alleen bij eerste occurrence per expert,
totaal-`need` == aantal unieke experts, 9 van 12 in dit geval).

**Wat dit sluit of opent.** Sluit de vraag of roadmap-item 1 nieuwe
CUDA-code vereist voor de **up_proj-fetch-stap** specifiek: **nee** — de
bestaande, al-bitexact-geverifieerde productiekernels volstaan, mits
correct aangeroepen (met de nu-bekende `dev["ids"]`-valkuil vermeden). Dit
elimineert de host-sync voor DIE stap zonder één regel nieuwe kernel-code.
**Nog niet gedaan**: dezelfde aanpak voor de down_proj-maskerunie (die
vraagt nog steeds host-side groepering per expert om maskers te OR'en —
een apart, groter probleem, geen simpele hergebruik van `cache_assign`).
**Nog niet geïntegreerd** in `proto_multi_seq_moe_shared.py` zelf en dus
nog geen nieuwe tok/s-claim — dit is de geïsoleerde mechanismecontrole,
exact zoals dit project se eigen discipline voorschrijft vóór integratie.

**Poorten.** Correctheid: bitexact, 0/12 mismatches, dedup-patroon
correct, PASS. Geen tok/s-claim.

**Artefacten.** `pro_research/diag_device_only_union.py`,
`pro_research/diag_device_only_union.json`.

**Vervolg, zelfde dag — geïntegreerd in de echte staplus: bitexact, maar
GEEN nettowinst, een eerlijke negatieve uitkomst.** Het geverifieerde
mechanisme ingebouwd in `proto_multi_seq_moe_shared.py`'s up_proj-fetch-
stap: `route_topk` schrijft nu direct in slices van één platte
`all_ids_dev`/`all_w_dev`-buffer (geen per-sequentie array meer), en de
fetch gebruikt `cache_assign`+`cache_fetch` rechtstreeks op de RUWE
N×top_k-lijst (geen host-side `set()`/`dict()` meer voor de unie zelf) —
precies het in isolatie bewezen mechanisme. **Correctheidspoort GESLAAGD**
op de robuuste 40-stappen-schaal: bitexact, 40/40 tokens × 2 sequenties.
**Timing: 10,898 tok/s — een KLEINE REGRESSIE tegenover de vorige 11,234
(40 stappen), niet de verwachte winst.**

**Waarom, vermoedelijk.** De fetch-buffer (`batched_c`/`batched_s`) is nu
altijd **P-groot** (N×top_k=12, worst case) in plaats van **u-groot**
(het werkelijke aantal unieke experts, ≤12, vaak kleiner bij overlap) —
de oude aanpak alloceerde minder wanneer er overlap was; de nieuwe alloceert
en nullt altijd de volle worst-case-buffer, ook al voorkomt `need[]` nog
steeds overbodige PCIe-fetches voor duplicaten. Plus: een verse
`dev_union`-structuur (9 device-arrays) wordt elke laag/stap opnieuw
gealloceerd. De bespaarde host-syncs wegen dus niet op tegen deze nieuwe
allocatie-/nulkost in dit specifieke geval.

**Wat dit sluit of opent.** Sluit de vraag of minder host-syncs
automatisch sneller is: **nee, niet vanzelfsprekend** — hetzelfde soort les
als eerder bij gather-batching (PCIe-gebonden werk profiteert niet
automatisch van een op-zich-correcte optimalisatie als er een andere,
even grote nieuwe kost tegenover staat). Het mechanisme blijft correct en
architecturaal schoner (minder host-afhankelijkheid, dichter bij
routekaart-item 1's geest) — maar levert in DEZE vorm geen gemeten
tok/s-winst op. Beide versies (host-unie vs device-only-unie) zijn nu
bitexact geverifieerd; het verschil tussen ze is binnen ruis-orde (~3%),
geen van beide dus overtuigend beter puur op snelheid.

**Poorten.** Correctheid: bitexact, 40/40 tokens × 2 sequenties, PASS.
Timing: geen winstclaim — eerlijk gerapporteerd als vlak/lichte regressie.

**Artefacten.** `pro_research/proto_multi_seq_moe_shared.py` (bijgewerkt
met device-only up_proj-unie), `pro_research/proto_multi_seq_moe_shared.json`
(laatste run, 10,898 tok/s bij 40 stappen).

**Tweede poging, direct erna, zelfde dag — de gediagnosticeerde oorzaak
gefixt, resultaat blijft vlak: de diagnose was ook fout.** De voorgestelde
verklaring (worst-case-P-grote fetch-buffer) had een directe, kernel-vrije
fix: `cache_assign` produceert zelf al een gepakte, gededupliceerde
expertlijst als bijproduct (`expert_of[:filled]`, waarbij `expert_of[v]`
gezet wordt precies wanneer slot `v` nieuw wordt toegewezen, en
`state2[1]` het aantal bijhoudt) — geen nieuwe kernel nodig, alleen een
al-bestaande kernel-uitvoer gebruiken die nog niet gebruikt werd. Fetch nu
naar een `u`-grote buffer (werkelijke unie-grootte, niet P) i.p.v.
worst-case. **Bitexact opnieuw bevestigd (40/40 tokens × 2 sequenties).**
**Timing: 10,894 tok/s — vrijwel identiek aan de 10,898 hiervoor, geen
verbetering.** De voorgestelde oorzaak (bufferomvang) was dus **ook fout**
— de werkelijke resterende kost zit vermoedelijk in de verse
`alloc_device_cache`-toewijzing zelf (9 device-arrays, elke laag, elke
stap — 23×40=920 keer, ~8280 kleine allocaties totaal) of ergens anders
niet-geïdentificeerd, niet in de buffergrootte.

**Wat dit definitief vaststelt.** Twee onafhankelijke, elk plausibele,
elk bitexact-geverifieerde pogingen om de device-only-unie-route sneller
te maken dan de host-unie-route zijn **beide mislukt** (10,898 en 10,894
tok/s, versus 11,234 voor de host-unie-versie — een consistente ~3%
regressie in beide device-only-varianten). De **host-side-unie-versie
(11,234 tok/s) blijft het beste geverifieerde getal** voor deze
sessie se `proto_multi_seq_moe_shared.py` en is teruggezet als de
canonieke staat van het bestand — de device-only-varianten se code en
resultaten blijven hier gedocumenteerd als een reële, tweemaal-bevestigde
negatieve uitkomst, geen verborgen mislukking.

**Poorten.** Correctheid: bitexact, PASS, beide pogingen. Geen
winstclaim — twee eerlijk gerapporteerde nulresultaten.

**Artefacten.** `pro_research/proto_multi_seq_moe_shared.py`
(teruggezet naar de host-unie-versie, 11,234 tok/s, het beste
geverifieerde getal).

**Derde poging, een ander idee: combineer de twee apart-bewezen
mechanismen (unie-deling BINNEN een stap + warme cache OVER stappen) —
bitexact, maar een grote, onverklaarde REGRESSIE, niet de verwachte
winst.** Twee bevindingen deze sessie stonden nooit samen getest: (1)
unie-gevoede deling binnen één stap (bitexact, 11,234 tok/s met een VERSE
cache per aanroep) en (2) een warme, evoluerende cache over meerdere
stappen vermindert missers met 27,6% (`diag_batch_warm_cache.py`, losstaand
van een echte GEMV/down_proj-pijplijn). `pro_research/proto_multi_seq_moe_shared_warmcache.py`
combineert ze: **één persistente cache per laag (cap=72, productie se
eigen standaard), gebouwd vóór de decode-lus, hergebruikt en evoluerend
over alle 40 echte stappen** — in plaats van `fused.alloc_device_cache`
vers per aanroep zoals in beide eerdere pogingen.

**Correctheidspoort GESLAAGD**: bitexact, 40/40 tokens × 2 sequenties,
PASS — de eviction-dynamiek van een echte, over stappen evoluerende LRU
verandert dus niets aan de berekende waarden, zoals verwacht (cache-status
beïnvloedt alleen welke bytes wanneer over PCIe gaan, nooit het resultaat).

**Timing: 1,725 tok/s — een REGRESSIE van ~6,5× tegenover de 11,234-
baseline, veel groter en in de tegenovergestelde richting van wat verwacht
werd** (een warme cache zou minder missers moeten geven, dus minder PCIe-
verkeer, dus sneller — niet 6,5× trager). **De oorzaak is NIET vastgesteld.**
Een snelle redenering (cap=72 maakt `cache_assign`'s eviction-scan duurder)
is al **tweemaal deze sessie direct getoetst en weerlegd**
(`diag_cache_assign_scan_cost.py`: vlak-tot-dalend tot cap=576, geen
stijgende kost) — dus die verklaring wordt hier NIET herhaald zonder
nieuwe toetsing. Andere kandidaten (grotere `codes`/`scales`-buffers, 72×
i.p.v. ~9-12× UP_CODE/UP_SCALE, met slechtere geheugenlocaliteit; méér
werkelijke fetches dan verwacht als de routing-diversiteit over 40 stappen
de 72-cap sneller vult dan gedacht) zijn **niet getoetst en dus expliciet
niet als verklaring geclaimd** — precies om niet een derde keer een
ongeverifieerde uitleg te geven die later weerlegd moet worden.

**Wat dit sluit of opent.** Sluit NIET de vraag of warme cache + unie-
deling ooit kan samenwerken — bitexact bewijst dat het mechanisme correct
is. Sluit WEL dat de simpele combinatie ("gewoon de cache persistent maken")
in deze vorm een reële, grote regressie geeft, geen winst — een derde
eerlijk nulresultaat (in dit geval fors negatief) voor deze
optimalisatieronde. Opent een precieze vervolgvraag voor wie dit oppakt:
sectiegeprofileerd meten (zoals eerder voor de andere twee versies) om de
ECHTE oorzaak vast te stellen vóór er nog een keer geraden wordt.

**Poorten.** Correctheid: bitexact, 40/40 tokens × 2 sequenties, PASS.
Timing: expliciet GEEN winstclaim — een grote, eerlijk gerapporteerde
regressie, oorzaak onbekend.

**Artefacten.** `pro_research/proto_multi_seq_moe_shared_warmcache.py`,
`pro_research/proto_multi_seq_moe_shared_warmcache.json`.

**Vervolg, zelfde dag — de belofte ingelost (sectiegeprofileerd meten in
plaats van gissen), en een verrassend, nog niet verklaard resultaat.**
Dezelfde `PROFILE`-instrumentatie als het andere script toegevoegd en
gedraaid (40 stappen × 2 sequenties, bitexact opnieuw bevestigd).

| sectie | aandeel |
|---|---:|
| **1. routing + shared expert** | **54,1%** |
| 2a. ids-kopie | 0,2% |
| 2b. cache_assign | 0,2% |
| 2c. cache_fetch | 2,8% |
| 2d. host-sync | 0,4% |
| 3. up_proj-GEMV + panel_scan | 14,4% |
| 4. unie-maskerberekening | 2,4% |
| 5a. down_proj gather (gebatcht) | 11,1% |
| 5b. down_proj masked+reduce+accumulate (gebatcht) | 14,1% |
| 6. accumuleren | 0,2% |

**`cache_assign`/`cache_fetch` zijn definitief onschuldig bevonden** (2b+2c
samen slechts ~3%) — dit sluit de eviction-scan-hypothese een DERDE keer
uit, nu ook in de geïntegreerde context, niet alleen de losstaande
micro-benchmark. **Maar sectie 1 (routing + shared expert) — code die
LETTERLIJK IDENTIEK is aan de niet-warme versie, en die de persistente
cache-buffers helemaal niet aanraakt — domineert onverwacht met 54,1%**,
tegenover ~12-13% in de niet-warme versie. Dit is het enige structurele
verschil tussen de twee scripts dat dit zou kunnen verklaren: de
persistente `codes`/`scales`-buffers zijn nu **cap-groot (72 sloten)** per
laag i.p.v. `u`-groot, een permanente reservering van orde ~190 MB per
laag × 23 lagen (de hogere VRAM-bezetting in `nvidia-smi`, 7814 MiB tegen
6810 MiB, bevestigt een reëel groter permanent geheugenbeslag). Een
**aannemelijke maar NIET geverifieerde** kandidaat-verklaring: een grote
permanente VRAM-reservering kan CuPy se geheugenpool-allocator trager
maken voor de vele kleine, tijdelijke allocaties die sectie 1 en
verderop nog steeds doet (`cp.zeros(top_k)`-achtige buffers) — maar dit is
**expliciet niet getoetst**, en wordt hier bewust NIET als vaststaand
gerapporteerd, na twee eerdere keren dit sessie waarin een aannemelijk
klinkende verklaring bij toetsing bleek te kloppen noch te falen zoals
verwacht.

**Wat dit sluit en opent.** Sluit definitief de eviction-scan als oorzaak
(nu drie onafhankelijke bevestigingen: de losstaande micro-benchmark, en
nu twee keer binnen de geïntegreerde context). Opent een precieze,
overdraagbare vervolgvraag: is de grote persistente buffer-allocatie de
oorzaak van sectie 1 se trager worden? Dat zou getoetst moeten worden met
een geïsoleerde micro-benchmark (kleine allocaties timen met en zonder een
grote, gelijktijdig actieve permanente VRAM-reservering) — niet gedaan
hier, met opzet, om niet een derde ongeverifieerde verklaring te
rapporteren.

**Poorten.** Correctheid: bitexact opnieuw bevestigd. Geen nieuwe
tok/s-claim (zelfde 1,7-tok/s-orde als hiervoor, binnen ruis van de
extra profilerings-syncs).

**Artefacten (bijgewerkt).** `pro_research/proto_multi_seq_moe_shared_warmcache.py`
(nu met `PROFILE`-vlag, standaard uit).

**Directe toetsing van de kandidaat-verklaring, zelfde dag: OOK weerlegd.**
`pro_research/diag_alloc_pressure.py` — een losstaande, modelvrije
micro-benchmark: kleine allocaties (`cp.zeros`, dezelfde ordegrootte als
sectie 1 se buffers) getimed, eerst zonder, dan MET een permanente ~4,33
GiB VRAM-reservering (matcht de werkelijke 23-lagen-`codes`/`scales`-
belasting van het warme-cache-script). **Resultaat: kleine allocaties
worden juist SNELLER met de grote reservering aanwezig (0,0363 → 0,0197
ms/ronde, verhouding 0,54×) — het tegenovergestelde van de aanname.**
Geheugenpool-druk van de grote persistente buffer is dus **ook** geen
verklaring.

**Balans van deze deelinvestigatie.** Drie op elkaar volgende, elk
aannemelijke, elk direct getoetste verklaringen voor de 6,5×-regressie
zijn stuk voor stuk weerlegd: (1) `cache_assign`'s eviction-scan (losstaand
én in context, tweemaal), (2) geheugenpool-druk van de grote persistente
reservering. **De werkelijke oorzaak van waarom sectie 1 (routing + shared
expert, ongewijzigde code) 54,1% van de tijd inneemt in het warme-cache-
script blijft onbekend.** Dit wordt hier eerlijk als open vraag gelaten in
plaats van een vierde ongeverifieerde gok te rapporteren — verder
onderzoek zou serieuzere profileringsgereedschappen vereisen (Nsight
Compute/Systems) die niet beschikbaar zijn binnen dit script-gebaseerde
onderzoekskader.

**Poorten.** Geen PRO-poorten (read-only micro-benchmark). Hypothese
expliciet weerlegd, `hypothesis_supported_gt_2x: false`.

**Artefacten.** `pro_research/diag_alloc_pressure.py`,
`pro_research/diag_alloc_pressure.json`.

**Vierde ronde, fijnere profilering: het mysterie verplaatst zich, en dat
IS de bevinding.** Sectie 1 verder opgesplitst in zijn zeven losse
kernel-aanroepen (`use_state`, `norm`, `acc.fill`, twee shared-expert-
GEMV's, `mv_f32`, `route_topk`) — **elk daalt naar bijna niets** (samen nog
geen 2% van de tijd). Het "mysterie" verplaatste zich niet weg, het
verplaatste zich naar de sync-grens zelf: `use_state()` (triviale Python
`setattr`-lus, geen GPU-werk) ving alsnog 52,6% op — een teken dat de
`cp.cuda.Device(0).synchronize()` in `_prof_mark` daar **wachtend werk uit
eerdere, nog niet gesynchroniseerde aanroepen** opving, niet de kost van
`use_state()` zelf.

**Dus ook `multi_step`'s Mamba/attentie-lagen geprofileerd** (nooit eerder
gedaan in beide MoE-scripts — die lagen liggen BUITEN `shared_moe_layer`).
**Resultaat: het "mysterie" verschijnt OOK daar, en in de triviale
MoE-add-back-stap** (`for s: use_state(); k.add_(...)` — twee bijna-gratis
operaties): `M_mamba_layer` 27,0% (31,58 s), `E_moe_layer_addback` 35,9%
(41,93 s) — de GROOTSTE losse post nu, ondanks dat de code erin triviaal
is. **Dit is de belangrijkste bevinding van deze hele deelinvestigatie**:
de vertraging is NIET gelokaliseerd in de MoE-cache-code — hij verschijnt
overal, ook in Mamba-verwerking die he-le-maal niets met de cache te maken
heeft. Dat wijst op een **globaal, doorlopend effect** (vermoedelijk
werkelijk méér totaal GPU-/PCIe-werk over de hele stap, of een
systeembrede contentie die alleen zichtbaar wordt bij dit soort
ongelijk verdeelde, sync-gebaseerde profilering) in plaats van een bug in
één specifieke sectie.

**Eerlijke grens van dit onderzoeksinstrumentarium.** Sync-gebaseerde
sectieprofilering (het enige gereedschap beschikbaar zonder Nsight
Compute/Systems) blijkt zijn eigen grens te hebben zodra het werk zo
ongelijk verdeeld en asynchroon opgestapeld raakt: elke sync-grens kan
wachtend werk van willekeurig welke eerdere, nog niet gesynchroniseerde
aanroep opvangen, wat sectie-toewijzing onbetrouwbaar maakt bij grote,
onregelmatige vertragingen zoals deze. Dit is zelf een waardevolle,
eerlijke conclusie: verder verfijnen van DEZE profileringsaanpak zal het
mysterie niet oplossen — dat vraagt echt Nsight Compute/Systems (buiten
bereik van dit script-gebaseerde kader), of een volledig herontwerp naar
CUDA-graph-residentie (routekaart-item 2) die het hele probleem van
"waar precies gaat tijd verloren binnen een asynchrone stapel" irrelevant
maakt door alles in één opname te vangen.

**Poorten.** Correctheid: bitexact bevestigd bij elke stap in deze hele
investigatie. Geen tok/s-winstclaim.

**Artefacten (definitief bijgewerkt).** `pro_research/proto_multi_seq_moe_shared_warmcache.py`
(fijnkorrelige `PROFILE`-instrumentatie in zowel `shared_moe_layer` als
`multi_step`, standaard uit).

---

## 2026-08-16 — Robuustheidscontrole van de N=2-naive-baseline: het cijfer krimpt bij een langere horizon — eerlijke bijstelling, geen tegenspraak

**Vraag.** De N=2-naive-baseline (+5,4% aggregate) en de latere expliciete-
deling-meting kregen allebei een robuustheidscontrole bij 40 stappen — de
expliciete-deling-versie hield stand (11,12→11,23, ~1% verschil). Voor een
eerlijke, gelijkwaardige vergelijking verdient de NAIVE-baseline dezelfde
controle, die tot nu toe alleen bij 15 stappen gemeten was.

**Opzet.** `DECODE_STEPS` in `proto_multi_seq_full_model.py` van 15 naar 40
(80 echte tokens i.p.v. 30), verder ongewijzigd. **Bitexact, 40/40 tokens ×
2 sequenties, PASS.**

**Uitkomst — het cijfer krimpt, geen ruis-toeval.** Solo: 31,020 tok/s
(ruis-consistent met eerdere solo-metingen ~29,8-31,0). **N=2 naive
aggregate bij 40 stappen: 31,656 tok/s — 1,0205× (+2,05%), niet de eerder
gerapporteerde +5,4%.**

**Waarom dit geen tegenspraak is, maar een verwachte verfijning.**
`diag_batch_warm_cache.py` (eerder deze sessie) mat expliciet dat het
gedeelde-cache-voordeel **afneemt** van cold-start (missers-reductie hoog
in de eerste stappen) naar steady-state (28,0% in het laatste kwart,
lager dan het 40-staps-gemiddelde van 27,6%). Een 15-staps-meting is meer
cold-start-gedomineerd dan een 40-staps-meting; het kleinere +2,05%-cijfer
bij 40 stappen is dus precies wat die eerdere diagnostiek al voorspelde —
geen fout, een preciezere meting over een langere, representatievere
horizon.

**Wat dit sluit of opent.** Corrigeert het gerapporteerde naive-cijfer naar
het robuustere **+2,05%** (niet +5,4%) als de betere schatting voor een
langlopende sessie. Sluit de vraag of de eerdere 15-staps-metingen een
toevalstreffer waren: nee (bitexact, reproduceerbaar patroon, consistent
met de aparte cold-vs-warm-diagnostiek) — maar het cijfer zelf hoort bij
de langere horizon vervangen te worden.

**Poorten.** Correctheid: bitexact, PASS. Eerlijke bijstelling van een
eerder gerapporteerd getal, geen nieuwe claim voorbij wat hierboven staat.

**Artefacten.** `pro_research/proto_multi_seq_full_model.py` (DECODE_STEPS
40), `pro_research/proto_multi_seq_full_model.json` (bijgewerkt, 40-staps-
resultaat).

---

## 2026-08-16 — N=8 naive baseline: de dalende trend wordt geen "vlak" meer, maar een INSTORTING — 4× TRAGER dan solo

**Vraag.** N=2 gaf +2,05% (robuust), N=4 gaf +4,7% (15 stappen, vlak tot
licht lager dan N=2). Zet die dalende trend door bij N=8, wordt hij erger,
of keert hij om? Dit test het direct, op precies hetzelfde geverifieerde,
tot nu toe best-presterende mechanisme (het naive pad, geen expliciete
deel-logica — dat blijft in absolute tok/s beter dan elke "slimme"
unie-gevoede variant die deze sessie geprobeerd is).

**Opzet.** `pro_research/proto_multi_seq_full_model_n8.py` — identiek
mechanisme als N=2/N=4, nu N=8, 8 diverse prompts, 30 stappen.
**Correctheidspoort GESLAAGD**: bitexact, 30/30 tokens × 8 sequenties.

**Uitkomst — geen vlakke trend meer, een instorting.**

| N | solo tok/s | naive aggregate tok/s | speedup |
|---:|---:|---:|---:|
| 2 | 29,798 (robuust: 31,020) | 31,411 (robuust: 31,656) | 1,054× / **1,021×** |
| 4 | 29,820 | 31,215 | 1,047× |
| 8 | 29,743 | **7,521** | **0,253× — 4× TRAGER dan solo** |

**Dit is geen geleidelijke verslechtering meer zoals N=2→N=4 — het is een
instorting.** Bij N=8 is de aggregate doorvoer over alle 8 sequenties
samen LAGER dan wat één enkele sequentie alleen al haalt.

**Waarschijnlijk verband met vandaag se andere N-schaal-bevindingen.** Deze
sessie vond al twee andere gevallen waar het vergroten van "hoeveel er
tegelijk door één vaste cache-capaciteit (72) moet" een REGRESSIE gaf in
plaats van een verbetering (grotere cache-capaciteit bij N=4: 0,706×;
persistente warme cache gecombineerd met unie-deling: 0,17× (6,5× trager)).
Bij N=8 met cap=72 en top_k=6 kunnen tot 48 experts per stap nodig zijn
(8×6) — bijna twee derde van de hele cache-capaciteit, elke stap opnieuw,
met waarschijnlijk hoge omloop (verschillende sequenties routeren naar
andere experts). Dit past bij een coherent, opkomend beeld: **het naive
gedeelde-cache-mechanisme (zonder expliciete unie-logica) schaalt niet
goed voorbij kleine N** — de cache verandert van incidenteel voordeel (bij
N=2, weinig contentie) naar een reëel knelpunt (bij N=8, veel contentie),
niet door een bug maar door een fundamentele capaciteit-versus-vraag-
mismatch.

**Wat dit sluit of opent.** Sluit definitief de vraag of "gewoon N
verhogen" met de naive aanpak een pad naar hogere aggregate doorvoer is:
**nee, integendeel** — bevestigt dat de vroege positieve resultaten bij
N=2 een klein-N-fenomeen zijn, geen schaalbaar patroon. Versterkt de
conclusie in `PATH_TO_100_TOKS.md` dat een echte batch>1-winst een
ECHTE, doordachte architectuur vereist (expliciete deling, juiste
cache-dimensionering voor grotere N, niet zomaar "meer sequenties door
dezelfde cache duwen"). Bevestigt ook waarom de eerdere cache-capaciteit-
en warme-cache-experimenten regressies vonden: dat waren geen
geïsoleerde bugs, maar vroege signalen van hetzelfde onderliggende
schaalprobleem.

**Poorten.** Correctheid: bitexact, PASS. Eerlijk gerapporteerde, forse
regressie — geen tok/s-winstclaim voorbij N=2/N=4.

**Artefacten.** `pro_research/proto_multi_seq_full_model_n8.py`,
`pro_research/proto_multi_seq_full_model_n8.json`.

**Vervolg, zelfde dag — de concrete hypothese (cache-thrashing) direct
getoetst en VERWORPEN, en dat lost het mysterie eindelijk op.** In plaats
van nog een keer te gissen: `_moe_dev`'s eigen device-cache houdt al
hit/miss-tellers bij (`dev["stats2"]`, opgeteld door `cache_assign`'s
kernel). `pro_research/diag_n8_cache_hitrate.py` leest deze rechtstreeks
uit ná een echte solo-run en een echte N=8-naive-run (zelfde mechanisme
als het N=8-script, geen nieuw correctheidsrisico — puur een teller
uitlezen).

**Resultaat: de hitrate bij N=8 is HOGER, niet lager, dan solo — 77,1%
tegen 69,7%.** De cache-thrashing-hypothese is dus **verworpen**: de cache
werkt beter bij N=8, niet slechter, en toch stortte de doorvoer in tot
0,253×. **Dit betekent dat de instorting NIETS met cache-missers/PCIe-
verkeer te maken heeft.**

**Wat dit WEL verklaart — en het sluit ook het eerdere, onopgeloste
warme-cache-mysterie af.** Gegeven dat (a) de hitrate beter is, niet
slechter, en (b) de eerdere sectieprofilering van de warme-cache-regressie
al liet zien dat de vertraging **globaal** was — óók zichtbaar in
ongerelateerde Mamba-lagen, niet gelokaliseerd in cache-code — wijst alles
nu naar de **Python-orkestratie-/kernel-launch-overhead van het state-
wisselmechanisme zelf**, niet naar caching. Het naive mechanisme roept
`rt.step()`-equivalente logica APART aan voor elke van de N sequenties,
elk met zijn eigen volledige doorloop van alle 52 lagen — geen enkele
niet-MoE-laag (Mamba, attentie) deelt ooit een kernel-launch tussen
sequenties. Deze sessie se eigen vroege bevinding (N1,
`N1_N5_OWN_HYPOTHESES_REPORT_2026-08-15.md`: **23,7% van een token is
kernel-uitgifte, geen rekenwerk**, voor ÉÉN sequentie) vermenigvuldigt zich
met N: bij N=8 is dat potentieel duizenden kernel-launches en tienduizenden
Python-`setattr`-aanroepen (~30 attributen × 52 lagen × 8 sequenties ≈
12.480 per decode-stap) per stap — puur CPU-gebonden overhead die NIET
overlapt met GPU-rekenwerk en linear-tot-slechter-dan-lineair schaalt met
N, onafhankelijk van hoe goed de cache het doet.

**Wat dit sluit.** Sluit definitief de cache-gerelateerde verklaringen
(eviction-scan, geheugendruk, thrashing — alle vier nu direct getoetst en
verworpen) voor zowel de N=8-instorting als het warme-cache-mysterie.
**De werkelijke, nu voor het eerst coherente verklaring: Python-
orkestratie-/kernel-launch-overhead van het state-wisselmechanisme zelf
schaalt met N en domineert bij grotere N** — een architecturale eigenschap
van HOE deze prototypes gebouwd zijn (aparte Python-aanroepen per
sequentie per laag), niet een bug in één specifiek onderdeel. Dit is
precies het soort probleem dat een echte CUDA-graph-integratie
(routekaart-item 2 in `PATH_TO_100_TOKS.md`) zou oplossen — een graph
vangt de hele multi-sequentie-staplus in één opname, waarna replay geen
Python-launch-overhead per kernel meer kost, ongeacht N.

**Poorten.** Geen PRO-poorten (read-only teller-uitlezing, geen
runtime-wijziging). Hypothese expliciet weerlegd,
`thrashing_hypothesis_supported: false`.

**Artefacten.** `pro_research/diag_n8_cache_hitrate.py`,
`pro_research/diag_n8_cache_hitrate.json`.

**Kritieke tegencontrole, zelfde dag: is de N=8-instorting zelf wel echt,
of een meetartefact van `cp.cuda.Event()`?** Een directe, onafhankelijke
`time.perf_counter()`-gebaseerde herhaling van dezelfde berekening
(`diag_n8_dispatch_vs_exec.py`) gaf GEEN instorting — bijna perfect
lineaire schaling (7,9-8,1× kost voor 8× werk, geen 4× regressie). Dit is
een reële tegenspraak tussen twee metingen van "dezelfde" berekening, en
dit project se eigen methodologie eist dat zulke tegenspraken opgelost
worden, niet genegeerd.

**Onderzoekstraject (eerlijk, inclusief wat niet meteen lukte).** Een
zorgvuldig gereconstrueerd script dat de VOLLEDIGE structuur van
`proto_multi_seq_full_model_n8.py` naboötste (Fase 2-correctheidspoort
verbatim, Fase 3a-solo-timing in dezelfde volgorde, zelfs de
`tokens_by_seq[s].append(...)`-regel binnen de getimede lus) **reproduceerde
de instorting NIET** (steeds ~1,01× i.p.v. 0,25×) — ondanks regel-voor-regel
vergelijking. In plaats van door te blijven gissen wat er nog ontbrak: het
**ORIGINELE, ONGEWIJZIGDE bestand zelf geïnstrumenteerd** (`diag_n8_instrumented_original.py`,
een letterlijke kopie van `proto_multi_seq_full_model_n8.py` met alleen een
`time.perf_counter()`-meting TOEGEVOEGD rond de bestaande, ongewijzigde
`cp.cuda.Event()`-lus — geen enkele regel logica veranderd).

**Uitkomst: de instorting reproduceert (0,243×, consistent met eerdere
runs), EN `cp.cuda.Event()` en `time.perf_counter()` komen binnen deze ene
run PERFECT overeen** (`ratio(wall/event)=1,0000`, beide 32.478,277 ms/32.478,636 ms
voor exact dezelfde 30-staps-N=8-lus). **Dit bewijst dat de instorting geen
meetartefact van `cp.cuda.Event()` is — hij is fysiek reëel**, bevestigd
door twee onafhankelijke tijdmethoden die binnen één run volledig
overeenstemmen.

**De aparte, kleinere puzzel (waarom de handmatige reconstructie de
instorting niet reproduceerde) blijft onopgelost** — vermoedelijk een
subtiel verschil dat ondanks zorgvuldige regel-voor-regel-vergelijking niet
gevonden is. Dit wordt hier expliciet als open, secundaire vraag gelaten
(niet als verklaring verzonnen) — het doet niets af aan de hoofdconclusie,
die nu met twee onafhankelijke methoden binnen dezelfde run is bevestigd.

**Wat dit vaststelt.** De N=8-instorting (en bij uitbreiding de N=4-cache-
en warme-cache-regressies eerder vandaag, gemeten met dezelfde
`cp.cuda.Event()`-methode) is een **reëel, fysiek fenomeen**, geen
tijdmeet-artefact. De eerdere conclusie (Python-orkestratie-/kernel-
launch-overhead schaalt met N, cache-hitrate juist beter bij N=8) blijft
staan, nu met een extra, onafhankelijke bevestiging dat het onderliggende
getal zelf klopt.

**Poorten.** Geen PRO-poorten (read-only reconciliatie-diagnostiek). Geen
tok/s-winstclaim — bevestigt een eerder gerapporteerde regressie, verandert
hem niet.

**Artefacten.** `pro_research/diag_n8_dispatch_vs_exec.py`,
`pro_research/diag_n8_dispatch_vs_exec.json`,
`pro_research/diag_n8_timing_reconcile.py`,
`pro_research/diag_n8_timing_reconcile.json`,
`pro_research/diag_n8_instrumented_original.py` (letterlijke kopie van
`proto_multi_seq_full_model_n8.py` plus één toegevoegde tijdmeting, geen
logicawijziging).

---

## 2026-08-16 — Grotere cache bij groter N: hersteld het voordeel? Verrassend: nee, het wordt WÉÉR erger — een echte hypothese verworpen

**Vraag.** De N=4-meting hierboven liet het incidentele voordeel vlak
blijven (zelfs licht dalen) i.p.v. groeien met N, met als aannemelijke
verklaring: vaste cache-capaciteit (72) geeft meer eviction/contentie bij
groter N. Expliciet als open vraag genoteerd: zou een met N meeschalende
capaciteit dit herstellen? Dit test dat direct.

**Opzet (één variabele: cache-capaciteit voor de N=4-arm).**
`pro_research/proto_multi_seq_full_model_n4_bigcache.py` — identiek
mechanisme, N=4, maar de N=4-arm gebruikt cap=144 (2×, overeenkomend met de
N=4/N=2-verhouding) i.p.v. 72. De solo-N=1-controle blijft bewust op de
standaard 72 (vergelijkbaar met alle eerdere resultaten). **Een echte bug
gevonden en gefixt vóór meting**: de eerste versie verwisselde per ongeluk
`rt.reset()` (nodig om h/pos/kv-cache tussen ground-truth-sequenties te
wissen) met `rt.enable_cache()` (wist alleen de MoE-cache, niet de
dynamische toestand) in de correctheidspoort-lus — zou sequentie 1 se
ground truth laten starten vanaf sequentie 0 se restanttoestand. Gefixt
vóór de eerste run. **Correctheidspoort GESLAAGD**: bitexact, 15/15
tokens × 4 sequenties.

**Uitkomst — hypothese VERWORPEN, en niet een beetje.** Solo (cap=72):
27,013 tok/s. N=4 met cap=144: **19,071 tok/s aggregate — 0,706×, een
ECHTE REGRESSIE**, niet een herstel of verbetering. De grotere cache maakte
het dus **slechter**, niet beter.

**Waarom, vermoedelijk.** `cache_assign`'s eigen kernel (`fused_nvfp4.py`)
doet bij elke misser een **lineaire scan over `cap` sloten** om het
minst-recent-gebruikte slot te vinden voor eviction (`for cix in 1..cap: if
last_used[cix] < mnv: ...`). Een grotere `cap` maakt die scan **duurder per
misser**, en die extra kost kan de winst van minder missers overtreffen —
vooral als het aantal missers toch al niet dramatisch daalt. Dit is een
reëel, niet eerder gedocumenteerd afwegingspunt: **cache-capaciteit
vergroten is niet gratis**, zelfs los van het VRAM-kostenaspect dat elders
deze sessie al gemeten werd (`diag_batch_vram_cost.py`) — er is ook een
reken-/latentiekost aan een grotere LRU-structuur zelf.

**Wat dit sluit of opent.** Sluit de hypothese "meeschalende cache-
capaciteit herstelt het naive-voordeel bij groter N" — verworpen, niet
bevestigd. Versterkt een bredere les: dit project se eerdere
capaciteitssweep (`diag_capacity_sweep.py`, "near-optimal, alle
agressievere splits slechter") ging over HERVERDELEN bij een VASTE totale
capaciteit; dit is een ANDER effect (absolute capaciteit vergroten heeft
een eigen, reële kost) dat nooit eerder apart gemeten was. Voor toekomstig
werk aan cache-afmetingen (batch=1 of batch>1): groter is niet automatisch
beter, en dit kost-mechanisme (lineaire eviction-scan) is nu expliciet
bekend in plaats van impliciet aangenomen.

**Poorten.** Correctheid: bitexact, PASS. Geen tok/s-claim voorbij wat
hierboven staat — dit is een duidelijk gerapporteerde REGRESSIE, geen
verborgen of afgezwakt negatief resultaat.

**Artefacten.** `pro_research/proto_multi_seq_full_model_n4_bigcache.py`,
`pro_research/proto_multi_seq_full_model_n4_bigcache.json`.

**Correctie, direct erna, zelfde dag: de VOORGESTELDE verklaring
(lineaire eviction-scan) is getoetst en WEERLEGD — de regressie zelf
blijft staan, maar de oorzaak is onbekend, niet wat hierboven beweerd
werd.** `pro_research/diag_cache_assign_scan_cost.py`: geïsoleerde
micro-benchmark van `fused.cache_assign` zelf, cap ∈ {72,144,288,576}, elke
aanroep een gegarandeerde volle-cache-eviction (worst case voor de
lineaire scan), 200 herhalingen per cap. **Resultaat: de kost per aanroep
STIJGT NIET met cap — hij daalt licht** (0,1012 → 0,0902 → 0,0882 → 0,0695
ms/aanroep van cap 72 naar 576). Dit weerspreekt de eerder voorgestelde
verklaring rechtstreeks: de lineaire scan is in de praktijk **niet** de
dominante kost bij deze cap-groottes (vermoedelijk simpel genoeg om
vast-overhead-gedomineerd te blijven, of de daling weerspiegelt iets
anders zoals GPU-klokgedrag tussen opeenvolgende, zeer korte aanroepen —
niet verder onderzocht).

**Wat dit betekent voor de eerdere claim.** De REGRESSIE zelf
(0,706×, binnen dezelfde run gemeten, bitexact gepoort) blijft een
geldige, betrouwbare meting — die staat niet ter discussie. Maar de
VERKLARING ervoor die ik voorstelde (lineaire eviction-scan) is nu
**expliciet weerlegd door een gerichte test**, niet bevestigd zoals eerst
gerapporteerd. **De werkelijke oorzaak van de bigcache-regressie is dus
nog onbekend** — mogelijk cache_fetch-gedrag, geheugenlay-out-effecten, of
iets anders, niet verder onderzocht. Dit is exact de reden waarom dit
project bij elke claim een aparte toetsing eist in plaats van op de eerste
plausibele verklaring te vertrouwen: de intuïtieve, uit de kernelbroncode
afgeleide verklaring bleek fout bij directe meting.

**Poorten.** Geen PRO-poorten (read-only micro-benchmark).

**Artefacten.** `pro_research/diag_cache_assign_scan_cost.py`,
`pro_research/diag_cache_assign_scan_cost.json`.

---

## 2026-08-16 — N=4 naive baseline: groeit het incidentele voordeel mee met N, zoals losstaande diagnostiek suggereerde? Verrassend: nee, vlak tot licht lager

**Vraag.** De N=2 naive baseline (hierboven/verderop) gaf +5,4% aggregate uit
puur incidenteel warm-cache-hergebruik. Losstaande diagnostiek eerder deze
sessie (`diag_batch_warm_cache.py`, N=4: 27,6% minder missers;
`diag_cross_sequence_union.py`: overlap groeit met N) suggereerde dat een
groter N meer deel-kans zou moeten geven. Klopt dat ook in de ECHTE,
geverifieerde eindtotmeting, niet alleen in geïsoleerde routing-tellingen?

**Opzet.** `pro_research/proto_multi_seq_full_model_n4.py` — identieke
mechanisme en correctheidsdiscipline als de N=2-versie (letterlijk
hergebruikt, niet herschreven), nu N=4, 4 diverse prompts, 15 stappen.
**Correctheidspoort GESLAAGD**: bitexact, 15/15 tokens × 4 sequenties.

**Uitkomst — verrassend vlak, niet groeiend.**

| N | solo tok/s (zelfde config) | naive aggregate tok/s | speedup |
|---:|---:|---:|---:|
| 2 | 29,798 | 31,411 | 1,054× (+5,4%) |
| 4 | 29,820 | 31,215 | 1,047× (+4,7%) |

**Geen groei — zelfs een lichte daling** (binnen ruis-orde, maar zeker geen
duidelijke stijging zoals de losstaande metingen deden vermoeden). Meest
aannemelijke verklaring: de cache-capaciteit (72 sloten) is **vast**,
onafhankelijk van N — bij N=4 concurreren vier sequenties se werksets om
dezelfde 72 sloten, wat meer eviction/contentie geeft dan bij N=2, en dat
compenseert (mogelijk volledig) de grotere ruwe overlap-kans die de
unie-tellingen alleen voorspelden. Bovendien schaalt de reken-kost ook mee
met N (geen deel voor compute, alleen voor fetch) — bij een vast
cache-budget kan het netto-effect dus vlak blijven ondanks een grotere
theoretische overlap-kans.

**Wat dit sluit of opent.** Nuanceert de eerdere aanname dat "meer N =
meer voordeel" voor het INCIDENTELE (niet-expliciete) deel-mechanisme.
Sluit niet uit dat de EXPLICIETE unie-gevoede deling (hierboven, wel
bitexact bewezen maar nog niet snel genoeg) wél zou schalen met N — dat is
een apart mechanisme met een andere dynamiek (garandeert deling i.p.v.
incidenteel hergebruik onder cache-druk). Opent een niet-gedane vervolgvraag:
zou een GROTERE cache-capaciteit (die met N meeschaalt) het naive-voordeel
alsnog laten groeien? Niet gemeten — cache=72 bleef vast in beide runs, met
opzet (één variabele: N).

**Poorten.** Correctheid: bitexact, PASS. Geen tok/s-claim voorbij wat
hierboven staat.

**Artefacten.** `pro_research/proto_multi_seq_full_model_n4.py`,
`pro_research/proto_multi_seq_full_model_n4.json`.

---

## 2026-08-16 — Expliciete MoE-deling geïntegreerd in de echte staplus: bitexact correct, maar 12× TRAGER — een belangrijke, eerlijke negatieve uitkomst die precies verklaart waarom dit een meerdere-weken-taak is

**Vraag.** De vorige meting (hieronder, "EERSTE ECHTE END-TO-END METING")
liet een naive N=2-staplus zien met +5,4% aggregate winst, puur uit
incidenteel warm-cache-hergebruik — géén expliciete unie-gevoede deling.
De voor de hand liggende vervolgstap: integreer de al bitexact-bewezen
unie-gevoede deling (`proto_batch_moe_layer_combined.py`, één laag, +20,9%)
in de ECHTE staplus, over alle 23 MoE-lagen, meerdere stappen.

**Opzet.** `pro_research/proto_multi_seq_moe_shared.py`. Bouwt direct voort
op de geverifieerde state-wisselinfrastructuur. Twee correctheids-
subtiliteiten vooraf uitgedacht (niet vanzelfsprekend uit eerdere
prototypes, die nooit tegen `_moe_dev` zelf vergeleken):
1. `_moe_dev` routeert via `fused.route_topk` (een CUDA-kernel), niet via
   `_route_device` (een aparte, cupy-argsort-gebaseerde berekening die
   eerdere prototypes gebruikten voor hun EIGEN naive-vs-batched-vergelijking,
   nooit tegen `_moe_dev`). Voor bitexactheid tegen `_moe_dev` moest dit
   script `route_topk` gebruiken.
2. `_moe_dev` accumuleert via `fused.accumulate_indirect` (gewicht van een
   DEVICE-buffer-slice), niet `accumulate_into` (een host-float-variant uit
   het oudere niet-cache-pad). Verschillende kernels zijn niet gegarandeerd
   bit-identiek ondanks algebraïsche gelijkwaardigheid (D1-les) — dit script
   gebruikt daarom `accumulate_indirect`, in exact dezelfde volgorde als
   `_moe_dev` (route-volgorde, niet unie-expert-volgorde).

**Correctheidspoort: GESLAAGD, bitexact.** N=2, 12 stappen, volledig
geïnterleaved, vergeleken tegen onafhankelijke `rt.reset()`-gebaseerde
`_moe_dev`-referentieruns — **12/12 tokens per sequentie, beide sequenties,
exact identiek.** Dit bevestigt terzelfdertijd iets dat GEEN eerder prototype
toetste: `gemv_into` (gebruikt in alle `proto_batch_*`-prototypes) en
`gemv_ervf_indirect` (gebruikt door productie-`_moe_dev`) zijn **bitexact
gelijk** voor dezelfde gewichten/invoer — een stilzwijgende aanname tot nu
toe, nu voor het eerst getoetst en bevestigd.

**Uitkomst — TIMING, en die is slecht: 12× TRAGER, geen sneller.**

| | ms/token (aggregate) | tok/s (aggregate) |
|---|---:|---:|
| N=1 solo (zelfde kale configuratie) | 33,559 | 29,798 |
| N=2 naive (alleen incidenteel warm-cache-hergebruik) | 31,836 | 31,411 |
| **N=2 met expliciete unie-gevoede MoE-deling** | **376,641** | **2,655** |

**Waarom, precies.** `nvidia-smi` tijdens deze meting: 26,98 W, 1785 MHz,
`pstate P1` — een heel ander (lager-throughput) regime dan de andere
metingen se ~55-60 W/P4. De oorzaak is duidelijk uit de code zelf: dit
script bouwt de unie-gevoede deling **puur in Python**, met per MoE-laag,
per stap: `cp.asnumpy()`-aanroepen (host-sync) per sequentie voor
route-ids, `.get()`-aanroepen (host-sync) per unie-expert voor de
maskerunie, en losse, kleine `cp.zeros()`-allocaties + kernel-launches per
(sequentie, expert)-paar EN per unie-expert. Over 23 MoE-lagen × 12 stappen
× tot ~12 unie-experts betekent dat **honderden host-device-syncs en
duizenden kleine kernel-launches** — precies het soort overhead dat
`_moe_dev`'s eigen ontwerp (device-only routing, geen host-sync, gepijplijnde
copy-stream) zorgvuldig vermijdt. Het onderliggende deel-MECHANISME is niet
fout — het is bitexact bewezen — maar een **naïeve Python-orkestratie**
van dat mechanisme verliest alle PCIe-besparing (en veel meer) aan
launch-/sync-overhead.

**Wat dit sluit en opent — de belangrijkste conclusie van de hele batch>1-
lijn tot nu toe.** Sluit definitief de vraag "is een snelle Python-prototype-
integratie mogelijk": **nee**, niet met deze aanpak. Bevestigt precies wat
`BATCH_ARCHITECTURE_DESIGN.md` van meet af aan zei: een echte integratie is
**echt CUDA-engineeringwerk** (device-only routing-unie-berekening zoals
`_moe_dev` al doet voor één sequentie, gebatchte kernel-launches over
unie-experts in plaats van een Python-for-loop, geen host-syncs in de hete
lus) — geen kwestie van "de al bewezen stukken aan elkaar plakken in
Python." De correctheids-uitkomst (bitexact) is nog steeds waardevol: het
bewijst dat de WISKUNDE van het deel-mechanisme klopt en dat integratie
"slechts" een prestatie-engineeringprobleem is, geen correctheidsprobleem.
**Niet gedaan, met opzet** (buiten scope van een sessie): een kernel-niveau
herimplementatie (device-only unie-routing-kernel, gebatchte gather/GEMV
over de unie in één launch) die deze overhead zou wegnemen.

**Poorten.** Correctheid: bitexact, 12/12 tokens × 2 sequenties, PASS. Timing:
geen claim van winst — expliciet en eerlijk gerapporteerd als een 12×
regressie, met oorzaak.

**Direct vervolg, zelfde dag — één overduidelijke inefficiëntie gevonden en
gefixt: 3,57× sneller, nog steeds bitexact.** Bij het herlezen van de eigen
code viel een echte fout op: de unie-masker-berekening zette een numpy-array
(`acc_mask`, al op host, kant-en-klaar) om naar een cupy-array en las hem
dan **element-voor-element terug via `cp.asnumpy()` BINNEN een lus over
`npanel` panelen** — een volledige host-sync per paneel-iteratie, in plaats
van gewoon de al-aanwezige numpy-data te gebruiken. Bij `npanel` in de orde
van honderden en tientallen unie-experts per laag × 23 lagen × 12 stappen
liep dit op tot **honderdduizenden overbodige host-syncs** in totaal.
Gefixt: puur numpy houden voor de hostzijde-berekening (`np.flatnonzero`
i.p.v. een Python-lus met `cp.asnumpy()` per paneel), pas naar cupy
converteren voor wat écht op device moet (de uiteindelijke `plist`/`nz`/
`pcount`-buffers voor de kernel-aanroep). **Correctheidspoort opnieuw
gedraaid: nog steeds bitexact, 12/12 tokens × 2 sequenties.** Timing: **2,655
→ 9,469 tok/s aggregate, 3,57× sneller** — nog steeds **3,3× trager dan de
naive baseline (31,411)**, dus nog geen netto winst, maar bevestigt dat de
overhead grotendeels uit **vermijdbare** inefficiënties bestaat, niet uit
iets fundamenteel onoplosbaars binnen Python-orkestratie. Toont ook hoeveel
ruimte er zit tussen "naïef Python-prototype" en "volledig CUDA-
geëngineerd" — waardevol voor wie dit oppakt, ook al is de conclusie
hierboven (echte kernel-integratie nodig voor nettowinst) onveranderd.

**Tweede vervolg, zelfde dag — geprofileerd per sectie: waar zit de
resterende 3,3× nu precies?** Lichtgewicht, module-vlag-gestuurde
instrumentatie toegevoegd (`PROFILE`-vlag, standaard `False`, verandert
niets aan de berekening zelf — puur `time.perf_counter()` +
`cp.cuda.Device(0).synchronize()` op sectiegrenzen, dus geen
correctheidsrisico). Eén geprofileerde run (12 stappen × 2 sequenties):

| sectie | aandeel |
|---|---:|
| 1. routing + shared expert | 12,0% |
| 2. gedeelde up_proj-fetch | 17,0% |
| 3. up_proj-GEMV + panel_scan | 8,9% |
| 4. unie-maskerberekening | 11,4% |
| **5. down_proj gather+masked+reduce** | **48,9%** |
| 6. accumuleren | 1,8% |

**Sectie 5 (down_proj) domineert verreweg — bijna de helft van alle
resterende tijd.** Dit is precies de sectie met de meeste kleine
kernel-launches per unie-expert (gather + per sequentie die hem koos:
down_masked + reduce_partials) én een verse 2,68 MB `mirror`-buffer-allocatie
per unie-expert. **Concrete, al beschikbare volgende hefboom, nog niet
toegepast**: deze sessie bouwde en verifieerde al **gebatchte** varianten
van precies deze kernels tijdens V5/V6-ontwikkeling
(`gather_down_sparse_ind_batched`, `gemv_down_masked_partial_ind_batched`,
`reduce_partials_batched` in `down_gather_batch_kernels.py`/
`down_proj_batch_kernels.py`, gebruikt in productie-V6 voor de top_k-
binnen-één-sequentie-dimensie) — nooit toegepast op de unie-over-sequenties-
dimensie hier. Vereist het herstructureren van de data naar de
samenhangende buffervorm die deze kernels verwachten (in plaats van
per-paar losse buffers) — een reële, maar nu precies afgebakende
vervolgtaak, niet langer een vaag "meer engineering nodig."

**Poorten (dit vervolg).** Correctheid: niet opnieuw getoetst na het
uitzetten van `PROFILE` — de instrumentatie raakt geen berekende waarde aan,
dus de eerdere bitexact-poort blijft geldig. Schone hertiming zonder
profiling: **9,692 tok/s** (ruis-consistent met de eerdere 9,469).

**Artefacten (dit deel).** `pro_research/proto_multi_seq_moe_shared.py`
(bijgewerkt, `PROFILE`-vlag toegevoegd, standaard uit),
`pro_research/proto_multi_seq_moe_shared.json` (schone eindmeting, 9,692
tok/s), `pro_research/proto_multi_seq_moe_shared_profile.json`
(sectie-uitsplitsing).

**Derde vervolg, zelfde dag — de al gebouwde gebatchte V5/V6-kernels
toegepast op de unie-dimensie: nog eens sneller, en een fysiek onderscheid
gevonden tussen wat batching wél en niet oplost.** De profiling wees
`down_proj gather+masked+reduce` aan als 48,9% van de resterende tijd. Deze
sessie had al **gebatchte, bitexact-geverifieerde varianten** van precies
deze kernels gebouwd tijdens V5/V6-ontwikkeling
(`gather_down_sparse_ind_batched`, `gemv_down_masked_partial_ind_batched`,
`reduce_partials_batched`, `weighted_accumulate_ind_batched`), maar nooit
toegepast op de unie-over-sequenties-dimensie hier — alleen op de
top_k-binnen-één-sequentie-dimensie in productie.

**Stap 1 — down_masked+reduce+accumulate batchen over alle (sequentie,
expert)-paren.** Vereist per paar een EIGEN mirror-slot (de gebatchte
kernel indexeert `bank + s*mirror_bytes` direct, geen indirectie) — dus een
gedeelde unie-expert se al-gegathered mirror wordt **apparaat-naar-
apparaat gekopieerd** (goedkoop, VRAM-bandbreedte) in elk paar-slot dat hem
nodig heeft, in plaats van opnieuw van host gehaald (het dure deel, dat
gededupliceerd blijft). **Correctheidspoort GESLAAGD**: bitexact, 12/12
tokens × 2 sequenties. **Timing nauwelijks veranderd (9,469 → 9,789
tok/s)** — verrassend weinig, gegeven hoeveel launches wegvielen.

**Herprofilering verklaart waarom**: de vroegere gecombineerde 48,9%
sectie splitste in `5a_gather` (37,4%!) en `5b_masked+reduce+accumulate`
(12,3%, sterk gekrompen dankzij het batchen). **Het batchen van
masked/reduce/accumulate werkte dus wél goed** — maar gather (nog
ongebatcht) bleek de nieuwe dominante kost, ongeveer gelijk in omvang aan
wat er bij masked/reduce bespaard werd, dus het totaal bleef vlak.

**Stap 2 — gather ook batchen over de unie-experts** (`gather_down_sparse_ind_batched`,
natuurlijk passend op de unie-dimensie, geen duplicatie nodig zoals bij
masked). **Correctheidspoort GESLAAGD**, nog steeds bitexact. **Timing:
9,789 → 10,17 → schone hermeting 10,72 tok/s** — een reële maar
**bescheiden** verbetering (~4-10%), duidelijk minder dramatisch dan de
masked/reduce-batching gaf.

**Het fysieke onderscheid, en waarom dit de belangrijkste les van deze hele
optimalisatieronde is.** Gather batchen hielp weinig omdat gather
fundamenteel **PCIe-bandbreedte-gebonden** is (leest van host-gemapte
`down_base`-geheugen — hetzelfde soort trage strided-host-lezen dat E2/
NERVF-4 eerder al identificeerden en waarvoor die eerdere sporen werden
**weerlegd**) — hetzelfde aantal bytes moet over de bus, of dat nu in 1 of
in u aparte launches gebeurt. Masked/reduce/accumulate zijn daarentegen
**reken-/VRAM-gebonden, kleine kernels** waarvoor launch-overhead wél de
dominante kost was — batching hielp daar wél substantieel. **Launch-
overhead-batching is dus geen universele oplossing: het lost precies één
klasse van inefficiëntie op (te veel kleine launches), niet de andere
(bandbreedte-gebonden data-beweging).**

**Cumulatieve balans van deze hele optimalisatieronde.** 2,655 → 9,469
(sync-fix, 3,57×) → 9,789 (masked/reduce/accumulate-batching, marginaal
op zichzelf maar noodzakelijk om gather als nieuwe bottleneck te
onthullen) → 10,72 tok/s (gather-batching erbij), **4,04× totaal sneller
dan de eerste werkende versie, nog steeds bitexact op elke stap.** Nog
steeds **2,93× trager dan de naive baseline (31,411)** — en dat resterende
gat is nu fysiek verklaard (gather + up_proj-fetch, samen ~49% van de
tijd, zijn beide bandbreedte-gebonden en waarschijnlijk dicht bij hun
fysieke vloer) in plaats van een vage "meer engineering nodig". Een
volgende stap zou PCIe-transfers moeten OVERLAPPEN met rekenwerk (zoals
V4-V6's eigen graph-residentie al doet voor batch=1) in plaats van ze te
verkleinen — een wezenlijk ander soort hefboom, niet gedaan hier.

**Poorten (dit hele derde vervolg).** Correctheid: bitexact bij elke stap,
12/12 tokens × 2 sequenties, telkens opnieuw getoetst na elke wijziging.

**Vierde, kleine vervolgstap, zelfde dag — pure Python/numpy-vectorisatie
van de unie-nz-lijstberekening.** De unie-maskerberekening bouwde de
platte "nz"-indexlijst via een geneste Python-lus (`for p in plist: for c
in range(16): if bit gezet: append(...)`) — puur CPU-overhead, geen
GPU-semantiek. Vervangen door numpy-bitvectorisatie (`(mask[:,None] >>
np.arange(16)) & 1` + booleaanse indexering, exact dezelfde rij-major-
volgorde als de geneste lus, dus geen wijziging aan WAT berekend wordt).
**Bitexact bevestigd (Phase B opnieuw PASS), timing 10,72 → 11,12 tok/s**
— een kleine maar reële verdere winst, puur uit minder CPU-side Python-
overhead.

**Eindstand van deze hele optimalisatieronde: 2,655 → 11,12 tok/s, 4,19×
sneller dan de eerste werkende versie, bitexact geverifieerd bij elke
stap.** Nog steeds 2,82× trager dan de naive baseline (31,411) — het
resterende gat blijft, zoals hierboven vastgesteld, grotendeels
PCIe-bandbreedte-gebonden (gather+up_proj-fetch) en dus niet verder te
sluiten met dezelfde klasse optimalisatie (launch-batching,
Python-vectorisatie). Een volgende, wezenlijk ander soort hefboom
(stream-overlap van PCIe-transfers met rekenwerk, CUDA-graph-residentie
voor de multi-sequentie-lus) is niet in deze sessie geprobeerd — een
reële, afgebakende vervolgtaak.

**Artefacten.** `pro_research/proto_multi_seq_moe_shared.py` (definitieve
versie, gather+masked/reduce/accumulate gebatcht + numpy-gevectoriseerde
maskerberekening), `pro_research/proto_multi_seq_moe_shared.json`
(eindmeting), `pro_research/proto_multi_seq_moe_shared_profile2.json`
(sectie-uitsplitsing na gather-batching, vóór de laatste vectorisatiestap).

**Robuustheidscontrole, zelfde dag: 12 stappen is klein voor deze sessie se
eigen maatstaf (V6's eigen record steunt op 765 samples).** `DECODE_STEPS`
verhoogd van 12 naar 40 (80 echte tokens totaal i.p.v. 24), zelfde script,
geen andere wijziging. **Bitexact, 40/40 tokens × 2 sequenties, PASS.
Timing: 11,234 tok/s** — binnen ~1% van de eerdere 12-staps-meting (11,12),
dus geen ruis-artefact, het cijfer is representatief.

---

## 2026-08-16 — EERSTE ECHTE END-TO-END METING: N=2, volledig model, meerdere stappen, geverifieerd bitexact — +5,4% aggregate, nog vóór expliciete deel-logica

**Waarom dit anders is dan alles hiervoor in de batch>1-lijn.** Elke
batch>1-meting tot nu toe was óf één MoE-laag (`proto_batch_moe_layer*.py`),
óf een geïsoleerde kernel-schalingstest, óf een read-only routing-diagnose.
Nooit het **echte, volledige 52-lagen model**, meerdere **echte** decode-
stappen, met een **echt gemeten** aggregate tok/s-getal. Dit sluit die
laatste, belangrijkste kloof — precies wat via de Stop-hook herhaaldelijk
als ontbrekend werd aangewezen.

**Mechanisme (geen productie-code aangepast).** `LightningRuntime.step()`
werkt uitsluitend via `self.X`-instantie-attributen; niets anders. Dus: vang
alle ~30 per-sequentie DYNAMISCHE buffers die `_alloc_state()` alloceert
(door die methode gewoon N keer echt aan te roepen en het resultaat te
snapshotten via `getattr` — geen handmatige herimplementatie van 30
buffervormen, dat zou transcriptierisico geven) in een per-sequentie
dict, dan met plain `setattr` wisselen vóór een aanroep van de
**ongewijzigde, echte** `rt.step()`. De GEWICHTEN (`rt.layer`, `rt.bank`,
`rt.fused`, `rt.k`) en de MoE-device-cache (`rt.cache`/`rt._dev_cache`)
blijven gewoon gedeeld (niet gewisseld) — precies het onderscheid dat
`BATCH_ARCHITECTURE_DESIGN.md` stap 1 als het fundamentele verschil
benoemde tussen dynamische en statische buffers.

**Een echte bug gevonden en gefixt vóórdat er iets gemeten werd:** `pos` is
een plain Python int; `step()` doet `self.pos += 1`, wat `rt.pos` REBINDT
naar een nieuw int-object in plaats van een bestaand object in place te
muteren (in tegenstelling tot elke `cp.ndarray`-buffer in de lijst, die wél
correct gealiased blijft via kernel-schrijvers). Zonder een expliciete
`state[s]["pos"] = rt.pos`-terugschrijving na elke stap zou wisselen-weg-en-
terug de positie van een sequentie stilzwijgend resetten — de KV-cache-
leesoffset corrumperend. Gevonden door **na te denken over welke attributen
gereassigned in plaats van gemuteerd worden**, niet door een falende test —
dezelfde discipline als de eerdere `bank["globals"]`-indexeringsbugs.

**Correctheidspoort (verplicht vóór elke tijdclaim).** Niet alleen één
ongebroken sequentie getest (had de `pos`-bug niet gevangen) — de
EXACTE interleaving-patroon van fase 3 zelf: N=2 sequenties, wissel-stap-
wissel-stap, elk vergeleken tegen zijn EIGEN onafhankelijke, ongewisselde
`rt.reset()`-gebaseerde controlerun. **Resultaat: bitexact, 15/15 tokens
per sequentie, beide sequenties, onder volledige interleaving.**
`pro_research/proto_multi_seq_full_model.py`.

**Uitkomst — het eerste echte getal.** Eén variabele (N), zelfde
configuratie (device-cache eager, `contexts_max=4096`, cap=72, GEEN graph,
GEEN selectieve ERVF, GEEN gebatchte kernels — bewust de kale E1-fase-2.1-
laag, niet V6, om een schone controle te houden):

| | ms/token | tok/s |
|---|---:|---:|
| N=1 (solo, zelfde configuratie/code-pad) | 33,559 | 29,798 |
| N=2 (naive, GEEN expliciete deel-logica) | 31,836 (aggregate) | **31,411 (aggregate)** |

**Aggregate speedup: 1,054× (+5,4%)** — reëel, positief, ondanks dat dit de
"naive"-arm is (geen unie-gevoede `cache_assign`, geen gedeelde fetch zoals
`proto_batch_moe_layer_combined.py` bewees). De winst komt uitsluitend van
**incidenteel** warme-cache-hergebruik: `self.cache`/`self._dev_cache`
worden niet gewisseld, dus sequentie B's stap profiteert toevallig van
experts die sequentie A's vorige stap al warm gemaakt heeft — hetzelfde
mechanisme als `diag_batch_warm_cache.py`, nu voor het eerst binnen een
echte staplus gemeten in plaats van als geïsoleerde LRU-simulatie.

**Wat dit sluit of opent.** Sluit de vraag of het hele idee praktisch
haalbaar is als state-management-mechanisme: **ja**, bitexact, geen enkele
gemiste buffer, geen enkele gemiste positie-desynchronisatie. Opent de
directe vervolgstap (nog niet gedaan): dezelfde verificatiediscipline
toepassen op een staplus die de **expliciete** unie-gevoede MoE-deling van
`proto_batch_moe_layer_combined.py` integreert in plaats van alleen
incidenteel warm-cache-hergebruik — zou de winst voorbij 5,4% moeten
brengen, gebaseerd op wat elke eerdere geïsoleerde meting al liet zien.

**Poorten.** Geen PRO-poorten (scoped integratieprototype). Correctheid:
bitexact onder interleaving, expliciete controle-arm (solo N=1, zelfde
configuratie). Geen tok/s-claim voorbij wat hierboven staat — dit is GEEN
V6-vergelijkbaar getal (andere, kalere configuratie, met opzet, voor een
schone N=1-vs-N=2-vergelijking).

**Artefacten.** `pro_research/proto_multi_seq_full_model.py`,
`pro_research/proto_multi_seq_full_model.json`.

---

## 2026-08-16 — Oorzaak van de supra-lineaire straf gevonden: reëel, groot klokverval onder aanhoudende belasting (36%)

**Vraag.** De synthese hierboven liet de oorzaak van de Mamba/lm_head-straf
expliciet open ("vermoedelijk klok-/stroomthrottling of geheugencontentie,
oorzaak niet vastgesteld"). Is er echt klokverval op deze hardware onder
aanhoudende belasting, groot genoeg om de gemeten straffen te verklaren?

**Opzet.** `pro_research/diag_lmhead_throttle_check.py`. `nvidia-smi
--query-gpu=clocks.sm,power.draw,temperature.gpu,pstate -lms 200` als
achtergrondproces gestart, dan **6 seconden aanhoudend** lm_head-werk
gedraaid (272 opeenvolgende batches van N=16, geen pauzes) — veel langer dan
één enkele schalingstest-ronde, met opzet, om het klokgedrag ondubbelzinnig
zichtbaar te maken. 31 nvidia-smi-samples gevangen tijdens de run.

**Uitkomst — reëel en groot.** SM-klok: **2685 MHz** (eerste ~0,6 s) →
daalt binnen ~1 seconde naar **~1642-1665 MHz** → stabiliseert op
**~1710-1717 MHz** voor de resterende ~5 seconden. Eerste-helft-gemiddelde
1887,9 MHz vs. tweede-helft-gemiddelde 1714,4 MHz — **een daling van ~174
MHz tussen de twee helften, en ~970 MHz (36%) tussen de piek en het
stabiele niveau.** Temperatuur steeg gestaag (70°C→73°C) terwijl
stroomverbruik nagenoeg vlak bleef (~55-60 W) en `pstate` steeds `P4` bleef
— consistent met een boost-klok die afbouwt naar een houdbaar duurzaam
niveau, niet met een pstate-overgang of een stroomlimiet-crisis.

**Wat dit vaststelt, en wat niet.** Bevestigt **direct en fysiek gemeten**
dat klokverval onder aanhoudende belasting een reëel, groot fenomeen is op
deze hardware — groot genoeg (36%) om de eerder gemeten 15-24%-tijdstraffen
voor Mamba/lm_head volledig te kunnen verklaren. **Niet vastgesteld:** of de
klokstaat exact hetzelfde was tijdens de oorspronkelijke vier
schalingstests zelf (die deden geen gelijktijdige nvidia-smi-polling) — dit
is dus sterk ondersteunend bewijs voor het throttling-mechanisme, geen
regel-voor-regel reconstructie van elke eerdere meting. Geheugencontentie
als aanvullende factor is hiermee niet uitgesloten, maar throttling alleen
al is aannemelijk voldoende om de waargenomen orde van grootte te verklaren.

**Belangrijke vervolgcorrectie, zelfde dag: het raakt de roofline NIET.**
Script uitgebreid om ook `clocks.mem` te pollen (niet alleen `clocks.sm`) —
cruciaal, want dit hele onderzoeksproject definieert de roofline als
**geheugenbandbreedte-gebonden** (338,4 GB/s streaming-leesbandbreedte, N5).
Herhaalde meting: `clocks.sm` daalt opnieuw substantieel (eerste-helft-
gemiddelde 1827,7 → tweede-helft-gemiddelde 1702,6 MHz), maar **`clocks.mem`
blijft exact 9001 MHz, geen enkele afwijking, alle 31 samples** — geen
geheugenklok-throttling. **Dit betekent dat de kern-roofline (338,4 GB/s /
165 tok/s ctx0) niet bedreigd wordt door dit fenomeen** — de SM-klok-daling
verklaart de reken-gebonden kernel-tijdstraffen (Mamba se in_proj, lm_head se
GEMV — beide doen echt rekenwerk, niet puur streamen) zonder de
geheugenbandbreedte-gebonden plafondaanname aan te tasten. Geruststellend
voor het hele sessie-kader: V6's eigen 47,41 tok/s-record en de
165 tok/s-roofline zelf blijven staan zoals gemeten.

**Wat dit wél opent.** Reken-gebonden kernels (lm_head, Mamba, en
vermoedelijk andere zwaar-compute-onderdelen) ondervinden een reële
duurzame-klok-straf die niet in eerdere kortlopende metingen zichtbaar was.
Voor toekomstig werk aan zulke kernels specifiek (niet de PCIe/HBM-
streaming-hefbomen die dit hele project tot nu toe domineerden) is dit
relevant; voor de PCIe-streaming-gedreven hefbomen (device-cache-fetch,
batch>1-deling) niet, want die zijn bandbreedte- niet compute-gebonden.

**Poorten.** Geen PRO-poorten (read-only root-cause-diagnostiek).

**Artefacten.** `pro_research/diag_lmhead_throttle_check.py`,
`pro_research/diag_lmhead_throttle_check.json`.

---

## 2026-08-16 — Synthese van de vier N-schalingstests: het patroon is niet "Mamba/lm_head zijn uitzonderingen", het is "duurdere kernels schalen slechter"

**Waarom dit een aparte vermelding verdient.** Vier losse metingen deze
sessie (`diag_attention_n_scaling.py`, `diag_mamba_n_scaling.py`,
`diag_shared_expert_n_scaling.py`, `diag_lmhead_n_scaling.py`) testten elk
één component apart. Naast elkaar gelegd (geen nieuwe GPU-run — puur
hergebruik van de vier bestaande JSON-resultaten) ontstaat een patroon dat
in geen van de vier losse write-ups genoemd werd:

| component | ms/aanroep bij N=1 | verhouding vs. ideaal-lineair @N=8 | @N=16 |
|---|---:|---:|---:|
| shared-expert | 0,0378 | 0,848 | 0,912 |
| attentie | 0,0961 | 0,958 | 0,970 |
| Mamba in_proj | 0,1767 | 1,164 | 1,148 |
| lm_head | 1,1537 | 1,208 | 1,194 |

**Monotoon: hoe duurder de kernel per aanroep, hoe slechter (meer
supra-lineair) de schaling bij herhaalde back-to-back-aanroepen.** De twee
goedkope/snelle kernels (shared-expert, attentie) schalen vlak of zelfs
iets ONDER ideaal-lineair (vaste per-launch-overhead die relatief kleiner
wordt bij grotere N); de twee duurdere kernels (Mamba, lm_head) schalen
merkbaar SLECHTER naarmate ze duurder zijn. Dit suggereert dat het probleem
niet Mamba- of lm_head-specifiek is (niet de FP8-tensor-kernel, niet de
vocab-vorm) maar een **algemene eigenschap van herhaalde back-to-back
GPU-kernellaunches** op deze hardware — vermoedelijk klok-/stroomthrottling
onder aanhoudende belasting of geheugencontrole-contentie die groter wordt
naarmate de working set van de kernel groter is. **Oorzaak niet
vastgesteld** (geen in-run klokmeting gedaan, alleen een snapshot ná afloop)
— dit is een correlatie, geen bewezen mechanisme.

**Wat dit sluit of opent.** Herkadreert de eerdere Mamba- en lm_head-
correcties: het zijn geen twee losstaande, toevallige uitzonderingen op een
verder betrouwbare "niet-gedeelde componenten schalen lineair"-aanname — het
is één onderliggend patroon dat zich toevallig het sterkst manifesteert bij
de twee duurste kernels. Voor een toekomstige batch>1-integratie betekent dit
dat de "rest-profiteert-niet-mee"-correctie waarschijnlijk **breder van
toepassing is** dan alleen Mamba/lm_head — elke kernel die duur genoeg is,
verdient dezelfde check vóór hij als "triviaal lineair" wordt aangenomen.
Opent een niet-uitgevoerde vervolgvraag: is dit klok-throttling (meetbaar
via `nvidia-smi --query-gpu=clocks.sm --loop-ms=...` tijdens een run i.p.v.
één snapshot erna) of geheugencontentie (meetbaar via Nsight Compute)? Geen
van beide gedaan deze sessie — puur een analytische synthese van bestaande
metingen, geen nieuwe GPU-tijd gebruikt.

**Poorten.** N.v.t. — synthese-observatie, geen nieuwe meting, geen
tok/s-claim.

**Artefacten.** Geen nieuwe (hergebruikt `diag_attention_n_scaling.json`,
`diag_mamba_n_scaling.json`, `diag_shared_expert_n_scaling.json`,
`diag_lmhead_n_scaling.json`).

---

## 2026-08-16 — lm_head-schaling: een NIEUW, nog nooit genoemd risico — grotere straf dan Mamba, op de duurste GEMV van het hele model

**Vraag.** `BATCH_ARCHITECTURE_DESIGN.md` noemt lm_head nergens expliciet als
risico-item (alleen attentie en Mamba worden apart behandeld in stap 7). Maar
lm_head is, net als attentie/Mamba/shared-expert, **niet expert-geselecteerd**
(hetzelfde gewicht voor elke sequentie) — dus zou volgens dezelfde redenering
~lineair moeten schalen. Gegeven dat de Mamba-aanname eerder al fout bleek en
lm_head de **grootste enkele GEMV van het hele model** is (output=vocab,
131.072 — de duurste stap per token, 1,15 ms bij N=1, tegenover Mamba se
0,177 ms en shared-expert se 0,038 ms), verdiende dit een check, ook al stond
het niet op de risicolijst.

**Opzet.** `pro_research/diag_lmhead_n_scaling.py` — zelfde methode als de
Mamba/shared-expert-schalingstests: de bestaande lm_head-GEMV (`fused.
gemv_into`, geen nieuwe kernel) N keer sequentieel gedraaid tegen N echte
gevangen post-`norm_f`-activaties (gevangen door `fused.gemv_into` tijdelijk
te wrappen en de aanroep met `rows == rt.vocab` te herkennen — lm_head se
eigen unieke signatuur, anders dan elke andere `gemv_into`-aanroep in het
model), N ∈ {1,2,4,8,16}, 30 herhalingen.

**Uitkomst — een reële, grotere straf dan Mamba.** ms/sequentie: 1,1537
(N=1) → 1,4333 (N=2) → 1,4222 (N=4) → 1,3934 (N=8) → 1,3776 (N=16) ms.
Verhouding gemeten/ideaal-lineair: **1,242 / 1,233 / 1,208 / 1,194** —
stabiliseert rond **+19 à +24% boven ideaal-lineair**, consistent over alle
vier N-waarden (lage spreiding in de percentielen, geen ruis-artefact).
**Groter dan Mamba se ~15% straf bij N=8-16**, en dit op de **duurste**
GEMV in het model — in absolute tijd betekent dit ~0,22-0,28 ms extra per
sequentie bij grotere N, wat zwaarder weegt dan Mamba se eigen absolute
straf (Mamba se GEMV is 6,5× goedkoper per aanroep).

**Wat dit sluit of opent.** Een **nieuw** risico, niet eerder benoemd: lm_head
hoort net als Mamba bij "niet-gedeeld en toch duurder per sequentie bij
grotere N", niet bij "vlak zoals attentie/shared-expert" — de aanname dat
alleen Mamba een uitzondering was, klopt dus niet meer. Dit maakt de
al-eerder-gecorrigeerde ~114 tok/s-bovengrensrekensom in dit document nóg
iets optimistischer dan de Mamba-correctie alleen al aangaf (geen nieuw
getal herberekend — zou weer aanname-op-aanname zijn — maar de richting is
eenduidig: nog iets lager dan wat de Mamba-correctie alleen impliceerde).
Praktisch: "lm_head+shared-expert" (10,1% van het token) bestaat dus uit één
component die zich netjes gedraagt (shared-expert, net bevestigd) en één die
dat niet doet (lm_head, hier ontdekt) — de eerdere gecombineerde 10,1%-cijfer
verhulde dat onderscheid.

**Poorten.** Geen PRO-poorten (read-only diagnostiek).

**Artefacten.** `pro_research/diag_lmhead_n_scaling.py`,
`pro_research/diag_lmhead_n_scaling.json`.

---

## 2026-08-16 — Shared-expert-schaling: bevestigt de "triviaal"-aanname dit keer wél (in tegenstelling tot Mamba)

**Vraag.** `BATCH_ARCHITECTURE_DESIGN.md` stap 6 stelt dat de shared expert
"triviaal" is voor batch>1 (niet expert-geselecteerd, dus niets te
dedupliceren, gewoon `[N, hidden]` i.p.v. `[hidden]`) — maar dat was, net als
de oorspronkelijke Mamba-aanname, een claim bij analogie (met attentie),
nooit fysiek gemeten voor de shared expert specifiek. `diag_mamba_n_scaling.py`
bewees eerder dat zo'n analogie fout kan zijn (Mamba bleek mild
supra-lineair). Zelfde discipline hier toegepast.

**Opzet.** `pro_research/diag_shared_expert_n_scaling.py` — identieke
methode als `diag_mamba_n_scaling.py`/`diag_attention_n_scaling.py`: de
bestaande, ongewijzigde shared-expert-up_proj-GEMV (`fused.gemv_into`, geen
nieuwe kernel) N keer sequentieel gedraaid tegen N echte gevangen
`normed`-activaties, N ∈ {1,2,4,8,16}, 30 herhalingen, MoE-laag 24.

**Uitkomst.** ms/sequentie blijft nagenoeg vlak: 0,0378 (N=1) → 0,0347 (N=2)
→ 0,0327 (N=4) → 0,0321 (N=8) → 0,0345 (N=16) ms. Verhouding
gemeten/ideaal-lineair: 0,92 / 0,87 / 0,85 / 0,91 — **iets ONDER 1,0**, dus
zelfs licht efficiënter dan perfect lineair (vaste per-launch-overhead die
relatief kleiner wordt bij grotere N), geen straf zoals bij Mamba.

**Wat dit sluit of opent.** Bevestigt (in tegenstelling tot de Mamba-
correctie) de oorspronkelijke aanname: de shared expert schaalt inderdaad
~lineair, geen verborgen kost. Dit is het derde stuk van de token-tijd
(naast attentie) dat nu fysiek bevestigd is te schalen zoals aangenomen;
Mamba blijft de enige uitzondering. Geen wijziging nodig aan
`BATCH_ARCHITECTURE_DESIGN.md`'s claim voor dit onderdeel.

**Poorten.** Geen PRO-poorten (read-only diagnostiek).

**Artefacten.** `pro_research/diag_shared_expert_n_scaling.py`,
`pro_research/diag_shared_expert_n_scaling.json`.

---

## 2026-08-16 — Eerste gecombineerde meting: up_proj- en down_proj-deling tegelijk in één laag — en het geheel is minder dan de som der delen

**Vraag.** `proto_batch_moe_layer.py` (up_proj-deling) en
`proto_batch_down_proj.py` (down_proj-unie-van-maskers-deling) bewezen elk
apart correct en sneller — maar down_proj se meting deed zijn eigen
up_proj-stap nog **naïef** (elke (sequentie, expert)-paar haalde zijn eigen
up_proj-gewichten opnieuw op, met opzet, om het down_proj-mechanisme te
isoleren). Delen die elk apart winnen, hoeven niet automatisch samen te
winnen — een eerste **gecombineerde** meting was nog niet gedaan, dezelfde
soort stap als V4 (2 mechanismen samen) en V6 (5 mechanismen samen) al
namen voor batch=1.

**Opzet (één laag, N=8, echte productiekernels voor beide stadia tegelijk).**
`pro_research/proto_batch_moe_layer_combined.py`. NAIVE: per
(sequentie, expert)-paar een eigen `cache_fetch` (up_proj) + GEMV, dan
`panel_scan` + eigen `gather_down_sparse_ind`-fetch + `down_masked` +
`reduce_partials`. BATCHED: één gedeelde `cache_fetch` over de unie-experts
voor up_proj (zoals `proto_batch_moe_layer.py`), dan per-sequentie GEMV op de
gedeelde buffer, dan de **unie van nonzero-maskers** per expert (OR over de
sequenties die hem kozen, zoals `proto_batch_down_proj.py`) voor **één**
gedeelde down_proj-`gather`, met daarna elke sequentie se eigen
`down_masked`-aanroep (eigen maskers/plist, niet de unie) tegen die gedeelde
mirror. **Belangrijke methodologiecorrectie tijdens het bouwen**: de eerste
versie mat de NAIVE up_proj-arm alleen als GEMV-tijd (de host→device-fetch
zelf viel buiten het getimede venster), terwijl de BATCHED-arm fetch+GEMV
samen mat — een oneerlijke vergelijking die de batched-arm kunstmatig
trager deed lijken (0,888×, "verlies"). Gecorrigeerd door de NAIVE-arm
dezelfde productie-`cache_fetch`-kernel te laten gebruiken als de
BATCHED-arm, beide getimed inclusief fetch — exact dezelfde discipline als
elders deze sessie (eerlijke armen, één variabele).

**Uitkomst (na correctie, bitexact: 0/48 mismatches).**

| stadium | NAIVE (ms) | BATCHED (ms) |
|---|---:|---:|
| up_proj (fetch+GEMV) | 7,651 | 4,041 (fetch 2,591 + GEMV 1,450) |
| down_proj (fetch+masked-GEMV+reduce) | 7,931 | 8,849 (fetch 1,812 + GEMV 7,037) |
| **totaal** | **15,582** | **12,890** |

**Gecombineerde winst: 1,209× (+20,9%), 2,692 ms bespaard** — reëel, maar
**kleiner dan de afzonderlijke metingen deden vermoeden** (up_proj alleen
eerder tot 2,89× bij N=16; down_proj alleen +2,56%/1,91×-fetch bij ander N).
**Opmerkelijk: down_proj se GEMV-stadium werd LANGZAMER in de gecombineerde
meting** (7,037 ms batched vs. impliciet minder in de naive-som), ondanks
dat de fetch zelf sneller werd (1,812 vs. een groter naïef equivalent) en de
FLOP's per sequentie **identiek** blijven (elke sequentie rekent nog steeds
alleen over haar eigen maskers/plist, niet de unie). Meest aannemelijke
verklaring: de gedeelde mirror-buffer is groter (unie-grootte in plaats van
één sequentie se eigen subset), wat de geheugenlocaliteit voor de
`down_masked`-kernel verslechtert zelfs als het rekenwerk gelijk blijft — een
reëel, niet eerder gedocumenteerd interactie-effect tussen de twee
deel-mechanismen, dat pas zichtbaar wordt als ze **samen** draaien.

**Wat dit sluit of opent.** Bevestigt dat het combineren van twee apart
bewezen mechanismen **niet gratis** is — precies de reden waarom deze sessie
`proto_batch_moe_layer_combined.py` bouwde in plaats van de twee losse
cijfers simpelweg te vermenigvuldigen (wat de werkregel toch al verbiedt).
Het netto-effect blijft positief en bitexact, maar een toekomstige
volledige-integratieschatting moet met dit gecombineerde getal rekenen
(1,209×), niet met de afzonderlijke up_proj/down_proj-cijfers los. Opent een
kleine, scherp afgebakende vervolgvraag (niet gedaan): is de
mirror-locatie-straf te verzachten door de mirror te sorteren op
paneel-volgorde in plaats van unie-invoegvolgorde? Puur een
prestatie-optimalisatie, geen correctheidsvraag.

**Poorten.** Geen PRO-poorten (scoped feasibility-prototype, geen
runtime-wijziging, geen tok/s-claim). Correctheid: bitexact, 0/48 mismatches.

**Artefacten.** `pro_research/proto_batch_moe_layer_combined.py`,
`pro_research/proto_batch_moe_layer_combined.json`.

---

## 2026-08-16 — VRAM-kost per extra sequentie: eindelijk een echt getal, en het is niet wat het risico-document impliceerde

**Vraag.** `BATCH_ARCHITECTURE_DESIGN.md` se risico #4 noemde VRAM als
mogelijke blokkade voor batch>1 ("N-voudige KV-cache/SSM-state kost VRAM die
er niet is... een nieuwe afweging, niet gemeten") maar had nooit een
werkelijk getal — alleen de constatering dat de GPU tijdens V4 al op 0 MiB
vrij stond. Wat kost één EXTRA sequentie nou precies, en hoeveel ruimte is
er echt, op welk punt in de stack?

**Opzet.** `pro_research/diag_batch_vram_cost.py`. Twee delen, geen kernels
gebouwd: (1) host-side aritmetiek die exact `runtime.py`'s eigen
`_alloc_state`-formules natrekt voor de twee buffer-klassen die volgens het
ontwerpdocument een batch-dimensie nodig hebben zonder deel-mogelijkheid
(KV-cache, FP8, per attentielaag; Mamba ssm+conv-state, FP32, per
Mamba-laag) — dit levert de **exacte** bytekost per extra sequentie, geen
schatting. (2) één echte `nvidia-smi`-meting bij het eager+device-cache-
bedrijfspunt (`contexts_max=4096`, `cache_capacity=72`, geen graph-capture)
— hetzelfde bedrijfspunt dat elk `diag_*`/`proto_batch_*`-script deze sessie
gebruikt.

**Uitkomst — de kost per extra sequentie.**

| component | lagen | bytes/laag | totaal |
|---|---:|---:|---:|
| KV-cache (FP8, K+V) | 6 attentielagen | 2.097.152 B | 12.582.912 B (12,0 MiB) |
| Mamba ssm+conv-state (FP32) | 23 Mamba-lagen | 2.195.456 B | 50.495.488 B (48,2 MiB) |
| **totaal per extra sequentie** | | | **63.078.400 B ≈ 60,16 MiB** |

**Verrassend deel: Mamba-state domineert, niet KV-cache** (48,2 MiB vs
12,0 MiB) — het omgekeerde van de gebruikelijke transformer-intuïtie. Dit
model heeft slechts 6 van 52 lagen volledige attentie (de rest Mamba/MoE),
dus de KV-cache-kost is klein; elke Mamba-laag draagt zijn eigen volle
ssm+conv-state, en er zijn er 23.

**Uitkomst — hoeveel past er echt.** Bij het eager+device-cache-bedrijfspunt:
**1.771 MiB vrij van 8.151 MiB totaal** (6.380 MiB in gebruik na model +
cache-laden, vóór graph-capture). Tegen 60,16 MiB/sequentie: **ruimte voor
29 extra sequenties (N tot 30)** zonder iets te verlagen. Bij het volledige
V6-bedrijfspunt (mét graph-capture) staat er, uit een eerdere meting deze
sessie (V4/V6-preregistratie, `RESEARCH_NOTEBOOK.md`/`TODO.md`, "0 MiB vrij
tijdens V4"), **0 MiB vrij** — dus daar past geen enkele extra sequentie
zonder de graph-capture-overhead of cache-capaciteit te verlagen.

**Wat dit sluit of opent.** Herkadreert risico #4 fundamenteel: het probleem
is **niet** dat N-voudige KV/Mamba-state duur is (60 MiB/sequentie is klein
— 29 sequenties zouden passen buiten de graph om) — het probleem is dat
**CUDA-graph-capture zelf** al het budget opeet, ver vóór batch>1 er ook nog
iets bij zou vragen. Dat betekent: een batch>1-integratie zonder
graph-residentie (eager-modus, zoals V1-V3 vóór graph-safe-residency) zou
ruim VRAM-budget hebben voor realistische N; een integratie MET
graph-residentie (V4-V6's eigen winst) zou eerst de graph-capture-kost zelf
moeten verlagen (kleinere `contexts_max`, lagere cache-capaciteit) vóór er
ook maar N=2 bij kan. Dat is een reële afweging tussen twee reeds bewezen
hefbomen (graph-residentie vs. batch>1) die nooit eerder zo expliciet
gekwantificeerd was.

**Poorten.** Geen PRO-poorten (read-only diagnostiek/aritmetiek, geen
runtime-wijziging, geen tok/s-claim). De "0 MiB vrij bij volledige graph"-
constatering is een eerder-gemeten feit (V4-preregistratie), hier
hergebruikt, niet in dit script opnieuw geverifieerd.

**Artefacten.** `pro_research/diag_batch_vram_cost.py`,
`pro_research/diag_batch_vram_cost.json`.

---

## 2026-08-16 — Staggered posities: overleeft de expert-unie continuous batching, of was "alle N op dezelfde stap" een gunstige aanname?

**Vraag.** Elke batch>1-meting tot nu toe (`diag_cross_sequence_union.py`,
`diag_batch_warm_cache.py`, beide `proto_batch_*.py`) vergeleek N sequenties
op **dezelfde stap-index** — met opzet, als eerste isolatie. Maar een echte
continuous-batching serving-runtime heeft sequenties op **onafhankelijke,
willekeurige posities** (sequentie A genereert token 40, sequentie B token 5,
tegelijk op dezelfde wall-clock-batchtick). Risico #3 in
`BATCH_ARCHITECTURE_DESIGN.md` vroeg zich expliciet af of dat de unie-
overlap zou vergroten (minder deling) omdat inhoudelijk-ongerelateerde
generatiediepten misschien breder uiteen routeren.

**Opzet (één variabele: lockstep vs. staggered posities, zelfde sequenties/
lagen/T).** `pro_research/diag_staggered_position_union.py`. N=4 sequenties
(dezelfde 4 diverse prompts), MoE-laag 24, elk **T+max_offset=53 echte**
stappen gevangen via `_route_device`. Vaste, deterministische offsets
`[0, 7, 15, 23]` (niet willekeurig — reproduceerbaar). Voor elke van T=30
wall-clock-ticks: LOCKSTEP-view = elke sequentie se stap `t`; STAGGERED-view
= sequentie `s` se stap `offset_s + t` — beide views komen uit **dezelfde**
onderliggende echte traject-data, dus het verschil isoleert precies de
staggering, niets anders.

**Uitkomst.**

| | gem. unie | % van max (N×top_k=24) |
|---|---:|---:|
| LOCKSTEP (zelfde stap-index) | 21,47 | 89,4% |
| STAGGERED (offsets 0/7/15/23) | 21,93 | 91,4% |

Verschil: **+1,9 procentpunt** grotere unie (dus iets minder overlap/deling)
onder staggering — een reële maar kleine verslechtering, geen ineenstorting
van het mechanisme. Ter vergelijking: `diag_cross_sequence_union.py` mat
eerder voor N=4, andere prompts/laag, 90,3% — dezelfde orde van grootte als
beide cijfers hier, een consistentiecheck dat dit geen toevalstreffer is.

**Wat dit sluit of opent.** Sluit risico #3 uit `BATCH_ARCHITECTURE_DESIGN.md`
voor de **routing-overlap-dimensie**: staggered posities (continuous batching)
vernietigen het deel-potentieel niet, ze verzwakken het licht (~2 procentpunt).
De volledige runtime-integratie (routing-unie ingebed in een staplus die
zelf N onafhankelijke posities bijhoudt) blijft ongebouwd — dit sluit alleen
de aanname dat lockstep-metingen een kunstmatig gunstig scenario waren, niet
de bouwvraag zelf.

**Poorten.** Geen PRO-poorten (read-only diagnostiek, geen runtime-wijziging,
geen tok/s-claim).

**Artefacten.** `pro_research/diag_staggered_position_union.py`,
`pro_research/diag_staggered_position_union.json`.

---

## 2026-08-16 — Warme-cache-dynamiek: houdt de fetch-deling stand over meerdere stappen, of was het een cold-cache-artefact?

**Vraag.** Alle batch>1-prototypes tot nu toe (`proto_batch_moe_layer.py`,
`proto_batch_moe_multilayer.py`, `proto_batch_down_proj.py`) maten een
**enkele cold-cache-snapshot** — met opzet, om het deel-effect te isoleren
van LRU-hitrate-dynamiek. Dat laat een open vraag: zodra de device-LRU-cache
al warm is (het normale geval in een lopende serving-runtime, niet de eerste
stap), profiteert elke sequentie al individueel van temporele lokaliteit
(dezelfde experts blijven vaak meerdere stappen relevant) — verdwijnt het
cross-sequentie-deel-voordeel dan grotendeels, omdat er toch al weinig
missers overblijven?

**Opzet (één variabele: gedeelde vs. onafhankelijke cache, zelfde stappen).**
`pro_research/diag_batch_warm_cache.py`. N=4 sequenties (uiteenlopende
prompts, zelfde 8 als elders), T=40 opeenvolgende **echte** stappen per
sequentie op MoE-laag 24, top_k=6 routes gevangen via `_route_device`
(identieke vangmethode als de cold-cache-scripts). Gebruikt de **echte**
productie-kernel `cache_assign`/`alloc_device_cache` (geen herimplementatie)
— dus de LRU-semantiek (tick-gebaseerde eviction, hit/miss-telling via
`need[]`) is exact wat productie zou doen.

- **GEDEELD**: één cache (cap=72, zelfde budget als productie-default),
  gevoed per stap met de **unie** van alle 4 sequenties se ids (24 ids/stap,
  ongededupliceerd — `cache_assign`'s kernel dedupliceert zelf correct
  binnen één aanroep omdat het sequentieel over de lijst loopt).
- **NAIVE**: 4 **onafhankelijke** caches, elk cap=72 (exact wat 4 losse
  batch=1-runtime-instanties vandaag zouden hebben — geen gereduceerd
  per-sequentie-aandeel), elk over dezelfde 40 stappen gevoed met zijn eigen
  sequentie se ids.

**Uitkomst (getallen, geen extrapolatie).**

| | totaal missers (960 aanroepen) | hitrate | laatste kwart (stap 30-40) |
|---|---:|---:|---:|
| GEDEELD | 142 | 85,2% | 18 missers |
| NAIVE (4×) | 196 | 79,6% | 25 missers |

Missers-reductie: **27,6%** over de volle 40 stappen, **28,0%** in het
laatste kwart (stap 30-40, het meest representatieve "warm steady-state"-
venster). Het voordeel **verdwijnt niet** naarmate de cache warmt — het
blijft nagenoeg constant procentueel, ook al dalen de absolute missers voor
beide armen sterk (cold start ~22-24 missers/stap → steady state ~1-3
missers/stap).

**Waarom kleiner dan de cold-cache-unie-cijfers hierboven (bijv. 90,3% van
no-overlap bij N=4 daar).** Dat eerdere cijfer was de unie-fractie van een
**enkele cold-cache-stap** — hier concurreert cross-sequentie-deling met
**temporele lokaliteit binnen elke sequentie zelf**, die de NAIVE-arm ook al
gratis krijgt zodra de cache warm is. Het gedeelde voordeel is dus reëel maar
kleiner dan de cold-cache-metingen deden vermoeden — een eerlijke correctie
van wat je zou verwachten als je cold-cache-cijfers naïef zou extrapoleren
naar een lopende runtime.

**Wat dit sluit of opent.** Sluit de vraag of het fetch-deel-mechanisme een
cold-cache-artefact is: **nee**, het houdt stand (27,6% minder missers,
stabiel in steady state) onder de echte productie-LRU-kernel. Opent geen
nieuwe bouwstap — dit blijft een geïsoleerde, read-only diagnostiek zoals de
rest van de batch>1-lijn (zie `BATCH_ARCHITECTURE_DESIGN.md`), maar bevestigt
dat de eerdere aanname ("het deel-voordeel is niet slechts een cold-start-
fenomeen") correct was, met een realistischer (lager, maar nog steeds
substantieel) getal dan de cold-cache-scripts alleen suggereerden. Ook
relevant: GEDEELD gebruikt 1×72-slot cache tegen NAIVE se 4×72-slot cache —
dus naast minder missers ook ~4× minder VRAM voor hetzelfde budget per
sequentie, een tweede, apart voordeel dat de cold-cache-scripts niet maten.

**Poorten.** Geen PRO-poorten (read-only diagnostiek, geen runtime-wijziging,
geen tok/s-claim). Geen VRAM- of correctheids-claim gedaan buiten wat direct
uit de echte kernel-aanroepen volgt.

**Artefacten.** `pro_research/diag_batch_warm_cache.py`,
`pro_research/diag_batch_warm_cache.json`.

---

## 2026-08-16 — Nieuwe, nog niet aangepakte hypothese: batch>1 zou de single-stream-roofline zelf kunnen doorbreken

**Waarom dit een andere categorie is dan alles hiervoor.** Alle metingen
deze sessie — V4 t/m V6, elke batched kernel, de capaciteitstuning — hebben
één ding gemeen: ze verlagen **tijd per byte** binnen een architectuur die
altijd **batch=1** aanneemt. Het 165 tok/s-roofline-plafond (Y-lijn) is zelf
berekend ONDER die aanname (bandbreedte-gebonden bij één sequentie). Geen
enkele hefboom die *binnen* batch=1 blijft kan boven dat plafond komen — dat
is precies waarom 100 tok/s bij batch=1 een factor 2,1× weg blijft ondanks
alle winst tot nu toe.

**De hypothese.** Als de runtime **N sequenties gelijktijdig** zou
verwerken (batch>1), zou het dure deel — expert-gewichten van host naar
device halen via PCIe — maar **één keer per uniek geselecteerde expert per
stap** hoeven te gebeuren, niet één keer per sequentie. Een expert die eenmaal
geladen is kan tegen N verschillende activatievectoren rekenen (analoog aan
hoe deze sessie's MoE-batching zes experts in één launch liet meerekenen in
plaats van zes losse launches — maar nu geamortiseerd over **sequenties**
i.p.v. over **experts binnen één sequentie**). Zolang de experts die N
sequenties gezamenlijk selecteren niet allemaal verschillend zijn, daalt de
PCIe-kost per nuttige token, en het aggregate tok/s-plafond zou boven de
165 tok/s single-stream-roofline kunnen uitkomen — dat is een **ander,
hoger plafond**, geen violatie van het bestaande.

**Bevestigd: dit bestaat nul in de huidige runtime.** Geen enkele buffer
heeft een batch-dimensie (`self.h = cp.zeros(self.hidden)`,
`self.normed`, `self.qv`, KV-cache, SSM/conv-state — allemaal 1D, puur
single-sequence). Dit is dus geen kleine uitbreiding maar een fundamenteel
andere architectuur: elke state-buffer, elke kernel (attentie, Mamba, MoE-
GEMV's), de graph-capture, en de device-LRU-cache-toewijzing zouden een
batch-dimensie moeten krijgen, en de router/cache-logica zou de **unie** van
experts over N sequenties moeten bepalen (dezelfde klasse berekening als
N7-A's opeenvolgende-token-overlap, maar nu over *sequenties* i.p.v. over
*tokens van dezelfde sequentie* — en die overlap is vermoedelijk veel lager,
want ongerelateerde content routeert vermoedelijk breder uiteen dan
opeenvolgende tokens van één tekst).

**Niet gebouwd, met opzet.** Dit is geen middag werk zoals de kernel-
batching hiervoor — het is een meerdere-weken-herontwerp van de hele
runtime. Een prototype forceren binnen deze sessie zou het risico lopen iets
half werkends achter te laten, wat tegen alles ingaat wat deze sessie
verder heeft gedaan (isoleer, verifieer, integreer pas als het klopt).

**Wat er WEL eerst gemeten zou moeten worden, vóór er gebouwd wordt** (zelfde
discipline als S10's eigen aanpak voor MTP): de **cross-sequentie expert-
unie** — laad N onafhankelijke prompts, laat de router voor elk zijn top-6
bepalen op hetzelfde tijdstip, en tel hoeveel unieke experts dat samen geeft
bij N=2, 4, 8, 16. Bij weinig overlap (unie dicht bij N×6) levert batchen
weinig op vergeleken met de complexiteit; bij aanzienlijke overlap (unie
duidelijk onder N×6) is er een reëel, mogelijk groot plafond-doorbrekend
potentieel. Dit is puur leesbaar uit bestaande route-logica
(`_route_device`/`capture_routes`, al gebruikt voor de MTP-route-unie-meting
eerder vandaag) — geen bouw nodig voor deze eerste meting.

**Deze meting is meteen ook gedaan** (`pro_research/diag_cross_sequence_union.py`):
16 inhoudelijk uiteenlopende prompts (geschiedenis, code, recept, fictie,
financieel verslag, biologie, recht, netwerkconfiguratie, archeologie,
muziek, aandelenmarkt, klimaat, schaken, OOP, archeologie, ML) — bewust
divers zodat overlap geen artefact is van gelijkaardige content — elk 20
stappen gestapt met `capture_routes`. Unie berekend over 30 willekeurige
deelverzamelingen per N, alle 23 MoE-lagen, alle 20 stappen.

| N | gem. unie | van max N×6 | % van no-overlap | amplificatie t.o.v. 1 token |
|---:|---:|---:|---:|---:|
| 2 | 11,58 | 12 | 96,5% | 1,93× |
| 4 | 21,67 | 24 | 90,3% | 3,61× |
| 8 | 38,90 | 48 | 81,0% | 6,48× |
| **16** | **63,90** | 96 | **66,6%** | **10,65×** |

**De overlap groeit met N, en wordt bij N=16 substantieel:** 96 experts nodig
zonder deling, maar gemiddeld maar 63,9 unieke experts om 16 sequenties elk
één token te geven — **33% minder unieke PCIe-gebonden expert-loads** voor
evenveel nuttige output.

**Waarom dit fundamenteel anders is dan MTP (dat wél gesloten werd).**
MTP's speculatieve drafts kostten een aparte MTP-forward (19,10 ms) die
soms weggegooid werd (bij verwerping) — de winst moest die kost eerst
terugverdienen, en deed dat niet. Bij batch>1 is **elke geproduceerde token
al opgevraagd** door een echte, andere sequentie — er is geen speculatieve
kost, geen weggegooid werk, geen "draft tax" om terug te verdienen. Elke
gedeelde expert is pure winst.

**Waarom dit de moeite waard is om vast te leggen, ook onaangepakt.** Dit is
de enige hypothese deze sessie die het **fundamentele plafond zelf** zou
kunnen verleggen in plaats van dichter naar het bestaande plafond toe te
werken, én de eerste, goedkope meting bevestigt reëel potentieel (geen
"weinig overlap, niet de moeite waard"-uitkomst). Alles binnen batch=1
nadert nu een asymptoot (V6 zit op 28,7% van 165; zelfs een perfecte
batch=1-implementatie zou nooit boven de 165 komen, en 100 vraagt 60,6%).
Wie hier verder aan werkt: de meting is klaar, de volgende stap is
architectuurontwerp (welke buffers krijgen een batch-dimensie, hoe wordt de
expert-unie per stap bepaald en experts geladen/gedeeld), niet meteen
kernels schrijven.

**Artefacten.** `pro_research/diag_cross_sequence_union.py` ·
`pro_research/diag_cross_sequence_union.json`.

### Vervolg, zelfde dag: het mechanisme fysiek getest, niet alleen geteld

De unie-telling hierboven is een projectie ("minder unieke experts nodig"),
geen meting van wat dat werkelijk kost. Om dat te onderscheiden van
speculatie is één laag van het mechanisme **geïsoleerd gebouwd en fysiek
getest** — niet de volledige batch>1-runtime (dat blijft een
meerdere-weken-taak), maar precies het stuk dat het meeste belooft: het
delen van de expert-**fetch** over sequenties.

**Opzet** (`pro_research/proto_batch_moe_layer.py`, cold-cache worst case —
bewust losgekoppeld van LRU-hitratedynamiek, die al apart bestudeerd is).
Eén echte laag (laag 24), N=16 echte sequenties (dezelfde 16 diverse
prompts), elk zijn eigen echte `normed`-activatie en top-6-routes gevangen
via `_route_device`. **NAIVE**: voor elke sequentie apart, elk van zijn 6
experts vers ophalen (96 fetches, geen deling — wat 16 losse batch=1-
runtimes vandaag zouden doen). **BATCHED**: alleen de unieke experts in de
unie ophalen (33 stuks — 65,6% deduplicatie, consistent met de eerdere
tellingsmeting), dan voor elke sequentie-expert-paar dezelfde productie-
ERVF-GEMV-kernel (`gemv_into`, ongewijzigd) draaien tegen de gedeelde
buffer.

**Correctheid: bitexact, 0 mismatches** tussen naive en batched over alle
96 sequentie-expert-paren — het delen van de fetch verandert niets aan wat
elke sequentie afzonderlijk berekent, zoals verwacht (routing en expert-
wiskunde zijn onafhankelijk van hoe de gewichten geladen worden).

**Timing, fysiek gemeten (`cp.cuda.Event`, dezelfde GPU, dezelfde sessie):**

| | fetches | totale tijd |
|---|---:|---:|
| NAIVE | 96 | 12,60 ms |
| **BATCHED** | 33 | **4,36 ms** |

**2,89× sneller, 8,25 ms bespaard — voor de fetch-fase van ÉÉN laag, voor
16 sequenties.** Dit is een reële, bitexact geverifieerde meting, geen
projectie.

**Claim-grens.** Dit bewijst het mechanisme voor één laag se fetch-fase; het
bewijst NIET de volledige batch>1-doorvoer (attentie, Mamba, KV-cache,
graph-capture en 22 andere lagen zijn niet meegenomen, en compute-tijd voor
de GEMV's zelf schaalt wél met N — die is hier niet apart gemeten). Simpele
optelling over 23 lagen zou een aanname zijn, geen meting — precies de fout
die werkregel 7 verbiedt. Wat dit wél vaststelt: het kernmechanisme is
**correct** en **fysiek sneller**, niet alleen theoretisch veelbelovend.

**Artefacten.** `pro_research/proto_batch_moe_layer.py` ·
`pro_research/proto_batch_moe_layer.json`.

### Vervolg: alle 23 lagen, en compute apart van fetch gemeten

Eén laag bewijst niet dat het patroon overal geldt, en de vorige meting
mat fetch+compute samen — als compute evenredig duurder zou worden bij
batchen, zou dat de fetch-winst deels opeten. Beide vragen nu apart
beantwoord (`pro_research/proto_batch_moe_multilayer.py`, zelfde opzet,
alle 23 MoE-lagen, N=16, cold-cache, fetch en compute elk apart getimed
met eigen `cp.cuda.Event`-paren).

**Correctheid: bitexact op alle 23 lagen, 0 mismatches totaal** — niet
alleen laag 24.

**Fetch-winst is overal aanwezig maar wisselt per laag** (1,42× tot 3,15×),
consistent met de eerder gemeten niet-uniforme missrate/lokaliteit per
laag (laag 24, 27, 13, 20 met hoge dedup — 60-66% — versus laag 1, 47, 51
met lagere dedup — 31-41%).

**Compute-tijd blijft vlak tussen naive en batched** (soms licht lager,
soms licht hoger in batched, binnen ruis — geen systematische straf voor
het batchen). Bevestigt de aanname uit de vorige meting: alleen de fetch
profiteert, compute schaalt gewoon met het aantal (sequentie,expert)-paren
ongeacht hoe de gewichten geladen zijn — logisch, want de GEMV zelf raakt
niet aan hoeveel keer een gewicht is opgehaald.

**Opgeteld over alle 23 gemeten lagen (niet geëxtrapoleerd):**

| | fetch+compute totaal |
|---|---:|
| NAIVE | 367,05 ms |
| **BATCHED** | **214,43 ms** |
| **Versnelling** | **1,71×** |

Dit is een preciezer, minder toevallig-gunstig getal dan de 2,89× van de
losse laag 24 (die zat aan de gunstige kant van de spreiding). **Claim-
grens, ongewijzigd streng:** dit dekt alleen de up_proj-GEMV+fetch,
opgeteld over de 23 daadwerkelijk gemeten lagen — nog steeds geen
down_proj, shared expert, attentie, Mamba, KV-cache, graph-capture of
routing/argmax/norm-overhead. Geen doorvoerclaim, wel een steviger
bevestiging dat het mechanisme consistent werkt, niet toevallig op één
laag.

**Artefacten.** `pro_research/proto_batch_moe_multilayer.py` ·
`pro_research/proto_batch_moe_multilayer.json`.

### Vervolg: down_proj — architecturaal anders dan up_proj, ook getest

Beide voorgaande metingen dekten alleen up_proj. down_proj is een **andere
soort deling**, niet dezelfde truc gekopieerd: up_proj's hele gewicht wordt
sowieso opgehaald ongeacht de activatie, dus delen is simpele deduplicatie
op expert-id. down_proj is **gemaskeerd/sparse** (`gather_down_sparse_ind`
haalt alleen de kolommen op die in de activatie na ReLU2 niet-nul zijn) —
twee sequenties die dezelfde expert kiezen kunnen alsnog **andere**
niet-nul-kolommen nodig hebben, dus simpelweg "dezelfde expert = deel de
fetch" zou fout zijn (een sequentie zou kolommen kunnen missen die niemand
voor haar ophaalde).

**De juiste generalisatie, gebouwd en getest**
(`pro_research/proto_batch_down_proj.py`): voor een expert die door meerdere
sequenties gekozen is, de **unie** van niet-nul-kolommen ophalen (OR van hun
`panel_masks`, een boven-verzameling — dus elke sequentie se eigen kolommen
zitten er gegarandeerd in), en daarna de gemaskeerde-som voor élke sequentie
met **haar eigen** `panel_masks`/`panel_list` draaien (niet de unie) tegen
die gedeelde mirror — de som raakt alleen die sequentie se eigen kolommen
aan, dus dit verandert welke bytes over PCIe gaan, nooit de berekende
waarde.

**Echte data, geen synthetische.** De post-ReLU2-activaties zijn echt
berekend (de echte up_proj-ERVF-GEMV gedraaid per (sequentie,expert)-paar,
met echte gewichten) — deze sessie leerde al eerder dat synthetische random
testdata onrepresentatieve randgevallen kan raken.

**Uitkomst (laag 24, N=16, 96 paren).** **Bitexact, 0 mismatches.** Fetch:
6,44 ms (naive, apart per paar) → **3,37 ms (unie-gedeeld), 1,91× sneller.**
Bytes over PCIe: 54,09 MB → 24,90 MB, **54,0% minder.** Kleiner dan
up_proj's dedup-fractie op dezelfde laag (65,6%) — logisch, want
verschillende sequenties se sparsity-patronen "verdunnen" de overlap
gedeeltelijk (de unie van twee gedeeltelijk-overlappende kolommenverzamelingen
is groter dan elke aparte verzameling) — maar nog altijd een reële, forse
winst, geen synthetisch artefact.

**Stand van het batch>1-mechanisme na deze sessie.** Beide helften van de
MoE-laag (up_proj: 1,71× opgeteld over 23 lagen; down_proj: 1,91× op één
laag, nog niet over alle 23 herhaald) zijn nu **bitexact bewezen correct en
fysiek sneller**, met de juiste — niet de simplistische — generalisatie
voor elk. Claim-grens ongewijzigd: dit is nog steeds geen doorvoerclaim,
en de volledige runtime-integratie (attentie, Mamba, KV-cache, graph-
capture, batch-dimensie op alle buffers) is niet gestart.

**Artefacten.** `pro_research/proto_batch_down_proj.py` ·
`pro_research/proto_batch_down_proj.json`.

### Architectuurontwerp: de brug tussen "mechanisme bewezen" en "runtime gebouwd"

Met beide helften van MoE's PCIe-kost fysiek bewezen deelbaar, is verdere
geïsoleerde prototyping van hetzelfde soort (nog een laag, nog een variant)
aan afnemende meeropbrengst toe. De echte volgende stap — al aangewezen in
`TODO.md` — is architectuurontwerp, geen code: `agents/BATCH_ARCHITECTURE_DESIGN.md`
zet dit uiteen (welke buffers een batch-dimensie nodig hebben, hoe de
routing-unie in de staplus zou komen, `cache_assign` voor N×top_k, graph-
implicaties voor continuous batching).

**De belangrijkste eerlijke waarschuwing daarin:** attentie, Mamba en
KV-cache hebben **geen** deel-mogelijkheid van dit soort — ze zijn niet
expert-geselecteerd, elke sequentie gebruikt toch al dezelfde gewichten, dus
er is niets te dedupliceren en de kost schaalt ~lineair met N. Aangezien MoE
"maar" 57,8% van het token is (componentafbraak, eerder vandaag), zal de
**aggregate** doorvoerwinst van batch>1 kleiner zijn dan de MoE-cijfers
alleen suggereren. Het document geeft een expliciet-als-rekensom-niet-
meting gelabelde grove bovengrens (~114 tok/s aggregate bij aanname van
volledige MoE-deling en ongewijzigde rest) — bewust net zo behandeld als
S10's MTP-voorcalculatie, die achteraf te optimistisch bleek. Aanbevolen
eerstvolgende meting: attentie-GEMV's voor N=2 batchen (geen deel-mechanisme
nodig) en checken of de kost echt lineair schaalt.

**Artefacten.** `agents/BATCH_ARCHITECTURE_DESIGN.md`.

### De aanbevolen meting gedaan: attentie schaalt inderdaad ~lineair met N

Het ontwerpdocument beval als eerstvolgende, geen-bouw-nodig-meting aan: klopt
de aanname dat attentie ~lineair schaalt met N (geen deel-mogelijkheid, dus
geen MoE-achtige winst te verwachten)? `pro_research/diag_attention_n_scaling.py`
— de bestaande, ongewijzigde productie-Q-projectie-GEMV (`rt.k.mv_bf16`, al
ERVF-gedispatcht zoals in V4/V6) N keer na elkaar gedraaid tegen N echte
gevangen activaties, N ∈ {1,2,4,8,16}, 30 ronden per N.

**Uitkomst: vrijwel perfect lineair.** ms/sequentie blijft nagenoeg constant
(0,091-0,096 ms) over de hele N-reeks; gemeten tijd is 94-97% van het ideale
lineaire model (N×N=1-kost) — dus **geen noemenswaardige launch-overhead-
speling** zoals MoE's `panel_scan`/`reduce_partials` die wél hadden. Dit
bevestigt het ontwerpdocument se belangrijkste waarschuwing rechtstreeks:
attentie heeft geen vergelijkbare batch>1-hefboom, en de conservatieve
aanname in de grove ~114 tok/s-bovengrensrekensom (attentie/Mamba/rest
blijven ongewijzigd bij batchen) was de juiste aanname, niet te pessimistisch.

**Wat dit betekent voor de prioriteit.** Een batch>1-integratie zou zijn
winst vrijwel uitsluitend uit MoE moeten halen (57,8% van het token) — de
overige 42,2% (attentie, Mamba, lm_head+shared, overhead) profiteert niet
op dezelfde manier. Dat maakt de zaak niet minder de moeite waard (MoE
alleen al is de grootste post), maar wel duidelijk begrensder dan een naïeve
"batch alles en alles wordt N× goedkoper"-aanname zou suggereren.

**Artefacten.** `pro_research/diag_attention_n_scaling.py` ·
`pro_research/diag_attention_n_scaling.json`.

### Correctie: Mamba schaalt NIET zoals attentie — mild supra-lineair, geen aanname meer

Het ontwerpdocument stelde "attentie/Mamba" op één lijn — beide zonder
deel-mogelijkheid, dus verondersteld ~lineair. Dat was voor Mamba een
**aanname bij analogie, nooit apart gemeten.** Deze sessie se eigen regel
("verifieer aannames, beweer ze niet") gold hier niet — hersteld.

`pro_research/diag_mamba_n_scaling.py`: dezelfde opzet als de attentie-
meting, nu op Mamba se `in_proj` (FP8-per-tensor-kernel, dus een fysiek
andere kernel dan attentie se BF16-ERVF-pad — geen reden om aan te nemen
dat dezelfde conclusie automatisch overdraagt). N∈{1,2,4,8,16}, 30 ronden.

**Uitkomst: mild supra-lineair, niet lineair.** ms/sequentie **stijgt** met
N: 0,177 ms (N=1) → 0,170 (N=2, nog binnen ruis) → 0,196 (N=4) → 0,206
(N=8) → 0,203 (N=16, ratio 1,15 t.o.v. ideaal lineair). Dat is een reële
**~15% straf** bij grotere N, geen neutrale schaling. Vermoedelijke oorzaak
(niet onderzocht): de FP8-tensor-kernel heeft een grotere rijenvorm
(rows=10304 tegenover attentie se 4096) en dus minder "vrije capaciteit"
per launch — herhaalde back-to-back launches zonder batch-bewust ontwerp
kunnen elkaar meer in de weg zitten (geheugenbank-conflicten, L2-druk,
niet geverifieerd welke).

**Gevolg voor de bovengrensrekensom.** De ~114 tok/s-rekensom nam aan dat
"de rest" (attentie+Mamba+overig) **ongewijzigd** blijft bij batchen. Voor
attentie klopt dat (gemeten, bevestigd). Voor Mamba is de aanname
**optimistisch** — de rest zou bij batch>1 niet gelijk blijven maar iets
duurder worden per sequentie, dus de werkelijke aggregate bovengrens ligt
naar verwachting **lager** dan de eerdere rekensom suggereerde. Geen nieuw
getal berekend (zou weer een aanname-op-aanname zijn) — de eerlijke
conclusie is dat de eerdere bovengrens nu bevestigd te optimistisch is voor
het Mamba-deel, niet dat er een preciezer getal klaarligt.

**Artefacten.** `pro_research/diag_mamba_n_scaling.py` ·
`pro_research/diag_mamba_n_scaling.json`.

---

## 2026-08-16 — Per-laag cachecapaciteit fysiek gemeten en geïntegreerd: 47,41 tok/s

**Vervolg op de hitrate-diagnose** (−14,3% missers, hitrate 85,6%→87,7%,
budget-neutraal). Nu fysiek causaal getest en in V6 geïntegreerd.

**Causale A/B** (`pro_research/v_capacity_realloc_ab.py`, productiekernels,
géén batching erbij — één variabele: capaciteitsverdeling), volgens
hetzelfde precedent als A1 (capaciteit veranderen bewijsbaar bitexact met
D1 aan). Full mode, 765 samples: BASE_A 31,1651 → NONUNIFORM 31,2058 →
BASE_B 31,7189 ms. Poorten: `nonuniform_equals_base_bitexact` ✅ (bitexact,
bevestigt het A1-precedent) · `base_drift_le_1ms` ✅ · `ctl_diverges` ✅ ·
winst **0,2362 ms/token (0,75%)** — kleiner dan de ruwe schatting (~0,31 ms)
maar reëel en positief, alle poorten groen.

**Geïntegreerd in V6** (`pro_research/layer_capacity.py`, herbruikbare
`apply_nonuniform_capacity(rt)`, aangeroepen na élke `enable_cache()`-call
in `graph_v6_full_stack.py` inclusief de CTL-herbouw — capaciteit is
budget-neutraal, dus **geen VRAM-kost**, in tegenstelling tot de
gather/down_masked-poging hierboven).

**Uitkomst (full, 765 samples).**

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,4289 ms | 31,82 |
| **V6 (nu met per-laag capaciteit)** | **21,0923 ms** | **47,41** |

Alle poorten groen (bitexact, deterministisch, controle-arm wijkt af, VRAM
binnen budget — ongewijzigd t.o.v. eerder, want budget-neutraal). Nieuw,
klein record: 47,41 tok/s (was 47,08-47,37 in eerdere runs, sessie-variantie
in die orde, maar dit is de eerste run MET de capaciteitswinst erbij).

**Stand.** 47,41/165 = **28,7% van roofline**, feitelijk ononderscheidbaar
van vóór deze toevoeging gezien de sessie-variantie (~0,2-0,3 ms/token is
klein t.o.v. de ruis tussen losse volledige-graph-builds) — maar wel een
**bitexact geverifieerde, budget-neutrale, reële winst**, geen ruis in de
eigen geïsoleerde A/B (die had een BASE_A/BASE_B-drift van 0,55 ms, dus de
0,24 ms winst zit ruim binnen wat de eigen poort als betekenisvol aanmerkt).

**Vervolg: is −20/+30 wel goed gekozen?** Sweep over vijf budget-neutrale
varianten (`diag_capacity_sweep.py`, hitrate-only, één subproces per
kandidaat wegens hetzelfde pinned-memory-probleem als eerder bij de
componentafbraak):

| reduce/boost | missers | hitrate |
|---|---:|---:|
| 0 / 0 (uniform) | 5182 | 85,61% |
| **−20 / +30 (verscheept)** | **4443** | **87,66%** |
| −30 / +45 | 4549 | 87,37% |
| −40 / +60 | 4960 | 86,23% |
| −50 / +75 | 5645 | 84,33% |
| −60 / +90 | 6865 | 80,94% |

De verscheepte −20/+30-verdeling is **het laagste punt van de sweep** — elke
agressievere verdeling (zelfde 10 lagen, grotere delta) doet het slechter,
niet beter. LRU-hitrate is dus sterk niet-lineair: voorbij een bepaald punt
kost het weghalen bij de "lage-miss"-lagen meer dan het geven aan de
"hoge-miss"-lagen oplevert. **Geen makkelijke extra winst door simpelweg
harder te trekken aan dezelfde hefboom.** Een grondigere optimalisatie zou
andere lagen moeten kiezen (niet alleen de 10 uit deze eerste analyse) en
over meerdere prompts moeten middelen — niet gedaan, maar de sweep laat zien
dat het potentieel hier klein is, niet dat er nog een grote winst wacht.

**Artefacten (aanvullend):** `pro_research/diag_capacity_sweep.py` ·
`pro_research/diag_capacity_sweep.json`.

**Artefacten.** `pro_research/layer_capacity.py` ·
`pro_research/v_capacity_realloc_ab.py` ·
`pro_research/results/PRO_CAPACITY_REALLOC_AB.json` ·
`pro_research/graph_v6_full_stack.py` (uitgebreid) ·
`pro_research/results/PRO_V6_FULL_STACK.json` (bijgewerkt).

---

## 2026-08-16 — Correctie: gather/down_masked batchen bleek WÉL correct — maar niet de moeite waard om te integreren

**Correctie op het eerdere "race condition"-blok.** Dat was voorbarig.
Herbeoordeling: de referentiekernel in die test is een **letterlijke kopie**
van de exacte kernel die V5/V6 al duizenden keren correct heeft gedraaid op
het echte model (bitexacte causale A/B's, geen enkele NaN daar). Als
diezelfde kernel op ECHTE data altijd correct is maar op MIJN synthetische
data NaN geeft, ligt het probleem hoogstwaarschijnlijk bij de synthetische
testdata (ongeconstrainde `standard_normal`-LUTs/bytepatronen die een
combinatie raken die met échte ReLU2-sparse activaties nooit voorkomt), niet
bij een race.

**Bevestiging.** Twee nieuwe scripts testen met **echte, gevangen
modeldata** in plaats van synthetische random data:
`verify_down_gather_batch_real_data.py` (top_k=1, één echte laag-aanroep,
via monkeypatchen van `fused.down_masked_ind_k` om de exacte argumenten van
een echte `rt.step()`-aanroep te onderscheppen) en
`verify_down_gather_batch_real_full.py`/`verify_gather_batch_real_full.py`
(top_k=6, alle zes echte expert-aanroepen van één laag). Uitkomst in alle
gevallen: **bitexact, nul NaN**, zowel voor `gather_down_sparse_ind_batched`
als `gemv_down_masked_partial_ind_batched`. De kernels zijn dus wel degelijk
correct — de eerdere "race"-diagnose was fout.

**Geïntegreerd en fysiek getest.** `moe_dev_batched.py` uitgebreid met een
optionele `gather_kernels`-parameter (`down_gather_batch_kernels.py`).
Causale A/B apart van V5/up-proj gehouden (`v_gather_batched_ab.py`, één
variabele: gather-batching aan/uit, bovenop de al geverifieerde stack).
Full mode: bitexact, controle-arm wijkt af, **+0,6826 ms/token (+2,56%)**
bovenop V5+up-proj. Alle poorten groen — in isolatie is dit dus een échte,
kleine winst.

**Maar niet geïntegreerd in V6 — twee redenen, gemeten, niet aangenomen.**
1. **VRAM.** Gather/down_masked-batchen vereist `top_k` onafhankelijke
   mirror-buffers i.p.v. één hergebruikte (`self.mstate["mirror"]`) — dat is
   `top_k × 23 lagen × 2,68 MB ≈ 387 MB`, tegenover het bestaande
   64 MiB-budget. **De VRAM-poort faalde**, en is niet verruimd.
2. **Marginale winst verdampt bij volledige integratie.** V6 met gather-
   batching erbij: 21,1129 ms/token (47,36 tok/s, full mode) — **binnen
   ruis identiek** aan V6 zonder deze toevoeging (21,1118 ms/token, 47,37
   tok/s, eerder gemeten). De +0,68 ms uit de geïsoleerde A/B is dus al
   grotendeels aanwezig via andere weg zodra de rest van de graph-
   residentie/batching al actief is (vergelijkbare les als de eerdere
   ablatiecorrectie: eager-gemeten componentwinsten vertalen zich niet
   1-op-1 naar extra winst bovenop een al sterk geoptimaliseerde graph).

**Besluit.** `graph_v6_full_stack.py` draait `gather_kernels` bewust NIET
mee (expliciet becommentarieerd in de code, niet stilzwijgend weggelaten).
V6 blijft op zijn eerder geverifieerde configuratie: **~47,1-47,4 tok/s**,
alle poorten groen, VRAM ruim binnen budget. `moe_dev_batched.py`'s
uitbreiding blijft wel beschikbaar (`gather_kernels=None` is de default) —
bruikbaar als op een GPU met meer VRAM-marge, of als de cache-capaciteit
elders wordt teruggebracht om ruimte te maken, dit alsnog de moeite waard
wordt.

**Artefacten.** `pro_research/verify_down_gather_batch_real_data.py` ·
`pro_research/verify_down_gather_batch_real_full.py` ·
`pro_research/verify_gather_batch_real_full.py` ·
`pro_research/v_gather_batched_ab.py` ·
`pro_research/results/PRO_GATHER_BATCHED_AB.json` ·
`pro_research/moe_dev_batched.py` (uitgebreid, optionele
`gather_kernels`-parameter, default `None`).

---

## 2026-08-16 — Per-laag cachecapaciteit herverdelen: −14,3% missers, nog niet fysiek gemeten

**Vraag.** De hitrate-diagnose (eerder vandaag) liet sterk niet-uniforme
missrates per laag zien bij uniforme capaciteit 72 (laag 1/3/6/51: 25-42%,
de rest 6-15%). Helpt het om capaciteit weg te halen bij lage-miss-lagen en
te geven aan hoge-miss-lagen, bij **gelijkblijvend totaal budget**?

**Methode.** `pro_research/diag_per_layer_capacity.py` — puur hitrate,
geen timing-claim. `enable_cache()`'s eigen allocatielogica (runtime.py:
324-378) letterlijk hergebruikt om specifieke lagen na de normale
`enable_cache(72)`-aanroep te herallokeren (geen wijziging aan runtime.py;
`_moe_dev` leest `c["cap"]` toch al per laag dynamisch, dus heterogene
capaciteit is architecturaal al ondersteund — alleen `enable_cache`'s eigen
gemaksfunctie is uniform). Test: −20 op de 6 laagste-miss lagen (72→52),
+30 op de 4 hoogste-miss lagen (72→102), totaal ongewijzigd (1656 slots).

**Uitkomst.** Missers: **5.182 → 4.443 (−739, −14,3%)**. Hitrate: 85,61% →
87,66%. De geboosde lagen verbeteren dramatisch (laag 1: 665→300 missers,
meer dan gehalveerd), de verlaagde lagen verslechteren mild (laag 13:
100→164) — LRU-hitrate is sterk niet-lineair bij lage capaciteit, dus
weghalen bij een al goed bediende laag kost weinig terwijl geven aan een
slecht bediende laag veel oplevert.

**Voorzichtige tijdschatting (rekenwerk, geen meting).** Bij M1's
bulk-tempo (24,93 GB/s) en 2,68 MB per up_proj-miss: 739 minder missers over
256 tokens ≈ 739/256 ≈ 2,89 missers/token minder × 2,68 MB ≈ 7,75 MB/token
minder ÷ 24,93 GB/s ≈ **~0,31 ms/token** potentieel — reëel maar klein
vergeleken met de kernel-batchingwinsten van deze sessie (1-9 ms/token).

**Nog niet gedaan.** Geen fysieke causale A/B (alleen hitrate, geen
tokentiming), niet geïntegreerd in V6, geen verfijning van de −20/+30-keuze
(die was een eerste ruwe gok, geen optimum). Kleine maar reële hefboom,
lagere prioriteit dan wat al gebouwd is gezien de bescheiden geschatte
omvang.

**Artefacten.** `pro_research/diag_per_layer_capacity.py` ·
`pro_research/diag_per_layer_capacity.json`.

---

## 2026-08-16 — `gather_down_sparse_ind`/`gemv_down_masked_partial_ind` batchen: NIET gelukt, eerlijk gesloten

**Poging.** Na vier geslaagde batchingen (panel_scan, reduce_partials,
accumulate, up-proj-GEMV) leek het logisch de laatste twee down_proj-
subkernels ook te proberen — beide bleken bij nader inzien óók een vaste
(niet data-afhankelijke) grid-formule te hebben, dus in principe dezelfde
veilige klasse. Nieuw bestand `pro_research/down_gather_batch_kernels.py`
met referentie- en batched varianten van `gather_down_sparse_ind` (slot via
`blockIdx.y`) en `gemv_down_masked_partial_ind` (slot via `blockIdx.z`,
`blockIdx.y` blijft de bestaande chunk-dimensie).

**Uitkomst: niet bitexact, en het patroon wijst op een race, geen simpele
adresseerfout.** Geïsoleerde test
(`verify_down_gather_batch_kernels.py`, synthetische data): de **mirror-
data** (gather's output) is **bitexact identiek** tussen referentie en
batched — dus `gather_down_sparse_ind_batched` zelf is aantoonbaar correct.
Maar `gemv_down_masked_partial_ind`'s uitvoer verschilt daarna alsnog, mét
NaN's — **ook in de referentie-arm** (318 NaN's), niet alleen de batched
arm (2756-3799 NaN's), en de **exacte tellingen wisselen tussen
proces-runs** bij identieke code en identieke invoerdata. Dat patroon —
zelfde logica, zelfde data, ander resultaat, niet-reproduceerbaar tussen
runs — wijst op een **race condition of synchronisatieprobleem**, niet op
een simpele transcriptiefout in de adressering (die zou wél deterministisch
verkeerd zijn, elke run hetzelfde).

**Besluit: niet verder geforceerd, niet geïntegreerd.** Geen enkele wijziging
raakte `moe_dev_batched.py` of `graph_v6_full_stack.py` — V6 blijft op zijn
volledig geverifieerde 47,37 tok/s staan. Dit is precies de reden waarom
elke stap in deze sessie eerst geïsoleerd bitexact getest werd vóór
integratie: deze twee kernels faalden op die eerste, goedkope stap, dus is
er niets stuk gemaakt. Verder debuggen vraagt gereedschap dat deze sessie
niet heeft (compute-sanitizer/cuda-gdb) — eerlijk gesloten in plaats van
doorgeduwd met een niet-geverifieerd resultaat.

**Wat dit niet raakt.** De al bitexact geverifieerde down_proj-batching
(`panel_scan`+`reduce_partials`, in V5/V6) gebruikt nog steeds de
ORIGINELE, ongewijzigde `gather_down_sparse_ind`/
`gemv_down_masked_partial_ind` per slot — dat pad is en blijft correct.

**Artefacten.** `pro_research/down_gather_batch_kernels.py` ·
`pro_research/verify_down_gather_batch_kernels.py` ·
`pro_research/verify_down_gather_batch_kernels.json` (status: gefaald, niet
verwijderd — een weerlegging is ook DONE).

---

## 2026-08-16 — Up-proj ERVF-GEMV gebatcht: 47,37 tok/s, roofline 28,7%

**Aanleiding.** Componentafbraak liet zien dat MoE 57,8% van het token is,
en down_proj (V5) daar maar 6,51 van de 12,07 ms van uitmaakt. De
up-proj-GEMV (`gemv_nvfp4_ervf_ind`, de ERVF-kernel die de NVFP4-experts
projecteert) wordt zes keer per laag sequentieel aangeroepen, elk naar een
**onafhankelijke** outputregio (`bs["act"][s]`) — geen gedeelde accumulator,
dus géén race, in tegenstelling tot `accumulate_indirect`. `x` (de
genormaliseerde hidden state) is bovendien identiek voor alle zes slots.
Grid-grootte is vast (niet data-afhankelijk). Dit is dus dezelfde veilige
batch-klasse als `panel_scan`/`reduce_partials`, alleen op een grotere,
belangrijkere kernel — de zorgvuldig met NERVF-2 tegen een referentie
geverifieerde WIDTH-16-subwarp-butterfly-reductie.

**Aanpak — extra voorzichtig gezien het gewicht van deze kernel.** Nieuw
bestand `pro_research/up_proj_batch_kernels.py`: de referentiekernel
(`gemv_nvfp4_ervf_ind_ref`) staat **letterlijk gekopieerd** naast de
batched versie (`gemv_nvfp4_ervf_ind_batched`, `blockIdx.y`=slot toegevoegd,
verder geen enkele regel rekenkunde gewijzigd) om transcriptiefouten in de
reductie-boom te vermijden. Geïsoleerde bitexacte test
(`verify_up_proj_batch_kernel.py`, synthetische data op echte dimensies
1856×2688, drie trials incl. `apply_relu2` aan/uit): **bitexact.**

**Causale A/B, apart van V5's eigen resultaat gehouden** (één variabele:
up-batching aan/uit, bovenop de al geverifieerde down_proj-batching —
`pro_research/v_up_proj_batched_ab.py`). Full mode: bitexact,
**+1,7423 ms/token (+6,11%)** bovenop V5. Alle poorten groen.

**V6 opnieuw gedraaid** (`graph_v6_full_stack.py` uitgebreid met
`up_proj_batch_kernels.UpProjBatchKernels`, geïnstalleerd vóór
`setup_graph()` net als de andere twee). Full mode, 765 samples:

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,0973 ms | 32,16 |
| **V6 (nu vijf mechanismen)** | **21,1118 ms** | **47,37** |

Winst 9,9855 ms/token (32,1%) t.o.v. zelfde-sessie EGR. Alle poorten groen:
bitexact, deterministisch, controle-arm wijkt af, dot-graph bevat nu alle
vijf kernelnamen (beide ERVF-dense, beide down_proj-batch, plus de nieuwe
up_proj-batch), VRAM binnen budget.

**Stand.** 47,37/165 = **28,7% van roofline** (was 26,9%). Nog een factor
**2,11×** te gaan tot 100 tok/s.

**Artefacten.** `pro_research/up_proj_batch_kernels.py` ·
`pro_research/verify_up_proj_batch_kernel.py` ·
`pro_research/verify_up_proj_batch_kernel.json` ·
`pro_research/v_up_proj_batched_ab.py` ·
`pro_research/results/PRO_UP_PROJ_BATCHED_AB.json` ·
`pro_research/moe_dev_batched.py` (uitgebreid, optionele `up_kernels`-
parameter) · `pro_research/graph_v6_full_stack.py` (uitgebreid) ·
`pro_research/results/PRO_V6_FULL_STACK.json` (bijgewerkt).

---

## 2026-08-16 — `weighted_accumulate_ind` veilig gebatcht en geïntegreerd: 44,37 tok/s

**Vervolg op de componentafbraak hierboven.** `accumulate_indirect` is
gebouwd volgens precies het aangewezen veilige patroon: **niet** een
mechanische kopie van `panel_scan`/`reduce_partials` (die schrijven
onafhankelijk per slot), maar een nieuwe `weighted_accumulate_ind_batched`-
kernel die de exacte `s=0..top_k-1`-fmaf-volgorde uit één kernel-launch
reproduceert (`acc = fmaf(contrib[s], w[s], acc)` in een vaste lus,
startend vanuit `dst[i]` dat al de shared-expert-term bevat) — bit-identiek
aan `top_k` losse launches, geen parallelle/atomic reductie die de
FP-optelvolgorde zou veranderen.

**Stap 1 (geïsoleerd, `verify_down_proj_batch_kernels.py` uitgebreid):**
bitexact tegen de sequentiële referentie, 3 trials met willekeurige
`dst`/`contrib`/`w`. **Stap 2 (`moe_dev_batched.py` uitgebreid, causale A/B
via `v5_batched_downproj_ab.py`, dat automatisch meelift):** full mode,
bitexact, **−3,1552 ms/token (−9,88%)** eager t.o.v. eerder −2,2126 ms
(−7,07%) met alleen panel_scan+reduce_partials — de accumulate-batching
draagt dus zelf nog eens ~0,94 ms/token bij, in lijn met de verwachting dat
dit een kleinere losse post was.

**V6 opnieuw gedraaid (pakt de uitbreiding automatisch mee via dezelfde
`install_batched_moe_dev`).** Full mode, 765 samples: **22,5354 ms/token,
44,37 tok/s** (was 44,19 met alleen panel_scan+reduce_partials), winst 8,7972
ms/token (28,1%) t.o.v. zelfde-sessie EGR. Alle poorten opnieuw groen:
bitexact, deterministisch, controle-arm wijkt af, VRAM binnen budget.

**Stand.** 44,37/165 = **26,9% van roofline**. Nog een factor **2,25×** te
gaan tot 100 tok/s.

**Artefacten.** `pro_research/down_proj_batch_kernels.py` (uitgebreid),
`pro_research/verify_down_proj_batch_kernels.py` (uitgebreid),
`pro_research/moe_dev_batched.py` (uitgebreid),
`pro_research/results/PRO_V5_BATCHED_DOWNPROJ_AB.json` (bijgewerkt),
`pro_research/results/PRO_V6_FULL_STACK.json` (bijgewerkt).

---

## 2026-08-16 — Componentafbraak van V6: MoE is 57,8% van het token, niet alleen down_proj

**Vraag.** Down_proj (V5) was de eerste hefboom, maar waar gaat de rest van
V6's ~20,9-22,6 ms/token naartoe? Nodig om de volgende stap gericht te
kiezen i.p.v. te blijven graven op down_proj alleen.

**Methode.** Zelfde ablatietechniek als `diag_down_ablation_timing.py`
(wall-clock, want `cp.cuda.get_elapsed_time` op graph-gevangen events faalt
op deze stack). Vier hele subblokken om de beurt vervangen door een no-op
vóór `setup_graph()` vangt, elk in een **apart proces** gedraaid
(`diag_v6_component_breakdown.py --drive`) — het bouwen van vijf volledige
30B-runtimes ná elkaar in één proces liep vast op pinned-host-geheugen dat
niet volledig vrijkwam tussen builds, ook niet met expliciete
`free_all_blocks()`+`gc.collect()`.

| stub | wat wordt overgeslagen | bovengrens | aandeel |
|---|---|---:|---:|
| `_attention` → no-op | 6 attentielagen (Q/K/V/O, KV-write, attention-kernel) | 3,0987 ms | 14,9% |
| `_mamba` → no-op | 23 Mamba-lagen (in_proj, conv, ssm_step, gated_norm, out_proj) | **−0,4287 ms** | ~0% (ruis, apart-proces-vergelijking heeft meer variantie dan de eerdere in-proces down_proj-ablatie) |
| `_moe` → no-op | **alle** 23 MoE-lagen: shared expert + routed (up+down) | **12,0669 ms** | **57,8%** |
| `fused.gemv_into` → no-op | lm_head (1×/token) **plus** shared-expert up+down (46×/token) — zelfde methode, dus samen gemeten, niet apart | 2,1023 ms | 10,1% |

**MoE is dus verreweg de grootste post — meer dan de helft van het hele
token.** Kruisverwijzing met de eerdere down_proj-ablatie (6,5058 ms
in-graph): MoE-bovengrens (12,0669) − down_proj (6,5058) = **~5,56 ms**
overige MoE-kosten (shared-expert-GEMV's, up-proj ERVF-GEMV, de batched
panel_scan/reduce_partials-kernels zelf, `accumulate_indirect`,
routing/cache-kernels) — nog niet apart uitgesplitst.

**Let op — de `moe`- en `lmhead_plus_shared_expert`-bovengrenzen
OVERLAPPEN** (beide bevatten de shared-expert-kosten); niet optellen alsof
ze disjunct zijn.

**Wat dit opent voor de volgende stap.** `accumulate_indirect` (de
gewogen-optel-kernel die de zes expert-bijdragen in `out` optelt) leek eerst
een voor de hand liggende volgende batch-kandidaat naast `panel_scan`/
`reduce_partials` — maar is dat **niet zomaar**: anders dan die twee is dit
een **sequentiële** accumulatie in dezelfde `out`-buffer
(`dst[i] = fmaf(src[i], w, dst[i])`, zes keer na elkaar). Simpelweg batchen
over slots zou een race-conditie geven (alle zes blocks lezen/schrijven
dezelfde `dst[i]` tegelijk) en zou atomics of een aparte
"schrijf-per-slot-dan-in-vaste-volgorde-optellen"-kernel vereisen — precies
de klasse fout die D1 al een keer blootlegde (optelvolgorde is niet
vrijblijvend bij FP-optelling). Een veilige batched variant zou, net als
`reduce_partials_batched`, een **apart, nieuw ontwerp** nodig hebben dat de
volgorde `s=0..5` expliciet bewaart — geen mechanische kopie van het
patroon dat voor `panel_scan`/`reduce_partials` wél veilig was. Niet
gebouwd deze sessie; opgenomen in `TODO.md` met deze precieze
kanttekening zodat niemand de D1-fout herhaalt.

**Artefacten.** `pro_research/diag_v6_component_breakdown.py` ·
`pro_research/diag_v6_component_breakdown.json` ·
`pro_research/diag_v6_component_breakdown_arm_*.json` (vijf, één per arm).

---

## 2026-08-16 — PRO V5 + V6 — batched down_proj gebouwd, geverifieerd en geïntegreerd: 44,19 tok/s, nieuw record

**Aanleiding.** De ablatiemeting hierboven (`diag_down_ablation_timing.py`)
gaf een realistische, in-graph bovengrens: down_proj kost hooguit 6,51
ms/token (28,9%) binnen V4's graph, waarvan `panel_scan`+`reduce_partials`
(vaste, data-onafhankelijke grid-groottes, dus veilig te batchen zonder de
lastigere PCIe-gather aan te raken) een deel is. In plaats van bij een
preregistratie te blijven steken is dit nu gebouwd, in drie stappen, elk
apart geverifieerd vóór de volgende.

**Stap 1 — geïsoleerde kernel-unittest (`verify_down_proj_batch_kernels.py`,
geen model/runtime nodig).** Twee nieuwe kernels
(`pro_research/down_proj_batch_kernels.py`): `panel_scan_batched` (grid
`(top_k,)` i.p.v. `top_k` losse `(1,)`-launches, elk block bewerkt zijn eigen
slot) en `reduce_partials_batched` (grid `(blocks_x, top_k)`, `blockIdx.y` =
slot). Beide zijn een mechanische transformatie: identieke per-block/per-
thread logica, alleen geadresseerd via een extra slot-index — geen
rekenkunde gewijzigd. Getest op synthetische data bij zes sparsity-niveaus
(0/30/50/70/95/100%) voor panel_scan en drie willekeurige trials voor
reduce_partials: **bitexact op alle gevallen.**

**Stap 2 — causale A/B/A/CTL in eager modus (`v5_batched_downproj_ab.py`,
volgens `PRO_V5_PREREGISTRATION.md`).** `install_batched_moe_dev`
(`pro_research/moe_dev_batched.py`) vervangt `rt._moe_dev` via
`types.MethodType` (zelfde niet-invasieve patroon als `_install_selective`)
— `gather_down_sparse_ind`/`gemv_down_masked_partial_ind` blijven ONGEWIJZIGD
per-slot (buiten scope, zoals de prereg voorschreef), alleen `panel_scan` en
`reduce_partials` worden éénmaal per laag aangeroepen i.p.v. zes keer.

*Eerste run had een bug*: een NIEUWE `mirror`-buffer per laag i.p.v.
hergebruik van `self.mstate["mirror"]` (die toch al bestaat, want gather/
down_masked blijven sequentieel per slot). Kostte 23 lagen × 2,68 MB ≈ 61,6
MB — precies waarom de VRAM-poort bestaat: `extra_vram_lt_64MiB` **faalde**
bij V6 hieronder (zie verderop), gevonden, begrepen, gefixt (hergebruik
`self.mstate["mirror"]`), opnieuw geverifieerd bitexact vóór verder te gaan.

Full-mode uitkomst (256×3, 765 samples): BASE_A 31,049 → BATCHED 29,0799 ms
(**−2,2126 ms, −7,07%**), BASE_B 31,5359 ms (drift 0,487 ms). Poorten:
`batched_equals_base_bitexact` ✅ · `base_drift_le_1ms` ✅ · `ctl_diverges`
✅ (bad_pick-sabotage wijkt af zoals vereist) · `candidate_gain_ge_1ms_or_3pct`
✅ · `full_samples_ge_500` ✅.

**Stap 3 — V6: alle drie mechanismen in één graph (`graph_v6_full_stack.py`,
niet apart gepreregistreerd — volgt rechtstreeks uit V4's en V5's eigen
poorten, geen nieuwe beleidskeuze).** `_install_selective`
(patcht `rt.k.mv_bf16`/`mv_fp8_tensor`, gebruikt in `_attention`/`_mamba`)
en `install_batched_moe_dev` (vervangt `rt._moe_dev` volledig, alleen MoE-
routering) raken **verschillende aanroeppunten** — beide vóór
`rt.setup_graph()` installeren vangt dus beide mechanismen in dezelfde
graph, naast de al aanwezige device-routing en veilige prompt-staging.

*Eerste V6-run*: alle correctheidspoorten slaagden, maar
`extra_vram_lt_64MiB` **faalde** (de V5-mirror-bug hierboven, opgespoord via
precies déze poort — het systeem werkte zoals bedoeld). Na de fix: alle
poorten groen.

**Uitkomst (full, 256×3, 765 samples).**

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,1595 ms | 32,09 |
| **V6 (device routing + graph-safe + selectieve ERVF + batched down_proj)** | **22,6306 ms** | **44,19** |

Winst: **8,5289 ms/token, 27,4%** t.o.v. dezelfde-sessie EGR. Extra VRAM
16 MiB (< 64 MiB-poort, ruim binnen budget na de fix). Alle poorten:
`argmax_direct_tie` ✅ · `graph_dot_contains_all_mechanisms` ✅ (dot-graph
bevat alle vier kernelnamen: beide ERVF-kernels én beide batched-kernels) ·
`v6_equals_egr` ✅ (bitexact over 256 tokens × 3 prompts) · `v6_deterministic`
✅ · `bad_pick_control_diverges` ✅ · `extra_vram_lt_64MiB` ✅ ·
`full_speed_gain_ge_2_5ms` ✅ · `full_samples_ge_500` ✅ (765).

**Dit is het nieuwe record, en verbetert op V4 (41,13 tok/s smoke/full) met
+7,5%.** Vergelijk met de eerdere V4-notitie: V6's winst (8,5289 ms) is
groter dan V4's eigen winst (6,8634 ms) plús V5's eigen eager-winst
(2,2126 ms) opgeteld zou suggereren (6,8634+2,2126=9,0760 — dicht bij 8,5289,
een kleine overlap-tax van ~0,55 ms, niet negatief) — de drie mechanismen
blijven dus grotendeels optellen, net als V4's eigen bevinding over de
eerste twee.

**Waar staan we nu t.o.v. het doel.** 44,19 tok/s is **26,8% van het
ctx0-roofline-plafond** (165 tok/s) — op van 24,9% (V4) en 17% (de oude
Nano-lijn). Naar 100 tok/s is nog een factor **2,26×** te gaan vanaf hier.
De resterende, nog niet gebouwde hefboom (PCIe-gather-herstructurering van
`gather_down_sparse_ind`, apart gehouden van dit V5-batchen per de
preregistratie) zou volgens de ablatiemeting hooguit nog eens een paar
ms/token kunnen opleveren — niet genoeg om alleen daarmee 100 te bereiken.
Er is nog geen geïdentificeerd pad naar 100 tok/s; dat blijft open.

**Artefacten.** `pro_research/down_proj_batch_kernels.py` ·
`pro_research/verify_down_proj_batch_kernels.py` ·
`pro_research/verify_down_proj_batch_kernels.json` ·
`pro_research/moe_dev_batched.py` · `pro_research/v5_batched_downproj_ab.py` ·
`pro_research/results/PRO_V5_BATCHED_DOWNPROJ_AB.json` ·
`pro_research/graph_v6_full_stack.py` ·
`pro_research/results/PRO_V6_FULL_STACK.json`.

---

## 2026-08-16 — MTP-route-unie gemeten: S10 stap 2 (speculatief decoderen) sluit, met cijfers

**Vraag.** `S10A_MTP_ACCEPTANCE_REPORT_2026-08-15.md` (bestond al — S10-A mat
de acceptatiegraad `A=2,114` over 360 stappen, poort G-S10-1 ≥1,5 **gehaald**)
identificeerde zelf de **enige nog onbekende term** die beslist of stap 2
(de speculatieve lus daadwerkelijk bouwen) de moeite waard is: hoeveel
**unieke** experts raakt een verificatie-sweep van `D+1=5` tokens per
MoE-laag, tegenover 6 voor één token? De prereg gaf alleen een theoretische
bovengrens (≤3,66× via N7-A's paarsgewijze overlap 2,011/6) en schreef
expliciet: "meet die unie eerst... het kost geen bouw."

**Methode.** Puur read-only analyse, geen nieuwe kernel, geen runtime-
wijziging. `LightningRuntime.step(token_id, capture_routes=...)` **bestaat
al** (regel 803-823) en registreert per MoE-laag de top-6 expert-ids per
stap. De exacte greedy-gegenereerde tokenreeksen van de drie S10-A-
gate-prompts staan al op schijf in `s10a_mtp_acceptance.json`
(`gate_prompts[*].sequence`). Teacher-forced replay van die reeksen door de
backbone (capaciteit 72, `device_cache=False` — nodig omdat `_moe_dev`
`(None, None)` teruggeeft en `capture_routes` dus niet vult; routing zelf is
onafhankelijk van cache-modus) geeft de exacte routes die de échte generatie
gebruikte. Sliding window van 5 opeenvolgende posities, unie van de
verzamelde expert-ids per laag, gemiddeld over alle vensters/lagen/prompts.
Script: `pro_research/diag_mtp_route_union.py`.

**Uitkomst.** Gemiddelde unie over 5 tokens: **19,88 van de 128 experts per
laag** (mediaan 20, p95 26, max 30 = het theoretische maximum 5×6). Dat is
**3,313×** t.o.v. de 6 van één token — ruim boven de eigen voorgestelde
grens van de prereg (>12 = "verdubbeling" ⇒ stap 2 negatief), en zelfs iets
onder de theoretische bovengrens (3,66×) maar in dezelfde orde. Stabiel over
de drie domeinen (expository 19,25 · narrative 19,48 · code 20,87 — code
route't net iets breder, consistent met code's hogere acceptatiegraad
elders).

**De rekensom van het rapport, met dit echte getal in plaats van de
schatting.** MoE-verificatiesweep = 39,523 ms × 3,313 = 130,95 ms; volledige
backbone-sweep (attentie 18,634 + Mamba 8,309 + lm_head 2,106 amortiserend +
MoE 130,95) ≈ **160,0 ms**. Eén speculatieve ronde = MTP-keten (19,10 ms) +
verificatiesweep (160,0 ms) = 179,1 ms, voor gemiddeld `A+1 = 3,114` tokens
⇒ **57,51 ms/token**. Niet-speculatief (gemeten S8 @262K): **54,28 ms/token**.

**Speculatief decoderen zou dus ~6,0% TRAGER zijn, niet sneller — met de
gemeten unie, niet de aanname.**

**Wat dit sluit.** S10 stap 2 (de speculatieve lus bouwen) sluit hiermee
netjes, zonder dat er één regel productiekernel voor geschreven hoefde te
worden — precies zoals de S10-A-prereg voorschreef. De oorzaak is
architectureel, niet toevallig: bij 128 routed experts en top-6-sparsity per
token routeren opeenvolgende tokens naar grotendeels verschillende
expert-sets, dus een gezamenlijke verificatie-sweep beweegt bijna evenveel
MoE-bytes als losse tokens genereren — terwijl de MTP-drafting daar nog eens
19,10 ms bovenop legt. Dit was, volgens `S10_MTP_SPECULATIVE_PREREGISTRATION_2026-08-15.md`,
"de enige overgebleven hefboom" die bytes-per-token verlaagt in plaats van
tijd-per-byte. Met deze meting is die hefboom nu ook dicht — de eerdere
tabel "Wat weerlegd is" in `STATE_OF_THE_WORK.md` (die nog de oude
Nano-sluiting citeerde) is hiermee opnieuw en definitief bevestigd, nu voor
het juiste model en met een directe meting i.p.v. een architectuurvergelijking.

**Wat dit niet doet.** Verandert niets aan S10-A's eigen bevindingen (de
wiring, de acceptatiegraad, de kostenmeting van de MTP-keten) — die blijven
correct en gedocumenteerd. Sluit alleen stap 2 op basis van de nu bekende
route-unie.

**Artefacten.** `pro_research/diag_mtp_route_union.py` ·
`pro_research/diag_mtp_route_union.json` (read-only, geen PRO-poort, leest
alleen bestaande `reports/lightningstream_nemotron/s10a_mtp_acceptance.json`
en roept alleen bestaande runtime-API's aan).

---

## 2026-08-16 — Diagnose device-cache hitrate onder V4 — down_proj is nooit gecached in `_moe_dev`

**Vraag.** Waar gaat de resterende 24,3 ms/token na V4 nog naartoe — PCIe-
missverkeer, of iets anders? Nodig om de volgende hefboom gericht te kiezen
in plaats van te gokken.

**Methode.** Read-only diagnostiek (`pro_research/diag_hitrate_v4.py`, geen
PRO-poort, geen preregistratie — puur meetkundig). `_moe_dev` (het pad dat
`device_cache=True` gebruikt, dus ook wat V4's graph capture) accumuleert per
laag hits/misses al in een device-buffer (`dev["stats2"]`, gevuld door de
`cache_assign`-kernel in `fused_nvfp4.py`) — nergens in de bestaande runners
uitgelezen. Dit script sommeert die buffer over alle 23 MoE-lagen na een
rollout van 256 tokens (prompt "The history of computing began when",
capacity 72, top_k 6).

**Uitkomst.** Hitrate **85,6%** (30.836 hits / 5.182 misses), **20,24 misses
per token** over 138 expert-selecties per token (23 lagen × top_k 6). Sterk
niet-uniform: laag 1/3/6 missen 42,5% / 36,5% / 25,2%, de meeste middenlagen
zakken naar 6–10%, laag 51 (laatste MoE-laag) piekt weer naar 25,3%.

**De structurele vondst.** `cache_mode` is `up_only` — de device-LRU-cache
(`c["cap"]`, gevuld via `cache_fetch`) dekt alléén de up_proj-codes/scales.
`_moe_dev`'s down-projectie loopt via
`down_masked_into_indirect(..., bank["down_base_ptr"], ...)` — dat leest
**bij elke expert-aanroep, hit of miss**, rechtstreeks uit de host-gemapte
bank. Er bestaat wél een `cache_mode="full"` met een `c["slot_down"]`
device-slot (zichtbaar in het oudere `_moe_cached_fast`, regel 659-660), maar
`_moe_dev` gebruikt dat pad nergens. Dus zelfs bij 85,6% up_proj-hitrate
bewegen **alle 138 down_proj-aanroepen per token** nog over PCIe (zij het via
de S5-masked/sparse-gather-techniek, dus minder dan een volle rij — het exacte
aantal bytes per aanroep is activatie-sparsity-afhankelijk en is hier niet
geschat om geen precisie te claimen die de statische code-lezing niet
onderbouwt).

**Wat dit opent — geprioriteerde volgende hefboom.** Twee onafhankelijke,
goed onderbouwde vervolgexperimenten, allebei niet in deze sessie gebouwd:

1. **Device-cache down_proj in `_moe_dev`.** `_moe_cached_fast` bewijst het
   patroon al (device-slot voor down_proj bij `cache_mode="full"`) — het moet
   nog naar het `device_cache`/graph-pad worden overgezet. Als down_proj-hits
   evenveel PCIe-verkeer schelen als up_proj-hits, is dit een tweede,
   onafhankelijke hefboom van vergelijkbare orde als V4's ERVF-winst, zonder
   dat er een nieuwe kernel bij hoeft (alleen bestaande device-cache-
   infrastructuur hergebruiken).
2. **Per-laag capaciteitstuning.** De miss-rate-ongelijkheid tussen lagen
   (laag 1/3/6/51 vs de rest) suggereert dat een uniforme capaciteit van 72
   per laag niet optimaal is — lagen met herhaaldelijk hoge missrate kunnen
   baat hebben bij een grotere cap, ten koste van lagen die toch al bijna
   nooit missen. Vereist eerst bevestiging dat dit patroon stabiel is over
   meerdere prompts/rollouts, niet een toeval van deze ene 256-token-run.

**Artefacten.** `pro_research/diag_hitrate_v4.py` ·
`pro_research/diag_hitrate_v4.json` (niet gecommit als PRO-resultaat, puur
diagnostisch).

**Correctie achteraf, zelfde dag.** De hierboven voorgestelde hefboom
("device-cache down_proj net als up_proj") bleek op een verkeerde
byte-aanname te steunen. `DOWN_PANEL_BYTES` (= `HALF_CODE + HALF_SCALE` =
2.494.464 + 311.808 B) is **2,68 MB per expert**, niet ~1 kB — bevestigd via
`CODE_BYTES`/`SCALE_BYTES` in `runtime.py` en `load_routed_bank`'s
`n * DOWN_PANEL_BYTES`-slicing. Vol cachen bij cap 72 × 23 lagen zou ~4,4 GiB
VRAM kosten; de GPU stond tijdens de V4-run al op **0 MiB vrij**. Dat maakt
"gewoon net als up_proj cachen" onhaalbaar zonder de up_proj-capaciteit fors
te verlagen (een echte trade-off, geen gratis winst) — vandaar de vervolgmeting
hieronder in plaats van dit alsnog te bouwen. Zie ook: up_proj's eigen missen
bewegen dus ook ~2,68 MB per miss (niet ~1 kB) — bij 20,24 missen/token is dat
**~54,2 MB/token**, wat bij het M1-tempo (24,93 GB/s) ~2,17 ms/token kost; dat
zit *niet* in de eerder gemeten "up_gemv"-component hieronder, want die meet
alleen de GEMV, niet de `cache_fetch`-staging.

---

## 2026-08-16 — Componentmeting V4-decodepad — down_proj-pijplijn is de grootste losse kostenpost

**Vraag.** Waar gaat de 24,3–34 ms/token precies naartoe? Nodig om de
down_proj-hefboom hierboven te vervangen door iets dat op echte metingen
rust, niet op statische bytegrootte-aannames (die al twee keer fout bleken).

**Methode.** Twee read-only diagnostieken, beide eager (`device_cache=True`,
géén graph — zodat elke aanroep echte Python is en met `cp.cuda.Event`-paren
precies te timen valt):

1. `diag_component_timing_v4.py` — omwikkelt `fused.gemv_ervf_indirect`
   (up_proj ERVF-GEMV) en `fused.down_masked_into_indirect` (de hele
   down_proj-pijplijn) als geheel, 128 tokens.
2. `diag_down_subkernels_v4.py` — splitst die down_proj-pijplijn verder open
   in zijn vier subkernels (`panel_scan`, `gather_down_sparse_ind`,
   `down_masked_ind`, `reduce_partials`), 96 tokens.

**Uitkomst (1).** Token p50 33,46 ms (deze eager config, iets hoger dan G0S's
EGR door instrumentatie-overhead). Per token, 138 expert-aanroepen (23 lagen
× top_k 6):

| component | ms/token | aandeel van token |
|---|---:|---:|
| up_gemv (ERVF, device-cache) | 5,003 | 14,6% |
| down_masked (hele pijplijn) | 9,573 | 27,9% |

down_masked is dus al zonder verdere opsplitsing de grootste single measured
component — groter dan up_gemv, en dat zonder up_proj's eigen
`cache_fetch`-staging (~2,17 ms/token, apart, hierboven geschat) mee te tellen.

**Uitkomst (2) — opsplitsing van die 9,57–11,39 ms (iets hoger token-p50
door extra instrumentatie: 36,28 ms):**

| subkernel | ms/token | aandeel van down-pijplijn |
|---|---:|---:|
| `gather_down_sparse_ind` (PCIe host-gemapte masked read) | 4,740 | 41,6% |
| `panel_scan` (mask/nonzero-scan, device-only) | 2,737 | 24,0% |
| `reduce_partials` (device-only) | 1,999 | 17,6% |
| `down_masked_ind` (de eigenlijke GEMV-compute) | 1,914 | 16,8% |
| **totaal down-pijplijn** | **11,390** | **29,9% van token** |

**Interpretatie.** Twee onafhankelijke aanwijzingen, geen enkele
overweldigend dominant:
- `gather_down_sparse_ind` is de grootste losse subkernel en bevestigt de
  PCIe-hypothese (host-gemapte, gemaskeerde/verstrooide lezing — dezelfde
  klasse toegang die E2/NERVF-4 al traag bleek, ~6,7-7,27 GB/s i.p.v.
  bulk-tempo 24,93-85,9 GB/s) — maar is met 41,6% niet de hele lading.
- `panel_scan` + `reduce_partials` samen (41,6% van de pijplijn, 4,74 ms/token)
  zijn beide kleine, device-only kernels zonder PCIe-component — hun kosten
  wijzen eerder op **launch-overhead**: 4 subkernels × 138 expert-aanroepen =
  **552 kernellaunches per token** alleen al voor down_proj. Fusie van deze
  vier kernels (of batchen over de 6 experts per laag in plaats van 6×4
  losse launches) is een onafhankelijke, mogelijk net zo grote hefboom als de
  PCIe-kant.

**Bovengrens, voorzichtig.** Zelfs als de hele down-pijplijn naar nul zou
gaan (onrealistisch), zou dat token-tijd van ~24,3 ms (V4, met selectieve
ERVF en graph) hooguit naar iets in de orde van ~13-15 ms brengen — ruw
richting 65-75 tok/s, niet 100. Dit is dus een substantiële maar geen
op-zichzelf-voldoende hefboom voor het einddoel.

**Waarom niet meteen gebouwd.** Twee echte CUDA-engineeringtaken (PCIe-
toegangspatroon herstructureren; vier kernels fuseren/batchen), allebei op
het kritieke pad van een 30B-productiemodel. Gezien de projectcultuur
("exactness before speed", nooit een poort verruimen, controle-armen
verplicht) verdienen die een eigen preregistratie en zorgvuldige
bitexact-verificatie, niet een haastige poging binnen dezelfde sessie waarin
al twee keer een snelle bytegrootte-aanname fout bleek.

**Artefacten.** `pro_research/diag_component_timing_v4.py` ·
`pro_research/diag_component_timing_v4.json` ·
`pro_research/diag_down_subkernels_v4.py` ·
`pro_research/diag_down_subkernels_v4.json` (alle vier read-only, geen
PRO-poort).

**Correctie achteraf — de eager meting overschat het absolute in-graph
budget (proportie klopte wel).** De 9,57-11,39 ms/token hierboven is eager
gemeten (`device_cache=True`, géén graph). V4 draait dit dezelfde pijplijn
**binnen** een gevangen CUDA-graph, die specifiek gebouwd is om
host-launch-overhead weg te nemen. Een poging om dat rechtstreeks te meten
met `cp.cuda.Event`-paren gevangen ín de graph liep vast op een echte
technische grens: `cp.cuda.get_elapsed_time` op events die binnen graph-
capture zijn opgenomen geeft op deze stack `cudaErrorInvalidValue`
(`pro_research/diag_down_ingraph_timing.py` — zelfde klasse bevinding als
G2's `cudaGraphLaunch`-restrictie, geen bug in het script).

In plaats daarvan: **ablatiemeting** — V4's exacte graph twee keer bouwen,
één keer ongewijzigd (REAL), één keer met `down_masked_into_indirect`
vervangen door een no-op (`out.fill(0)`, STUB — levert bewust **foute
tokens**, nooit als correctheidsresultaat te lezen, timing-only) vóór
`setup_graph()` vangt. Verschil = harde bovengrens op wat down_proj
maximaal in-graph kost. Uitkomst (200 replays per arm,
`pro_research/diag_down_ablation_timing.py`):

| arm | p50 ms/token |
|---|---:|
| REAL (ongewijzigd, zoals V4) | 22,5302 |
| STUB (down_proj = no-op) | 16,0244 |
| **verschil (bovengrens down_proj in-graph)** | **6,5058 (28,9% van het token)** |

De **proportie** komt opvallend overeen met de eager-meting (~28-30%), maar
het **absolute** bedrag is veel kleiner (6,51 ms in-graph tegenover
9,57-11,39 ms eager) — graph-replay amortiseert dus al een groot deel van
de launch-overhead die de eager meting toeschreef aan `panel_scan`/
`reduce_partials`. **Gevolg voor de V5-preregistratie hieronder:** de eerder
genoemde bovengrens ("~65-75 tok/s als de hele pijplijn zou verdwijnen") was
te optimistisch — de eager-pijplijnkosten direct op de V4-basislijn
plakken was appels-met-peren. De juiste bovengrens is: volledige eliminatie
van down_proj zou V4 van ~22,5 naar 16,0 ms/token brengen (**~62,4 tok/s**,
optimistisch/onhaalbaar aangezien STUB fout is). V5 dekt bovendien alleen de
launch-overhead-helft, niet de PCIe-kant — geschat (dezelfde verhouding als
de eager subkernel-split, 41,6% van de pijplijn) op hooguit **~2,7 ms/token**
haalbaar, oftewel richting **~46-50 tok/s** in het beste geval. Bijgewerkt
in `PRO_V5_PREREGISTRATION.md`'s claim-boundary sectie.

**Aanvullend, zelfde blok — batchen is architecturaal veilig, geverifieerd
uit de code (nog niet gebouwd).** `_moe_dev`'s `for s in range(top_k):`-lus
roept `down_masked_into_indirect` zes keer sequentieel aan met **hetzelfde**
scratch-object `self.mstate` (`alloc_masked_state`, één-expert-grootte,
éénmalig aangemaakt in `runtime.py:311`), en schrijft elk resultaat naar een
eigen slice `self.contrib[s*hidden:(s+1)*hidden]` (`self.contrib` is al
`top_k*hidden` vooraf gealloceerd, regel 314). Er is dus **geen
cross-slot-afhankelijkheid** — de zes expert-aanroepen zijn embarrassingly
parallel, alleen toevallig sequentieel geïmplementeerd met hergebruikt
scratch-geheugen. Batchen (6-voudig scratch + kernels die intern over de
slot-dimensie gridden i.p.v. zes losse launches van 4 kernels) zou de
berekende waarden niet veranderen — alleen de launch-granulariteit. Dat maakt
de correctheidskant van deze hefboom aantoonbaar behapbaar; de
implementatie-kant (kernels herschrijven om over 6 slots te gridden, met
correcte per-slot scratch-indexering) is nog steeds echt engineeringwerk,
maar niet blind engineeringwerk. Extra aanwijzing: `panel_scan_k` lanceert
met grid `(1,)`, block `(256,)` — één block, 138× per token — een
klassiek launch-overhead-bound patroon (nauwelijks parallellisme benut per
launch).

---

## 2026-08-16 — PRO G2 — K-token epoch-graph: technisch gesloten, geen bug

**Vraag.** Kan de bestaande token-graph (`rt._graph`, een geïnstantieerde
`cudaGraphExec_t` via CuPy) K keer als child in één parent-graph gevangen
worden, zodat één host-launch K tokens vooruitbrengt?

**Uitkomst.** Nee, met de huidige aanpak. `pro_research/epoch_graph.py --mode
smoke` (ongewijzigd, al aanwezig, nooit eerder gedraaid) faalt voor k=2 én k=4
met `cudaErrorStreamCaptureUnsupported: operation not permitted when stream is
capturing`, bij `rt._graph.launch(stream)` binnen `stream.begin_capture()`.

**Waarom.** `cudaGraphLaunch()` — het aanroepen van een reeds
geïnstantieerde/uitvoerbare graph — is zelf geen capturable API-call. Om een
graph als child-node in een andere capture op te nemen moet je de
graph-**template** (`cudaGraph_t`, vóór instantiatie) doorgeven aan
`cudaGraphAddChildGraphNode()`, niet de uitvoerbare `cudaGraphExec_t` die
`.launch()` gebruikt. `runtime.py`'s `setup_graph()` bewaart alleen het
geïnstantieerde object (`s.end_capture()`); de template wordt niet apart
vastgehouden. Dit is een CUDA-API-beperking, geen CuPy-instelling of bug in
deze pack.

**Status: `technical_blocked`, poort `parent_graph_ids_exact` niet bereikt —
eerlijke sluiting per de eigen regel van de pack** ("Unsupported nested
capture is a valid technical closure"). Niet geforceerd, geen alternatief
mechanisme stilletjes gesubstitueerd.

**Wat dit open laat.** De K-token-amortisatie-hypothese zelf is niet weerlegd
— alleen déze implementatiestrategie. Een diepere vervolgstap (niet in deze
sessie gedaan) zou de graph-template apart moeten bewaren tijdens
`setup_graph()`'s eigen capture en `cudaGraphAddChildGraphNode` rechtstreeks
via CuPy's lage-niveau runtime-bindings aanroepen — een aparte,
substantiëlere CUDA-engineeringtaak, geen kleine reparatie.

**Artefacten.** `pro_research/results/PRO_G2_EPOCH_GRAPH.json` (status
`technical_blocked`).

---

## 2026-08-16 — PRO V4 — graph-safe + selectieve ERVF fysiek geïntegreerd: 41,13 tok/s, alle poorten groen

**Vraag.** V3-G0S (graph-residentie alleen, +10,1%) en V3-G1B (selectieve ERVF
alleen, +10,73%) zijn los gemeten en mogen niet worden opgeteld (Amdahl-
interactie onbekend). Wat levert één fysiek geïntegreerde arm echt op?

**Mechanisme.** `_step_body_graph()` roept `self._attention`/`self._mamba` aan,
die zelf via `self.k.mv_bf16`/`mv_fp8_tensor`/`mv_f32` dispatchen
(`runtime.py:401-474`). CUDA-graph-capture legt vast welke kernel op
capture-moment achter dat Python-attribuut zit. Dus: eerst
`selective_ervf_v3._install_selective(rt, dense)` draaien (herbindt die drie
attributen voor de vier bevroren winnende vormen), **dan pas** `rt.setup_graph()`
— de ERVF-kernels worden dan mee vastgelegd in de graph, terwijl K/V/router op
productiekernels blijven binnen dezelfde graph. Nieuwe runner:
`pro_research/graph_selective_v4.py`, preregistratie
`PRO_V4_PREREGISTRATION.md` (bevroren vóór meting).

**Opzet.** EGR (productie, device-cache, geen graph) vs GRAPH_SELECTIVE
(selectieve dispatch geïnstalleerd vóór capture) vs DET (twee rollouts) vs CTL
(`bad_pick=1`-sabotage herbinnen dezelfde graph, moet falen). Structurele
verificatie: `rt._graph.debug_dot_str()` moet `pro_gemv_bf16_ervf16` én
`pro_gemv_fp8_tensor_ervf16` bevatten — bewijst dat de ERVF-kernels echt in de
graph zitten, niet alleen tijdens de warmup zijn aangeroepen.

**Uitkomst (full, 3 prompts × 256 tokens, 765 getimede samples).**

| arm | p50 | tok/s |
|---|---:|---:|
| EGR (zelfde sessie) | 31,1786 ms | 32,07 |
| **GRAPH_SELECTIVE** | **24,3152 ms** | **41,13** |

Winst: **6,8634 ms / 22,0%** — meer dan de grootste losse mechanisme-winst
(3,3841 ms) en dicht bij de naïeve som van beide losse smoke-winsten
(2,8931 + 3,3841 = 6,2772 ms), dus nagenoeg volledig additief met slechts een
kleine overlap-tax. GRAPH_SELECTIVE's eigen p50 (24,3152) ligt ook onder béíde
eerder apart gemeten mechanismen (28,6063 en 28,158 ms) — dat is het eerste
directe bewijs dat de twee winsten fysiek samen bestaan zonder elkaar op te
eten.

**Poorten.** `argmax_direct_tie` ✅ · `graph_dot_contains_ervf` ✅ (beide
kernelnamen aangetroffen) · `graph_selective_equals_egr` ✅ (bitexact op alle
drie prompts, 256 tokens) · `graph_selective_deterministic` ✅ ·
`bad_pick_control_diverges` ✅ (2 van 3 prompts wijken af zoals vereist — de
controle heeft dus onderscheidend vermogen) · `extra_vram_lt_64MiB` ✅ (4 MiB)
· `full_speed_gain_ge_2_5ms` ✅ · `full_samples_ge_500` ✅ (765).

**Wat dit betekent voor het 100 tok/s-doel.** 41,13 tok/s is 24,9% van het
ctx0-roofline-plafond (165 tok/s, hardware-eigenschap, modelonafhankelijk) —
op van ~17% (Nano-lijn) naar ~25%. Nog altijd een factor 2,4× te gaan tot 100.
Dit is wél het eerste fysiek geïntegreerde, onafhankelijk gepoorte resultaat op
het **juiste** doelmodel, en het bevestigt dat losse mechanismewinsten hier
grotendeels blijven optellen in plaats van elkaar te kannibaliseren — een
gunstig signaal voor het toevoegen van een derde/vierde mechanisme (K-token
epoch-graph, MTP) op dezelfde manier.

**Artefacten.** `pro_research/PRO_V4_PREREGISTRATION.md` ·
`pro_research/graph_selective_v4.py` ·
`pro_research/results/PRO_V4_GRAPH_SELECTIVE.json`.

---

## 2026-08-16 — PRO V3 anchor-onderzoek — verklaard: twee verschillende modellen, geen bug

**Vraag.** De gebruiker vroeg expliciet: V3-G0S/G1B (`pro_research`) draaien
bit-identiek EGR vs GRAPH_SAFE/SELECTIVE, maar beide wijken al bij token 1 af
van het bevroren `V36_DETERMINISTIC_ANCHOR.json`. Komt dat door
`nemotron_3_5_lightning_v35` versus het oudere ankerpad, of door een
runtime/model-identiteitswijziging?

**Antwoord: het ankerpad.** Twee fysiek verschillende checkpoints:

| | `models/nemotron_3_5_lightning` (het ankerpad) | `models/nemotron_3_5_lightning_v35` (pro_research default) |
|---|---|---|
| werkelijke identiteit | **NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4**, verkeerd gedownload en misleidend hernoemd | NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4, sha `6dbbd757…` |
| `max_position_embeddings` | 262 144 (Nano) | 1 048 576 (Lightning) |
| quantisatie | — | MIXED_PRECISION: experts+lm_head NVFP4, Mamba in/out FP8-per-tensor, attentie BF16 |
| bron | — | modelopt 0.44.0rc5, drieweg `quant_kind()` |

Bewijsketen: `reports/lightningstream_nemotron/N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md`
("De hele lijn draait op NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4") →
`N2R_V35_LAYOUT_REPORT_2026-08-14.md` (adjudicatie van de échte Lightning-download,
`models/nemotron_3_5_lightning_v35`) → `HANDOVER_TO_KIMI_2026-08-15.md`
("Model staat nu in `models/nemotron_3_5_lightning_v35`. Zet
`LS_MODEL_DIR=nemotron_3_5_lightning_v35` om ermee te draaien."). Bevestigd
lokaal: `config.json` van het ankerpad heeft `max_position_embeddings: 262144`
(Nano-plafond), het `_v35`-pad heeft `1048576` (Lightning-plafond).
`A1_ADOPTION_PRECONDITION.json.environment.model_dir` = `"nemotron_3_5_lightning"`
— A1/D1/E1-E6/NERVF-0..5 en `V36_DETERMINISTIC_ANCHOR.json` zijn dus **allemaal
gemeten op Nano**, ondanks dat de correctie al op 2026-08-14 bekend was. De
`scripts/treesweep200/*.py`-scripts defaulten nog steeds naar
`nemotron_3_5_lightning` (geen `LS_MODEL_DIR` override in die lijn) — de
"herhaal de meetketen op het juiste model"-stap uit N0R_CORRECTION is voor de
closed-namespace-lijn **nooit uitgevoerd**.

**Gevolg — geen bug, wel een scope-correctie.**
- V3's eigen interne vergelijkingen (EGR vs GRAPH_SAFE, BASE_A/SELECTIVE/BASE_B)
  zijn methodologisch geldig: alle armen laden hetzelfde model in hetzelfde
  proces. De 28,61 ms / 28,16 ms resultaten staan.
- De externe ankervergelijking in V3 is terecht `informative`, niet gating —
  precies zoals de V3-preregistratie het al voorzag.
- **Alles in dit bestand vóór 2026-08-16 (ERVF 1,936×, D1, A1, E1 fase 2.1,
  "37,49 ms/token", "26,7–29,5 tok/s") is gemeten op Nemotron-3-Nano, niet op
  Nemotron-3.5-Lightning.** `HANDOVER_TO_KIMI_2026-08-15.md` mat vóór A1/D1 al
  een kale Lightning-baseline (27,743 tok/s @ctx0, vóór ERVF/D1/A1) die al in de
  buurt zit van Nano's volledig-geoptimaliseerde 29,5 tok/s — Lightning's
  kleinere `lm_head` (NVFP4 i.p.v. BF16, 704→198 MB) en FP8-Mamba bewegen
  minder bytes per token. De kernelwinsten (ERVF, v4-attentie, D1) zijn
  architectuur-onafhankelijk aannemelijk overdraagbaar (N2R: "shape-identiek
  op 8 byte na"), en V3's eigen bitexacte pariteit op het echte Lightning-model
  bevestigt dat ze *fysiek* overdragen — maar de closed-namespace tok/s-tabel
  in `STATE_OF_THE_WORK.md` beschrijft strikt genomen Nano, niet het
  opdrachtdoel.

**Wat dit niet doet.** Geen enkele eerder gemeten winst wordt ingetrokken —
D1/A1/E1F21's eigen interne A/B's zijn ook allemaal single-model, dus intern
geldig. Het is puur een naamgevings-/scopefout die al één keer eerder werd
gedocumenteerd (N0R_CORRECTION) maar niet is doorgezet naar de
adoptiemetingen erna.

**Aanbeveling voor de volgende fase.** `pro_research` blijft op
`nemotron_3_5_lightning_v35` (correct) draaien — geen wijziging nodig. Wie ooit
weer in de closed `treesweep200`-lijn werkt, moet `LS_MODEL_DIR=nemotron_3_5_lightning_v35`
zetten, of nog beter: het ankerpad hernoemen zodat de mismatch niet blijft
terugkomen (`models/nemotron_3_5_lightning` → `models/nemotron_3_nano`, zodat de
naam niet meer liegt). Niet in deze sessie gedaan omdat schrijfrechten buiten
`agents/`+`pro_research/` niet zijn opgeëist.

**Artefacten.** `pro_research/PRO_V4_PREREGISTRATION.md` (model-identiteitsnoot
erin opgenomen) · bronnen: `N0R_CORRECTION_WRONG_CHECKPOINT_2026-08-14.md` ·
`N2R_V35_LAYOUT_REPORT_2026-08-14.md` · `HANDOVER_TO_KIMI_2026-08-15.md`.

---

## 2026-08-15 — E1 fase 2.2 — graph-replay GEBOUWD maar ongemeten (sessie gestopt op quota)

**Vraag.** Kan de hele token — embedding t/m argmax — als één CUDA-graph
replays worden nu het MoE-pad sync-vrij is?

**Status.** Preregistratie bevroren (`E1F22_GRAPH_CAPTURE_PREREGISTRATION`),
graph-API gesmoketest (capture → launch → correct over 5 replays), alle
kernels en runtime-methoden geschreven en op syntax gecontroleerd — maar de
A/B is **niet gedraaid**. Niets hieronder is een meting.

**Ontwerpkeuzes die de volgende agent niet opnieuw hoeft te bedenken.**
- `attn_decode_warp_fp8_gqa4_dp`: vaste grid (2,256); elke block schrijft áltijd
  zijn 4 partials — dode splits schrijven neutraal (m=-inf, l=0) — dus nooit
  stale data, en `attn_decode_combine` (vaste 1024 slots) slaat l≤0 al over.
  Zelfde optelvolgorde als eager → bitexact te verwachten, te bewijzen door
  de verifier.
- Embedding: tabel wordt bij `setup_graph()` naar pinned+mapped gekopieerd
  (+0,656 GiB host-RAM); `embed_gather_bf16` leest hem in-graph via tok_dev.
- Token-flow: argmax schrijft tok_dev aan het einde van replay N; embed_gather
  leest het aan het begin van replay N+1. Prompt-tokens staged de host met een
  stream-geordende 4-byte H2D (geen sync). Ids oogsten via pinned ringbuffer
  (`ring_harvest`); gegenereerde ids staan vanaf ring-index P-1.
- Kill-criteria staan in de prereg: K1 = event-fork weigert → single-stream
  fallback als aparte arm.

**Artefacten.** Prereg + code in `gpu_kernels.py`/`runtime.py` (zoek
"E1 fase 2.2"). Nog te schrijven: runner, verifier, rapport.

---

## 2026-08-15 — E1 fase 2.1 — device-resident routing werkt: −4,54 ms/token eager, alle poorten groen

**Vraag.** Kan de MoE-laag zonder één device→host-sync draaien (routerkop,
LRU, miss-staging als kernels), zodat graph-capture (fase 2.2) überhaupt kan —
en wat levert alleen dat al op?

**Opzet.** Eén variabele: `device_cache` aan/uit op de geadopteerde stack.
BASE = default, DEV = device routing+LRU (cap 72), INV = cap 56 moet dezelfde
tokens geven, CTL = `bad_pick`-sabotage moet falen. Pariteit tegen het
bevroren A1/V36-anker, 2 prompts × 64 tokens, contexts_max=4096. De
staging-kernel volgt het M1-patroon uit de microbench (bulk-read uit pinned
host = 24,93 GB/s, 96% van DMA), NIET de M2-variant (GEMV leest zelf van host:
7,27 GB/s — dood).

**Uitkomst.** p50 41,540 → **36,998 ms/token (−4,542 ms, −10,9%)**, pariteit
behouden. Dat is 51% van het 8,925-ms-budget uit fase 1; de rest is pure
launch-overhead voor fase 2.2. Verifier 14/14, inclusief bitexacte spiegels
(indirecte GEMV == directe ERVF; accumulate_ind == accumulate_into) en een
exacte Python-LRU-spiegel van `cache_assign`.

**Bug gevonden.** `enable_cache` resette `_dev_cache` niet → INV-arm draaide
cap-56-semantiek over vuile LRU-staat en faalde terecht. Fix: `_dev_cache = {}`
in `enable_cache`; INV+CTL opnieuw gedraaid met schone staat, beide groen.
Ook de verifier had zelf zo'n staat-desync (verse cache-buffers tegen oude
slot-tabellen) — zelfde les: *cache-inhoud en slot-staat zijn één invariant*.

**Poorten.** C1 ✅ · INV ✅ · CTL ✅ (schone attributie) · S1 ✅ (−4,542 ≥ 1,5)
· V1 ✅ (113 KiB analytisch).

**Wat dit opent.** Fase 2.2 (graph-capture van de hele token) heeft nu een
sync-vrij MoE-pad. Resterend capture-werk: embedding-gather, argmax,
pos-afhankelijke kernels (kv_write_fp8, attentie-splits) op device-pos.

**Artefacten.** `E1F21_DEVICE_ROUTING_AB.json` · `E1F21_INV_CTL_RERUN.json` ·
`e1f21_independent_verification.json` · rapport
`E1F21_DEVICE_ROUTING_REPORT_2026-08-15.md`.

---

## 2026-08-15 — A1 — de bewezen stack staat nu default aan, en het anker is opnieuw bevroren

**Vraag.** E6 mat dat ERVF + v4 + D1 sneller én exact is. Mag dat de default worden?

**Waarom niet meteen.** E6 vergeleek twee armen binnen één proces met dezelfde
cachegeschiedenis — precies het regime waarin de exactheid vóór D1 óók léék te
kloppen. Vier fasen (NERVF-3, NERVF-4, E4, S11) haalden hun pariteitspoort over
2×64 tokens terwijl de runtime niet deterministisch was. Een adoptie mag niet op
een blinde test rusten.

**Opzet.** Verander de **cachecapaciteit** (72 vs 56). Dat verandert het
hit/miss-patroon op vrijwel elke laag van elke token, dus de optelvolgorde, zonder
een gewicht/route/kernel aan te raken. Plus een **controle-arm die moet falen**:
dezelfde vergelijking zonder D1.

**Uitkomst.** Met D1: identiek over 2 × 256 tokens. Zonder D1: divergeert
(expository, token 224; narrative niet). De test heeft dus vermogen — maar smal,
en dat verklaart waarom de fout vier fasen lang onzichtbaar bleef.

**Poorten.** G-A1-CAP ✅ · G-A1-CTL ✅ (faalde zoals vereist) · G-A1B-DEFAULT ✅
(runtime zonder enige vlag reproduceert bit-identiek wat A1 mat) · G-A1B-FLAGS ✅.

**Wat dit opent.** Defaults om: `use_ervf=True`, `rt.attn=attention_fp8_gqa4`,
`deterministic_accum=True`. De attentiekernel wordt nu via `rt.attn` gekozen;
oude scripts die `rt.k.attention_fp8_gqa` overschrijven meten voortaan een
nul-verschil in plaats van stil iets verkeerds — bewust zo gekozen.

**Het anker.** V35 wordt niet meer gereproduceerd, divergentie al bij token 1.
Dat komt van D1, niet van v4 (E4 reproduceerde V35 wél). Het anker legde een
ordeafhankelijk artefact vast. Vooraf vastgelegde regel gevolgd: `V36_DETERMINISTIC_ANCHOR.json`
bevroren, V35 ongewijzigd bewaard, niet-vergelijkbaarheid opgeschreven.

**Artefacten.** `A1_ADOPTION_PREREGISTRATION_2026-08-15.md` ·
`A1_ADOPTION_REPORT_2026-08-15.md` · `A1_ADOPTION_PRECONDITION.json` ·
`A1B_ADOPTION_VERIFY.json` · `V36_DETERMINISTIC_ANCHOR.json`

---

## 2026-08-15 — E6 — geïntegreerd 41,98 → 37,49 ms per token, exact

**Opzet.** Drie armen `base_a / integrated / base_b`, 3 domeinen × 512 causale
tokens, D1 in **alle** armen (zonder D1 zijn twee armen niet eens vergelijkbaar).
Eén variabele: ERVF + v4-attention.

**Uitkomst.** +4,169 (expository) / +4,443 (narrative) / +4,858 (code) ms, elk
boven zijn eigen drift. Bit-identieke uitvoer. VRAM ongewijzigd.

**Poorten.** Exactheid ✅ · latency ✅ · VRAM ✅ · eindpoort ≥50 tok/s ❌ (26,7).

**Wat dit sluit.** Niets — maar het laat zien dat de resterende afstand tot 50
tok/s niet uit de gebouwde componenten kan komen. E2 is weerlegd en E1 fase 2 is
ongebouwd; dat zijn de twee posten die het plan ervoor had ingeboekt.

**Artefacten.** `E6_INTEGRATED_REPORT_2026-08-15.md` · `E6_INTEGRATED_RUN.json`

---

## 2026-08-15 — D1 — de runtime was niet deterministisch, en dat is nu opgelost

**Vondst.** `_moe_cached` accumuleert de zes routed experts in
**hit-dan-miss-volgorde**. Welke expert een hit is hangt van de LRU-staat af, dus
twee runs met andere cachegeschiedenis tellen in andere volgorde op. FP-optelling
is niet associatief → twee armen met **identieke configuratie** divergeerden over
512 tokens. Ontdekt doordat NERVF-5 `base_b` tegen `base_a` zette.

**Ingreep.** Rekenvolgorde en optelvolgorde gescheiden: rekenen blijft hit-eerst
(latencywinst blijft), bijdragen naar aparte slotbuffers, na de lus optellen in
routevolgorde `s = 0..5`. Kosten: `top_k × hidden` floats (64 KB), nul extra
kernels.

**Uitkomst.** `base_b` nu identiek aan `base_a` over 3 × 512 tokens; NERVF-5
slaagt alsnog op alle drie zijn poorten. ERVF-winst met D1 aan: +2,771 / +5,008 /
+5,395 ms.

**De les.** Vier eerdere exactheidspoorten waren waar voor hun eigen run maar
bewezen minder dan ze leken. Vandaar werkregel 8: bouw een controle-arm die moet
falen.

**Artefacten.** `D1_DETERMINISM_REPORT_2026-08-15.md` · `d1_determinism.json`

---

## 2026-08-15 — NERVF-0 t/m 5 — ERVF gerepliceerd op een tweede model

**Vraag.** Reproduceert de Qwen3-30B ERVF-doorbraak op een architecturaal andere
NVFP4 hybrid-Mamba MoE?

**Antwoord: ja.** 1,936× bitexact op het projectievlak, bij **dezelfde** gekozen
subwarp-breedte 16 die Qwen selecteerde, op een ander model, een andere
quantisatie (NVFP4 vs Q5/Q8) en een andere shape.

**Het mechanisme.** w-lane subwarps per rij, 256/w rijen per 256-thread block,
per lane aparte virtuele accumulatoren, en een reductie die de DAG van de
referentiekernel **exact** reconstrueert — de eerste butterfly-stap (offset 16)
wordt een lane-lokale optelling, offsets 8/4/2/1 blijven shuffles binnen de
subwarp, en de acht warp-sommen vouwen in registers in exact de volgorde
`((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

**Fout onderweg.** w=4 en w=8 gaven eerst 72/72 mismatches doordat ik de virtuele
accumulatoren *sequentieel* vouwde in plaats van in butterfly-volgorde. Na de fix
alle vier breedtes bitexact.

**NERVF-1 valstrik.** RAW_SCAN sprong van 9,77 naar 51,67 µs tussen runs: het
2,81 MiB record past in L2. Opgelost door alle armen door een 254 MiB pool van 95
replica's te cyclen → spreiding 0,1%. Elke bandbreedtemeting hierna doet dit.

**NERVF-4.** Weerlegd, zie E2.

**Poorten.** Doorbraakladder LEVEL 2 gehaald (≥1,35× exact). LEVEL 3 niet: het
volledige expertpad bevat de down-projectie, die ERVF niet raakt en waar NERVF-4
de voor de hand liggende route sloot. LEVEL 4 (≥35 tok/s) niet: 29,45.

**Niet geclaimd.** Geen nieuwheidsclaim — geen prior-art-audit, geen stock
llama.cpp-vergelijking, geen tweede GPU.

**Artefacten.** `NERVF_NEMOTRON_FINAL_REPORT.md` · verifier 66/66 in
`nervf_independent_verification.json`
