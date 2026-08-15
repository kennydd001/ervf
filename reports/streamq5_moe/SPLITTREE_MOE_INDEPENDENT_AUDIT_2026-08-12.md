# SplitTree-MoE / TriRail-Q5 — onafhankelijke audit

Datum: 2026-08-12  
Bron: `C:/Users/de_do/Downloads/SPLITTREE_MOE_SMOKING_GUN_REPORT_2026-08-12.md`  
Scope: lokale bewijsstukken en mechanische haalbaarheid; geen GPU-run en geen
wijziging van een experimentregistry.

## Kort verdict

**SplitTree is een geldige, mechanistisch nieuwe onderzoekshypothese, maar het
smoking-gunrapport overschat de huidige bewijssterkte.** De double-copydiagnose
is numeriek plausibel en OneCopy80 is de juiste eerstvolgende engineeringproef.
De grotere SplitTree-projectie rust echter op drie nog onbewezen aannames:

1. de Intel-iGPU haalt met een direct uit gedeeld hostgeheugen lezende,
   bitexacte Q5-kernel ongeveer dezelfde effectieve snelheid als de bestaande
   resident-FP16/OpenVINO-microbenchmark;
2. Intel-iGPU- en NVIDIA-dGPU-deelresultaten kunnen met dezelfde FMA-,
   denormal-, afrondings- en signed-zero-semantiek worden gemaakt;
3. de per-laag synchronisatie en de noodzakelijke tussenresultaten van
   gate/up, SwiGLU en down blijven klein genoeg.

Geen van die drie punten is door ERVF/ERGV, P11B of PORT80B bewezen. Daarom is
de juiste status **open hypothesis / high-value falsification target**, niet
Eureka en ook nog geen aannemelijke 15,16-ms expert-planevoorspelling.

Een belangrijk onderscheid:

- **OneCopy80** is direct afgeleid van PORT80B en moet vóór SplitTree worden
  getest;
- **WholeExpert-NearData** (vaste experts op de iGPU) past bij de genoemde
  393.216 bytes activatieverkeer;
- **SplitTree** (iedere GEMV-reductie over beide devices) heeft extra
  deelvectoren, quotient-merges en minstens twee synchronisatiebarrières per
  expertpipeline nodig.

## 1. Audit van de transport-"smoking gun"

### Wat lokaal hard is gemeten

De relevante lokale artefacten bevestigen:

- PORT80B verplaatste bij zero cache exact `973.209.600` actieve bytes per
  token via 480 expertrecords;
- PORT80B p50/p95 waren `63,034 / 73,544 ms`;
- de afzonderlijke pinned-H2D-roofline mat voor 512 MiB `26,159 GB/s`
  mediaan;
- de WSL2 STREAM-achtige CPU-proef mat bij acht threads `48,119 GB/s` copy;
- PORT80B had `Page Reads/sec p50 = 0`, maar p95 `93,406` en maximum `7.759`;
- de volledige bank had 46,845 GiB peak working set en bleef een uur actief.

De rekenkunde in het smoking-gunrapport klopt:

```text
973.209.600 / 48,119e9 = 20,225 ms
973.209.600 / 26,159e9 = 37,204 ms
som                       57,429 ms
57,429 / 63,034           91,11%
rest                        5,605 ms
```

### Wat daar nog niet uit volgt

De componenten komen uit verschillende processen en meetpaden. De
`48,119 GB/s` is expliciet een WSL2 STREAM-achtige arraycopy, niet de native
Windows-copy van 480 verspreide file-maprecords naar acht pinned windows. De
`26,159 GB/s` is een grote aaneengesloten pinned transfer, niet 480 transfers.
Het numerieke sluiten van de som is dus een sterke hypothese, geen causale
decompositie.

Verder is `42,81 ms = 37,204 + 5,605` een **medianeschatting**. Zij kan niet
worden vergeleken alsof zij de bevroren **p95**-poort van 45 ms bewijst. De
tails van registratie, Windows memory management, 480 bronranges en GPU-
dispatch zijn nog onbekend.

`Page Reads/sec p50 = 0` maakt continu hard disk-paging als verklaring voor de
mediaan onaannemelijk. De teller is echter systeembreed en niet per token
gealigneerd; hij ziet ook geen soft faults. De verdedigbare formulering is dus
"geen bewijs voor dominante continue hard paging", niet "disk paging kan de
mediaan niet verklaren".

**Conclusie transport:** ST0 en ST1 uit het rapport zijn methodologisch juist.
OneCopy80 is een directe PORT80B-vervolgproef en geen bewijs voor SplitTree.

## 2. Audit van de iGPU-roofline

P11B mat werkelijk op `Intel Arc Pro 140T GPU (32GB) (iGPU)`:

| resident FP16 OpenVINO, acht experts | p50 | p95 |
|---|---:|---:|
| latency | 1,1991 ms | 1,983525 ms |
| nominale FP16-parameterbytes / tijd | 62,962 GB/s | 38,062 GB/s |

De nominale bytecount is
`8 × 3 × 2048 × 768 × 2 = 75.497.472 bytes`, dus de omrekening klopt.
Maar P11B:

- compileerde constante FP16-matrices via OpenVINO;
- herhaalde honderdmaal dezelfde resident graph en dezelfde input;
- implementeerde geen Q5-unpack, BF16-schalen of ERGV-reductie;
- bewees geen zero-copy uit een grote host-USM-bank;
- vergeleek alleen de **NPU**-uitvoer met CPU; voor `GPU.0` werd geen
  correctheidsveld opgeslagen.

De `38,06 GB/s` is daarom geen gemeten Q5-hostbandbreedte. Het kan zowel te
optimistisch zijn (Q5-decode, USM-faults, compilersemantiek) als een verkeerde
kostenmaat zijn (FP16 GEMM-compute en caches zitten in dezelfde tijd).

Ook `32GB` in de OpenVINO-device-string is geen bewijs dat 27,56 GiB
modelpagina's naast OS, 46,5-GiB file mapping, dGPU-staging en runtimebuffers
resident blijven. Als USM dezelfde fysieke file-backed pagina's werkelijk
aliaset, hoeft er geen duplicaat te bestaan; dat moet met working-set-,
migration- en allocationtelemetrie worden aangetoond.

## 3. Audit van de 59/41-split

Onder ideale, onafhankelijke en volledig gelijktijdige kanalen is de algebra
correct:

```text
f_i = 38,06 / (38,06 + 26,159) = 0,592659
t   = 0,9732096 / (38,06 + 26,159) = 15,155 ms
bank_i = 46,496887 GiB × f_i = 27,557 GiB
```

Dit is uitsluitend een transport/nominale-work lower bound. De twee
"bandbreedtes" zijn niet gelijksoortig: de iGPU-waarde bevat een resident
FP16-operator, terwijl de dGPU-waarde alleen pinned H2D is. Voor de dGPU komen
Q5-kernelcompute en scheduling er nog bij tenzij zij volledig met DMA kunnen
overlappen. Beide devices delen bovendien CPU/DDR-verkeer; hun snelheden mogen
niet zonder een gelijktijdige controlemeter worden opgeteld. Dense Q8-shell,
router en attention gebruiken tegelijkertijd de dGPU.

De bronreductieboom telt 256 logische accumulators. Een exacte 59,27%-cut is
niet één subtree. De dichtste eenvoudige cut is `152/256 = 59,375%`; de
kleinste complete-subtreecover van de kleinere 104 bladeren bevat minstens
`64 + 32 + 8`, dus drie frontierwaarden per outputrij. Een 50/50-rootcut heeft
wel één frontierwaarde en is semantisch en communicatief veel eenvoudiger,
maar geeft als ideale transportbottleneck ongeveer:

```text
max(0,5 × 0,9732 / 38,06; 0,5 × 0,9732 / 26,159) = 18,60 ms
```

Dat is minder fraai dan 15,16 ms, maar nog steeds de rationele eerste
SplitTree-configuratie.

"Hele quantization groups" en "hele reductiesubtrees" zijn bovendien niet
automatisch hetzelfde. In de huidige ERGV-bronboom worden eerst
accumulatoren met indexafstand 128 samengevoegd. Q5-groepen zijn daarentegen
opeenvolgende blokken van 16 packs. Een groep-gebaseerde shard is dus geen
enkele bron-subtree. Exactheid vereist ofwel:

- een subtree-layout met gedupliceerde schaalprovenance en exacte repacking;
- of veel meer quotient-frontierwaarden voor een group-layout.

## 4. De 393-KiB-claim hoort niet bij SplitTree

De berekening zelf is exact:

```text
2 × 48 × 2048 × 2 = 393.216 bytes/token
973.209.600 / 393.216 = 2.475×
```

Maar dit telt één inputvector en één geaggregeerde output per laag. Dat is de
dataflow wanneer de iGPU **hele experts** uitvoert. Bij SplitTree wordt binnen
ieder geselecteerd expert gate, up én down gesplitst:

1. gate/up-partials moeten exact worden samengevoegd vóór SwiGLU;
2. de resulterende 512-dimensionale hiddenvector moet voor down op beide
   devices beschikbaar zijn;
3. down-partials moeten opnieuw exact worden samengevoegd.

Zelfs met slechts **één** cross-device FP32-partial per outputrij is een harde
ondergrens voor top-10 per laag:

```text
x naar iGPU                         2048 × 2         =   4.096 B
gate+up partials              2 × 512 × 10 × 4      =  40.960 B
expert-hidden naar tweede device    512 × 10 × 2     =  10.240 B
down partials                       2048 × 10 × 4    =  81.920 B
minimum per laag                                      137.216 B
minimum over 48 lagen                               6.586.368 B
```

Dat is 16,75× meer dan 393.216 bytes, al blijft het nipt onder 8 MiB. Voor de
voorgestelde ongelijke 152/104-cut zijn minimaal drie frontier-subtrees nodig;
dan stijgt deze eenvoudige ondergrens tot ongeveer 18,4 MB/token en faalt de
voorgestelde `<=8 MiB`-poort al vóór API-, padding- en synchronisatieverkeer.

De poort kan dus alleen logisch samen met een 50/50-rootcut blijven staan, of
de techniek moet worden hernoemd naar whole-expert near-data execution.

## 5. Exactheid: wat ERGV wel en niet overdraagt

ERVF/ERGV bewijst lokaal dat één CUDA-kernel dezelfde 256-accumulatorboom op
andere lane/warpgeometrieën kan uitvoeren. ERGV-C2 is sterk bewijs:

- alle gegenereerde Q5/Q8-widths bitexact;
- 63/63 onafhankelijke controles;
- 16,30% Q8- en 7,87% Q5-p50-winst tegenover uniforme P7;
- parity met de handgetunede N1C-graph.

De huidige `ExactReductionIR` heeft echter:

- één root en één fysieke CUDA-schedule;
- geen device-domain, forest-cut, transfernode of quotient-merge;
- een FMA-contract dat letterlijk
  `cuda-default-contract-same-reference-and-candidate` heet;
- geen Intel Level Zero/SYCL-codegenerator;
- geen expliciet cross-vendor FTZ/DAZ-, NaN- of signed-zerobewijs.

Daarom is SplitTree een echte uitbreiding van ERGV, geen reeds bewezen
toepassing. Wiskundig blijft bronboomgelijkheid mogelijk als uitsluitend
complete subtrees worden afgesneden, alle frontiernodes apart blijven en de
originele quotientboom ze in dezelfde operandvolgorde samenvoegt. Eén scalar
per device samenvoegen is voor een 59/41-cut in het algemeen **niet**
bitexact.

De ST3-regel "anders nul extra kwaliteitsregressie" mag niet als alternatief
binnen dezelfde exacte claim gelden. Een niet-bitexacte cross-vendorvariant is
een afzonderlijke approximate candidate en vereist een vooraf vergrendelde
modelkwaliteitstest.

## 6. Overlap met bestaand werk

| Onderdeel | Bestaand lokaal bewijs | Nieuwe SplitTree-gap |
|---|---|---|
| STREAMQ5 / PORT80B | echte 46,497-GiB synthetische hostbank; 480 records en actieve bytes fysiek gemeten | geen directe source pages, geen iGPU-compute, geen real-80B-kwaliteit |
| ERVF/ERGV | exacte bronboomvirtualisatie en CUDA-codegen binnen één NVIDIA-device | distributed forest IR, Intel-codegen, transfer- en quotientnodes, cross-vendor bewijs |
| P11B | Intel-iGPU is snelste host-side device in resident FP16 OpenVINO-proef | geen Q5, geen grote shared-USM-bank, geen iGPU-correctheidsaudit |
| TierFlow | 4,1577× verkeersreductie is trace-aritmetisch mogelijk bij 32,03% routesubstitutie | orthogonale trainingstechniek; niet nodig voor SplitTree en geen runtimebewijs |
| GaugePack | behoud van code/scale/group/leaf-provenance was als codecidee uitgewerkt | pruningpremisse is gesloten; lossless repacking voor deviceshards blijft bruikbaar zonder pruning |
| N4B-R | exacte synthetische 80B-vorm en resident dGPU-expertcompute p95 8,869 ms | geen echte H2D/iGPU/concurrency/full-stackmeting |

TriRail's NPU-leg heeft momenteel geen prioriteit. P11B liet voor acht experts
NPU `1,936 / 3,270 ms` p50/p95 zien, faalde zijn relatieve foutpoort en was
trager dan de iGPU. Een always-active shared expert op de NPU is een aparte
latere hypothese, niet gratis extra capaciteit voor de huidige SplitTree-claim.

## 7. EntropyPin-audit

De gemeten Qwen30-code-entropie `4,442757 bit/code` is echt; P8D telde
28.991.029.248 codes in de volledige lokale Q5-bank. De berekening
`4,442757 + 0,125 = 4,567757 bpp` tegenover `5,125 bpp` geeft inderdaad een
ideale zero-ordergrens van 10,873% voor die verdeling.

Niet bewezen zijn:

- dezelfde codeverdeling voor Qwen3-Coder-Next-80B;
- random-access tilemetadata, ANS-tabellen, offsets en alignment;
- worst-case tile-expansie en decodebuffers;
- fused decode-throughput boven het bespaarde H2D-/DDR-volume.

`46,497 -> 41,441 GiB` is daarom een lineaire scenarioanalyse, geen exacte
codec- of capaciteitsuitkomst.

## 8. Minimale falsifieerbare test

De goedkoopste test die de belangrijkste nieuwe empirische aanname isoleert
is **ST2-mini: iGPU Q5 shared-USM truth test**. Hij bewijst SplitTree nog niet,
maar kan de geciteerde iGPU-premisse direct sluiten zonder een 27,6-GiB-shard
of cross-vendor scheduler te bouwen.

### Vast protocol

1. Gebruik echte records uit de bestaande Qwen30-Q5-bank; minimaal gate/up en
   down, met een circulaire working set van minstens 512 MiB zodat dezelfde
   75-MiB OpenVINO-constanten niet uit cache worden herhaald.
2. Map de records als host/shared USM. Houd vóór en na de run de host working
   set, deviceallocaties en page-/migrationtelemetrie bij. Een verborgen
   duplicaat in private iGPU-memory is een hard fail.
3. Implementeer precies de bestaande Q5-code-, BF16-scale-, FMA- en
   256-leaf-ERVG-semantiek. Test normale, subnormale, signed-zero,
   cancellation- en willekeurige activaties.
4. Vergelijk ieder outputbit met de bestaande CPU/ERVG-referentie. Sla raw
   hashes en alle timings op.
5. Meet na warm-up minstens 1.000 gerandomiseerde recordselecties, AB/BA tegen
   een resident/private controlegroep, en rapporteer p50/p95 in effectieve
   **feitelijk gelezen Q5-bytes/s**.

### Harde stop/go

```text
GO alleen als:
  bitverschillen                    = 0
  verborgen private weightcopy      = 0 bytes
  post-warmup hard page-ins          = 0 in de meetvensters
  shared-USM full-path p95          >= 21,63 GB/s

anders: sluit de huidige SplitTree-roofline.
```

Na een pass volgt pas een **50/50 one-layer root-cut**: top-10
Qwen3-Coder-Next-vorm, één frontierwaarde per output, twee expliciete
SwiGLU/down-barrières, concurrent versus serieel, en bitexact tegen een
single-device ERGV-reference. Een 60/40-test hoort pas na een expliciete
frontier-trafficrekening.

## Eindbesluit

Het rapport bevat twee bruikbare routes, maar zij moeten uit elkaar worden
getrokken:

1. **OneCopy80:** hoge prioriteit; de double-copyverklaring is sterk genoeg
   voor ST0/ST1, maar nog niet bewezen en de p95-poort blijft open.
2. **WholeExpert-NearData:** eenvoudiger heterogene hypothese die werkelijk
   bij 393 KiB activatieverkeer past; route-imbalance en expertplacement worden
   dan de centrale vragen.
3. **SplitTree-ERVG:** wetenschappelijk interessanter, maar vereist een
   distributed-forestcompiler en heeft meer communicatie/synchronisatie dan
   het rapport telt. Begin met ST2-mini en daarna een 50/50-rootcut.

De lokale data openen dus een serieuze nieuwe onderzoeksbranch, maar de
15,16-ms-, 27,56-GiB- en 2.475×-argumenten vormen samen nog geen consistent
fysiek ontwerp.

## Geraadpleegde lokale artefacten

- `reports/streamq5_moe/PORT80B_P0_PHYSICAL_HOST_BANK_REPORT_2026-08-12.md`
- `reports/streamq5_moe/port80b_p0_physical_host_bank_result.json`
- `reports/offload_roofline/P_C_H2D_REPORT.md`
- `reports/offload_roofline/p_c_h2d_result.json`
- `reports/streamq5_moe/p11a_cpu_stream.json`
- `reports/streamq5_moe/P11B_NPU_EXPERT_MICROBENCH_PREREGISTRATION.md`
- `reports/streamq5_moe/p11b_npu_expert_microbench.json`
- `scripts/streamq5_moe/run_p11b_npu_expert_microbench.py`
- `src/moe_lab/ergv_compiler.py`
- `reports/streamq5_moe/ERGV_C2_PERFORMANCE_AUTOTUNER_REPORT_2026-08-12.md`
- `reports/streamq5_moe/TIERFLOW_F0_REPORT_2026-08-12.md`
- `reports/streamq5_moe/GAUGEPACK_P9D1_P9B_MUTATION_AUDIT_2026-08-12.md`
- `reports/streamq5_moe/N4BR_SYNTHETIC_80B_EXACT_REPLICATION_REPORT_2026-08-12.md`
- `reports/streamq5_moe/P8D_Q5_CODE_AUDIT.md`

