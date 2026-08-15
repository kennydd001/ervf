# Alle ideeën — definitieve lokale sluiting

Datum: 2026-08-12

## Antwoord

Ja, alle 48 geïnventariseerde ideeën uit de prompt, attachment,
`KERNEL_INVERSIE`, `RICHTINGEN_DOORGEREKEND`, `doorbraak.txt`, `info/pro` en de
overdraagbare `ds4-8gb-cpu`-ideeën hebben nu een expliciete status. Er staat
geen lokaal uitvoerbaar item meer op `queued`.

Dat betekent niet dat alle 48 fysiek konden worden uitgevoerd. De eerlijke
verdeling is:

| status | aantal | betekenis |
|---|---:|---|
| verified pass | 14 | fysieke/operationele poort gehaald |
| verified pass — alleen kwaliteit | 3 | full-depth kwaliteit gehaald; fysieke runtime nog niet gebouwd |
| verified negative | 19 | hypothese fysiek of empirisch gefalsificeerd |
| superseded | 3 | premisse door sterker bewijs vervangen of weerlegd |
| blocked artifact | 4 | checkpoint/runtimecomponent ontbreekt |
| blocked hardware | 2 | tweede/andere hardware ontbreekt |
| blocked measurement | 1 | vergelijkbare energiesensor ontbreekt |
| blocked scope | 2 | vereist training, niet alleen checkpoint-evaluatie |
| **totaal** | **48** | geen `queued` |

De volledige item-voor-itemadministratie staat in
`ALL_IDEAS_CLOSURE_REGISTRY_2026-08-12.yaml`. De onafhankelijke slotverifier
herberekende 23/23 geselecteerde provenance-, reken-, gate- en registrychecks.
De volledige Python-testsuite passeert 156/156.

## Sterkste bewezen systeemresultaat

P13C blijft de lokale Eureka:

- Qwen3-30B-A3B, batch 1, 4K context;
- harde Windows Job Object-limiet van 32 GiB;
- eigen Q5-expertbank, INT8-trunk, gepinde hostvensters en exacte EVT-PM-
  attention;
- 10.000 tokens zonder OOM;
- 14,2348 tok/s gemiddeld;
- 69,862 ms mean, 91,984 ms p95 en 100,498 ms p99;
- piek process-commit 10,185 GB (9,486 GiB) en piek working set circa 19,645 GiB;
- voorspellingen, missreeks en 4K-KV-digest exact gelijk aan P12R2.

Dit bewijst een sterke lokale systemscombinatie op deze laptop. Het bewijst
geen wereld-SOTA of brede nieuwheid.

## Nieuwe resultaten uit de slotcampagne

### Uniform Q5-kwaliteit is steviger geworden

De 10× grotere full-depth audit gebruikte 100 contexten en 12.700 labels:

- relatieve CE-toename +1,4517%;
- top-1-overeenkomst 92,9528%;
- 95%-contextbootstrap [+1,1542%, +1,7619%].

De 2%-kwaliteitspoort is dus niet alleen een toevalstreffer van de kleine test.
De dataset is wel corroboratief en niet volledig onaangeraakt.

### Replicatie op DeepSeek-V2-Lite

Q5 passeerde full-depth kwaliteit over 26 MoE-lagen:

- validation +0,716% relatieve CE;
- test +1,493%;
- test top-1 tokenovereenkomst 94,922%;
- mediane routeroute-overlap 96,745%.

Dit ondersteunt overdraagbaarheid naar top-6-routing en shared experts. Een
fysieke DeepSeek-bank/cache/runtime is nog niet getest.

### Mixed Q4/Q5

Een vaste selectie van twaalf Q4-lagen en 36 Q5-lagen passeerde validation en
test en projecteert 5% minder expert-codebytes. Dit is een geldige
kwaliteitskandidaat, nog geen snelheidswinst.

### 50% structured pruning: half positief, fysiek negatief

De vaste 384/768-neuronselectie passeerde full-depth kwaliteit op validation en
test. De logische nulmaskervariant lijkt dus kwaliteitsveilig. De naïeve echte
compactie naar nieuwe dichte Q5-groepen faalde echter hard: +48,027% relatieve
CE en 60,472% top-1 op validation. De test bleef terecht dicht.

De les is nieuw en belangrijk: kanaalselectie en quantisatiegroepindeling zijn
niet onafhankelijk. Een eventuele compacte opvolger moet oorspronkelijke
down-groepschalen en groepsidentiteit bewaren, of opnieuw gekalibreerd worden.

### GPU-router

De GPU-kernel reproduceerde op 480 echte routervectoren ids en BF16-gewichten
bitexact. De volledige routebarrière was desondanks 4,392× trager op p50 en
2,064× trager op p95. Zolang cacheplanning op de host plaatsvindt, is deze
variant gesloten.

### Externe runtimeanker

Actuele `llama.cpp` Q5_K_M op dezelfde machine, CPU-only met 16 threads, haalde
0,225149 decode-tok/s; P13C haalde 14,234758 tok/s, een verhouding van 63,22×.
Dit is geen vergelijking met de beste hybride llama.cpp/kTransformers-runtime:
de lokale build had geen CUDA-toolkit en gebruikte andere quantisatiesemantiek.

### Energie

P13C's 40 GPU-powerpunten geven gemiddeld 48,32975 W, oftewel een GPU-only
projectie van 3,395 J/token. Een volledige joule/tokenvergelijking is geblokkeerd
omdat CPU/DRAM/wandenergie voor beide paden niet meetbaar was.

## Doorbraakverdict

`breakthrough_claim_allowed` blijft **false**. Expertstreaming/caching,
CPU–GPU-MoE-samenwerking en getegelde exacte attention hebben duidelijke prior
art. De waarschijnlijke eigen bijdrage is smaller: de specifieke combinatie van
exacte numerieke semantiek, fysieke Q5-bank, pinned-window streaming, expliciete
RN-reducties, harde 32-GiB-gate en een ongewoon strikte falsificatieketen.

Voor een verdedigbare bredere claim ontbreken nog:

1. een optimale publieke hybride baseline op dezelfde laptop;
2. een volledige fysieke DeepSeek-replicatie;
3. onafhankelijke uitvoering door een andere machine/persoon;
4. publieke benchmarks met vooraf verzegelde prompts en kwaliteitsevaluatie;
5. een gerichte patent/code-literatuurreview van de exacte combinatie.

## Wat nu rationeel is

De volgende hoofdregistry moet geen nieuwe vrijblijvende ideeënlijst zijn. De
hoogste informatiewaarde zit in twee afgebakende projecten:

1. een **group-identity-preserving compacte pruningbank** die geselecteerde
   down-kanalen bewaart zonder ze tot nieuwe Q5-groepen te herschikken;
2. de **volledige fysieke DeepSeek-V2-Lite-replicatie**, inclusief bank, cache,
   kernels, 4K decode en harde geheugenlimiet.

Beide moeten als nieuwe registries starten. De huidige 48-itemcampagne is lokaal
gesloten en mag inhoudelijk niet achteraf worden aangepast.
