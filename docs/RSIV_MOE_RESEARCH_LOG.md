# RSIV-MoE / GhostWeights research log

## 2026-08-11 — P0 onafhankelijk geopend

De gesloten CRAFT-MoE-registry blijft onaangeraakt. `RSIV_MOE_V1` is als
afzonderlijke hypothese geopend met een nieuwe namespace, vaste model- en
datasetrevisions en een vooraf vastgelegde P1-rankcensus.

De eerste proef gebruikt lagen 1, 13 en 26 van DeepSeek-V2-Lite, natuurlijke
top-6-routes en onafhankelijke WikiText-blokken. Rank, threshold, splitregels,
partiële koude-byteboekhouding, exacte algebraïsche controles en het
validation-naar-testprotocol zijn vastgelegd voordat nieuwe activaties worden
geanalyseerd. P1 kan uitsluitend een rank-/page-faultscreen opleveren; snelheid,
kwaliteit en Eureka blijven buiten bereik totdat latere fasen die fysiek meten.

## 2026-08-11 — P1 capturepoging 001 veilig gestopt

De eerste capturepoging stopte bij de aanvullende layer-1-fallbackregressie,
voordat een capture- of resultaatbestand was geschreven. De officiële MLP werd
buiten `torch.inference_mode()` opnieuw aangeroepen met een inference-tensor;
PyTorch blokkeerde terecht een mogelijk autogradpad. Er zijn geen meetregels,
data, thresholds of gates gewijzigd. De enige codecorrectie plaatst deze
controleaanroep in `torch.inference_mode()`; poging 002 hergebruikt exact de
preregistratie.

## 2026-08-11 — Validationpoging 001 ongeldig door extra BF16-control

De validationgrid en alle vooraf geregistreerde algebraïsche controles zijn
berekend zonder testslices te lezen. Route/count/bound-controls en de echte
full-rank FP32-operatorimagecontroles voor `x/g/u/z/y` slaagden. De lock werd
toch ongeldig verklaard doordat de evaluator aanvullend bit-exactheid eiste van
opgeslagen BF16-`z` na hergroepering naar andere per-expert GEMM-batchvormen.

Die extra eis stond niet in de preregistratie en is numeriek onjuist: dezelfde
BF16-rijen kunnen door een andere GEMM-batchvorm een andere accumulatievolgorde
krijgen. De bewaarde verschillen waren maximaal 0,125/0,25/1,0 per laag,
terwijl alle vooraf geregistreerde FP32-imagecontroles ruim binnen hun
`2e-5`-relatieve en `2e-4`-absolute grenzen bleven. De ongeldige lock blijft
behouden onder SHA-256
`1ff70f0657b873b9cebbed19dc93b6c95c4a0ad739bf1da129ab3dd0713e3eb1`.

Addendum 001 verwijdert uitsluitend deze niet-vooraf-geregistreerde
batchvorm-bitexactheid uit de verplichte controls. Zij blijft als diagnostiek
zichtbaar. Data, grid, gates, selectieregel en testslot blijven ongewijzigd en
test is nog niet geopend.

## 2026-08-11 — Verifierpoging 001 bewaard

De eerste P1-verifier rapporteerde 7.741 geslaagde inhoudelijke controles, één
waarschuwing en één fout. De enige fout was een fragiel tekstanker: de code
zocht de aaneengesloten woorden `geen Eureka`, terwijl het analyserapport de
claimgrens als `geen Eureka-verdict` formuleert en elders een regelafbreking
tussen beide woorden heeft. Dit raakt geen numeriek resultaat of gate.

De mislukte JSON/Markdown-audit blijft behouden met hashes
`1f3a42af8ce397654a28a36cf1846865352ecc63d0d5992168705f12ed3e7230` en
`1dbccaec28084a378bcc9483fa6bdfc54e2a043b2067907ec634ce72b467ea74`.
Verifier v2 controleert het bestaande, expliciete woord `Eureka-verdict` en
verifieert bovendien de hashes van de bewaarde v1-uitvoer.

## 2026-08-11 — P1 V2 terminaal negatief, RSIV cross-architectuur open

De eenmalig geopende test bevestigt het validationbesluit. De bevroren
diagnostische kandidaat (`rank=4`, `threshold=0,001`) bereikt voor zowel de
offline trainbasis als causale 96→32-prefixtransfer `0%` double-gate-fast en
`1,000×` geprojecteerde koude-bytereductie op validation én test. Hij mist de
`92%`/`10×`-gate volledig.

Dit is niet alleen een gevolg van de conservatieve lock. Na de selectie haalt
zelfs `rank=128`, `threshold=0,10` slechts `0,629%/0,749%` offline double-fast
op validation/test en `1,006×/1,008×` cold reduction. Causale prefixtransfer
blijft nul. Bij de primaire rank-32-cap is test `0,412%` en circa `1,006×`.

De exacte rangcensus verklaart waarom. De opgeslagen `x`- en `z`-matrices zijn
vrijwel steeds full row-rank; de full-rank expertatlas verbruikt
`99,902–100,000%` van de expert-count-cancellationbound. Bij threshold 0,001 en
cap 128 voegt ieder van de 3.072 testinvocations per laag een nieuwe input- én
intermediairrichting toe. Binnen de 128-tokenblokken is de groei dus lineair,
niet verzadigend.

Alle verplichte controles slagen. De onafhankelijke v2-verifier herberekent
7.744 controles: 7.744 geslaagd, nul fouten en één gedeclareerde waarschuwing
over niet-bitexacte BF16-hergroepering tussen verschillende GEMM-batchvormen.
Die waarschuwing raakt geen route, rank, FP32-operatoridentiteit of gate.

Besluit: P1 sluit op V2 als `screen_negative_v2`; P2-operatorimages en runtime
worden hier niet gebouwd. Het overkoepelende RSIV-verdict blijft
`inconclusive_higher_e_replication_required`, omdat de vooraf vastgelegde
schaalvoorspelling nog op een hogere-E-familie moet worden getest. Zo'n vervolg
vereist vóór download een eigen registry-item en preregistratie.

## 2026-08-11 — P1B V2-long-prefix vooraf geregistreerd

P1A gebruikte per causaliteitsblok 96 prompt- en 32 futuretokens. Dat is een
geldige negatieve pilot, maar niet dezelfde conditie als de centrale
T=1.024-promptcompilatie uit de GhostWeights-hypothese. Vóór een checkpoint van
57–98 GB wordt gedownload, test P1B daarom de goedkoopste ontbrekende verklaring:
kan een prompt-specifieke basis na 1.024 causale tokens de volgende 128 tokens
wel opvangen?

Twee vaste validation- en twee vaste testcontexten van elk 1.152 tokens worden
op lagen 1/13/26 gemeten. De rank-/thresholdgrid, `92%` double-fastgate,
`10×`-cold-bytegate, partiële G/U/D-boekhouding en validation→testdiscipline
blijven identiek. P1A wordt niet heropend of hernoemd. Bij een geldige P1B-
failure gaat de volgende rankcensus naar een afzonderlijk vooraf geregistreerde
hogere-E-familie; bij succes is uitsluitend een nieuwe P2-preregistratie
toegestaan.

## 2026-08-11 — P1B validationpoging 001 ongeldig door extra extreemcheck

Alle vooraf geregistreerde P1B-controls slaagden: routes en routergewichten
sluiten exact, counts en opslagbound sluiten, en volledige prefix-`x/z`-bases
reconstrueren met maximale relatieve L2 rond `2,1e-15`. De vereiste upstream
P1A-operatoridentiteit bleef eveneens geldig.

De implementatie maakte de lock toch ongeldig doordat zij vrijwillig de oude
absolute FP32-imagegrens opnieuw op tweemaal zoveel promptactivaties toepaste.
Op laag 26 bleven relatieve `z/y`-fouten rond `9,2e-7/1,0e-6`, maar de grootste
absolute uitschieters werden `4,88e-4/3,97e-4` in plaats van maximaal `2e-4`.
P1B had uitsluitend de upstreamidentiteit en nieuwe FP64-prefixprojectie als
vereiste vastgelegd; deze extra sample-extreemcheck was niet geregistreerd.

De ongeldige v1-lock blijft behouden onder SHA-256
`192be7621be51d3679f0e65613b5aae5539b4e75779ad5534f81083594a8de84`.
Addendum 001 houdt de extra long-prefixmeting zichtbaar als diagnostiek, maar
laat haar de P1B-lock niet langer ongeldig maken. Data, grid, gates en
validationselectie blijven onveranderd; test is nog niet geopend.

## 2026-08-11 — P1B long-prefix terminaal negatief en onafhankelijk bevestigd

De eenmalig geopende P1B-test bevestigt dat een causale prefix van 1.024 tokens
V2-Lite niet redt. De op validation vergrendelde diagnostische kandidaat
`rank=32`, `threshold=0,10` bereikt op test slechts `0,434%` double-fast en
`1,007×` geprojecteerde koude-bytereductie. Zelfs de na afloop gerapporteerde
ruime diagnose `rank=128`, `threshold=0,10` blijft op `1,866%` en `1,033×`.

De exacte prefixcensus geeft in alle zes context-laagcombinaties gemiddeld 96
input- en 96 intermediaire richtingen per expert: exact de gemiddelde
observatiebound. De rang groeit dus tot de beschikbare steekproeflimiet en
vertoont geen bruikbare saturatie. De onafhankelijke verifier herberekent 3.905
verplichte controles; alle 3.905 slagen, zonder fouten en met één reeds
gedeclareerde waarschuwing over een niet-verplichte absolute FP32-extreemcheck.

P1B sluit als `long_prefix_screen_negative_v2`. P2 op V2 blijft geblokkeerd.
Volgens de vooraf vastgelegde cross-architectuurregel gaat P1C naar een
afzonderlijk gepinde hogere-E-familie.

## 2026-08-11 — Qwen3-30B-A3B gekozen en P1C vooraf geregistreerd

De officiële Base-checkpointgegevens zijn vóór acquisitie vastgelegd. Qwen3
heeft 128 experts, top-8 routing, `d=2048`, `m=768` en 48 lagen. De exacte
revisie `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9` bestaat uit 16 BF16-shards
van samen 61.066.575.648 bytes. Bij 1.024 prefilltokens daalt de verwachte
routerbelasting van V2's 96 naar 64 invocaties per expert.

Model, commit, lagen `[0,23,47]`, contextsplit, grid, gates, validationlock,
resourceplafonds en hard-falsificatieregel zijn bevroren in P1C. De test sluit
als `falsified_rank_working_set` wanneer rank 32 opnieuw minder dan 80%
double-fast haalt. Een positieve P1C geeft uitsluitend toestemming voor P2 en
is geen Eureka- of runtimeclaim.

De pre-downloadcompatibiliteitscontrole vond dat de bestaande
`transformers==4.46.3` geen native Qwen3-MoE-klasse exporteert. Daarom is vóór
gewichtacquisitie `transformers==4.51.3` gepind, passend bij de officiële
Qwen-configuratie (`transformers_version=4.51.0`). De upgrade is alleen geldig
als alle bestaande V2-regressietests daarna blijven slagen.

## 2026-08-11 — Qwen-acquisitiepreflight 001 veilig gestopt

De eerste manifest-only aanroep stopte vóór enige checkpointdownload met een
lokale `TypeError`: een lijst dictionaries werd zonder expliciete sorteersleutel
gesorteerd. Er zijn nul gewichtbytes opgehaald. Poging 001 is als JSON bewaard;
de enige correctie sorteert hetzelfde manifest op bestandsnaam. Model, commit,
shardverwachtingen, gates en protocol veranderen niet.

## 2026-08-11 — Qwen-transportpoging 002 hervatbaar onderbroken

Het exacte manifest slaagde, maar vier standaardtransfers bleven na circa twee
minuten sterk ongelijk en leverden te weinig totale throughput. Het
transportproces is daarom bewust met `KeyboardInterrupt` gestopt. De lokale map
bevatte daarna 284.258.163 bytes metadata en hervatbare sharddelen; niets is
verwijderd. Poging 003 gebruikt hetzelfde model, commit en bestandspatroon met
de reeds geïnstalleerde officiële Xet-backend in high-performance-modus. Dit is
uitsluitend een transportinstelling en verandert het experiment niet.

High-performance Xet met vier gelijktijdige bestandstaken verbeterde de
doorvoer niet: na ruim twee minuten was geen shard compleet en bleven de vier
verbindingen vrijwel stilstaan. Transportpoging 003 is daarom eveneens bewust
en hervatbaar gestopt. Poging 004 serialiseert de bestandstaken (`max_workers=1`)
zodat één 4-GB-shard zonder cross-file contention kan voltooien. De officiële
hashcontrole na afloop blijft ongewijzigd.

Ook één Xet-bestandstaak schreef gedurende meerdere minuten geen nieuwe bytes.
Transportpoging 004 is vastgelegd en gestopt. Poging 005 schakelt Xet uit en
gebruikt de standaard HTTP-backend van dezelfde Hub-client, nog steeds met één
hervatbare worker. Alleen het transportpad wijzigt; de lokale shards moeten na
afloop nog altijd bytegrootte én officiële LFS SHA-256 evenaren.

## 2026-08-11 — Echte Qwen-laagstreaming vóór data-opening bevestigd

Met het reeds complete officiële shard 1 is laag 0 geladen en uitgevoerd op
batch 2 × 1.152 deterministische synthetische token-ID's. Validation en test
bleven ongeopend. De BF16-forward duurde 0,542 s na 0,924 s laden, gebruikte
maximaal 1.399.409.152 CUDA-bytes en 2.138.644.480 proces-RSS. Output was
eindig; route-ID's, routergewichten en routerlogits sloten exact met de native
Qwen-blockforward. De geplande laagstreaming past dus ruim binnen beide
resourceplafonds. Dit is geen rankscreen of runtimevergelijking.

## 2026-08-11 — Qwen-checkpoint volledig en lokaal geverifieerd

Transportpoging 005 voltooide via de standaard HTTP-backend met één worker.
Alle 16 officiële BF16-shards zijn aanwezig: exact 61.066.575.648 bytes. Ieder
lokaal shard is opnieuw gehasht en alle 16 SHA-256-waarden zijn gelijk aan de
officiële LFS-manifestwaarden op commit
`1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`. De acquisitie duurde van
08:21:27 tot 09:00:42 UTC. Validation en test waren gedurende deze fase nog
niet geopend; P1C mag nu de validation-capture starten.

## 2026-08-11 — P1C validation vergrendeld en test één keer geopend

De echte Qwen-captures gebruiken lagen 0, 23 en 47, twee contexten per split,
1.024 causale prefilltokens en 128 toekomstige tokens. Alle route-ID's,
routergewichten en logits sloten exact met de officiële forward; iedere laag
bevat exact 18.432 routed invocaties per split en alle verplichte controls
slaagden.

Geen enkele validationkandidaat met rank maximaal 32 haalde een niet-nul
double-fast bottleneck. De vooraf bepaalde selectieregel vergrendelde daarom de
diagnostische kandidaat `rank=4`, `threshold=0,001` vóór de testcapture. Pas
daarna werd de test één keer geopend. De validation-lock heeft SHA-256
`19038b6a718ebb5b6607dd52c4c8ca59b1542b746711c77ff0568e8a72804f3f`.

## 2026-08-11 — Qwen hard-falsificatie en verifier-addendum 001

De vergrendelde kandidaat behaalt op test 0% double-fast en 1,000× reductie.
De vooraf geregistreerde hard-falsificatiediagnose `rank=32`,
`threshold=0,10` behaalt 0% op validation en slechts 1,742% double-fast met
1,034× reductie op test. Zelfs rank 128 bij threshold 0,10 bereikt maar 5,762%
en 1,108×. De inputrang is op alle censuscheckpoints exact gelijk aan het
aantal observaties; de exacte atlas gebruikt 98,98–100% van zijn theoretische
observatiebound.

Verifierpoging 001 faalde vijf extra normalisatiesomchecks doordat haar
zelfgekozen BF16-tolerantie 0,002 kleiner was dan de geldige unit-roundoff
`eps/2=0,00390625`; de maximale gemeten afwijking was 0,00244140625. De
ongeldige outputs zijn onveranderd en hash-verankerd bewaard. Addendum 001
wijzigt alleen deze extra verifiercontrole. De definitieve audit bevestigt
4.429/4.429 verplichte checks, nul fouten en één gedeclareerde waarschuwing.

## 2026-08-11 — RSIV_MOE_V1 terminaal gesloten

Hard-falsificatieregel 1 is nu gereproduceerd op DeepSeek-V2-Lite én de vooraf
geregistreerde hogere-E-familie Qwen3-30B-A3B. De exacte
expert-count-cancellationbound blijft waar, maar levert in de gemeten data
geen lage-rank future-working-set op. P2 tot en met P7 worden niet geopend:
hun noodzakelijke page-faultvoorwaarde is al met ruime marge onmogelijk en
achteraf Kimi/V4 kiezen zou de bevroren stopregel schenden.

Registry-uitkomst: `falsified`. Eureka: nee. Het terminale rapport staat in
`reports/rsiv_moe/RSIV_MOE_FINAL_VERDICT.md` met SHA-256
`c5a77b3c767255d93d02d6bc7ecf19660cd2078e8482b89db5f391df543a840a`.
