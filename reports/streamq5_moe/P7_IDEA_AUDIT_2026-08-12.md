# P7 idee-audit — van kernel-inversie naar ERVF

Datum: 2026-08-12

## Besluit

De sterkste aangeleverde hypothese zat niet in extra quantisatie of een nieuwe
cachelaag, maar in de launchgeometrie. De oorspronkelijke formulering van
`KERNEL_INVERSIE_2026-08-12.md` veronderstelde dat P6B eerst naar een
gedequantiseerde BF16-scratch schreef. Broninspectie falsificeerde dat: Q5 en Q8
worden al uit codes en schalen in registers gereconstrueerd en direct met de
activatie vermenigvuldigd. De echte fout was één volledig 256-thread block per
outputrij, inclusief acht blockbrede synchronisaties.

P7A mat dit vervolgens fysiek:

| Bank | Raw scan | Row-pattern | GEMV | Pattern/raw | Diagnose |
|---|---:|---:|---:|---:|---|
| Q8 | 356,81 GB/s | 96,43 GB/s | 91,16 GB/s | 27,03% | geometrie/reductie/launch dominant |
| Q5 | 361,32 GB/s | 89,41 GB/s | 65,10 GB/s | 24,75% | geometrie/reductie/launch dominant |

Hieruit ontstond **Exact-Reduction Virtual Fusion (ERVF)**: één 16-lane subwarp
verwerkt één rij, zestien rijen delen een block, maar iedere lane emuleert de
oorspronkelijke virtuele CUDA-threads en exact dezelfde FP32-reductieboom. Het
is daardoor een architectuurwijziging zonder numerieke wijziging.

## Beoordeling van de aangeleverde richtingen

| Richting | Uitkomst | Reden |
|---|---|---|
| Kernel-inversie / fused dequant | Premisse gefalsificeerd; kernidee omgebouwd | Directe registerdequantisatie bestond al. ERVF pakte de echte reductie- en occupancyfout aan. |
| ds4/DwarfStar subwarp-rows | Nuttige onafhankelijke aanwijzing, geen vergelijkbare benchmark | De bron bevat warp/quarter-warp row-kernels. Dat ondersteunt de geometrie, maar de repo demonstreert vooral CPU/NVMe-capaciteit voor een ander model. |
| ds4 NVMe demand paging | Capaciteitsrichting, niet de huidige snelheidsoplossing | P6B heeft een resident Q8-trunk en een pinned Q5-bank; NVMe vervangt hier geen getimede GPU-GEMV. |
| Domeingeconditioneerde dynamische cache | Geldig en reeds actief in P6B/P7C | Het cachebeleid, de echte missers en H2D-kopieën bleven in de end-to-end-test aanwezig en exact gelijk. |
| Alleen CUDA Graphs | Eerder gesloten | P3B validation gaf graph/eager host-p50-ratio 1,0076; graph replay versnelde de echte keten niet. |
| CPU-compute bij cachemiss | Niet geselecteerd | Een STREAM-bandbreedtegetal bewijst geen snelheid voor verspreide Q5-expert-GEMV. Na ERVF moet dit tegen de nieuwe 7,61 ms expertvloer worden getest, niet tegen de oude kernel. |
| Statische expertpruning | Niet geschikt voor de exacte P7-claim | Kan bytes/tijd besparen, maar verandert modelsemantiek en vereist een nieuwe grote kwaliteitsevaluatie. |
| Q5 als INT4-kern plus overflowstaart | Geldige capaciteits-/kernelhypothese voor P8 | Vereist eerst een volledige codehistogram- en tail-layouttest. Hij was niet nodig om P7-snelheid te winnen. |
| Groter model / Qwen3-Coder-Next | Replicatierichting | Kan generaliteit en active-set-invariance testen, maar bewijst niet de huidige kerneloorzaak. |
| TierFlow-training | Langetermijnonderzoek | Interessant als hardware-aware trainingsdoel, maar niet falsifieerbaar met alleen deze bestaande checkpointbank. |

## Eigen vervolgideeën

1. **Projection-Adaptive ERVF** — kies subwarpbreedte apart voor Q8-projectietype
   en voor Q5 gate/up versus down. P7B koos nu één globale breedte; dat kan lokale
   optima verbergen.
2. **Scale Broadcast ERVF** — laad een Q5-schaal één keer per 16-lane subwarp en
   distribueer de ongewijzigde BF16-bits met een shuffle. Dit kan redundante
   schaalloadinstructies verwijderen en bitexact blijven.
3. **Miss-Budget Controller** — optimaliseer cachebudget op de nu versnelde
   kernelvloer. Het doel wordt p95-latentie onder een expliciet miss-bytebudget,
   niet alleen hitrate.
4. **Exact INT4-core + sparse overflow** — hercodeer Q5-codewaarden binnen
   `[-7, 7]` als vier bits en bewaar uitsluitend de exacte overschrijdingen. De
   hypothese gaat alleen door als de volledige bankhistogrammen een voldoende
   dunne staart aantonen en de extra lookup de ERVF-winst niet opeet.

## Wetenschappelijke grens

P7 bewijst een substantiële, bitexacte lokale versnelling ten opzichte van onze
eigen P6B-runtime. Het is geen bewijs dat niets bestaands sneller is. Daarvoor
zijn nog dezelfde-hardwaremetingen nodig tegen actuele llama.cpp/MLX/vLLM- of
andere relevante runtimes en, voor nieuwheid, een bredere literatuur- en
prior-artscan.
