# Primaire-bronnenonderzoek: GaugePack, ERGV en Qwen3-Coder-Next-80B-baselines

Datum: 2026-08-12  
Status: evidence report, geen patent- of volledige prior-artsearch

## Kort verdict

1. **GaugePack ligt in een druk onderzoeksgebied, maar de voorgestelde
   semantische combinatie blijft zinvol om te testen.** Neuronsparsity,
   pruning plus quantization, sparse low-bit-kernels en offline weight
   repacking bestaan al. De hier relevante afbakening is veel smaller:
   fysieke compaction van een *reeds bevroren*, groupwise-gequantiseerde
   SwiGLU/MoE-expert, met behoud van de oorspronkelijke codes, schalen,
   quantization-group-ID's én de bestaande floating-point-reductieboom.
2. **ERVG/ERVF is niet hetzelfde als gewone deterministische of
   reproduceerbare reductie.** Bestaande libraries fixeren een boom of maken
   de uitkomst juist onafhankelijk van de boom. ERGV wil een gekozen oude
   boom op een andere fysieke threadindeling exact emuleren. Dat is de
   verdedigbare onderzoeksgrens, mits zij algemeen wordt gemaakt en mechanisch
   wordt geverifieerd.
3. **Er zijn geloofwaardige publieke 80B-resultaten op 8-GB-GPU's, maar geen
   rechtstreeks vergelijkbare meting voor de lokale RTX PRO 2000 Blackwell
   Laptop gevonden in de geraadpleegde primaire bronnen.** De beste 8-GB-
   referenties lopen van 1,75–1,80 tok/s voor een experimentele FP8-
   safetensorsstreamer tot 20–21 tok/s voor llama.cpp Q4_K_M met veel host-RAM.
   Dat bereik toont vooral hoe bepalend runtime, quantformaat, context en
   cachebeleid zijn. Een synthetische 80B-vormprojectie is hiermee niet
   vergelijkbaar.

Een niet aangetroffen exacte match is **geen bewijs van nieuwheid**. De
formuleringen hieronder zijn daarom novelty boundaries en testadviezen, geen
claims dat de ideeën als eerste bestaan.

## 1. GaugePack: wat bestaat al?

### Nabije primaire bronnen

| Bron | Wat al bestaat | Waarom dit GaugePack niet volledig afdekt |
|---|---|---|
| [SparseGPT](https://arxiv.org/abs/2301.00774) | Eenmalige post-training pruning tot circa 50%, inclusief 2:4/4:8 en compatibiliteit met weight quantization. | Pruning is approximatief; de bijdrage specificeert niet het bitexact bewaren van de originele group-ID/scale-topologie én een gekozen oude reductieboom voor frozen Q5 SwiGLU-MoE-experts. |
| [SliceGPT](https://arxiv.org/abs/2401.15024) en [officiële implementatie](https://github.com/microsoft/TransformerCompression) | Fysiek kleinere dense matrices door transformationele invariantie en het verwijderen van rijen/kolommen. | Verandert de modelbasis en accepteert kwaliteitsverlies. Het is geen fysieke encoding van exact dezelfde reeds geprunde, groupwise-gequantiseerde operator. |
| [MARLIN](https://arxiv.org/abs/2408.11743) en [officiële repo](https://github.com/IST-DASLab/marlin) | Offline reshuffling van quantized weights en scales naar een hardwarevriendelijke layout; snelle INT4-kernels. | Toont dat logische quantdata en fysieke layout losgekoppeld kunnen worden, maar niet arbitrary neuronmask-compaction met originele group-ID's en legacy-reductiebitpatronen. |
| [vLLM GPTQ-Marlin-2:4](https://docs.vllm.ai/en/v0.15.0/api/vllm/model_executor/layers/quantization/gptq_marlin_24/) | Een concrete gecombineerde 2:4-sparse plus 4-bit kernel/serialization path. | Vaste 2:4-structuur en bijbehorend formaat; niet het P9B arbitrary 384/768-masker en niet het behoud van een specifieke 768-leaf floating-point-graaf. |
| [Enabling Dynamic Sparsity in Quantized LLM Inference / SpQt](https://arxiv.org/abs/2511.04477) | Combineert dynamische activatiesparsity met groupwise K-quantization via een zigzaglayout, sparse-index gathering en een gespecialiseerde GEMV; rapporteert tot 1,55× decodewinst. | De sparsity is inputafhankelijk en kwaliteit-equivalent, niet bitexact ten opzichte van één frozen masked operator; de layout is niet gedefinieerd als behoud van originele scale-group-identiteiten én oorspronkelijke reductiepositie. |
| [PermuQuant](https://arxiv.org/abs/2605.09503) | Laat rechtstreeks zien dat channel ordering de fout van per-groupquantization beïnvloedt en groepeert kanalen vóór quantization op statistiek. | Ondersteunt juist de stelling dat group topology semantisch relevant is, maar herquantiseert na een gekozen permutatie; het bewaart de oorspronkelijke codes/schalen niet. |
| [Mixture of Neuron Experts](https://arxiv.org/abs/2510.05781) | Neuron-granulaire selectie binnen experts en circa 50% actieve MoE-layerparameters. | Een getrainde architectuur/dynamische selectie, geen post-hoc bitexact pack van een frozen quantized expert. |
| [DERN](https://arxiv.org/abs/2509.10377) | Retraining-free neuron-level pruning en recombinatie in bestaande sparse MoE-modellen. | Richt zich op kwaliteitsbehoud na structurele verandering, niet op bitgelijkheid met een vooraf bestaand quantized-maskoracle. |

De primaire bronnen begrenzen de claim dus als volgt:

- **Niet claimen:** nieuwe neuronsparsity, nieuwe MoE-pruning, eerste combinatie
  van sparsity en low-bit quantization, of eerste offline repacking van
  quantized weights.
- **Wel testbaar als eigen mechanisme:** een codec plus kernel die exact de
  bestaande P9B-operator uitvoert terwijl survivors fysiek compact zijn, en
  die voor elke survivor `(code, oorspronkelijke scale, oorspronkelijke
  group-id, oorspronkelijke leaf-position)` behoudt.

### Harde bewijsvoorwaarden voor de smalle GaugePack-claim

Een overtuigend bewijs moet minimaal tonen:

1. geen requantization: opgeslagen survivorcodes en BF16-schalen zijn
   byte-voor-byte uit de frozen bronrecord afkomstig;
2. expliciete group provenance voor down-projectiekanalen die uit verschillende
   oorspronkelijke 128-groepen komen;
3. expliciete nul-leaves op alle geprunde logische posities, inclusief
   `+0/-0`, FTZ/DAZ, BF16-decode en FMA-contract;
4. bitexacte gate-, up-, SiLU-, product- en downoutputs versus P9B, eerst met
   adversariële input en daarna over alle bedoelde expert-laagrecords;
5. fysiek gemeten bankbytes, H2D-bytes en kernel-events—geen telling van
   “overgeslagen MAC's” als snelheid;
6. een vooraf geregistreerde volledige P13 AB/BA-run. Alleen dan mag de
   componentwinst worden doorgetrokken naar tokens/s.

De regelmatige `64/128 per group`-variant is een **nieuw pruningmodel** en moet
opnieuw door de kwaliteitsgate. Alleen de arbitrary-maskvariant kan kwaliteit
van P9B erven, en dan uitsluitend nadat bitexacte operatorgelijkheid bewezen is.

## 2. ERGV: verschil met bestaand reproducibilitywerk

### Drie verschillende garanties

| Garantie | Betekenis | Voorbeelden |
|---|---|---|
| Run-to-run determinisme | Dezelfde implementatie en tuning geven telkens dezelfde bits. | [NVIDIA CCCL/CUB determinism](https://nvidia.github.io/cccl/unstable/cccl/determinism.html); `DeviceReduce` gebruikt voor `run_to_run` een vaste boom. |
| Boom-onafhankelijke canonieke uitkomst | Verschillende partities/schedules worden door een ander accumulatorformaat naar dezelfde uitkomst gebracht. | [ReproBLAS](https://bebop.cs.berkeley.edu/reproblas/index.php), [ExBLAS](https://github.com/riakymch/exblas), [Microsoft RepDL](https://github.com/microsoft/RepDL). |
| Bronboom-preservatie | Een fysiek andere uitvoering produceert de bits van één vooraf gekozen bron-DAG, inclusief diens tussenafrondingen. | Dit is de relevante ERGV-doelstelling. De bronnen hierboven lossen bewust een ander probleem op. |

De noodzaak van die afbakening wordt door de toolchain zelf bevestigd:

- NVIDIA documenteert dat floating-point optelling niet associatief is en dat
  FMA één afronding gebruikt waar afzonderlijke multiply plus add er twee
  gebruiken: [CUDA Floating Point and IEEE 754](https://docs.nvidia.com/cuda/pdf/Floating_Point_on_NVIDIA_GPU.pdf).
- [MLIR `scf.parallel`](https://mlir.llvm.org/docs/Dialects/SCFDialect/)
  specificeert de reductievolgorde als onbepaald; floatreducties kunnen daardoor
  niet-deterministisch zijn.
- LLVM/MLIR kunnen reassociation en contraction toestaan of verbieden via
  [`strictfp` en fast-math flags](https://www.llvm.org/docs/LangRef.html) en
  [MLIR fast-math-attributen](https://mlir.llvm.org/docs/Dialects/ArithOps/).
  Dit bewaakt semantische vrijheid, maar kiest niet automatisch een efficiënte
  hardwaremapping die een externe legacyboom emuleert.
- [FPRev](https://arxiv.org/abs/2411.00442) kan de summatievolgorde van een
  black-boxfunctie numeriek reconstrueren. Dat is een nuttige mogelijke frontend
  voor ERGV, geen compiler voor de doelmapping.
- [LifeJacket](https://arxiv.org/abs/1603.09290) en
  [Alive2](https://github.com/AliveToolkit/alive2) tonen hoe precieze
  floating-pointtransformaties via SMT/translation validation gecontroleerd
  kunnen worden. Zij zijn relevante verifierprior-art, maar geen MoE/GEMV-
  reduction-graphvirtualizer.

### Verdedigbare ERGV-bijdrage

Niet “eerste reproduceerbare floatreductie” claimen. De sterkere en smallere
engineeringthese is:

> Een compiler neemt een getypeerde logische reductie-DAG met expliciete
> rounding-, FMA-, signed-zero- en denormalcontracten, zoekt een andere fysieke
> lane/warp/blockmapping en accepteert uitsluitend varianten die voor het
> gedefinieerde inputdomein de gekozen bron-DAG behouden.

Om boven een handgeschreven kerneltruc uit te komen, moet een ERGV-prototype:

- minstens Q8, Q5, BF16 en een sparse-zero-leafvariant in één IR beschrijven;
- width/lane/tile-mappings automatisch genereren in plaats van templates
  handmatig te dupliceren;
- een snelle differential verifier én voor kleine vormen een solver- of
  exhaustieve verifier bieden;
- compilerflags/PTX/SASS en target-architectuur aan het bewijsartefact binden;
- aantonen dat dezelfde bron-DAG op minstens twee fysieke mappings en liefst
  twee GPU-architecturen bitexact blijft;
- performance-autotuning strikt scheiden van semantische acceptatie.

Dat zou nieuwheidswaardig kunnen zijn als systeemcombinatie. De huidige
componentresultaten alleen bewijzen nog geen algemene compiler.

## 3. Publieke Qwen3-Coder-Next-80B referentiepunten

De officiële [Qwen-modelkaart](https://huggingface.co/Qwen/Qwen3-Coder-Next)
specificeert 80B totaal/3B actief, hidden 2048, 48 lagen, 512 routed experts,
top-10, één shared expert, intermediate 512 en 262.144 native context. Het
[technisch rapport](https://arxiv.org/abs/2603.00729) documenteert
modelkwaliteit, maar levert geen rechtstreeks 8-GB-offloadbenchmarkprotocol.

De volgende cijfers zijn primaire zelfrapportages in publieke code-/issue-
artefacten. Ze zijn bruikbaar als engineeringreferentie, niet als gecontroleerd
benchmarkpaper.

| Bron/configuratie | Gemelde decode | Bewijskwaliteit en beperking |
|---|---:|---|
| [Experimentele Qwen PR #562](https://github.com/QwenLM/Qwen3-Coder/pull/562/files): laptop RTX 3070 Ti 8 GB, 32 GB RAM, aangepaste Transformers/FP8 streamer | `1,80 tok/s` over 207 tokens; latere pinned-RAMconfig `1,75 tok/s`; SSD-only `1,07 tok/s` | Open PR met titel **DO NOT MERGE**; echte output en code, maar geen gestandaardiseerde prompt/context, herhalingen of thermische statistiek. Dit is de dichtste publiek gedocumenteerde 8-GB-FP8-offloadreferentie. |
| [llama.cpp discussie #21154](https://github.com/ggml-org/llama.cpp/discussions/21154): i9-13900K, RTX 3070 8 GB, ≥80 GB RAM, Q4_K_M, CUDA | `20–21 tok/s` bij kleine context; bij ongeveer 155K prompt en `--n-cpu-moe 26`: `10 tok/s` | Echte gebruikersconfig en contextgevoelige resultaten, maar één deelnemer, geen raw per-tokenarray of identieke lokale GPU. Laat zien dat de hoge waarde niet los van host-RAM, Q4 en context mag worden geciteerd. |
| [llama.cpp issue #19480](https://github.com/ggml-org/llama.cpp/issues/19480): Ryzen AI 9 HX PRO 370, 96 GB DDR5-5600, CPU-only, Q4_K_M 51 GB | `7,74 tok/s` | Goed omschreven CPU/runtime/build en een same-machine dense controle. Geen GPU-offload en dus geen lokale-hardwarebaseline. |
| Zelfde issue, gemelde Radeon 9070 XT 16 GB plus 64 GB RAM met CPU-MoE-placement | circa `25 tok/s` | Nuttige hybride-offloadreferentie maar dubbel zoveel VRAM en andere GPU/backend. |
| [ds4-8gb-cpu](https://github.com/baker27727/ds4-8gb-cpu) | geen sustained Qwen-decodescore | Demonstreert DeepSeek V4 Flash CPU-only NVMe demand paging met een 5,33-s gecontroleerde first-token-diagnostic. De repo claimt expliciet geen competitieve sustained throughput en ondersteunt in deze publicatie geen Qwen3-Coder-Next; dus geen directe baseline. |

### Consequentie voor de lokale 80B-claim

De N4B-R synthetische actieve-expertmeting en `36,946 ms` conservatieve
full-stackprojectie mogen **niet** als circa 27 tok/s tegenover de tabel worden
gezet: zij bevatten geen echt checkpoint, echte routing, fysiek gemeten
expert-H2D, Gated DeltaNet/attention of volledige decodetijd.

De eerste eerlijke publieke vergelijking vereist daarom één echte
Qwen3-Coder-Next-run met minimaal:

- exact checkpoint en quantization hash/variant;
- GPU, CPU, RAM, opslag, driver, CUDA en commit-SHA;
- fysieke VRAM/working-setpiek en feitelijke placement van trunk/experts;
- prompt-hash, contextlengte, 256–1.000 decode-tokens, greedy sampling en
  warm/cold status;
- TTFT, prefill tok/s, decode mean/p50/p95, page faults en cache-hit/missratio;
- een tweede AB/BA-run en een kwaliteit-/logitcontrole tegen de
  niet-gecompacteerde quantized referentie.

Voor een **industriële doorbraakclaim** is meer nodig dan één lokale topscore:
een onafhankelijk gereproduceerde echte-80B-run, minimaal één tweede
hardware-/backendconfiguratie, een openbare patch en testbank, en behoud van
modelkwaliteit. De 20–21 tok/s RTX-3070/Q4-zelfrapportage is een nuttig
competitief referentiepunt, maar geen formele universele drempel.

## Bron- en zoekgrens

Alle inhoudelijke vergelijkingen hierboven gebruiken papers, officiële
projectdocumentatie, modelkaarten of oorspronkelijke repository-issues/PR's.
Blogs, nieuwsaggregators, Reddit en YouTube zijn niet als bewijs gebruikt.
Doorzocht zijn onder andere combinaties van `groupwise quantization`,
`channel permutation`, `sparse quantized GEMV`, `neuron pruning`, `SwiGLU`,
`MoE`, `reduction tree`, `bitwise reproducibility`, `strictfp`,
`translation validation`, `Qwen3-Coder-Next`, `8GB`, `CPU offload` en
`llama.cpp`. Deze zoekdekking is niet volledig genoeg voor een juridische
novelty- of patentconclusie.
