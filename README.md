# ERVF — MoE-inferentieonderzoek op consumentenhardware

Een archief van gepreregistreerd onderzoek naar het draaien van grote
Mixture-of-Experts-modellen op één laptop-GPU. Elke hypothese heeft haar poorten
**vóór** de meting vastgelegd, en de weerlegde hypotheses staan er net zo
volledig in als de geslaagde — dat is het punt van het archief.

De naam komt van het belangrijkste resultaat: **ERVF, Exact-Reduction Virtual
Fusion.**

---

## ERVF in het kort

Een GEMV-kernel waarin elke rij door een **subwarp van 16 lanes** wordt
afgehandeld in plaats van door een heel warp, met per lane aparte *virtuele*
accumulatoren. De reductie is zo opgezet dat ze de reductieboom van de
referentiekernel **exact reconstrueert**: de eerste butterfly-stap (offset 16)
wordt een lane-lokale optelling zonder shuffle, de offsets 8/4/2/1 blijven
shuffles binnen de subwarp, en de acht warp-sommen vouwen in registers in precies
de volgorde `((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))`.

Daardoor is het resultaat niet "numeriek gelijkwaardig" maar **bit-identiek** —
en tegelijk bijna twee keer zo snel.

| | |
|---|---|
| kernel-versnelling | **1,936×** bij **0 van 72** numerieke verschillen |
| bandbreedte | 72,7 → **140,8 GB/s** |
| bitexact op | vier subwarp-breedtes (4/8/16/32) |

Eerst gevonden op Qwen3-30B-A3B, daarna **gerepliceerd op Nemotron 3.5 Lightning
30B-A3B** — een ander model, een andere quantisatie (NVFP4 tegen Q5/Q8), een
andere shape. De replicatie koos onafhankelijk dezelfde breedte 16.

Kernel: [`src/moe_lab/lightningstream_nemotron/fused_nvfp4.py`](src/moe_lab/lightningstream_nemotron/fused_nvfp4.py) ·
Eindrapport: [`reports/nervf_nemotron/NERVF_NEMOTRON_FINAL_REPORT.md`](reports/nervf_nemotron/NERVF_NEMOTRON_FINAL_REPORT.md)

## De runtime waar het in zit

**Nemotron 3.5 Lightning 30B-A3B NVFP4** draait causaal op een **8 GiB RTX PRO
2000 Blackwell laptop-GPU** door de 128 routed experts per laag vanaf host te
streamen. Hybride Mamba/MoE/attention over 52 lagen.

| meting | regime | resultaat |
|---|---|---|
| **E6** geïntegreerd | 3 domeinen × 512 causale tokens | 41,980 → **37,490 ms/token**, bit-identiek |
| **E1 fase 2.1** device-routing | 2 prompts × 64 tokens | 41,540 → **36,998 ms/token** |
| doorvoer | 512-token rollout / ctx 0 | **26,7 / 29,5 tok/s** |
| roofline-plafond | gemeten 338,4 GB/s streaming read | 165 (ctx 0) / 119 (lang) tok/s |

De runtime draait op ongeveer **17% van zijn roofline**. Componentwinsten worden
in dit archief **nooit bij elkaar opgeteld** — elke combinatie vraagt een eigen
fysieke A/B, en de twee regels hierboven zijn dus niet optelbaar.

## Begin hier

| bestand | waarvoor |
|---|---|
| [`agents/STATE_OF_THE_WORK.md`](agents/STATE_OF_THE_WORK.md) | stand van zaken: bewezen, weerlegd, hoeveel tok/s |
| [`agents/TODO.md`](agents/TODO.md) | de actieve takenlijst |
| [`agents/RESEARCH_NOTEBOOK.md`](agents/RESEARCH_NOTEBOOK.md) | logboek per fase, nieuwste bovenaan |
| [`agents/README.md`](agents/README.md) | de werkregels |
| [`reports/treesweep200/EXPERIMENT_REGISTRY.yaml`](reports/treesweep200/EXPERIMENT_REGISTRY.yaml) | alle experimenten met hun poorten en uitslag |

## De onderzoekslijnen

| lijn | onderwerp | uitkomst |
|---|---|---|
| [`nervf_nemotron`](reports/nervf_nemotron) | ERVF-replicatie op Nemotron | **geslaagd**, 1,936× bitexact |
| [`treesweep200`](reports/treesweep200) | roofline-herstel, attention, integratie, adoptie | deels geslaagd, zie registry |
| [`lightningstream_nemotron`](reports/lightningstream_nemotron) | de streamende 30B-runtime zelf | draait |
| [`streamq5_moe`](reports/streamq5_moe) | de 80B-lijn | apart spoor |
| [`qwen_gptq_bank`](reports/qwen_gptq_bank) | Qwen3-30B, waar ERVF vandaan komt | oorsprong |
| [`rsiv_moe`](reports/rsiv_moe) | GhostWeights/RSIV | **weerlegd**, `falsified_rank_working_set` |
| [`craft_moe`](reports/craft_moe) · [`hera_moe`](reports/hera_moe) · [`dhera_moe`](reports/dhera_moe) · [`fleq_moe`](reports/fleq_moe) · [`e2gq_moe`](reports/e2gq_moe) · [`coretail_moe`](reports/coretail_moe) · [`adhera_moe`](reports/adhera_moe) · [`bitflow_moe`](reports/bitflow_moe) | eerdere compressie- en cachehypotheses | grotendeels weerlegd, elk met eigen log in [`docs/`](docs) |
| [`baseline`](reports/baseline) | de eerste lijn: DeepSeek V2-Lite activation-space compressie | zie [`docs/BASELINE_V2LITE_README.md`](docs/BASELINE_V2LITE_README.md) |

## Wat hier weerlegd is

Even belangrijk als wat werkt, en met dezelfde strengheid gemeten:

- **Speculative decoding / MTP** — dicht langs drie onafhankelijke paden
  (X1-ratio 1,0017 · Z1-lineariteit R²=0,99986 · K0/S13 route-unie boven pariteit).
- **Gatherloze downflow** — −5,7 tot −7,4 ms per token. Strided host-reads halen
  6,7 GB/s over PCIe tegen 85,9 vanaf device; de gather verdient zichzelf terug.
- **Byte-reductie / betere compressie** — begrensd: halveren scheelt 34,2%, het
  OrbitANS-plafond is 7,23%.
- **1000 tok/s op deze hardware** — de roofline sluit het uit. 50 en 100 niet.

## Een methodologische vondst die het vermelden waard is

Vier opeenvolgende fasen haalden hun exactheidspoort terwijl de runtime in
werkelijkheid **niet run-to-run deterministisch was**. De MoE-laag telde de zes
routed experts op in hit-dan-miss-volgorde, wat van de LRU-staat afhangt, en
floating-point optelling is niet associatief — twee runs met een andere
cachegeschiedenis gaven andere tekst.

Alle vier de tests keken over 2×64 tokens binnen één proces en konden de fout
daarom niet zien. De reparatie was klein (reken hit-eerst, tel op in routevolgorde
— 64 KB, nul extra kernels), maar de les zit nu in de werkregels:

> **Bouw een controle-arm die moet falen.** Slaagt hij, dan heeft je test geen
> onderscheidend vermogen en bewijst je hoofdpoort niets.

## Werkwijze

Elke fase: preregistratie met bevroren poorten → runner → een **aparte verifier**
die alles herberekent zonder de runner te importeren → rapport met claim
boundary → registry-update. Poorten worden nooit achteraf verruimd, er is één
variabele per meting, en een componentmeting wordt nooit opgewaardeerd tot tok/s.

De volledige regels staan in [`agents/README.md`](agents/README.md).

## Wat niet in deze repo zit

Modelgewichten, virtualenvs, caches, `third_party/`, en vijf ruwe oracle-dumps
boven GitHubs limiet van 100 MB per bestand. De rapporten die ze samenvatten
zitten er wel in, en de runners in [`scripts/`](scripts) kunnen ze opnieuw
produceren.

## Installatie

De Nemotron-lijn gebruikt een eigen virtualenv (`.venv-nemotron`) met CuPy tegen
CUDA 12; zie [`requirements.txt`](requirements.txt) en [`pyproject.toml`](pyproject.toml).
De opzet en reproductiestappen van de eerste lijn (DeepSeek V2-Lite) staan
ongewijzigd in [`docs/BASELINE_V2LITE_README.md`](docs/BASELINE_V2LITE_README.md).
