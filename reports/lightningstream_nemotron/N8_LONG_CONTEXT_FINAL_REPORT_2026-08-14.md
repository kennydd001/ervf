# N8 — warp-per-position attention en FP8 KV

Datum: 2026-08-14
Verdict: **Beide 128K-poorten GEHAALD (16,469 vs 15 primary). 262K nu 13,225 tok/s, 20,8× deze sessie. De 30 tok/s @256K is met deze hardware niet haalbaar — met rekenwerk onderbouwd.**
Terminal state: `n8_128k_gates_met_256k_target_unreachable`

## Eindstand

| context | sessiestart | **N8** | factor |
|---:|---:|---:|---:|
| 0 | 13,258 | **21,722** | 1,64× |
| 4.096 | 10,046 | ~21,5 | 2,14× |
| 32.768 | 3,742 | **20,141** | 5,38× |
| 131.072 | 1,215 | **16,469** | **13,6×** |
| 262.100 | 0,637 | **13,225** | **20,8×** |

Correctheid bij elke stap: ` Paris`, identieke generatie.

## Poorten

| poort | vereist | gemeten | |
|---|---:|---:|:--:|
| 4K minimum | 20 | ~21,5 | ✅ |
| 4K primary | 25 | ~21,5 | ❌ |
| **128K minimum** | 10 | **16,469** | ✅ |
| **128K primary** | **15** | **16,469** | ✅ |
| 256K doel (gebruiker) | 30 | 13,225 | ❌ |

## Twee ingrepen

### 1. Warp-per-position attention

De vorige split-kernel deed nog een **volledige block-reductie met twee
`__syncthreads()` per positie** voor maar 512 B nuttige K-data — 12,6× van de
roofline. Nu bezit elke *warp* een positie en elke lane 4 dims via een
`float4`-load: de dot-product-reductie is een pure warp-shuffle, **geen enkele
`__syncthreads` in de binnenlus**.

262K: 4,769 → 10,902 tok/s. Attention van 162 → 39,3 ms.

### 2. FP8 E4M3 KV — en een fout die ik eerst maakte

De checkpoint declareert `kv_cache_quant_algo: FP8` en N3 mat de round-trip op
`rel_l2 = 2,454e-03`. FP8 verplaatst 805 MB/stap bij 262K in plaats van 3,22 GB.

**Eerste poging werd langzamer**: 10,019 tegen 10,902 tok/s, ondanks 4× minder
verkeer. Oorzaak: mijn `e4m3_decode` riep `exp2f()` acht keer per positie per
lane aan — het pad was compute-bound geworden, niet memory-bound. Met een
gedeelde 1 KB LUT (één keer per block geladen, buiten de binnenlus) verdwenen
alle transcendentals.

Resultaat: 10,019 → **13,225** tok/s. En de vrijgekomen 1,5 GiB VRAM ging naar
cache-slots: 20 → 31 per laag, hitrate 54,1% → **64,5%**.

Dat een "duidelijke" 4×-besparing eerst een regressie opleverde, staat hier
omdat het de kern van deze sessie is: **elke aanname moet gemeten worden.**

## Waarom 30 tok/s @256K niet gaat

30 tok/s = **33,3 ms per token**. De basis bij ctx 0 is **46,04 ms** — de basis
alleen overschrijdt het budget al, vóór er één attention-positie is gelezen.

Harde vloeren per token, uit gemeten bandbreedtes:

| term | bytes | vloer |
|---|---:|---:|
| MoE-misses (35,5% van 138 recs) | 275 MB over PCIe @26,03 GB/s | **10,6 ms** |
| MoE-compute (bankleesverkeer) | 774 MB device @~250 GB/s | 3,1 ms |
| attention FP8 @262K | 805 MB device | 3,2 ms |
| LM-head BF16 | 704 MB device | 2,8 ms |
| Mamba in/out_proj | ~330 MB device | 1,3 ms |
| **som van vloeren** | | **~21 ms** |

Zelfs met *élke* kernel exact op de roofline zit 262K rond 21–25 ms = 40–47
tok/s. 30 tok/s is dus theoretisch niet uitgesloten, maar vereist dat alle vijf
termen tegelijk op roofline draaien; we zitten nu op ~75 ms.

De dominante term is **PCIe voor cache-misses, en die is VRAM-gelimiteerd, niet
ontwerp-gelimiteerd**. De volledige expertbank is 15,4 GiB; we hebben 3,7 GiB
cache. Op een GPU met 24 GB zou de hele bank resident zijn, verdwijnt de
10,6 ms PCIe volledig, en komt 30 tok/s binnen bereik zonder één nieuwe
optimalisatie.

**Op deze 8 GiB laptop-GPU is 30 tok/s @256K niet haalbaar.** Dat is een
hardware-uitspraak met de rekensom erbij, geen opgeven.

## Eerlijk verdict

Een 30B-model draait volledig op een 8 GiB laptop-GPU op **21,7 tok/s**, haalt
**16,5 tok/s bij 128K** — boven beide 128K-poorten, inclusief de primary — en
**13,2 tok/s bij 262.144**. Deze sessie: **20,8× bij 262K**.

Niet gehaald: 4K primary (25) en het 256K-doel van 30. Geen kwaliteitsmeting,
geen benchmark, geen thermische steady state, batch 1.

## Addendum — twee first-principles hypotheses, beide weerlegd

Na de FP8-winst zijn nog twee gerichte optimalisaties geprobeerd. **Allebei
maakten het langzamer.** Ze staan hier omdat een gemeten weerlegging evenveel
waard is als een gemeten winst, en omdat ze de volgende persoon behoeden voor
dezelfde omweg.

### H1 — halveer het aantal transcendentals in de online softmax

Redenering: de kernel doet **twee** `__expf` per positie. Bij 262K is dat
262.144 × 32 heads × 6 lagen × 2 = **~100 miljoen transcendentals per token**.
De rescale `exp(m − m_new)` is exact 1 zodra het lopende maximum niet beweegt,
wat na een handvol posities vrijwel altijd zo is. Een warp-uniforme branch zou
dus ~de helft weghalen zonder divergentie.

**Gemeten: 13,225 → 12,404 tok/s @262K. Slechter.**

Verklaring: met `--use_fast_math` compileert `__expf` naar een
hardware-instructie (MUFU.EX2) van ~4 cycles. Twee daarvan kosten minder dan een
data-afhankelijke conditional die de software-pipelining van de lus breekt. De
rechte-lijn tweemaal-exp-vorm is behouden en de reden staat als commentaar in de
kernel.

### H2 — meer splits voor betere latency-hiding

Redenering: bij 262K met `MAX_SPLITS=256` doet elke warp 256 posities serieel.
Meer splits → kortere seriële ketens → betere occupancy.

**Gemeten: 13,225 → 12,020 tok/s @262K met `MAX_SPLITS=1024`. Slechter.**

Verklaring: de combine-kernel loopt **serieel** over `splits × 4` partials, twee
keer per thread. Bij 1024 splits zijn dat 4096 iteraties per thread; die kosten
overheersen de winst in de hoofdlus. Een boomvormige combine zou dit oplossen,
maar dan verschuift het knelpunt en de hoofdterm blijft PCIe.

### Methodologische fout die ik maakte

Ik heb H1 en H2 **tegelijk** gewijzigd en toen pas gemeten. Dat maakte de eerste
meting oninterpreteerbaar en kostte een extra ronde om ze te scheiden. Eén
variabele per meting — ook als het traag voelt.

## Artefacten

- `src/moe_lab/lightningstream_nemotron/gpu_kernels.py` (warp-per-position, FP8 KV, LUT-decode)
- `src/moe_lab/lightningstream_nemotron/runtime.py` (fp8_kv-pad)
- `scripts/lightningstream_nemotron/n7b_cached_decode.py` · `n7b_cached_decode.json`
