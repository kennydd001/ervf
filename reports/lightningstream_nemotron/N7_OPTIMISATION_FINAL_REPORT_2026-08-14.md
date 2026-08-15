# N7 — flash-decode, intra-layer overlap, LRU expert cache

Datum: 2026-08-14
Verdict: **1,97× sneller op 4K en 7,46× op 262K, correctheid behouden. Beide throughput-poorten NÉT niet gehaald: 19,744 vs 20 tok/s (−1,3%) en 7,762 vs 10 tok/s (−22%).**
Terminal state: `n7_optimised_gates_narrowly_missed`

## Sessieprogressie — alles gemeten

| context | N6-C start | N7 eind | factor |
|---:|---:|---:|---:|
| 0 | 13,258 | **19,577** | 1,48× |
| 4.096 | 10,046 | **19,744** | **1,97×** |
| 32.768 | 3,742 | 13,813 | 3,69× |
| 131.072 | 1,215 | **7,762** | **6,39×** |
| 262.100 | 0,637 | **4,750** | **7,46×** |

Correctheid bij elke stap ongewijzigd: `The capital of France is` → ` Paris`,
en coherente generatie (`' humans first needed to count and record information.
Early tools like the abacus and late'`).

## Drie ingrepen

### 1. Flash-decoding split — de lange-contextfix

De oude kernel liep één block per query-head sequentieel door alle `t` posities,
met twee `__syncthreads()` per positie: O(t) geserialiseerd, ~248 ms per
attention-laag bij 262K. Nu wordt het positiebereik over maximaal 256 blocks
gesplitst, elk met online softmax, gevolgd door een combine-kernel die de
partiële `(m, l, acc)`-tripels samenvoegt. De rescaling die de kernel al deed is
precies wat die splitsing legaal maakt.

Effect alleen hiervan: 262K van 0,637 → 4,506 tok/s (7,07×).

### 2. Intra-layer overlap — met een correctie op mijn eigen rapport

Het N6-B/C-rapport stelde dat het porten van N4-R2's overlap ~2× zou geven.
**Dat was fout en de correctie is inhoudelijk belangrijk:** cross-layer prefetch
is **causaal onmogelijk** — laag `L+1`'s route hangt af van laag `L`'s output.
N4-R2 kon over lagen heen overlappen omdat de routes daar vooraf vastlagen
(synthetisch). In een echte decode kan dat niet.

Wat wél kan is overlap *binnen* een laag over de zes experts, en in het
cache-pad: **hits eerst uitrekenen terwijl de misses binnenkomen**, met een event
per miss in plaats van één globale synchronisatie. Dat laatste alleen bracht
17,60 → 19,61 tok/s.

### 3. LRU-expertcache, gedimensioneerd op gemeten lokaliteit

N6-A vond alle 128 experts in gebruik met maar 8,7× spreiding, wat een *statische*
prior zwak maakt. Dat zegt niets over *temporele* lokaliteit, en die is er wel:

> **2,011 van 6 experts gedeeld tussen opeenvolgende tokens.**

Gesimuleerde LRU-hitrate op echte routes: 33,4% bij 8 slots/laag, 48,6% bij 16,
**65,0% bij 32**, 75,0% bij 48. Gemeten na implementatie: **66,0% bij 32** en
**69,4% bij 37** — de simulatie voorspelde de meting binnen 1 procentpunt.

De embedding is naar host verplaatst (N5-variant B): een token raakt 5.376 van
704.643.072 bytes aan, dus device-residentie kocht niets terwijl het 0,656 GiB
aan cache-slots kostte.

## Configuraties

| | 4K-config | 262K-config |
|---|---:|---:|
| shell (embed op host) | 2,211 GiB | 5,164 GiB |
| cache | 4,448 GiB (37/laag) | 2,404 GiB (20/laag) |
| vrij | 0,130 GiB | 0,000 GiB |
| hitrate | 69,4% | 54,2% |
| tok/s @0 | 19,577 | 19,175 |

VRAM is in beide gevallen volledig benut. De cache krijgt letterlijk wat de KV
overlaat, precies zoals N5 voorspelde.

## Tegen de poorten

| poort | vereist | gemeten | tekort |
|---|---:|---:|---:|
| 4K minimum acceptable | 20 | **19,744** | **−1,3%** |
| 4K primary | 25 | 19,744 | −21% |
| 128K minimum acceptable | 10 | **7,762** | **−22%** |
| 128K primary | 15 | 7,762 | −48% |

**Niet gehaald.** 19,744 is geen 20. Het verschil is 1,3% en het zou triviaal
zijn om er met een gunstiger promptkeuze, een warmere meetvenster of afgeronde
getallen overheen te komen — dat is precies waarom het hier zo blijft staan.

## Wat er nog ligt

1. **De routed transfer is de vloer.** Bij 69,4% hit blijft ~30,6% van 138
   records over: ~42 records × 5,6 MB = 237 MB per token, ~9,1 ms PCIe. Meer
   cache is de directe hefboom en die is VRAM-gelimiteerd, niet ontwerp-gelimiteerd.
2. **LM-head 5,4 ms** (10,6% van het token) — één BF16 GEMV over 131.072 × 2.688.
3. **Mamba 7,9 ms** over 23 lagen; de SSM-stap is nog niet geoptimaliseerd.
4. Een tweede GPU-generatie of meer VRAM verandert het beeld direct: dit is een
   8 GiB laptop-GPU waarop een 30B-model volledig draait.

## Eerlijk verdict

Wat er staat: een **correcte, volledige NVFP4 MoE-runtime op een 8 GiB
laptop-GPU** die coherente tekst produceert op **19,7 tok/s bij 4K** en nog
**4,75 tok/s bij 262.144 contextdiepte** haalt, met de volledige 15,4 GiB
expertbank in host-RAM en een gemeten 69,4% cache-hitrate.

Wat er niet staat: de throughput-poorten. Niet op 4K (−1,3%) en niet op 128K
(−22%). Geen kwaliteitsevaluatie, geen benchmark, geen thermische steady state,
geen uitspraak over andere hardware of batch>1.

De vooruitgang in deze sessie is 1,97× op 4K en 7,46× op 262K, met bij elke stap
dezelfde eerste token. Dat is echte engineering-winst. Het is geen doorbraak, en
de resterende hefbomen zijn benoemd en gemeten in plaats van geschat.

## Artefacten

- `scripts/lightningstream_nemotron/n7a_route_locality.py` · `n7a_route_locality.json`
- `scripts/lightningstream_nemotron/n7b_cached_decode.py` · `n7b_cached_decode.json`
- `src/moe_lab/lightningstream_nemotron/gpu_kernels.py` (flash-decode split)
- `src/moe_lab/lightningstream_nemotron/runtime.py` (LRU cache, hit-first overlap, host embedding)
