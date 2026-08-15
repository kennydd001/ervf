# N7-C — gevectoriseerde NVFP4 code-loads

Datum: 2026-08-14
Verdict: **4K-poort GEHAALD: 20,671 tok/s tegen een minimum van 20. 128K nog niet: 7,968 tegen 10.**
Terminal state: `n7c_4k_gate_met_128k_gate_missed`

## Kernresultaat

| context | N6-C start | N7-B | **N7-C** | totaal |
|---:|---:|---:|---:|---:|
| 0 | 13,258 | 19,577 | **21,540** | 1,62× |
| 4.032 | 10,046 | 19,744 | **20,671** | **2,06×** |
| 32.768 | 3,742 | 13,813 | **14,427** | 3,86× |
| 131.072 | 1,215 | 7,762 | **7,968** | 6,56× |
| 262.100 | 0,637 | 4,750 | **4,769** | **7,49×** |

Warme generatie: **21,72 tok/s**. Correctheid ongewijzigd bij elke stap —
` Paris`, identieke generatie, expert `rel_l2 = 1,605e-07`.

## Wat het was

De expert-GEMV kostte 13,76 ms terwijl de bandbreedte-roofline ~3,9 ms is:
**3,5× ernaast**. De oorzaak was banaal en pijnlijk — mijn kernel las de
gepakte codes **byte voor byte**:

```c
const unsigned char byte = crow[b];   // 1-byte load, 5,6 MB per expert
```

Gevectoriseerde low-bit loads staan letterlijk in mijn eigen research handoff
onder "herbruikbare mechanismen" uit het ERVF/ERGV-werk van dit project. Ik had
ze niet toegepast.

De fix: één `uchar4`-fetch dekt 8 codes en heeft **één** block-scale-lookup
nodig, omdat 16 codes = 8 bytes zijn en de bytes `[4v, 4v+3]` dus altijd in
groep `(4v)>>3` vallen. Vier keer minder geheugentransacties, acht FMA's per
load.

## Wat het níet was

Ik dacht eerst dat de router de boosdoener was: 0,339 ms per laag × 23 = 7,8 ms
voor een GEMV van 344 kFLOP, wat op latency van twee `asnumpy`-syncs per laag
leek te wijzen. Ik heb die samengevoegd tot één readback en de shared expert —
die niet van routing afhangt — ervóór gelaunched zodat de sync erachter wegvalt.

**Resultaat: 19,73 → 19,50 tok/s. Geen winst.** Die hypothese was fout en staat
hier omdat een gemeten niet-effect even hard is als een gemeten effect. De
router-tijd zit ergens anders; ik heb hem niet verder ontleed en benoem hem dus
niet.

## Poorten

| poort | vereist | gemeten | uitkomst |
|---|---:|---:|:--:|
| **4K minimum acceptable** | 20 | **20,671** | ✅ **GEHAALD** |
| 4K primary | 25 | 20,671 | ❌ −17% |
| 128K minimum acceptable | 10 | 7,968 | ❌ −20% |
| 128K primary | 15 | 7,968 | ❌ −47% |

De eerste throughput-poort van de opdracht is gehaald. De andere drie niet.

## Configuraties

| | 4K-config | 262K-config |
|---|---:|---:|
| shell (embed op host) | 2,211 GiB | 5,164 GiB |
| cache | 4,448 GiB (37/laag) | 2,404 GiB (20/laag) |
| vrij | 0,130 GiB | 0,000 GiB |
| hitrate | 69,4% | 54,2% |

VRAM is in beide gevallen volledig op. De cache krijgt exact wat de KV overlaat.

## Over de iGPU — analyse, geen claim

De vraag of de Intel Arc Pro 140T zou helpen verdient een echt antwoord.

**Mechanisch is het aantrekkelijk.** Bij 69,4% hit blijven ~42 misses per token
over: 42 × 5,612,544 B = 236 MB over PCIe = ~9,1 ms bij de gemeten 26,03 GB/s.
De iGPU deelt systeem-DDR5 en zou die 42 experts *in host-geheugen* kunnen
uitrekenen en alleen het resultaat terugsturen: 42 × 2.688 floats = **451 KB in
plaats van 236 MB**, een reductie van ~523×.

Ruwe schatting: 236 MB uit DDR5 op ~60 GB/s ≈ 3,9 ms, tegen 9,1 ms PCIe — en
parallel aan de RTX die de 96 hits doet. Dat is een reëel heterogeen voordeel,
en het is precies waar de beschermde HET-NEXT-lijn aan werkt.

**Daarom doe ik het niet.** De Arc Pro 140T is het actieve experiment van de
andere agent; hun Intel-helft is een formele PASS met een bevroren bundle en zij
zitten nu op NC0–NC13. Een tweede proces dat hun iGPU claimt, kan hun metingen
verstoren. De analyse staat hier als richting, niet als resultaat, en het
getal 3,9 ms is een **schatting uit specificaties**, geen meting.

Als dit wordt opgepakt hoort het via hun lijn te lopen, niet via de mijne.

## Eerlijk verdict

Een 30B-model draait volledig op een **8 GiB laptop-GPU** op **21,5 tok/s** bij
korte context en **20,7 tok/s** bij 4K — boven de minimumpoort — en haalt nog
**4,77 tok/s** op **262.144** contextdiepte, met de volle 15,4 GiB expertbank in
host-RAM en 69,4% cache-hitrate.

Niet gehaald: de 4K-primary (25), en beide 128K-poorten. Geen kwaliteitsmeting,
geen benchmark, geen thermische steady state, batch 1, één GPU.

Deze sessie: **2,06× op 4K en 7,49× op 262K**, met bij elke stap dezelfde eerste
token en dezelfde generatie. De grootste winst kwam van een fout in mijn eigen
kernel die ik in mijn eigen handoff had kunnen lezen.

## Artefacten

- `src/moe_lab/lightningstream_nemotron/fused_nvfp4.py` (uchar4-loads)
- `src/moe_lab/lightningstream_nemotron/runtime.py` (gepakte route-readback, shared-expert-vóór-sync)
- `scripts/lightningstream_nemotron/n7b_cached_decode.py` · `n7b_cached_decode.json`
