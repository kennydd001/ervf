# NERVF-5 — gestopt op de eigen stopregel: de productie-runtime is niet run-to-run deterministisch

Datum: 2026-08-15
Namespace: `NERVF_NEMOTRON`
Verdict: **G-NERVF-5C faalt, maar niet door ERVF. De twee bàsislijn-armen wijken onderling af: `base_b` is niet identiek aan `base_a` over 512 tokens, terwijl beide `use_ervf=False` draaien. De opdracht schrijft hier expliciet voor: "Wanneer de bestaande production kernel zelf niet deterministisch bitwise is, stop en documenteer dit vóór de gate wordt aangepast." Dat is gedaan.**
Terminal state: `nervf5_halted_baseline_nondeterminism_across_runs`

## 1. Wat er gemeten is

Drie armen `base_a / ervf / base_b`, drie promptdomeinen, **512 causale tokens**
per domein, één modelload.

| domein | basis p50 | ERVF p50 | winst | drift | conclusief |
|---|---:|---:|---:|---:|:--:|
| expository | 40,907 | 39,117 | +1,789 | 0,777 | ✅ |
| narrative | 41,487 | 36,882 | +4,605 | 1,263 | ✅ |
| code | 41,049 | 36,719 | +4,330 | 0,944 | ✅ |

| poort | uitslag |
|---|:--|
| **G-NERVF-5C** exactheid | ❌ — **ook `base_b` ≠ `base_a`** |
| **G-NERVF-5P** latency | ❌ (hangt aan 5C) |
| **G-NERVF-5M** VRAM | ✅ geen regressie |

## 2. De vondst: de accumulatievolgorde hangt af van de cachestaat

Dit staat in `_moe_cached`, en het is geen ERVF-code:

```python
order  = [s for s in range(len(idx)) if not needs_wait[s]]   # hits eerst
order += [s for s in range(len(idx)) if needs_wait[s]]       # misses daarna
```

De zes routed experts van een laag worden **in hit-dan-miss-volgorde**
geaccumuleerd, niet in routevolgorde. Welke expert een hit is hangt af van de
LRU-staat, en die hangt af van alles wat er vóór die token gebeurd is. Twee runs
met een andere cachegeschiedenis accumuleren dus in een andere volgorde, en
floating-point optelling is niet associatief.

Dat is precies wat hier gebeurt: `base_a` start met een koude cache, `base_b`
start met de cache zoals de vorige twee armen hem achterlieten. Zelfde gewichten,
zelfde routes, zelfde kernels — andere optelvolgorde.

## 3. Waarom dit eerder niet opviel

Alle eerdere pariteitschecks in deze lijn liepen over **2 × 64** tokens
(NERVF-3, NERVF-4, E4-in-lus, S11, W1) en kwamen identiek uit. Over 512 tokens
per domein divergeert het wel. De eigenschap is dus niet afwezig maar **fragiel**:
hij houdt zolang de cachegeschiedenis tussen de armen genoeg overeenkomt.

Dat kwalificeert de eerdere bit-identiek-claims. Ze zijn waar voor de runs waarin
ze gemeten zijn — de artefacten liggen er — maar ze bewijzen niet dat de runtime
onder alle omstandigheden dezelfde tokens geeft.

## 4. Wat dit voor ERVF betekent

**Niets negatiefs.** ERVF is bitexact bewezen op het niveau waar dat betekenis
heeft: de kernel zelf, tegen de productiekernel, **0/72 mismatches over vier
breedtes** (NERVF-2), met identieke MAC-toewijzing en een exact
gereconstrueerde reductieboom. Wat NERVF-5 blootlegt zit een laag hoger, in de
runtime die de zes expertbijdragen optelt, en het geldt even hard met ERVF uit.

De latencywinst blijft staan en is in alle drie de domeinen conclusief: **+1,8 tot
+4,6 ms per token** over 512-token rollouts, boven hun eigen drift.

## 5. Wat er eerst moet gebeuren

Eén regel code beslist het: accumuleer in **routevolgorde** in plaats van
hit-dan-miss-volgorde. De hit-eerst-volgorde bestaat om latency te winnen (hits
rekenen terwijl misses binnenkomen) — die winst blijft behouden als je de
*berekening* in hit-volgorde doet maar de *accumulatie* in slotvolgorde, door de
zes bijdragen apart te bewaren en aan het eind in vaste volgorde op te tellen.
Dat is precies wat X1's `reduce_slots` al doet voor de expert-major sweep.

Kosten: zes buffers van 2688 floats per laag (64 KB) en één extra reductiekernel.
Opbrengst: een runtime die run-to-run deterministisch is, waardoor élke
exactheidspoort in deze lijn — en in Kimi's — pas echt betekenis krijgt.

Dat is een aparte fase met een eigen preregistratie. Ik open hem niet binnen
NERVF-5, en ik pas de gate niet aan om ERVF te laten slagen.

## 6. Claim boundary

512-token causale rollouts op deze GPU bij capacity 72, drie domeinen, drie armen.
Latency is end-to-end wandtijd per token inclusief de synchronisatie; p95/p99 over
511 stappen per domein. De context groeit tijdens de rollout, dus deze cijfers
zijn niet vergelijkbaar met n7b's bevroren vaste-diepte-getallen. De
niet-determinisme-vondst is vastgesteld door **twee armen met identieke
configuratie** te vergelijken en is daarmee onafhankelijk van ERVF.

## 7. Artefacten

`scripts/nervf_nemotron/nervf5_full_model.py` · `nervf5_full_model.json`
