# ST2-mini — echte Intel host-USM Q5-gate

Datum: 2026-08-12  
Formeel verdict: **Q5/host-USM bewezen; bevroren p95-doorvoerpoort faalt**  
Onafhankelijke verificatie: **41/41 CPU-checks geslaagd**

## Uitkomst

De lokale Intel Arc Pro 140T kan echte Qwen3-30B-A3B STREAMQ5-records
rechtstreeks uit een host-USM-allocatie lezen en exact dezelfde Q5-reductie
uitvoeren. Dat is nu fysiek bewezen zonder OpenVINO-constantengewichten en
zonder NVIDIA-kernel of -transfer.

De vereiste stabiele doorvoer van `21,63 GB/s` wordt echter niet gehaald:

| fysieke mapping | event p50 | event p05 | wall p50 | wall p05 | formele conservatieve waarde |
|---|---:|---:|---:|---:|---:|
| 256-thread bronboom | 11,780 | 9,866 | 11,688 | 9,654 | **9,654 GB/s** |
| ERGV width 8 | **21,913** | 20,616 | 21,549 | 18,540 | **18,540 GB/s** |
| harde gate |  |  |  |  | **>=21,63 GB/s** |

De conservatieve waarde is, zoals vooraf vastgelegd, het minimum van de
event-p05- en wallclock-p05-bandbreedte. Width 8 faalt dus 14,29% onder de
formele gate. Zelfs event-only p05 faalt nog 4,69%. De mediaan komt zeer dicht
bij of boven de grens, maar een medianepass mag de bevroren tailgate niet
vervangen.

## Capability en fysieke plaatsing

De read-only capabilityprobe vond twee OpenCL-platforms, maar selecteerde
uitsluitend:

```text
Intel(R) OpenCL Graphics
Intel(R) Arc(TM) Pro 140T GPU (32GB)
driver 32.0.101.8517
```

De Intel-device meldde:

- `CL_DEVICE_HOST_UNIFIED_MEMORY = true`;
- `cl_intel_unified_shared_memory`;
- host-USM capabilities `3` (access + atomic access);
- werkende functiepointers voor `clHostMemAllocINTEL`,
  `clMemFreeINTEL`, `clSetKernelArgMemPointerINTEL` en
  `clGetMemAllocInfoINTEL`.

De daadwerkelijke weightallocatie was:

| veld | resultaat |
|---|---:|
| allocatietype | `CL_MEM_TYPE_HOST_INTEL` (`0x4197`) |
| grootte | 538.230.784 bytes |
| base-pointerattest | exact |
| alignment | 4.096 bytes |
| gate/up-ring | 531 records, 537.219.072 bytes |
| ringgrootte | **512,33203125 MiB** |
| extra downrecord | 1 |

De kernel kreeg de weightpointer met `clSetKernelArgMemPointerINTEL`.
Voor weights werden nul `cl_mem`-buffers, nul enqueue-writes, nul copies en nul
migrations aangevraagd. Hardwarecaches blijven normale caches; de audit bewijst
dat geen private volledige weightbank via de gebruikte API is gemaakt.

De 531 ringrecords kwamen fysiek uit laag 0–2 van de bestaande P1D-bank. De
drie bronlayer-SHA's zijn opnieuw volledig gecontroleerd. De compacte
USM-contentdigest was:

```text
50b2e5a7d9ef58cd5712cd8792f2a32fb3157ce9129577f72831caa00e94fb08
```

### Kleine preregistratiecorrectie

In de tekst van de eerste preregistratie staat één handmatige vermenigvuldiging
verkeerd als `537.519.072 bytes / 512,618 MiB`. De bevroren operationele
constant was steeds 531 records. Onafhankelijke herberekening geeft
`531 × 1.011.712 = 537.219.072 bytes = 512,33203125 MiB`. Dit blijft boven de
512-MiB-gate; geen threshold, recordaantal of runtimeparameter is aangepast.

## Exactheidsbewijs

Beide fysieke mappings werden gecontroleerd op echte laag-0/expert-0 gate-,
up- en downrecords onder drie vaste inputfamilies:

- normale power-of-two-input;
- cancellationgevoelige tegengestelde paren;
- `+0`, `-0`, subnormalen, kleinste normalen en tekenwissels.

Het onafhankelijke CPU-orakel decodeerde de little-order Q5-codes, gebruikte de
opgeslagen BF16 group-128-schalen, rondde ieder gewicht naar BF16, behield alle
256 accumulatoren, voerde strides `128,64,32,16,8,4,2,1` in volgorde uit en
rondde de output naar BF16-RNE.

| mapping | projecties × inputs | outputelementen | bitverschillen |
|---|---:|---:|---:|
| bronboom | 3 × 3 | 10.752 | **0** |
| ERGV width 8 | 3 × 3 | 10.752 | **0** |

Alle achttien expected/observed SHA-paren waren exact gelijk. Dit is sterker
dan de oude P11B-OpenVINO-proef, waarin de Intel-iGPU-output niet tegen CPU werd
opgeslagen.

## Meetprotocol

Per mapping:

- 531 echte gate/up-records in één host-USM-ring;
- batch 16;
- 34 untimed batches om de hele ring minstens eenmaal te raken;
- 1.000 timed events;
- startindex `(iteration × 17) mod 531`, waardoor de hele ring rouleert;
- 16.121.856 feitelijk gelezen code+scalebytes per event;
- OpenCL-eventtijd plus hostwallclock;
- geen header- of paddingbytes als effectieve throughput geteld.

Width 8 gebruikte geen post-hoc widthsearch. De keuze was vóór ST2 al de
bevroren Q5 gate/up-keuze in P7/ERGV en ERGV-C2. De mapping laat acht fysieke
Intel-subgrouplanes elk 32 bronaccumulatoren emuleren, met dezelfde lane-lokale
folds en ordered subgroup-shuffles als de logische bronboom.

Width 8 verbeterde de p50 met 1,860x en de conservatieve taildoorvoer met
1,920x. Dat bevestigt ERGV ook op een tweede GPU-vendor als exact
reductiemechanisme, maar niet als industriële performancepass.

## Page- en fouttelemetrie

De primaire 256-threadrun duurde lang genoeg voor één post-warmup PDH-sample;
`Page Reads/sec` en `Pages Input/sec` waren beide nul. De snellere width-8
timed sectie voltooide in minder dan één seconde en leverde daardoor geen
1-Hz-PDH-sample. Zij krijgt dus terecht geen page-readpass. Dit verandert het
verdict niet: width 8 faalt onafhankelijk al de event- en wallclockdoorvoer.

De eerste uitvoerpoging stopte vóór allocatie of kernelwerk door een foutieve
ctypes-signatuur voor `clHostMemAllocINTEL`. Die poging is behouden onder
`failed_attempts/st2_mini_abi_attempt_001`; alleen de ABI-binding werd
gecorrigeerd, waarna exact dezelfde bevroren parameters zijn uitgevoerd.

## Betekenis voor SplitTree

ST2-mini sluit de oorspronkelijke empirische redenering op twee manieren aan:

**Positief bewezen:**

1. een grote echte Q5-bank kan op deze Intel-iGPU als host-USM worden gebruikt;
2. exacte Q5-code-, BF16-scale- en bronboomsemantiek is cross-backend mogelijk;
3. ERGV width 8 generaliseert bitexact naar Intel-subgroups en verdubbelt
   vrijwel de fysieke snelheid.

**Niet gehaald:**

1. stabiele p95-zijde `>=21,63 GB/s`;
2. de uit FP16/OpenVINO afgeleide `38,06 GB/s` Q5-premisse;
3. enig gelijktijdig iGPU+dGPU-, quotientmerge- of laagresultaat.

Met de nu gemeten conservatieve width-8-snelheid en de bestaande dGPU-link is
een nieuwe ideale, nog steeds onvolledige roofline:

```text
B_i = 18,540 GB/s
B_d = 26,159 GB/s
f_i = B_i / (B_i + B_d) = 41,48%
ideale actieve-weightvloer = 0,9732096 / (B_i + B_d) = 21,77 ms
iGPU-bankshard = 19,29 GiB
```

Dit betekent dat heterogene uitvoering niet wiskundig dood is. De vroegere
59/41-split en 15,16-ms-vloer zijn wel gefalsificeerd door de echte Q5-tail.
De herziene 21,77-ms-vloer laat slechts circa 3,23 ms ruimte tot een 25-ms
expert-planegate, nog vóór quotientverkeer, twee SwiGLU/down-barrières,
dGPU-Q5-compute en DDR-contentie. Een volledige SplitTree-build is daarom nu
niet gerechtvaardigd zonder eerst een kleine 50/50 one-layer concurrencytest
die deze overhead fysiek meet.

## Formeel besluit

```text
host-USM capability                    PASS
>=512 MiB echte circulaire Q5-bank     PASS
geen aangevraagde private weightcopy   PASS
Q5/ERVG bitexactheid                   PASS
source-tree >=21,63 GB/s p95-side      FAIL (9,654)
width-8 >=21,63 GB/s p95-side          FAIL (18,540)

ST2-mini verdict:
exact_host_usm_q5_pass_but_p95_throughput_gate_fail
```

De huidige ST2-schedule wordt gesloten. Een claim dat de iGPU `38 GB/s` echte
Q5 uit hostgeheugen levert is niet langer toegestaan. De componentpass voor
cross-vendor ERGV-exactheid blijft een echte nieuwe positieve bevinding.

## Artefacten

- `ST2_MINI_PREREGISTRATION_2026-08-12.md`
- `st2_mini_opencl_capability_probe.json`
- `scripts/streamq5_moe/st2_mini_opencl_probe.py`
- `st2_mini_host_usm_q5_result.json`
- `scripts/streamq5_moe/run_st2_mini_host_usm_q5.py`
- `ST2_MINI_ERVG_W8_CONFIRMATION_PREREGISTRATION_2026-08-12.md`
- `st2_mini_ergv_w8_result.json`
- `scripts/streamq5_moe/run_st2_mini_ergv_w8.py`
- `st2_mini_independent_verification.json`
- `scripts/streamq5_moe/verify_st2_mini_host_usm_q5.py`
- `SPLITTREE_MOE_INDEPENDENT_AUDIT_2026-08-12.md`

