# PORT80B-D2 — onafhankelijke CPU-only verificatie

**Verdict:** `verified_negative_with_protocol_findings`  
**GPU-context geopend:** nee  
**Alle replaybare reken-, hash- en provenancechecks:** PASS  
**Volledige protocolconformiteit:** FAIL

## Herberekende fysieke resultaten

| prefix | n | mean ms | p50 ms | p95 ms | p99 ms | min ms | max ms | GB/s bij p95 | Page Reads/s max | bronmismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 307/512 (60.0%) | 120 | 51.912406 | 51.889791 | 52.372851 | 52.542288 | 51.438560 | 52.595074 | 18.582330 | 4.928408 | 0 |
| 358/512 (69.9%) | 120 | 51.795291 | 51.781857 | 52.046046 | 52.206916 | 51.448002 | 52.234177 | 18.699011 | 93.987077 | 0 |
| 410/512 (80.1%) | 120 | 51.774465 | 51.761280 | 52.064005 | 52.232546 | 51.439583 | 52.315105 | 18.692561 | 5.998743 | 0 |

Alle drie getimede prefixes hebben 120 eindige samples. Hun opgeslagen mean/p50/p95/p99/min/max, bandbreedte, gates en page maxima zijn bit-/floating-pointconsistent met de ruwe JSON-reeksen. Iedere prefix rapporteert 48 geslaagde registraties en nul unregisterfouten.

## Failure arithmetic

| experts | p95 boven 45 ms | factor | nodige latencyreductie | GB/s-tekort | tekort t.o.v. gate |
|---:|---:|---:|---:|---:|---:|
| 307 | 7.372851 | 1.163841× | 14.078% | 3.044670 | 14.078% |
| 358 | 7.046046 | 1.156579× | 13.538% | 2.927989 | 13.539% |
| 410 | 7.064005 | 1.156978× | 13.568% | 2.934439 | 13.568% |

De 21,627-GB/s-poort is de 45-ms-poort naar boven afgerond. Exact 973.209.600 bytes in 45,000 ms is 21,62688 GB/s: formeel zou dat de latencygate halen maar de bandwidthgate met 0,00012 GB/s missen. Deze minieme afrondingsinconsistentie beïnvloedt D2 niet; de gemeten tekorten zijn 2,93–3,04 GB/s.

## Capability en kleine mapped-hostprobe

- Device: `NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU`, compute capability `12.0`, discrete=`True`, unified addressing=`1` en async engines=`1`.
- `canMapHostMemory=1`, `hostRegisterSupported=1`, `hostRegisterReadOnlySupported=1`, `canUseHostPointerForRegisteredMem=0`, `pageableMemoryAccessUsesHostPageTables=0`.
- De 64-MiB-registratie rapporteert een niet-nulle devicepointer en gelijke eerste/laatste 4.096 bytes.
- `setDeviceFlags(cudaDeviceMapHost)` werd niet aangeroepen: CuPy miste de binding en bewaarde een `AttributeError`. De mapped probe werkte desondanks.
- **Protocolafwijking:** de preregistratie eiste een byte-voor-bytevergelijking van de 64-MiB-probe; de runner vergeleek slechts 8.192 randbytes. De volledige probe is dus niet bewezen.

## Mismatch- en routeaudit

Voor elk van 307/358/410 experts zijn de exacte correctnessroutes (`token=20.000+prefix`) onafhankelijk gereconstrueerd: 48 lagen × tien unieke experts, alle binnen de prefix. De verifier scande voor iedere prefix alle 973.209.600 geselecteerde bronbytes; headers, Q5-codes, BF16-schalen en padding hebben nul structurele mismatches. De volledige bank-SHA is opnieuw CPU-side `4a97af22833b239badc065d9c065ca259c791a84218640946d68c4e72e034462`.

De D2-run bewaart echter alleen `full_destination_mismatch_count: 0`, geen destinationhash of buffer. De tijdelijke GPU-bestemming kan daarom niet post-hoc CPU-only worden hervergeleken. De evaluatorbron dekt aantoonbaar alle bytes, maar de uitgevoerde GPU-uitkomst blijft een niet-replaybare scalarclaim.

## Registratie, OOM en destructorscope

| prefix | 48 ranges geregistreerd | unregisterfouten | status |
|---:|---|---:|---|
| 307 | ja | 0 | timed |
| 358 | ja | 0 | timed |
| 410 | ja | 0 | timed |
| 512 | ja | **48** | `cudaErrorMemoryAllocation` |

De 100%-prefix registreerde aanvankelijk alle 48 ranges, maar faalde vóór correctness/timing met OOM. Daarna rapporteerde ieder van de 48 unregistercalls eveneens OOM; de full-bankpoort faalt dus zowel op uitvoering als op verplichte succesvolle unregister.

Voor de destructorvraag gelden twee scopes:

- **Prefix-lokaal:** 307/358/410 waren al volledig getimed en zonder gemelde unregisterfout afgesloten. De latere 512-OOM verandert hun ruwe samples niet; hun lokale `no_cuda_or_runner_error=true` is intern consistent.
- **Strikt run-globaal:** dezelfde Python-run registreerde later een CUDA-OOM en 48 cleanupfouten. Een globale claim “geen CUDA/runner error in D2” is daarom fout. Exitcode en stderr/destructorlog zijn niet opgeslagen, zodat een afzonderlijke destructor-OOM niet post-hoc kan worden gereplayed. Als destructorstderr meetelt, moet de globale foutpoort voor alle prefixes als false/unverified worden gezien.

Dit verandert het verdict niet: iedere getimede prefix faalt al de p95- en bandwidthpoorten, plus de page-readpoort.

## Page telemetry en bewijsgrens

- Prefix 307: max 4,928408 Page Reads/s.
- Prefix 358: max 93,987077 Page Reads/s.
- Prefix 410: max 5,998743 Page Reads/s.

Telemetry was beschikbaar, zonder samplererror en met strikt oplopende monotone timestamps; de zero-page-readgate faalt bij alle getimede prefixes.

Dit blijft synthetisch registered scattertransport. Geen Q5-rekenkernel, echte 80B-router, kwaliteit, dense shell, daadwerkelijke mapped-host ERGV-uitvoering, tokens/s of endurance is bewezen.
