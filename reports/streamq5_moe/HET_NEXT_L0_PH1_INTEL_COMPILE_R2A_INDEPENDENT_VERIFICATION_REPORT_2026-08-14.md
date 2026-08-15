# PH1 Intel compile R2A — onafhankelijke verificatie

Datum: 2026-08-14  
Methode: uitsluitend CPU/read-only; geen OpenCL-, compiler- of devicecall door deze audit.

## Verdict

**PASS — 10/10 onafhankelijke controles.** R2A levert geldige compile-eligibility voor de exact gebonden Intel Arc-configuratie. De 7.852-byte R2-bron compileerde met code `0` tot een niet-lege native program binary van 186.352 bytes. Program en context zijn beide met code `0` vrijgegeven en de finale live-resourcecount is nul.

Dit resultaat bewijst uitsluitend dat de bevroren vier-kernelbron op deze driver kan bouwen en als native binary kan worden uitgelezen. Het bewijst nog geen host-USM-allocatie, kernelcreatie of -uitvoering, payloadcorrectheid, numerieke gelijkheid, timing of performance.

## Herberekende kernfeiten

| bewijs | onafhankelijk resultaat |
|---|---|
| pakketbestandenset | exact 6 bestanden; geen ontbrekende of extra bestanden |
| `result.json` | 8.404 bytes; SHA-256 `ac7c90e15c71cf2a481004f78954e9d78631078d3e08893d3f716120345df5cc` |
| `manifest.json` | SHA-256 `9be3e584a4bda6b827ef1f536ccbbff42f6b2bbd4cbf998a530e7133f0a369ff` |
| `commit.json` | 220 bytes; SHA-256 `c9f9ab3838d9d3d4ddd6e16a18f7989c16061901f08e046987081d9d975a152a` |
| R2 OpenCL-bron | 7.852 bytes; SHA-256 `f1b3ccdae6d202ed210810e3cd419f726ea89ffa8fba0c84df5c2bfca3a84d21` |
| buildlog | 1 byte; SHA-256 `6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d` |
| native binary | 186.352 bytes; SHA-256 `8b57db279fbb1d7d8df17ebab5cfb54203ef8da8cc31df2d136650820548f629` |
| buildopties | `-cl-std=CL3.0 -cl-fp32-correctly-rounded-divide-sqrt` |
| identiteit | Intel Arc Pro 140T 32GB; Intel; driver `32.0.101.8517`; PCI `0000:00:02.0`; host-USM-extensie aanwezig |
| ledger | exact 8 rijen: identity, context, program, build, binary-read, program-release, context-release, cleanup |

De R2-bron is onafhankelijk opnieuw afgeleid van de oudere bron. Exact drie beoogde tekstwijzigingen zijn teruggevonden: de waarschuwende required-subgroup-pragma verwijderd, de gereserveerde identifier `half` eenmaal naar `halfway` hernoemd en zijn predicaatgebruik aangepast. De pakketbron is bytegelijk aan deze afleiding.

De manifestrijen matchen alle payloadbestanden in lengte en digest. `commit.json` bindt exact de manifest- en result-digest. Alle in `result.json` opgegeven R1B/R2/R2P1/R2A-, CPU-package- en auditbestanden zijn opnieuw gehasht en matchen. De fysieke ledger bevat buildcode `0`, een enkel program device, exact opgegeven binarylengte bij query én read, schone program/context-releases en geen cleanupfouten. `payload_read=false`; queues, kernels, events, memory objects, allocaties en launches zijn alle exact nul.

## Exacte volgende poorten: Intel host-USM full expert

R2A opent **implementatie en statische audit**, niet automatisch een fysieke correctness-run. De eerstvolgende immutable Intel-revisie moet vóór één devicepoging minimaal deze poorten sluiten:

1. **Nieuwe binding en autorisatie.** Bind de onafhankelijk geverifieerde CPU-freeze, dit R2A commit/result/manifest, bron `f1b3ccda…`, binary `8b57db27…`, buildlog `6e340b9c…`, buildopties en exacte Intel-identiteit. Eis eerst frozen source-audit, daarna een no-device statische preflight en pas daarna een afzonderlijke one-attempt lock.
2. **Compiler/runtime-identiteit.** De volledige runner moet vóór kernelcreatie exact dezelfde bron, opties, buildlog en native-binarydigest reproduceren of een vooraf vastgelegde, geaudite binary-loadroute gebruiken; geen stilzwijgende rebuildvariant, fast-math, FTZ of contractie.
3. **Veilige predevice-ingang.** Verifieer commit, source/Q5-records, input, LUT en alle digests vóór de eerste OpenCL-call. Voer exact 22 fail-closed veilige controls uit; zij moeten ieder vóór compile/alloc/launch afwijzen. Houd de drie numerieke q-step-witnesses apart als CPU-diagnostiek.
4. **Echte copyless host-USM.** Maak exact 14 afzonderlijke, unieke, 4.096-byte-uitgelijnde host-USM-allocaties met samen `2.185.216` semantische bytes. Attesteer per pointer via `clGetMemAllocInfoINTEL`: type host, base exact pointer en grootte exact. `cl_mem`, buffer APIs, read/write/copy, migrate en prefetch blijven allemaal nul.
5. **Exacte uitvoeringsledger.** Eén in-order queue; vier kernels (`gate_linear`, `up_linear`, `activation`, `down_linear`); exact 18 pointerargs; globale/lokale groottes respectievelijk `4096/256`, `4096/256`, `512/256`, `16384/256`; geen events; exact één `clFinish`; CPU-directe outputreads uitsluitend daarna.
6. **Niet-vacuous bitcorrectheid.** Initialiseer vijf outputs met `0xffff` en vier counterarrays met nul. Vereis vier all-one counters, alle canaries overschreven en exacte BF16-stagehashes: gate `e8a00c17…`, up `f8dc1dc2…`, SiLU `a83041f1…`, activatie `762384a5…`, down `142607c8…`.
7. **Resources en cleanup.** Start-RAM minstens 16 GiB, na iedere fase minstens 2 GiB beschikbaar, piek-working-set maximaal 12 GiB en retained artifacts maximaal 16 MiB. Probeer in omgekeerde volgorde exact 21 releases (14 USM, 4 kernels, program, queue, context), ook na een eerdere releasefout; alle returncodes bewaren en eindigen met nul live owned resources.
8. **Onafhankelijke adjudicatie.** Een verifier zonder import van runner/backend/common moet pakket, controls, raw vijf stages, counters, verbodsmetingen, volledig geordende ledger, resourcegates en commit-last-transactie herberekenen. Elke afwijking blijft formeel negatief en krijgt atomische failure evidence.

Alleen een PASS op deze Intel-poorten bewijst één volledige officiële expert-50 Q5-route op Intel host-USM voor het ene bevroren inputvectorcontract. Zelfs dat is nog geen NVIDIA-, multi-expert-, router-, full-layer-, performance- of doorbraakclaim.

## Audit-artefacten

- verifier: `scripts/streamq5_moe/verify_het_next_l0_ph1_intel_compile_r2a.py`, SHA-256 `1bdff04b6615d8cc4be592a3ec879997cdc095f54317a52a43c33187ce48f720`;
- verification JSON: `reports/streamq5_moe/het_next_l0_ph1_intel_compile_r2a_independent_verification.json`, SHA-256 `99fc50814373c96e18254bd130a6b21797276c8a0922fb892782364deb5fafea`.
