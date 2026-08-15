# ST2-mini preregistratie — Intel iGPU Q5 host-USM truth test

Datum: 2026-08-12  
Status bij schrijven: protocol bevroren vóór OpenCL/Level-Zero capability-output,
kerneloutput of timingoutput werd geopend. Alleen de aanwezigheid van algemene
runtime/toolbestanden en de bestaande Q5-bank is vooraf als discovery bekeken.

## Doel en claimgrens

Test uitsluitend de empirische premisse van SplitTree dat de lokale Intel Arc
Pro 140T echte STREAMQ5-Q5-records rechtstreeks uit host/shared memory kan
uitvoeren zonder een verborgen private weightcopy, met bitexacte
256-accumulatorsemantiek en voldoende p95-doorvoer.

Dit is geen dGPU-, cross-device-, SplitTree-, modelkwaliteit- of volledige
decodeclaim. De NVIDIA-dGPU mag door deze proef niet worden gebruikt.

## Bevroren brondata

- `reports/runs/streamq5_moe/p1d_q5_bank/layer_*.q5bin`;
- echte Qwen3-30B-A3B Q5-records;
- record: 64-byte header, 983.040 codebytes, 24.576 BF16-scalebytes en 4.032
  paddingbytes; 1.011.712 bytes totaal;
- code: acht little-order 5-bitwaarden per vijf bytes, signed code = stored-15;
- scale: persisted little-endian BF16, group size 128;
- gate/up: 768 rijen × 2.048 kolommen;
- down-correctness-smoke: 2.048 rijen × 768 kolommen.

## Fase C — capability en bewijs van hostplaatsing

De test mag alleen doorgaan als één OpenCL/Level-Zero-device ondubbelzinnig de
Intel Arc Pro 140T-iGPU is en `cl_intel_unified_shared_memory` aanbiedt.

Vereist:

- host-USM allocation capability en device access tot host-USM;
- weights via `clHostMemAllocINTEL`;
- kernelargument via `clSetKernelArgMemPointerINTEL`;
- geen `clCreateBuffer` voor weightdata;
- geen migrate/copy/enqueue-write van weightdata na de eenmalige CPU-vulling;
- devicevendor bevat Intel en device type is GPU; NVIDIA wordt verworpen;
- exact 531 volledige gate/up-records in de host-USM-ring:
  `531 × 1.011.712 = 537.519.072 bytes = 512,618 MiB`, dus groter dan 512 MiB.

Als host-USM of het functiepad ontbreekt, eindigt ST2-mini als
`blocked_no_auditable_host_usm_path`. Een OpenVINO-constantengraph of
`CL_MEM_USE_HOST_PTR` telt niet als vervanging, omdat een private drivercopy
daarmee niet uitgesloten is.

## Fase Q — bitexacte Q5-poort

De OpenCL-kernel moet dezelfde logische operator uitvoeren als de lokale
STREAMQ5/ERVG-reference:

1. decode packed Q5;
2. BF16-scale naar FP32;
3. `weight = round_bf16(code × scale)`;
4. 256 logische accumulatoren, bronvolgorde per accumulator behouden;
5. binaire strides `128,64,32,16,8,4,2,1` in dezelfde operandvolgorde;
6. final output `round_bf16_rne`.

Correctness gebruikt ten minste één echte gate-, up- en downrecord en vaste
normale plus cancellationgevoelige input. De opgeslagen BF16-outputbits moeten
voor ieder element gelijk zijn aan een onafhankelijke CPU-orakelberekening.

Harde gate: `bit_differences = 0`. Iedere afwijking is
`negative_cross_backend_q5_semantics` en stopt performanceclaiming.

## Fase P — 512-MiB circulaire host-USM-doorvoer

- de 531 echte gate/up-records worden eenmalig op CPU in host-USM geladen;
- een volledige untimed sweep raakt ieder record;
- daarna minimaal 1.000 timed batch-events, met geroteerde startindex en
  wrap-around door de volledige ring;
- iedere event verwerkt een vooraf vast te leggen vast aantal volledige
  records; alle rijen worden berekend en een outputdigest wordt opgeslagen;
- tijd komt uit OpenCL profiling-events, met hostwallclock als controle;
- effectieve bytes zijn alleen werkelijk gelezen code- en scalebytes:
  `1.007.616 bytes/record`; headers en padding tellen niet als kernelbytes;
- rapporteer mean, p50, p95-latency en mean/p50/p95 effectieve GB/s. Voor de
  doorvoerpoort wordt de conservatieve vijfde bandbreedtepercentiel gebruikt,
  gelijk aan de 95e latencyzijde.

Harde gates:

```text
Intel iGPU geselecteerd                         true
host-USM pad aantoonbaar                        true
host-USM weightbytes                         >= 536.870.912
verborgen private weightbuffer/copy              0
Q5 bitverschillen                                0
p95-latentiezijde effectieve throughput       >= 21,63 GB/s
```

Een allocatie-, build-, timeout-, driver- of API-fout wordt letterlijk
gerapporteerd. Gates worden niet aangepast na output. Geen centrale registry
wordt door ST2-mini gewijzigd.

