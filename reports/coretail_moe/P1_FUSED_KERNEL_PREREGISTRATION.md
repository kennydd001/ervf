# CORETAIL-MoE P1 — exacte fused-kernel preregistratie

Gelockt op 2026-08-12, na de onafhankelijke P0-pass en vóór de eerste
CORETAIL-kernelsnelheidsmeting.

## Vereiste bron

`reports/coretail_moe/p0_full_bank_format_verification.json` moet `p0_pass`,
26/26 controles en `p1_authorized=true` bevatten. De kernel leest de bestaande
fysieke core/tail en mag nergens een volledige gedequantiseerde matrix
materialiseren.

## Toolchain en vaste kernelvorm

- CUDA 13.2 via CuPy 14.1.1 NVRTC op de RTX PRO 2000 Blackwell Laptop GPU;
- één CUDA-block per uitvoerrij, exact 256 threads;
- FP32-accumulatie en BF16-schaalbits met dezelfde group-128-semantiek;
- geen block-size-, layout- of expertselectiesweep op de testset;
- één voorafgaande toolchain-/correctheidssmoke is toegestaan, zonder
  gerapporteerde P1-snelheidsclaim.

## Gelockte microbenchmark

- lagen: 0, 24 en 47;
- experts: de acht route-IDs van token 0 uit `general_router_ids` in de reeds
  bevroren supplementtrace van iedere laag;
- matrices: gate, up en down afzonderlijk;
- invoer: deterministische PCG64-float32-vectoren uit de P1-inputlock;
- warm-up: 100 launches; meting: 500 launches per record;
- rapporteer p50/p95/p99, weight-applicaties/s, effectieve bytes/s, piek-VRAM,
  thermische toestand en outputfout.

Verplichte baselines:

1. BF16-matrixvectorreferentie;
2. echt fixed-width uint2 GPTQ;
3. entropy-packed exacte GPTQ met gemeten hostdecode en H2D;
4. CORETAIL exact: resident core plus werkelijk geselecteerde tail.

## Correctheid

- fixed uint2 en CORETAIL moeten exact dezelfde gehele codes en BF16-schaalbits
  representeren;
- outputs worden vergeleken met FP32-accumulatie over exact die quantized
  weights;
- tolerantie: `max_abs <= 5e-3` en `relative_l2 <= 1e-4`;
- iedere niet-eindige output of semantische mismatch sluit P1.

## Full-token tailbudget

Gebruik de eerste 1.024 tokens van elk van de vijf bevroren domeintraces over
alle 48 lagen. Tel de werkelijk geselecteerde compressed tailrecords en meet
hostdecode plus pinned H2D. Door de 27,2-Gweight/s-computegate resteert maximaal
33,3 ms van het 100-ms/tokenbudget voor taildecode en transfer; de p95 moet
daarbinnen vallen.

## Harde P1-gates

- alle correctheidscontroles slagen;
- routed throughput minimaal 27,2 miljard weight-applicaties/s;
- full-token taildecode plus H2D p95 maximaal 33,3 ms;
- geen gedequantiseerde matrix in de CORETAIL- of uint2-kernel;
- alle runtime scratch-, offset- en transferbytes worden gerapporteerd.

Een pass opent P2-kwaliteit. Een fail sluit alleen de huidige CPU-zlib/NVRTC-
runtimeconstructie; de fysieke P0-representatie blijft een afzonderlijk bewezen
resultaat.
