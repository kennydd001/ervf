# P11B preregistratie — Intel NPU SwiGLU-expertmicrobenchmark

Datum: 2026-08-12. OpenVINO 2026.2.1 ziet vóór preregistratie `NPU: Intel(R)
AI Boost`; geen benchmarkuitvoer is geopend.

## Hypothese

Een NPU kan acht cold experts sneller uitvoeren dan de gemeten GPU-route als
zelfs een gunstige resident-FP16-implementatie onder de fysieke 1,12-ms
ERVF+PCIe-vloer blijft. Test één en acht onafhankelijke experts met
`2048 -> gate/up 768 -> SwiGLU -> down 2048`, batch 1.

## Protocol

- Devices: NPU, CPU, Intel GPU en NVIDIA GPU voor zover OpenVINO compileert.
- FP16-constanten en FP16-input; vijf warmups, honderd synchrone inferenties.
- Compile-tijd en first-infer afzonderlijk rapporteren.
- Controleer NPU-uitvoer tegen CPU met `max_abs <= 0,05` en `max_rel <= 0,02`.

## Gate

De NPU opent alleen een packed-Q5-vervolg als de 8-expert p50 én p95 maximaal
95% van de gemeten all-cold GPU-grens zijn en de numerieke grens slaagt. Een
fail sluit de NPU-snelheidsrichting voor deze machine onder deze gunstige
best-case; het is geen universeel NPU-oordeel.

