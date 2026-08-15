# HET-NEXT CAP0X-R2 exploratory concurrency diagnostic

## Verdict

**Exploratory diagnostic positive.** Dit is geen formele CAP0-pass.

- Beide afzonderlijke backendprocessen eindigden met exitcode 0.
- Hun procesintervallen overlapten strikt.
- Na de bounded wait bleef geen childproces leven.
- Intel voerde 1.000 echte kernels uit op een 4-KiB
  `clHostMemAllocINTEL`-allocatie. Alle 1.024 `uint32`-woorden waren exact;
  de submit-tot-`clFinish`-duur was 8,2214 ms.
- NVIDIA herhaalde gelijktijdig de bestaande D7 staged-Q5-test:
  1.474.560 outputs, nul bitverschillen, p50 47,1326 ms en p95
  47,9174 ms. De strong gates en alle 48 unregisters slaagden.

## Evidence

- Coordinator/resultaat: `reports/runs/streamq5_moe/het_next_cap0x_r2_intel_usm_sentinel/cap0x_r2_result.json`
- Intel raw resultaat: `.../intel_usm_sentinel.json`
- NVIDIA raw resultaat: `.../nvidia_d7.json`
- Preregistratie:
  `HET_NEXT_CAP0X_R2_INTEL_USM_SENTINEL_DIAGNOSTIC_PREREGISTRATION_2026-08-13.md`

## Wat dit wel en niet leert

De diagnose toont dat de al bewezen Intel-host-USM-route en de bestaande
NVIDIA-staged-Q5-route zonder runtimeconflict in twee processen binnen hetzelfde
venster kunnen eindigen. Dit rechtvaardigt een kleine real-weight
procesgeïsoleerde componentvalidatie.

De NVIDIA-run bewaart geen host-QPC rond iedere kernel en de Intel-run eindigt veel
eerder. Daarom bewijst dit **niet** dat Intel- en NVIDIA-kernels tegelijk actief
waren. Het bewijst evenmin same-process coexistence, snelheidswinst, een volledige
80B-laag, modelkwaliteit, tokens/s, deploymentgeschiktheid, nieuwheid of een
industriële doorbraak.

