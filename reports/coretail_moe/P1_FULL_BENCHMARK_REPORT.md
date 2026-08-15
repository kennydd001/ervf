# CORETAIL-MoE P1 — volledige fused-kernelbenchmark

Uitkomst: **p1_pass** (5/5 gates).

CORETAIL aggregate p50: 33.319 Gweight/s; p95: 30.738 Gweight/s; gate: 27,2.
Pinned tail H2D bij de werkelijke p95-tokenomvang van 71.286 MiB: p95 3.314 ms; gate: 33,3 ms.
Correctheid: 72/72 gelockte matrixgevallen.

De runtime houdt de volledige tail eenmaal gedecomprimeerd in host-RAM; per token worden uitsluitend geselecteerde raw flags en row offsets gekopieerd. P1 bewijst geen modelkwaliteit of end-to-end tok/s.
