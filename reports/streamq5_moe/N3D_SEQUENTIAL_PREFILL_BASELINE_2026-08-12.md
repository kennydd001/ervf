# N3D — TTFT en sequentiële prefillbaseline

Datum: 2026-08-12

Dit is een post-hoc herberekening van de fysieke P13C-run; geen nieuwe testclaim.

De service-ready tijd voor het 7-tokenprompt tot de eerste vrije voorspelling was **459.244 ms**. Inclusief de eenmalige domeincacheactivatie was dit **655.005 ms**.

| tokens sequentieel | eerste cyclus wall | effectieve input tok/s | tweede cyclus wall |
|---:|---:|---:|---:|
| 1 | 82.510 ms | 12.120 | 95.912 ms |
| 7 | 459.244 ms | 15.242 | 524.664 ms |
| 128 | 7107.114 ms | 18.010 | 8023.848 ms |
| 512 | 30172.144 ms | 16.969 | 32964.357 ms |
| 1024 | 63091.121 ms | 16.230 | 67445.589 ms |
| 4096 | 284306.808 ms | 14.407 | 290415.558 ms |

De 4K-invoer kost sequentieel honderden seconden. Een echte GEMM-prefillkernel is dus nog open; deze tabel voorkomt dat decode-tok/s als prefillprestatie wordt gepresenteerd.

Claimgrens: geen cold-process-TTFT, geen batch-prefill, geen nieuwe ongeopende testpartition.
