# PRO-MAX V2 eindrapport

## Uitgangspunt

V6: **21.0923 ms/token = 47.41 tok/s**.

## Exacte final-mile kandidaten

| kandidaat | status | adopt | p50 ms | tok/s | winst ms |
|---|---:|---:|---:|---:|---:|
| Add+RMSNorm | gate_failed | False | 21.3962 | 46.74 | 0.3441 |
| Q/K/V one-launch | gate_failed | False | 21.4866 | 46.54 | 0.2386 |
| LM-head+argmax | gate_failed | False | 22.3746 | 44.69 | -0.5168 |

## Gecombineerde V10-run

Status: **gate_failed**.
P50: **21.5157 ms = 46.48 tok/s**.
Nog tot 50 tok/s: **1.5157 ms/token**.
Milestones: `{"E100_single_stream": false, "E50_single_stream": false, "E75_single_stream": false}`.

## Exacte child-graph epochs

Status: **gate_failed**; beste: `null`.

## Onafhankelijke verificatie

Verdict: **verification_failed**.

## Interpretatie

Een componentmicrobenchmark is geen tok/s-doorbraak. Alleen de gecombineerde causale V10-run kan E50 openen. Child-graph epochs zijn queued/offline throughput en veranderen de latency van het eerste token niet. 100 tok/s single-stream blijft een afzonderlijke, veel zwaardere eis; aggregate batch>1 blijft de voornaamste post-E50 architecturale route.
