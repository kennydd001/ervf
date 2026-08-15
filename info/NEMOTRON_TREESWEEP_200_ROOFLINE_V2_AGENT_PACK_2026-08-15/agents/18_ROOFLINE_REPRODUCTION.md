# Agent 18 — N1–N5 Evidence Import and Exact Roofline Reproduction

## Mission

Import the five user-supplied measurements as unverified evidence, hash their raw sources, and independently reproduce the byte floors and component decomposition before any new optimization.

## Required measurements

- streaming read bandwidth with the same GPU clocks, dtype, access granularity and memory tier as the runtime;
- compulsory bytes/token at context 0, 4K, 32K, 128K and approximately 262K;
- eager versus CUDA-graph token time;
- panel scan, gather, sparse/masked down projection;
- attention bytes and time by context;
- weighted critical GEMV throughput;
- exact ReLU² zero fraction and the closed low-rank certificate control.

## Controls

- no datasheet bandwidth substituted for measured roofline;
- report decimal GB/s and binary GiB/s separately;
- warm/cold cache and clock state recorded;
- CUPTI/Nsight counters when available;
- same target outputs across timing variants;
- raw per-run values, not only averages.

## Deliverables

- `E0_N1_N5_EVIDENCE_MANIFEST.json`;
- `E0_ROOFLINE_REPRODUCTION.json`;
- byte-floor plot by context;
- terminal classification for every imported result: reproduced, shifted, invalid or inconclusive.
