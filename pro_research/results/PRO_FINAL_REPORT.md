# PRO research — automatisch eindrapport

Gegenereerd: `2026-08-15T21:30:37.732091+00:00`

## Oordeel in gewone taal

De experimenten leverden nog geen onafhankelijk geverifieerde doorbraakkandidaat op. De negatieve of technische uitkomst is wel bruikbaar: ze sluit een concreet mechanisme.

## Resultaten

| spoor | status | kerngetal |
|---|---|---|
| G0 full-token graph | NEGATIEF / POORT GEMIST | p50 31.287 -> 25.954 ms; winst 5.333 ms |
| G1 ERVF voor BF16/FP8/FP32 | NEGATIEF / POORT GEMIST | micro 1.509x; geïntegreerd — ms; — tok/s |
| G2 K-token epoch graph | NIET GEDRAAID | — |

## Detail per spoor

### G0 — E1F22 graph
- `G_E1F22_PAR`: **False**
- `G_E1F22_CTL`: **None**
- `G_E1F22_DET`: **True**
- `G_E1F22_S1`: **True**
- `G_E1F22_VRAM`: **True**
- Eager: 31.962 tok/s op basis van p50.
- Graph: 38.530 tok/s op basis van p50.

### G1 — generalized ERVF
- `attn_q_bf16` (bf16 4096x2688): 2.419x, bitexact=True.
- `attn_k_bf16` (bf16 256x2688): 0.771x, bitexact=True.
- `attn_v_bf16` (bf16 256x2688): 0.809x, bitexact=True.
- `attn_o_bf16` (bf16 2688x4096): 3.052x, bitexact=True.
- `router_f32` (f32 128x2688): 0.795x, bitexact=True.
- `mamba_in_fp8` (fp8 10304x2688): 2.257x, bitexact=True.
- `mamba_out_fp8` (fp8 2688x4096): 2.154x, bitexact=True.

### G2 — epoch graph
Niet uitgevoerd.

## Breakthroughcontrole

De productdrempel blijft **minstens 50 tok/s in een geïntegreerde causale run**, niet een microbenchmark. Een snelle component wordt niet automatisch bij andere percentages opgeteld.

## Bestanden

- `PRO_G0_E1F22_GRAPH_AB.json`
- `PRO_G1_DENSE_ERVF.json`
- `PRO_G2_EPOCH_GRAPH.json`
- `PRO_VERIFICATION.json`
