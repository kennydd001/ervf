# E1 fase 2.1 — device-resident routing + device-LRU-cache (eager) — preregistratie (2026-08-15)

Bevroren vóór meting. Voort op `E1F2_ZEROCOPY_PREREGISTRATION_2026-08-15.md`
waarvan de microbench inmiddels is uitgevoerd:

- **G-E1F2-M0 PASS** (UVA correct), **G-E1F2-M2X PASS** (bitidentiek),
- M1 streaming bulk-read pinned host: **24,93 GB/s** (96% van DMA's 26,03),
- M2 in-kernel GEMV vanaf host: 7,27 GB/s → G-E1F2-K1 *marginaal* — maar het
  gebouwde ontwerp gebruikt het M1-pad (bulk-staging-kernel naar device-cache),
  niet het M2-pad. Accounting met gemeten getallen: misses 127 MB/token @ 24,9
  GB/s = 5,1 ms ≈ DMA-pariteit (4,9 ms); bespaarde route-readbacks + host-gap
  (S14: 4,7 ms GPU-idle + 23 syncs) zijn zuivere winst → netto positief → bouwen.

## Ontwerp (één variabele t.o.v. de A1-default-stack)

De MoE-laag zonder één enkele device→host-sync, nog steeds **eager** (geen
graph in deze fase):

1. `route_topk_f32` — router-head op device: sigmoid + bias + top-6 +
   gewichtsnormering, schrijft ids/gewichten naar device-buffer.
2. `cache_assign` — device-LRU: slot_of/expert_of/last_used-tabellen op
   device, serieel één-thread (deterministisch, tie → laagste slot).
3. `cache_fetch` — miss-experts bulk-gekopieerd vanaf pinned bank naar
   device-slot door een kernel (uint4, het M1-patroon, ~25 GB/s).
4. Indirecte varianten van de bestaande kernels (ERVF up-GEMV, gather_down,
   masked down-GEMV, weighted_accumulate) die slot/expert-id/global_scale/
   gewicht **uit device-buffers lezen** i.p.v. by-value args. Rekenkundig
   identieke bodies: zelfde bytes, zelfde volgorde, zelfde waarden.

De Python-lus lanceert alleen nog; hij leest nooit meer een route-waarde.

## Poorten (bevroren)

- **G-E1F21-C1** (correctheid): met `device_cache=True` reproduceert de
  runtime de A1/V36-tokenreeksen **exact** (alle prompts, 64 tokens,
  contexts_max 4096, capacity 72). Verwachting: bitidentiek; de enige
  toegestane verschilbron is de 6-element gewichtssom (associatie) — token-
  pariteit is de poort, zoals bij A1.
- **G-E1F21-INV** (cache-invariantie): capacity 56 vs 72, beide
  `device_cache=True` → identieke tokenreeksen. (De cache bewaakt waarden niet;
  hij mag de uitkomst niet raken.)
- **G-E1F21-CTL** (controle-arm, moet falen): `bad_pick=1` (slot s=5 kiest de
  7e beste expert) moet binnen 64 tokens op minstens één prompt van het anker
  afwijken. Faalt de controle-arm niet, dan heeft C1 geen onderscheidend
  vermogen en is C1 ongeldig.
- **G-E1F21-S1** (snelheid, eager): p50 tokentijd over de gegenereerde vensters
  ≤ default-stack p50 − 1,5 ms. (Verwachting −4 tot −8 ms: 23 readbacks +
  host-gap vallen weg. Eager houdt de launch-overhead; die is E1-2.2.)
- **G-E1F21-V1** (VRAM): device-extra < 32 MiB totaal.

## Beslisregels

- C1 faalt → niet adopteren, bug zoeken; geen toleranties verruimen.
- CTL slaagt (d.w.z. produceert toch het anker) → hele fase inconclusive.
- S1 faalt maar C1/INV/CTL passen → alsnog door naar E1-2.2 (graph), want het
  budget van fase 2 zit in uitgifte, niet in syncs alleen; dat wordt dan
  expliciet zo gerapporteerd.
- Claim boundary: S1 is een echte wandtijd-A/B (tok/s-niveau toegestaan omdat
  het een end-to-end meting is, geen componentoptelling).
