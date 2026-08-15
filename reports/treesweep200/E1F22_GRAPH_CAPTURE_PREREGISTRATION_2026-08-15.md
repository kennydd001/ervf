# Preregistratie E1 fase 2.2 — CUDA-graph-capture van de volledige token

**Datum:** 2026-08-15 · **Status: BEVROZEN vóór elke meting.** Poorten worden
niet verruimd na het zien van resultaten.

Voortbouwt op: E1 fase 1 (budget 8,925 ms host-issue-overhead, gemeten mét
ERVF) en E1 fase 2.1 (device-resident routing + device-LRU, eager; −4,542 ms
gemeten, restbudget ~4,4 ms). Anker: `V36_DETERMINISTIC_ANCHOR.json` +
bevroren A1-ids uit `A1_ADOPTION_PRECONDITION.json`.

## Hypothese

Met het MoE-pad sync-vrij (2.1) kan de hele decode-token — embedding t/m
argmax, alle 31 lagen, mamba/attentie/MoE — als één CUDA-graph worden
gecaptureerd en gereplayd. De host doet per token alleen nog:
`graphLaunch` + een asynchrone id-readback. Verwachte winst: het resterende
launch-overhead (~4,4 ms) minus wat de replay zelf kost.

## Ontwerp (vastgelegd, zodat de poorten iets hebben om tegen te toetsen)

1. **`embed_gather`-kernel** — leest de token-id uit een device-buffer,
   gathered de bf16-rij uit de embeddingtabel en schrijft f32 in `h`
   (16-bit left-shift, identiek aan de huidige omzetting). De embeddingtabel
   wordt daarvoor **pinned+mapped** gealloceerd (+0,656 GiB pinned host-RAM;
   wordt gerapporteerd, niet gepoort). Geen cupy-temporaries meer in `step()`.
2. **`argmax`-kernels** (two-pass: 256×256 partiële reductie + finale) over
   de logits, lage-index-wint-ties, schrijft de volgende token-id in het
   device-token-buffer. Verifier toetst bit-gelijkheid met `cp.argmax`,
   inclusief gefabriceerde ties.
3. **`pos` op device** — `pos_dev` int32; graph-varianten van
   `kv_write_fp8` en van de attentie lezen pos/t uit dat buffer; een
   increment-kernel sluit de graph af. `reset()` schrijft pos_dev host-zijdig
   (buiten de graph).
4. **Attentie met vaste grid** — clone van `attn_decode_warp_fp8_gqa4` met
   grid (2, 256) vast; elke block berekent t en `splits/chunk` in-kernel;
   blocks met `s*chunk >= t` schrijven een **neutraal** partial
   (m=-3e38, l=0) zodat geen enkele slot oude data bevat;
   `attn_decode_combine` draait over de vaste 1024 slots en slaat l≤0 al over
   (bestaand gedrag). Numeriek identiek aan de eager v4.
5. **Multi-stream in de graph** — de copystream-fork (evt[0]/evt[1]) uit 2.1
   wordt mee-gecaptureerd; events zijn `disable_timing=True`. Kill-criterium
   hieronder dekt het geval dat event-hergebruik over 23 lagen in één capture
   niet wordt toegestaan.
6. **Token-bookkeeping** — decode: argmax schrijft de id in-graph; de host
   leest ids asynchroon terug via een pinned ringbuffer met één sync per
   K tokens. Prompt-tokens: host schrijft de id (4 B H2D) vóór elke launch.

## Armen (één variabele: graph-replay vs eager, zelfde runtime)

- **EGR** — eager device_cache-pad (de 2.1-arm), referentie voor ids én tijd.
- **GRAPH** — gecaptureerde token, replay-lus.
- **CTL** — graph gecaptureerd met `bad_pick=1`: MOET pariteit breken
  (werkregel 8; bewijst dat de graph echt de gesaboteerde routing uitvoert).
- **DET** — dezelfde prompt twee keer achter elkaar gereplayd (met reset
  ertussen): ids moeten identiek zijn (replay-determinisme).

## Bevroren poorten

| poort | eis |
|---|---|
| **G-E1F22-PAR** | GRAPH-ids == EGR-ids over **3 × 256 tokens** (2 anker-prompts × 256 + 1 code-prompt × 256), én de eerste 64 ids van elke anker-prompt matchen de bevroren A1-ids. |
| **G-E1F22-CTL** | CTL-arm breekt pariteit met EGR op minstens één prompt. |
| **G-E1F22-DET** | twee identieke replays geven identieke ids (3 × 256). |
| **G-E1F22-S1** | p50(GRAPH) ≤ p50(EGR) − **2,5 ms** bij `contexts_max=4096`, ≥500 getimede tokens. Motivatie: restbudget 8,925 − 4,542 = 4,383 ms; de poort eist ≥57% daarvan, ruim boven de meetruis (p50-spreiding < 0,5 ms in 2.1). |
| **G-E1F22-VRAM** | device-geheugengroei t.o.v. de EGR-arm < **64 MiB** (graph-instantiatie + device-buffers). |

## Meetplan

- Zelfde regime als 2.1: `contexts_max=4096`, model `nemotron_3_5_lightning`,
  `embed_on_host=True`, `fp8_kv=True`, cache capacity 72, D1 aan.
- Timing: per-token `perf_counter_ns` rond `graphLaunch` + async readback,
  sync per token voor id-verzameling; p50 over ≥500 tokens (context groeit
  0→~768, dus de meting dekt ook groeiende t in de vaste-grid attentie).
- GPU-vrijheid gecheckt via `nvidia-smi --query-compute-apps` vóór de run
  (exit 4 bij bezetting). Runner schrijft ruwe samples + poortevaluatie naar
  `E1F22_GRAPH_CAPTURE_AB.json`.

## Kill-criteria (expliciet, vóór de meting)

- **K1:** stream-capture weigert de multi-stream MoE-fork (event-fout) →
  fallback: capture met geserialiseerde fetch (één stream). Dat wordt dan een
  tweede, eerlijk benoemde arm; als ook die faalt: fase 2.2 = technisch
  geblokkeerd, rapport beschrijft waarom.
- **K2:** pariteit breekt en is na inspectie niet toe te schrijven aan een
  implementatiefout maar aan intrinsieke graph-semantiek → fase stopt, geen
  poortverruiming.
- **K3:** p50-winst < 1,0 ms → de graph is niet de moeite; rapport als
  weerlegging van deze aanpak, budget blijft onbenut.

## Claim boundary (vooraf vastgelegd)

- Een PASS is een claim over **dit regime** (ctx ≤ ~768, 8 GiB-GPU, dit
  model), geen langecontext- en geen tok/s-eindclaim; die volgen pas uit het
  aparte langecontext-profiel.
- Componentmetingen worden niet opgewaardeerd. Alleen de end-to-end
  token-p50 telt voor S1.
