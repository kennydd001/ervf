# Opdracht: nieuwe wetenschap voor 50 tok/s bij lange context

Je neemt een werkende, gemeten MoE-runtime over. **Optimaliseren is klaar — er
moet nieuwe wetenschap uitgevonden worden.**

## Waar we staan (gemeten, niet geschat)

Nemotron 3 Nano 30B-A3B NVFP4 draait volledig op één **8 GiB laptop-GPU**
(RTX PRO 2000 Blackwell, cc 12.0), coherent (`The capital of France is` →
` Paris`):

| context | tok/s |
|---:|---:|
| 0 | 21,4 |
| 32.768 | 20,2 |
| 131.072 | **16,7** |
| 262.100 | ~13,2 |

Configuratie: 15,4 GiB routed expertbank pinned in host-RAM, 3,7 GiB
LRU-expertcache op device (31 slots/laag, **65% hitrate**), FP8 E4M3 KV,
embedding op host, fused NVFP4 decode+GEMV kernels.

## Waarom 50 tok/s nieuwe wetenschap vereist

Harde vloeren per token bij 262K, uit **gemeten** bandbreedtes
(PCIe 26,03 GB/s, device ~250 GB/s):

| term | vloer |
|---|---:|
| MoE cache-misses, 275 MB over PCIe | **10,6 ms** |
| MoE-compute, 774 MB device | 3,1 ms |
| attention FP8, 805 MB device | 3,2 ms |
| LM-head BF16, 704 MB | 2,8 ms |
| Mamba-projecties, ~330 MB | 1,3 ms |
| **som** | **~21 ms → 47,6 tok/s** |

**50 tok/s ligt bóven de roofline van deze aanpak.** Zelfs met elke kernel
perfect haal je het niet. Er moet dus iets veranderen aan *wat er beweegt*, niet
aan hoe snel het beweegt.

## Wat al weerlegd is — niet opnieuw proberen

Gemeten, met cijfers, in `N8_LONG_CONTEXT_FINAL_REPORT_2026-08-14.md`:

- **Minder transcendentals in de online softmax** (branch om de rescale-exp over
  te slaan): 13,225 → 12,404 tok/s. `__expf` is ~4 cycles met fast-math; de
  branch breekt de pipelining.
- **Meer flash-decode splits** (256 → 1024): 13,225 → 12,020. De combine-kernel
  loopt serieel over `splits×4` partials.
- **Cross-layer prefetch van experts**: causaal onmogelijk — laag `L+1`'s route
  hangt af van laag `L`'s output.
- **Statische expert-prior**: alle 128 experts worden gebruikt, spreiding maar
  8,7×. Alleen *temporele* lokaliteit (2,011 van 6 gedeeld tussen tokens) loont.

Verboden hypotheses uit eerder onderzoek staan in `EXPERIMENT_REGISTRY.yaml`
onder `forbidden_hypotheses` — o.a. pruning (lineair in fractie, +47,8% CE bij
50%), low-rank expert-surrogaten, Q2-semantiek.

## Waar je moet zoeken

De dominante term is **10,6 ms PCIe voor cache-misses**, en die is
VRAM-gelimiteerd. Richtingen die nog niemand gemeten heeft:

1. **Minder bytes per expert.** `relu2` maakt ~de helft van de 1856
   intermediates nul; die kolommen van `down_proj` worden met nul
   vermenigvuldigd. Sla `down_proj` getransponeerd op in de host-bank en
   transfer alleen de rijen waar `h[j] > 0`. Vereist een afhankelijke transfer
   ná `up_proj` — meet of dat de winst opeet.
2. **Batch > 1 / speculatief decoden.** Alle transferkosten amortiseren over
   meerdere tokens. Nemotron heeft mogelijk MTP-gewichten; controleer dat eerst.
3. **Expert-delta-codering.** Verschillen experts binnen een laag genoeg om een
   gedeelde basis + delta te rechtvaardigen? Meet de entropie voordat je bouwt.
4. **Route-voorspelling.** Als laag `L+1`'s route uit laag `L`'s hidden state
   voorspelbaar is boven toeval, wordt prefetch alsnog mogelijk. Meet de
   voorspelbaarheid, bouw pas daarna.

## Werkregels

- **Schrijf alleen in** `reports/lightningstream_nemotron/`,
  `scripts/lightningstream_nemotron/`, `src/moe_lab/lightningstream_nemotron/`,
  `tests/lightningstream_nemotron/`, `models/nemotron_3_5_lightning/`,
  `docs/LIGHTNINGSTREAM_NEMOTRON_RESEARCH_LOG.md`, `.venv-nemotron/`.
- **Alles daarbuiten is read-only.** Een tweede agent werkt aan de
  Qwen3-Coder-Next 80B-lijn in `reports/streamq5_moe/`. Draai
  `scripts/lightningstream_nemotron/protected_manifest.py verify` na elke fase;
  0 modified / 0 removed is de eis. Added is oké — dat is hun werk.
- **GPU delen:** blokkeer op `nvidia-smi --query-compute-apps`; nooit een proces
  killen. De Intel Arc iGPU is hún experiment — afblijven.
- **Protocol:** preregistratie mét gates vóór uitvoering, dan runner, dan een
  *aparte* onafhankelijke verifier die alles herberekent zonder de runner te
  importeren, dan rapport met claim boundary. Poorten worden nooit verruimd na
  het zien van een resultaat. Eén variabele per meting.
- **Nooit** een componentmeting opwaarderen naar tok/s.

## Startpunten

| wat | waar |
|---|---|
| volledige geschiedenis, alle lijnen | `LIGHTNINGSTREAM_RESEARCH_HANDOFF.md` |
| alle fases, gates, verboden hypotheses | `EXPERIMENT_REGISTRY.yaml` |
| huidige stand + weerleggingen | `N8_LONG_CONTEXT_FINAL_REPORT_2026-08-14.md` |
| runtime | `src/moe_lab/lightningstream_nemotron/runtime.py` |
| kernels | `gpu_kernels.py`, `fused_nvfp4.py` |
| meetrunner | `scripts/lightningstream_nemotron/n7b_cached_decode.py` |
| alles gebundeld | `ALL_RESEARCH_LINES_*.zip` |

Reproduceer eerst de huidige meting. Als je die niet haalt, ligt het aan de
omgeving en niet aan je idee.
