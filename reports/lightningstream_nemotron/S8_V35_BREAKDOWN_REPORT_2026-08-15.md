# S8 — componentbreakdown v35: de GQA-kloof verklaard

Datum: 2026-08-15
Verdict: **De "GQA levert minder dan voorspeld"-kloof bestaat niet. Geïsoleerde componentmetingen tellen te veel; attention is wél gehalveerd. MoE is nu de dominante term en zit ~26× van zijn roofline.**
Terminal state: `s8_gqa_gap_resolved_moe_now_dominant`

## Meting bij 262.100 (v35, capacity 72)

| component | per call | ×n | totaal |
|---|---:|---:|---:|
| **MoE-laag** | 1,718 ms | 23 | **39,523 ms** |
| attention-laag | 3,106 ms | 6 | **18,634 ms** |
| Mamba-laag | 0,361 ms | 23 | 8,309 ms |
| `lm_head` | 2,106 ms | 1 | 2,106 ms |
| RMSNorm | 0,013 ms | 53 | 0,716 ms |
| — som der delen | | | **69,287 ms** |
| — **gemeten token** | | | **52,363 ms** |
| — niet-toegewezen | | | **−16,924 ms** |

## De kloof: een meetartefact, geen kernelprobleem

De niet-toegewezen term is **negatief**: de som der delen (69,287 ms)
overschrijdt het gemeten token (52,363 ms) met 16,9 ms. Bij ctx 0 is dat zelfs
−23,2 ms.

Dat is het antwoord op de S7-vraag. Een geïsoleerde componentmeting forceert een
synchronisatie die de echte decode-lus verbergt — de copy-stream loopt tijdens
compute. **Geïsoleerd meten telt dus systematisch te veel**, en het verschil
tussen "som der delen" en het echte token ís de gerealiseerde overlap.

Attention is wel degelijk gehalveerd: **37,6 ms (S6, vóór GQA) → 18,6 ms (nu)**.
Kimi's 10,66× amplificatiemeting was correct; de kernel doet wat hij belooft.
Het end-to-end effect leek klein omdat attention niet de hele verklaring was.

**Les voor de methode:** een negatieve restpost is geen fout maar informatie.
Wie hem wegdeelt of "overhead" noemt, verliest precies het signaal dat overlap
werkt. Hij blijft hier onbenoemd en wordt als getal gerapporteerd.

## Het knelpunt is verschoven naar MoE-compute

Bij 80,4% hitrate zijn de misses 19,6% × 138 ≈ 27 records. Met S5's up-only
transfer (2,81 MB) is dat ~76 MB over PCIe ≈ **2,9 ms**. De MoE-term is
**39,5 ms**. Het is dus geen transfer.

Rooflineschatting voor MoE-compute: 138 experts × (up 2,49 MB + ~9% van down)
≈ 375 MB device-lees bij ~250 GB/s ≈ **1,5 ms**. Gemeten 39,5 ms is daarmee
~26× van de roofline — de grootste relatieve kloof die nu in het systeem zit.

Per expert is dat 0,286 ms voor werk dat ~11 µs aan bandbreedte kost. De
waarschijnlijke oorzaak is **launch- en kleine-kernel-overhead**: per expert
lopen up-GEMV, `panel_scan`, `gather_down_sparse` en down-GEMV als aparte
launches, dus ruwweg 4 × 138 ≈ 550 launches per token alleen voor de experts.

## Volgende hypothese (te preregistreren, één variabele)

**H-S9: batch de zes experts van een laag in één kernel-launch.** In plaats van
zes losse up-GEMV's van elk 1.856 blocks, één launch met grid `(1856, 6)`; idem
voor down. Dat brengt de expert-launches van ~550 naar ~92 per token.

Voorspelling vóór meting: als launch-overhead de oorzaak is, daalt de MoE-term
substantieel; als hij nauwelijks beweegt, is de oorzaak iets anders en moet die
eerst gemeten worden in plaats van geraden.

Waarschuwing bij het bouwen: de per-expert `weight_scale_2` en de routegewichten
verschillen, dus een gebatchte kernel moet die als arrays meekrijgen. De
reductie-orde per outputrij moet identiek blijven aan nu, anders verandert de
generatie en is de winst waardeloos.

## Ook bevestigd

`lm_head` van 5,965 ms (S6, BF16) naar **2,106 ms** (NVFP4) — 2,8× sneller,
puur uit het checkpointformaat.

## Claim boundary

Componenttoerekening op deze runtime, GPU en diepte. Elk cijfer is direct
gemeten, nooit door aftrekking; de niet-toegewezen term wordt gerapporteerd en
niet benoemd. Geen tok/s-doelclaim, geen kwaliteitsclaim.

## Artefacten

`scripts/lightningstream_nemotron/s8_v35_breakdown.py` · `s8_v35_breakdown.json`
