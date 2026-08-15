# P1C Qwen streaming smoke-test

**Datum:** 2026-08-11  
**Status:** geslaagd; uitsluitend synthetische token-ID's, geen validation- of
testdata geopend.

De native `transformers==4.51.3` Qwen3-MoE-decoderlaag 0 is rechtstreeks uit
het reeds complete officiële BF16-shard 1 geladen, naar de GPU verplaatst en
met exact dezelfde aanroep uitgevoerd als de P1C-streamer.

| Controle | Resultaat |
|---|---:|
| Inputvorm | batch 2 × 1.152 tokens |
| Hidden/outputvorm | 2 × 1.152 × 2.048 |
| Laadduur | 0,924 s |
| Forwardduur | 0,542 s |
| Piek CUDA allocated | 1.399.409.152 B |
| Proces-RSS na forward | 2.138.644.480 B |
| Output eindig | ja |
| Route-ID's exact | ja |
| Routerweight max-absolute fout | 0 |
| Routerlogit max-absolute fout | 0 |

De meting blijft ruim onder de vooraf geregistreerde plafonds van 7,5 GiB
CUDA-allocatie en 32 GiB RSS. Dit is alleen een uitvoerbaarheids- en
integriteitscontrole; er is geen rank-, kwaliteit-, runtime-baseline- of
Eureka-resultaat uit afgeleid.
