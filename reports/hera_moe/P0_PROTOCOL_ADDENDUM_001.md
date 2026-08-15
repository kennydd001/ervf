# HERA-MoE P0 protocoladdendum 001 — werkelijke router-topk

**Vastgelegd:** 2026-08-11, vóór de definitieve herhaling.

## Gevonden controlefout

Poging 001 berekende in een post-hook opnieuw `torch.topk` op de officiële
routerlogits. De logits waren bit-exact, maar veel BF16-logits zijn gelijk.
CUDA `topk` kiest bij ties niet gegarandeerd dezelfde indices tussen twee
aanroepen. Daardoor faalde `route_ids_exact` op alle lagen en weken zelfs de
general-counts af van de historische E2GQ-herberekening.

De volledige poging blijft bewaard:

- result SHA-256
  `5ed9cfc1c411b6e2d75dca51ba404edac3e6cee6c0250e96e294a0b1b7a74066`;
- report SHA-256
  `f502ba7f5bfac5e440a895e75eb657e66f352aab2b7d4c9416cf9011423fbd2f`;
- alle layerreports en route-artifacts onder `*_attempt_001`.

## Correctie

De definitieve run onderschept exact de ene `torch.topk(..., k=8)`-call die
het officiële Qwen MoE-block zelf uitvoert en bewaart die geretourneerde IDs en
waarden. Een tweede topk wordt niet langer als routebron gebruikt. De raw
negende probability wordt alleen gebruikt als numerieke marginwaarde; tied
indices beïnvloeden die waarde niet.

Domeinen, tokenarrays, threshold 128, unionregel, entropyrate en geheugengates
blijven ongewijzigd. De oude E2GQ-countovereenkomst wordt gerapporteerd als
reproduceerbaarheidsdiagnostiek, niet als inhoudelijke gate: de oude artifacten
bevatten eveneens een tweede topk en zijn daarom niet de autoriteit voor tied
expert-IDs.
