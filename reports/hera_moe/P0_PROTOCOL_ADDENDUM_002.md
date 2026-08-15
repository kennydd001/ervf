# HERA-MoE P0 protocoladdendum 002 — routecontrole zonder tweede softmax

**Vastgelegd:** 2026-08-11, vóór de definitieve herhaling.

Poging 002 onderschepte correct exact één officiële router-`topk`-call. De
opgeslagen general-routes reproduceerden daardoor alle 48 historische
E2GQ-countarrays exact (`L1=0`). De samengestelde `route_ids_exact`-boolean
bleef echter false omdat hij óók de onderschepte topk-values vergeleek met een
tweede softmaxevaluatie in de post-hook. Die redundante evaluatie hoeft bij
BF16-ties niet bit-exact te zijn en zegt niets over de reeds onderschepte IDs.

Poging 002 blijft volledig bewaard:

- result SHA-256
  `b7833d21454362ce90478626f3302868a1417cf3b0257ae0d66251d524dbebba`;
- report SHA-256
  `4458cdf57dab61a400c43f01034d9e88470b1a0deef95c8076b5e6f3f49d0df3`;
- alle routes/layerreports onder `*_attempt_002`.

De definitieve boolean vereist uitsluitend: exact één onderschepte officiële
topk-call en bit-exacte pre/post-routerlogits. Inputlock, routes, threshold,
unionregel, metrics en gates veranderen niet.
