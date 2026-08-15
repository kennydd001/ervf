# STREAMQ5-MoE P3B — dispatch-gap preregistration

Datum: 2026-08-12. Status bij vastlegging: geen P3B-timingoutput geopend.

## Hypothese

De vaste term van circa 18,67 ms in de P3A-regressie bevat een materiële
CPU/WDDM-dispatch-gap. CUDA Graph-replay van exact dezelfde fysieke all-hit
expertberekening verlaagt daarom de mediane host-wandtijd met ten minste 10%,
zonder de GPU-kerneltijd of uitvoer te veranderen.

Dit is bewust smaller dan de hypothese in `DATAPLANE_ONTLEDING_2026-08-12.md`.
P3A groepeert gate/up en down al over alle acht experts. De huidige keten heeft
vier launches per laag (gate+up, SwiGLU, down, reductie), dus 192 launches per
token en niet 768.

## Vastgelegde workload

- GPU, CuPy/CUDA-runtime en Q5-bank zijn dezelfde als P3A.
- 48 lagen, acht werkelijk uit de fysieke bank geladen experts per laag.
- De expertkeuze komt uit één reeds vastgelegde P3A-route:
  - smoke: general token 0;
  - validation: general token 512;
  - test: instruction token 768.
- De acht experts staan vooraf in de eerste acht slots van iedere laag; de
  getimede regio bevat dus nul H2D-cachemissen.
- De volledige P3A-cacheallocatie, INT8-trunkreservering en KV-reservering
  blijven gelijktijdig aanwezig. De trunk/KV-inhoud blijft een byteallocatie;
  dit experiment claimt geen volledig model.
- Iedere iteratie reset dezelfde 2048-dimensionale FP32-state via D2D-copy en
  voert daarna exact dezelfde vier kernels per laag uit.
- Eager en graph gebruiken dezelfde kernels, launchgeometrie, slots en bytes.
- 20 warmups; validation 120 gemeten iteraties; test 360 gemeten iteraties.
- Zowel host-wandtijd als CUDA-eventtijd wordt per iteratie bewaard.
- Een 192-launch no-op controlereeks meet de dispatchvloer apart.

## Primaire poorten

Validation opent test uitsluitend als alle poorten slagen:

1. alle fysieke bankrecords zijn uit de reeds geverifieerde P1D-bank geladen;
2. cache + trunk + KV zijn co-resident en minimaal 384 MiB blijft vrij;
3. eager en graph produceren bit-identieke eindstates;
4. alle uitvoer en timings zijn eindig;
5. `graph_host_p50 / eager_host_p50 <= 0.90`;
6. graph verandert de GPU-kerneltijd niet materieel:
   `graph_event_p50 / eager_event_p50` ligt in `[0.90, 1.05]`.

De dispatchhypothese is bevestigd als validation én een eenmalige test alle
poorten halen. Anders wordt hij gesloten. De ruwe tijden blijven ook bij falen
bewaard.

## Interpretatiegrens

Een pass bewijst alleen dat launchdispatch een materiële optimalisatiehefboom
voor de bestaande all-hit expertketen is. Een fail falsificeert niet dat een
nieuw handgeschreven fused/persistent kernel sneller kan zijn; hij
falsificeert wel dat het reeds gegroepeerde P3A-pad door graph-replay alleen
naar de voorgestelde 8–11 ms zakt. Transferoverlap, voorspelde routes,
attention, echte routergewichten, KV-mutatie, LM-head, sampling en end-to-end
tok/s worden hier niet getest.
