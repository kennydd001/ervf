# STREAMQ5-MoE P7C — strikte ERVF end-to-end-replicatie

Datum: 2026-08-12. Status bij vastlegging: geen P7C-output geopend.

## Hypothese

De in P7B geselecteerde 16-lane ERVF-kernels vervangen uitsluitend de Q8- en
Q5-GEMV-launchgeometrie. Zij emuleren exact de 256 virtuele threadaccumulatoren
en dezelfde FP32-reductieboom als P6B. Daarom blijft de volledige decoderoutput
gelijk, terwijl test- en rolloutlatentie minstens 20% dalen.

## Vergrendelde wijziging

- Q8: 16 rijen per 256-thread block in plaats van één rij per block.
- Q5 gate/up en down: 16 rijen per block.
- Codes, schalen, BF16-rounding, MAC-volgorde per virtuele thread, reductieboom,
  router, cachebeleid, transfers, attention, KV, LM-head en evaluator blijven
  semantisch ongewijzigd.
- De strikte P6B-stopwatch blijft vóór de fysieke host-embeddinglookup starten.

## Fasen

De bestaande gesloten volgorde blijft smoke → validation → test → 512-token
greedy rollout. De bestaande P6B-inputs, kwaliteitsreferenties, cache- en
residentiepoorten worden hergebruikt. P7B-resultaat, runner en input/evaluator
worden per SHA-256 vergrendeld.

## Primaire poorten

P7C slaagt alleen als:

1. alle bestaande P6B-gates opnieuw slagen;
2. validation en test exact dezelfde CE-waarden, voorspellingen, missreeksen en
   KV-digests opleveren als P6B;
3. rollout exact dezelfde prompt-, feedback- en gegenereerde tokenreeksen geeft;
4. `P7C test mean / P6B test mean <= 0,80` en dezelfde verhouding voor p95;
5. `P7C rollout mean / P6B rollout mean <= 0,80` en dezelfde verhouding voor p95;
6. dezelfde fysieke bank-, cache- en KV-bytes resident zijn en minstens 192 MiB
   vrije scratch resteert.

## Claimgrens

Een pass bewijst een bit-exacte, substantiële lokale versnelling van de reeds
werkende custom Qwen3-30B-A3B-decoder op deze RTX PRO 2000 8GB-machine. Hij
bewijst geen wereldrecord, geen winst tegen alle runtimes/hardware en geen
algemene winst voor andere modellen. ERVF is een werknaam en geen
prior-artclaim.
