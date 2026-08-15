# N1C2 — Generalized Reduction Graph end-to-end preregistratie

Datum: 2026-08-12. Registry-parent: N003. Status bij vastlegging: het
N1C2-resultaatbestand bestaat niet en de fysieke run is niet geopend.

## Hypothese

De op N1C-validation gekozen en vóór deze integratie bevroren exacte grafiek

- Q8: `head=16, k=64, o=16, q=16, router=64, v=64`;
- Q5: `gate_up=8, down=8`

verlaagt niet alleen de resident projection-plane-tijd, maar ook de volledige
P13 EVT-PM-decodetijd zonder één semantisch bit te veranderen.

## Vaste runtime en workload

- Eén fysiek geladen P13 EVT-PM-runtime met de bestaande mapped/pinned
  expertbank en één geactiveerde `general`-cache.
- De input is de verzegelde P7-testrollout: eerst het vaste prompt van 7 tokens,
  daarna 128 gepaarde fysieke decodes.
- Baseline is exact P13: Q8 ERVF-16 en Q5 ERVF-16.
- Kandidaat wijzigt uitsluitend de projectiekernels naar de hierboven bevroren
  N1C-grafiek. Attention, routing, cachekopieën, residentie en alle andere
  runtimecode zijn identiek.
- De eerste 16 tokenparen zijn warmup en tellen niet mee voor timing.

## Same-runtime pairing

Voor ieder token worden token, positie, router-LRU en alle tellers gesnapshot.
Baseline en kandidaat starten elk uit diezelfde snapshot. De uitvoervolgorde is
ABBA: even tokens baseline→kandidaat en oneven tokens kandidaat→baseline. Een
derde canonieke baseline-uitvoering bepaalt uitsluitend de toestand en tokeninput
voor de volgende stap; meetvolgorde kan de rollout dus niet veranderen.

## Exactheidspoort

Voor alle 128 paren moeten gelijk zijn:

1. next-token-prediction;
2. fysiek aantal cachemisses;
3. volledige KV-digest tot en met de huidige positie;
4. dynamische LRU-cachetoestand;
5. SHA256 van alle 151.936 FP32-logits;
6. SHA256 van de uiteindelijke 2.048-element state.

Het N1C-bronresultaat moet vooraf `overall_pass=true` bevatten en exact de
bevroren configuratie vastleggen. Anders wordt de run geweigerd.

## Timingpoorten

Over de 112 paren na warmup moet gelden:

- `candidate_mean / baseline_mean <= 0,98`;
- `candidate_p50 / baseline_p50 <= 0,98`;
- `candidate_p95 / baseline_p95 <= 1,00`.

Alle exactheids- en timingpoorten moeten samen slagen voor `overall_pass`.
De test wordt één keer geopend; een mislukking wordt niet post-hoc hertuned.

## Claimgrens

Dit is een same-runtime, gepaarde 128-token P13-integratietest op één GPU en één
domein. Een pass bewijst een lokale end-to-end-generalisatie van de bevroren
N1C-grafiek. Hij bewijst geen 10K-endurance, tweede model/GPU, publicatiepoort,
energiewinst, wereldwijde nieuwheid of SOTA.
