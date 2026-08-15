# DeepSeek-V2-Lite baselineplan

## Doel

De eerste baseline moet drie vragen los van elkaar beantwoorden:

1. Kan deze laptop de officiële teacher praktisch laden of laaggewijs uitlezen?
2. Kunnen we routerkeuzes en MoE-uitgangen correct en reproduceerbaar loggen?
3. Is expertgedrag op echte activaties beter comprimeerbaar dan eenvoudige
   baselines zoals top-1 en een route-onafhankelijke voorspelling?

Een lage reconstructiefout op teacher-forced activaties is niet voldoende.
Latere experimenten moeten student-rollouts gebruiken, omdat kleine fouten de
hidden states en daarmee de discrete routerkeuzes kunnen veranderen.

## Baselines

- Exacte top-6 teacher: referentie, fout nul.
- Zero predictor: normalisatiecontrole.
- Top-1 routed expert: eenvoudige sparsificatiebaseline.
- Shared mean expert: route-onafhankelijke baseline.
- Activation-aware shared basis met ranks 4, 8, 16, 32, 64.
- Route-conditioned aggregate surrogate.
- Hybride: exacte top-1 plus surrogate voor de resterende experts.
- MoBE-, MergeMoE/Sub-MoE- en quantisatiebaselines bij hetzelfde bytebudget;
  zie `docs/PRIOR_ART.md`.

## Eerste echte proef

Begin op lagen 1, 13 en 26 met 5.000 calibrationtokens. Bewaar niet standaard
alle individuele expertoutputs. Bewaar hidden state, top-k ids/gewichten,
aggregate teacheroutput en een beperkte steekproef individuele outputs. Dit
houdt opslag beheersbaar en maakt de eerste vergelijking snel.

Splits prompts vóór tracegeneratie in train/validation/test. Rapporteer zowel
teacher-forced reconstructie als vrije rollouts. Zeldzame experts krijgen een
afzonderlijk coverage-rapport.

## Stopcriteria

- Geen duidelijke winst boven top-1/shared-mean bij gelijk bytebudget: stop de
  huidige surrogatearchitectuur.
- Minder dan 4× bij de afgesproken foutgrens: niet opschalen naar alle lagen.
- Sterke routerdrift of generatie-instorting: eerst rollouttraining oplossen.
- Geen 8× reductie op V2-Lite: V4 Flash niet downloaden voor dit traject.

## Rapportagecontract

Ieder resultaat bevat minimaal:

- datum/tijd en git-commit;
- model-id en exacte Hugging Face-revisie;
- hardware, driver, Python/PyTorch/Transformers;
- configuratie en random seed;
- train/valid/test-databron en tokenaantallen;
- gemiddelde én p50/p95/p99 latency;
- kwaliteit tegenover bytes per outputtoken;
- bekende beperkingen en eerstvolgende falsificatietest.
