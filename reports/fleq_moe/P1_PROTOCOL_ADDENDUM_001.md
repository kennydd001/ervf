# FLEQ-MoE P1 protocoladdendum 001 — GSQ optimizerstappen

## Status

Dit addendum corrigeert de uitvoering vóór enige andere expert of laag is
geëvalueerd. Selectie, data, codebooks, initialisatie, tien epochs, LR's,
temperatuur, logit-scale, seed, resourcegrenzen en gates veranderen niet.

De eerste complete laag-0/expert-46-run interpreteerde één expertmatrix als één
batch en voerde daardoor slechts één optimizerstap per epoch uit. De officiële
Qwen3-30B-recipe gebruikt batchgrootte 64; tien epochs betekenen dus meerdere
optimizerstappen per epoch. De annealing en cosinescheduler lopen over alle
optimizerstappen, niet alleen over de epochs.

Vanaf de definitieve run geldt daarom:

- batchgrootte 64 routed expertrows;
- per epoch een seeded permutatie van alle volledige aaneengesloten batches,
  conform upstream `get_random_batch_indices`;
- de eventuele laatste onvolledige batch wordt conform upstream niet gebruikt;
- temperatuur, logit scale en LR volgen de globale optimizerstep;
- held-out context 1 wordt alleen na de harde codeassignments geëvalueerd.

De ongeldige complete poging blijft ongewijzigd bewaard:

- artifact:
  `reports/runs/fleq_moe/p1/attempt_002_layer_00_expert_046.safetensors`,
  SHA-256 `b128db74158af21b0d7e0d3a2acf6b436c563bf1f2256d142681de18b045eecf`;
- rapport:
  `reports/fleq_moe/p1_experts/attempt_002_layer_00_expert_046.json`,
  SHA-256 `8d45f54a3450e272db16a2d339994a82559d958274b787687e07cb8c090b4e78`.

De daarin geziene negatieve metric (`-34,23%` GSQ-verbetering versus GPTQ) is
diagnostisch en mag niet als definitief P1-resultaat worden gebruikt.

