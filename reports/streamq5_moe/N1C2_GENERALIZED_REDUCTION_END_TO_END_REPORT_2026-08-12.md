# N1C2 — Generalized Reduction Graph end-to-end-resultaat

Datum: 2026-08-12  
Parent: N003/N1C  
Verdict: **formeel NEGATIEF — één van drie timingpoorten nipt gemist**

## Uitkomst

De bevroren N1C-grafiek is volledig in dezelfde P13 EVT-PM-runtime geïntegreerd:

- Q8: `head=16, k=64, o=16, q=16, router=64, v=64`;
- Q5: `gate_up=8, down=8`.

Alle semantische controles slagen over 128 gepaarde tokens. De kandidaat is op
alle drie gerapporteerde timingstatistieken sneller, maar de vooraf vastgelegde
mean-grens wordt met 0,3437 procentpunt gemist. Daardoor is `overall_pass=false`.

| Metriek | P13 ERVF-16 | N1C-grafiek | Ratio | Vereist | Verdict |
|---|---:|---:|---:|---:|---|
| mean | 60,6614 ms | 59,6567 ms | **0,983437** | ≤ 0,98 | FAIL |
| p50 | 59,6852 ms | 58,1384 ms | **0,974085** | ≤ 0,98 | PASS |
| p95 | 84,2145 ms | 80,9575 ms | **0,961325** | ≤ 1,00 | PASS |

Dit komt overeen met een waargenomen winst van circa 1,66% op mean, 2,59% op
p50 en 3,87% op p95. Deze winst mag niet als bevestigde primaire pass worden
gepresenteerd, omdat alle vooraf geregistreerde poorten conjunctief waren.

## Sterke bitexactheid

Alle onderstaande waarden zijn `true` over 128/128 paren:

- next-token-prediction exact;
- missenaantal exact;
- volledige KV-digest exact;
- dynamische LRU-cachetoestand exact;
- SHA256 van alle 151.936 FP32-logits exact;
- SHA256 van de uiteindelijke 2.048-element state exact.

De controle gaat daardoor verder dan alleen dezelfde gegenereerde tokenreeks:
ook verborgen toestand en volledige logitvector zijn per stap identiek.

## Protocolcontrole

- 128 tokenparen aanwezig;
- 16 warmupparen uitgesloten, 112 paren getimed;
- exact 64 even baseline-eerst-paren;
- exact 64 oneven kandidaat-eerst-paren;
- beide varianten draaiden vanuit dezelfde token-, positie-, LRU- en
  teller-snapshot;
- uitsluitend een verse baseline-uitvoering bepaalde de volgende toestand.

De fysieke runtime gebruikte de lokale NVIDIA RTX PRO 2000 Blackwell Laptop GPU,
de bestaande 4.977.623.040-byte expertcache, 1.248.931.840-byte device-trunk en
402.653.184-byte KV-cache.

## Secundaire gepaarde diagnose

Dit deel was geen primaire preregistratiepoort en verandert het verdict niet.
Na warmup was de kandidaat sneller op 64 van 112 paren en trager op 48. Het
gemiddelde gepaarde verschil was `-1,0048 ms`; de mediaan `-0,5152 ms`.
Een deterministische 100.000-sample paired bootstrap gaf voor het gemiddelde
verschil een 95%-interval van ongeveer `[-2,0455, +0,0115] ms`. Het interval
raakt nul. Dat past bij “kleine waarschijnlijke winst, maar onvoldoende scherp
voor de streng vooraf gekozen mean-poort”, niet bij een robuuste grote sprong.

De mean-ratio was `0,98108` wanneer baseline eerst draaide en `0,98580` wanneer
de kandidaat eerst draaide. ABBA heeft dus een meetvolgorde-effect verkleind,
maar niet volledig geëlimineerd.

## Interpretatie

N1C2 bewijst wél dat de nieuwe 64-lane Q8-reductie samen met Q5-width-8 zonder
numerieke verschuiving in de volledige P13-decoder kan functioneren. De
geïsoleerde N1C-speedups van 18,4% en 7,0% vertalen zich echter maar gedeeltelijk
naar totale tokenlatency. Dat is consistent met Amdahl: aandacht, routing,
cachemissen, H2D/copy en overige laagglue blijven ongewijzigd.

Methodologisch wordt de hypothese gesloten als **verified negative op de
strenge end-to-end-poort**, met een afzonderlijk vastgelegd positief
exactheidsresultaat en kleine indicatieve snelheidswinst. Een herhaling met
versoepelde grens of geselecteerde uitsluiting van uitschieters is niet
toegestaan. Alleen een vooraf gemotiveerde nieuwe hypothese—bijvoorbeeld de
N1C-grafiek gecombineerd met een onafhankelijk bewezen orthogonale
optimalisatie—kan een nieuwe test rechtvaardigen.

## Artefacten en provenance

- Preregistratie:
  `reports/streamq5_moe/N1C2_GENERALIZED_REDUCTION_END_TO_END_PREREGISTRATION.md`
- Ruwe uitvoer inclusief 128 paren:
  `reports/streamq5_moe/n1c2_generalized_reduction_end_to_end.json`
- Reproduceerscript:
  `scripts/streamq5_moe/run_n1c2_generalized_reduction_end_to_end.py`
- Preregistratie-SHA256:
  `ac5f893d7588f1cb07ea0d2eda958484affa80f62b6ea1968ad59201096047d6`
- Script-SHA256:
  `05100124156d053b50f8b42ada5c261042c5917c3609198e4397240c5c7a1aee`
- N1C-resultaat-SHA256:
  `10312adcd295be15e502651a14988c0900aa0667ea19f1aaca686d842bc83a02`
- P7-testinput-SHA256:
  `24e9deab670ce003374d459afefd11f93d766837af6c436711c6d686b7de3e59`

Alle vier hashes in het uitvoerbestand zijn na afloop opnieuw tegen de fysieke
bestanden gecontroleerd en kwamen exact overeen.
