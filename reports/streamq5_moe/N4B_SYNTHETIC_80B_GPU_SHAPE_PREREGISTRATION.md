# N4B — synthetische 80B GPU-vormpoort

Vastgelegd vóór de fysieke timing.

## Scope

N4A verifieerde de officiële Qwen3-Coder-Next-vormen en geheugenbudgetten. N4B
meet zonder checkpointpayload de dominante actieve expertvorm fysiek:

- 48 lagen;
- top-10 routed plus één shared expert per laag;
- hidden 2.048, intermediate 512;
- echte STREAMQ5 Q5-recordlayout: 2.027.520 bytes per expert;
- een afzonderlijk adres voor ieder van de 528 actieve records: 1.070.530.560
  residentiële bytes;
- gate/up, exacte SwiGLU en down;
- ERVF-breedtes 8, 16 en 32.

De payloadbytes en BF16-schalen zijn synthetisch; alleen vorm, databeweging,
reductiegraaf en launchgeometrie worden geclaimd. Breedte wordt op validation
gekozen, daarna eenmaal tegen width-16 getest met 120 AB/BA-paren.

## Exactheid en poorten

- Iedere breedte moet bit-voor-bit gelijk zijn aan width-16 over de volledige
  48-laagse expertplane.
- Geselecteerde expertplane test-p95 `<=50 ms`.
- De Q8 dense-shell wordt byte-lineair geprojecteerd vanaf de fysieke N1C-Q8-
  test-p95 en officiële N4A-devicebytes. Zowel de primaire als conservatieve
  `2×`-projectie moet `<=40 ms` zijn.
- Expert-p95 plus conservatieve dense-p95 moet `<=90 ms` zijn.
- N4A host-58-GiB-, 4K- en 32K-cachepoorten moeten waar zijn.

## Claimgrens

Geen 80B-modelkwaliteit, echte routersporen, DeltaNet-kerneltijd,
checkpointpayload, cachemissverdeling, prefill of end-to-end tokens/s. Een pass
rechtvaardigt alleen de volgende fysieke port-/checkpointstap.
