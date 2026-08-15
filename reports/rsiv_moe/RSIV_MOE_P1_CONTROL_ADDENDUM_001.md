# RSIV-MoE P1 controladdendum 001

Vastgelegd op `2026-08-11T07:42:53.7659364Z`, vóór validationpoging 002 en
terwijl de testslices nog niet waren geopend.

## Aanleiding

Validationpoging 001 kreeg `invalid_controls_failed`, hoewel alle vooraf
geregistreerde route-, count-, bound- en full-rank FP32-imagecontroles slaagden.
De implementatie had een extra, niet-geregistreerde control toegevoegd:
opgeslagen BF16-`z` moest bit-exact zijn wanneer dezelfde rijen later per expert
in kleinere GEMM-batches werden herberekend.

Die eis is geen geldige capture-integriteitscontrol. BF16-matrixvermenigvuldiging
mag bij een andere batchvorm een andere kernel en accumulatievolgorde gebruiken.
De originele capture groepeerde alle 2.048 tokens; de validationcontrol alleen
de 1.024 train-tokens. Bitverschil bewijst dan niet dat een van beide paden een
andere algebraïsche operator gebruikt.

## Enige wijziging

`stored_bf16_z_batch_shape_bit_exact` blijft zichtbaar als diagnostiek maar
maakt `all_required_controls_pass` niet langer onwaar. De vooraf vastgelegde
full-rank operatorimagecontrol blijft ongewijzigd voor `x`, `g`, `u`, `z` en
`y`, met per laag en globaal:

```text
relative_l2 <= 2e-5
maximum_absolute_error <= 2e-4
```

Ook route-ID-bitexactheid, routergewichttolerantie `1e-6`, countidentiteit,
rankgrenzen, de expert-count-cancellationbound en het ongewijzigde officiële
fallbackpad blijven verplicht.

## Geen analytische vrijheid toegevoegd

- Geen data, split, rank, threshold, gate of selectieregel verandert.
- De raw capture blijft SHA-256
  `8c532434d50df8bd65691a72351a21084bdf00d5b68406f27806379ad9a67906`.
- De ongeldige v1-lock blijft bewaard als
  `reports/rsiv_moe/p1_validation_selection.json`, SHA-256
  `1ff70f0657b873b9cebbed19dc93b6c95c4a0ad739bf1da129ab3dd0713e3eb1`.
- Validationpoging 002 schrijft naar een nieuw bestand.
- De testslices blijven gesloten tot een geldige v2-lock bestaat.

