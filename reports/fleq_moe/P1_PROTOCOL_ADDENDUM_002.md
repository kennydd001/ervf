# FLEQ-MoE P1 protocoladdendum 002 — geldige ternary hard baseline

## Status

Dit addendum betreft uitsluitend de vooraf toegestane ternarydiagnostiek. De
primaire 2-bitresultaten en hun reeds gefaalde gate veranderen niet. Geen P2-
of testdata wordt geopend.

De eerste ternarydiagnostiek gebruikte de ruwe output van upstream
`rtn_quantize(..., trits=True)` als RTN-baseline. De assembler detecteerde dat
enkele Q/scalepaar-ratio's buiten `{−1,0,+1}` lagen. De raw initializer is
bedoeld als input voor `GumbelQuantizerTernary`; diens `get_hard_weights()` is
de werkelijke discrete codeprojectie. De definitieve baseline is daarom de
officiële harde initialisatie vóór enige optimizerstap. GSQ zelf blijft vanaf
dezelfde raw initializer trainen.

Daarnaast rapporteert de definitieve herhaling afzonderlijk:

- byte-exacte harde weights;
- exact gelijke lossreeks;
- de gezamenlijke determinismecontrole.

Poging 003 blijft volledig bewaard onder:

- `reports/runs/fleq_moe/p1_ternary_attempt_003/`;
- `reports/fleq_moe/p1_ternary_experts_attempt_003/`;
- aggregate JSON SHA-256
  `19f8d733da6acb2e66f673b3643dfc619fbc6d38d42dfae7fe7decd6a57edbb5`;
- aggregate Markdown SHA-256
  `0d386eee7e14455697ad90a9b9e521f4ac5b0bbf3d0b4e7e6579c33b95b6f722`.

De numeric values uit poging 003 zijn diagnostisch ongeldig en mogen niet als
definitieve ternaryvergelijking worden gebruikt.

