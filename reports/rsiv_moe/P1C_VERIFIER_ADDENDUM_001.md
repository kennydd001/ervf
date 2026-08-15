# P1C verifier-addendum 001 — BF16-normalisatiesom

## Status

Dit addendum wijzigt uitsluitend een extra controle in de onafhankelijke
verifier. Het verandert geen capture, data, selectie, kandidaat, threshold,
gate, grid, metric of onderzoeksverdict.

Verifierpoging 001 gebruikte `atol=0,002` voor de som van acht geselecteerde
routergewichten nadat Qwen die gewichten van FP32 naar BF16 had teruggecast.
Vijf van de zes laag-splitcombinaties overschreden die zelfgekozen tolerantie;
de grootste gemeten afwijking was `0,00244140625`. De capturecontrole had al
bevestigd dat de native router-ID's en routergewichten exact overeenkwamen met
de officiële Qwen-forward.

Voor een niet-negatieve vector die vóór de cast exact tot één is genormaliseerd,
is de som van de absolute eerste-orde BF16-castfouten begrensd door de BF16
unit roundoff `u = eps/2 = 0,00390625`: elke componentfout is maximaal
`u * |w_i|` en `sum(|w_i|) = 1`. De verifier gebruikt daarom voortaan deze
analytische grens, met `rtol=0`.

De ongeldige eerste verifieruitvoer blijft onveranderd bewaard:

- JSON SHA-256: `79e193decedd954f43ab0a4c80cc13ccb2451af7c341ee5194860357f0d70eb1`.
- Markdown SHA-256: `044a0b8bdc15b41bdef3c6b8b9fd31493b9de67f7f215643a92fa072180cc50b`.

De definitieve verifier moet beide hashes controleren en deze protocolcorrectie
als gedeclareerde waarschuwing rapporteren.
