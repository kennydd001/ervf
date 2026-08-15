# HERA-MoE P0 — multidomain-tieraudit

**Vooraf geregistreerd:** 2026-08-11  
**Status:** bronbestanden verworven en schema's gecontroleerd; geen HERA-routes,
hot-union of domeinmetrics geopend.

## Mechanische vraag

Blijft de statische `count >= 128` hot/cold-partitie klein genoeg voor de
8-GiB-VRAM-gate wanneer dezelfde vooraf vastgelegde selectie op vijf domeinen
wordt toegepast, of was de WikiText-coldtier voornamelijk domeinspecifiek?

De gesloten E2GQ-calibratie en GSQ-resultaten worden niet gewijzigd.

## Model, inputs en domeinen

- Qwen3-30B-A3B-Base revisie
  `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
- Per domein exact 32.768 tokens als 32 contexten van 1.024 tokens.
- `general`: bestaande WikiText-2 raw-v1 train, bestaande joinregel, eerste
  32.768 tokens.
- `code`: publieke CodeXGLUE line-completion train; exact 16.384 tokens uit
  Python `input`-rijen gevolgd door 16.384 uit Java `input`-rijen.
- `math`: GSM8K main/train in rijvolgorde als
  `Question: ...\nAnswer: ...`, eerste 32.768 tokens.
- `multilingual`: FLORES-200 dev via de gepinde publieke parquetmirror; exact
  4.096 tokens per taal in deze vaste volgorde:
  `arb_Arab, zho_Hans, hin_Deva, rus_Cyrl, spa_Latn, swh_Latn, jpn_Jpan,
  nld_Latn`.
- `instruction`: Dolly-15k in bestandsvolgorde als
  `Instruction/Context/Response`, eerste 32.768 tokens.
- Alle bronrevisies, lokale hashes en mislukte gated acquisities zijn
  append-only vastgelegd in `reports/hera_moe/`.

Een safetensors-inputlock met alle vijf exacte tokenarrays wordt geschreven
voordat de eerste HERA-route wordt berekend.

## Routermetingen

Per domein, laag en expert:

- top-8 invocation count;
- som van de officiële, binnen top-8 genormaliseerde routerweights;
- som van die routerweights in het kwadraat;
- gemiddelde en minimum van `raw p(selected expert) - raw p(ninth expert)`.

Per token wordt ook de raw boundary margin `p8-p9` vastgelegd. Route-IDs worden
opgeslagen zodat cold calls/token na het bevriezen van de union onafhankelijk
kunnen worden herberekend.

## Vooraf vastgelegde tierregel

1. Maak voor ieder domein de set van laag-expertparen met count `>=128`.
2. De HERA-hotset is de **union** van die vijf sets.
3. Alle overige paren zijn cold en blijven in de hypothese exact BF16.
4. Geen expert wordt op basis van validation/test, routermass of geheugen na de
   telling terug naar cold verplaatst.

## P0-gates

Gebruik voor de projectie uitsluitend:

- 4.718.592 routed parameters per expert;
- diagnostische hotrate `1,930708991156684 bpp`;
- alle 1.541.093.376 non-expertparameters op exact 4 bpp;
- alle cold expertparameters op 16 bpp.

P0 is `tier_positive` wanneer tegelijk:

1. hot pack + INT4 trunk maximaal **5,75 GiB** is;
2. exact cold BF16 plus gemapte backing weights maximaal **24 GiB** host-RAM is;
3. alle 48×128 paren en alle vijf domeinen zonder fallback zijn geteld;
4. router-IDs exact overeenkomen met de officiële blockoutput;
5. CUDA peak maximaal 7,5 GiB en proces-RSS maximaal 32 GiB is;
6. de onafhankelijke verifier alle artifacthashes, counts, union en
   cold-callpercentielen reproduceert.

Als de multidomainunion boven 5,75 GiB uitkomt, sluit de statische
count-threshold-HERA-hypothese vóór enige GPTQ-, kwaliteits- of runtimebouw.
P0 maakt geen Eureka- of snelheidsclaim.

