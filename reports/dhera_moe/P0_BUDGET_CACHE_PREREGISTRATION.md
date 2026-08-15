# DHERA-MoE P0 — vaste budgetcache op nieuwe routes

**Vooraf geregistreerd:** 2026-08-11  
**Status:** geen DHERA-validationroutes of cache-uitkomsten geopend.

## Afzonderlijke hypothese

De statische HERA-union van 6.081 experts en 7,167 GiB resident weights blijft
gefalsificeerd. DHERA test één andere mechaniek: een vaste resident basis onder
een harde bytecap plus een kleine exacte BF16-victimcache.

## Vast geheugenbudget en basis

- Totale resident-weightcap: 5,75 GiB.
- Non-experttrunkprojectie: 0,717627525330 GiB op 4 bpp.
- Exacte BF16-cache: 56 experts × 9 MiB = 0,4921875 GiB.
- Over voor entropy-GPTQ: 4,540185 GiB.
- Met de reeds gemeten diagnostische rate `1,930708991156684 bpp` passen exact
  **4.280** expertparen in de basis.
- Selecteer die 4.280 uitsluitend uit HERA-trainingroutes op aflopende som van
  `router_weight²` over de vijf domeinen; ties: hogere totale count, dan lager
  laagnummer, dan lager expert-ID.
- De overige 1.864 experts blijven exact BF16 in host-RAM.

Deze selectie wordt gelockt vóór nieuwe validationroutes worden berekend.

## Eén vaste cachepolicy

De 56 BF16-slots worden niet gesweept:

- één primary slot per laag (48 slots);
- acht globale LRU-victimslots;
- cache reset bij iedere nieuwe context van 1.024 tokens;
- tokenvolgorde, laagvolgorde 0→47 en top-k-rankvolgorde 0→7;
- base-experts veroorzaken geen transfer;
- een exact-cold hit in primary of victim veroorzaakt geen transfer;
- bij miss wordt exact één volledig BF16-expertbestand van 9 MiB geteld;
- een vervangen primary entry gaat naar de globale victim-LRU;
- geen prefetchoracle, lookahead of domaingestuurde policy.

## Nieuwe out-of-sample inputs

Per domein opnieuw 32×1.024 tokens:

- general: WikiText validation, eerste 32.768 tokens;
- code: per taal de tweede 16.384-tokenwindow uit dezelfde gepinde Python- en
  Java-trainbestanden;
- math: GSM8K main/test, eerste 32.768 tokens;
- multilingual: FLORES-200 devtest, 4.096 tokens per dezelfde acht talen;
- instruction: de tweede 32.768-tokenwindow uit Dolly-15k.

Exacte tokenarrays en hashes worden gelockt vóór routeruitvoering.

## P0-gates

P0 is `cache_trace_positive` wanneer tegelijk:

1. resident weights maximaal 5,75 GiB zijn en cold BF16 maximaal 24 GiB;
2. voor elk van vijf domeinen gemiddelde H2D maximaal 64 MiB/token is;
3. voor elk domein p95 maximaal 144 MiB/token en p99 maximaal 288 MiB/token;
4. alle officiële routecalls exact worden onderschept;
5. alle cache-events, hashes, byteformules en percentielen onafhankelijk
   worden gereproduceerd.

Alleen een positieve trace mag P1 openen. P0 bewijst geen PCIe-latency,
overlap, GPTQ-kwaliteit, packgrootte of 10 tokens/s.

