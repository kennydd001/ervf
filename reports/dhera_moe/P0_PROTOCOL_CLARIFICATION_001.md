# DHERA-MoE P0 — protocolverduidelijking 001

**Vastgelegd:** 2026-08-11, vóór enige cachesimulatie of cachemetriek.

De preregistratie noemt een primary slot per laag en acht globale
LRU-victimslots, maar specificeert de state transition bij een victim-hit niet
expliciet. DHERA gebruikt de standaard victimcache-transitie:

- bij een victim-hit wisselt de geraadpleegde entry met de primary entry van
  dezelfde laag;
- de vervangen primary entry wordt de meest recente victim-entry;
- is de primary entry leeg, dan wordt alleen de geraadpleegde victim-entry naar
  primary verplaatst;
- een victim-hit veroorzaakt geen host-to-device-transfer.

Voor p95 en p99 gebruikt DHERA de discrete nearest-rank-definitie: sorteer de
32.768 tokenwaarden en neem element `ceil(p * n)`, met een één-gebaseerde rang.
Het gemiddelde wordt berekend uit de exacte gehele miss-counts vóór omzetting
naar MiB (`9 MiB` per miss).

Deze verduidelijking verandert geen budget, selectie, contextlengte, cachemaat,
gate of route. De validationroutes waren op dit moment nog niet tot
cache-events of verkeerspercentielen verwerkt.
