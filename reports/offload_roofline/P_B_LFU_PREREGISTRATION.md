# P-B preregistratie — online cumulatieve LFU-residency

Deze algorithmische lock is gemaakt vóór de nieuwe P-B-simulatie, maar op reeds
geopende HERA-routertraces. De bewijsstatus is daarom exploratief, niet nieuw
confirmatoir bewijs.

## Input

- De officiële, onafhankelijk geverifieerde HERA-routebank: 48 lagen × 5
  domeinen × 32.768 tokens × top-8.
- Een cachekey is `(layer, expert)`; er zijn exact 6.144 mogelijke keys en 384
  route-invocations per token.
- Alle in het HERA-resultaat geregistreerde routehashes moeten kloppen.

## Vastgelegde policy

- Cache start leeg voor elke stationaire domeinrun.
- Cumulatieve LFU-tellers blijven bij eviction bestaan; er is geen decay of
  domeinlabel. Dit is een deterministische TinyLFU-achtige policy.
- Router-lookahead maakt de benodigde keys vóór expertcompute bekend. Een key
  die vóór de token niet resident is telt als één cold call; prefetch verbergt
  mogelijk latency maar verandert de miss-telling niet.
- Na elke token bestaat de cache uit de `N` keys met hoogste cumulatieve
  frequentie. Ties: recentste token eerst, daarna hoogste globale key.
- Capaciteiten: 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 4700, 5120,
  5632 en 6144.
- Stationair worden alle 32.768 tokens gebruikt. P99 is de `higher`
  nearest-rank-kwantieldefinitie.

## Domeinwissels

Voor elk van de 20 gerichte bron→doel-paren wordt de eindstaat na de volledige
brontrace gekopieerd. Tellers, recency en residentie worden niet gereset. De
eerste 512 doeltokens worden gevolgd. Herstel is het eerste token `t` waarvoor
zowel `[t,t+64)` als `[t+64,t+128)` gemiddeld ≤3 cold calls/token halen. Alleen
`t ≤ 200` passeert; geen herstel in het observatievenster faalt eveneens.

## Gates bij N=4700

- gemiddelde cold calls/token ≤3,0 op elk domein;
- p99 ≤12 op elk domein;
- herstel ≤200 tokens voor alle 20 wissels;
- resident expertgewicht + geregistreerde INT4-trunk ≤5,75 GiB.

Alle gates moeten slagen. De volledige per-token missreeksen worden als NPZ
bewaard en de samenvatting als JSON. Dit experiment meet routertrace-residency,
niet PCIe-overlap, kernelruntime, tok/s of modelkwaliteit.
