# CORETAIL-MoE P1 — definitief kerneloordeel

**Oordeel: P1 geslaagd; de exacte CORETAIL-microkernel en geselecteerde
tailtransfer halen de vooraf vastgelegde hardwaregates.**

De vaste NVRTC-kernel is zonder testset-tuning uitgevoerd op 72 gelockte
gate/up/down-matrices uit lagen 0, 24 en 47. Alle gevallen reconstrueren dezelfde
GPTQ-semantiek. De maximale CORETAIL-uitvoerfout tegenover FP32-accumulatie is
1,91×10⁻⁶ en de maximale onderlinge afwijking tegenover fixed uint2 is
1,19×10⁻⁶.

De aggregate CORETAIL-throughput meet 33,319 Gweight/s op p50 en 30,738
Gweight/s op de som van per-record-p95-latenties, tegenover de harde gate van
27,2 Gweight/s. De fixed-uint2-referentie meet 67,198 Gweight/s en de
gedequantiseerde BF16-referentie 61,826 Gweight/s.

Over 5.120 bevroren routed tokens is de conservatieve p95-omvang van de
row-aligned tail plus offsets 74.749.072 bytes. De pinned H2D-kopie meet 3,314 ms
p95, ruim onder de 33,3-ms-gate. De volledige tail wordt eenmaal bij modelstart
naar een conservatief maximaal 1,112-GiB hostcache gedecomprimeerd; daardoor is
de per-tokendecode nul en blijven uitsluitend de geselecteerde bytes dynamisch.

Alle vijf P1-gates slaagden. Een onafhankelijke audit herberekende alle hashes,
72 correctheidsgevallen, throughputrekenkunde, de 5.120-token tailverdeling en
de gatebeslissingen: 13/13 controles geslaagd.

De gemeten expert-plus-tailprojectie is 57,624 ms p50 en 62,262 ms p95. Dit is
geen end-to-end snelheid: trunk, router, activaties, scheduling en
autoregressieve orkestratie zijn niet in één runtime gemeten. De latere P2-
kwaliteitsfase sloot negatief op +42,943% relatieve test-CE. Daardoor is de
geïntegreerde wall-clocktest niet geautoriseerd; zie het masterverdict.
