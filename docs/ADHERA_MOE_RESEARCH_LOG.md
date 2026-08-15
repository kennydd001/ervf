# ADHERA-MoE onderzoekslog

## 2026-08-11 — causale warmup vooraf geregistreerd

De vaste domeinbasis faalde door code- en instructionstaarten. ADHERA test één
causale policy: 64 warmuptokens, daarna één contextbasis voor 960 tokens. De
selectie gebruikt alleen reeds verwerkte routes. Twee volledige basewissels per
context worden meegerekend. P0A gebruikt geopende routes en kan alleen een
verse P0B rechtvaardigen.

## 2026-08-11 — 64-tokenpolicy negatief bevestigd

De causale adaptatie brengt code-p99 van 567 naar 405 MiB/token, maar de gate
blijft 288. Math verslechtert naar 73,758 MiB/token gemiddeld en instruction
naar 72,267; hun staartgates falen eveneens. Alleen general en multilingual
passeren. De onafhankelijke state-machine reproduceert alle 17 controles.
P0B en P1 blijven gesloten; er is geen warmuplengtesweep uitgevoerd.
