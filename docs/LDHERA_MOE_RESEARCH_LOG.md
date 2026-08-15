# LDHERA-MoE onderzoekslog

## 2026-08-11 — laaglokale allocatie vooraf geregistreerd

LDHERA behoudt de domeinbasis en exact 56 cacheslots, maar leert de verdeling
over lagen uitsluitend uit HERA-trainingroutes. Exacte DP minimaliseert
training-LRU-misses; validation blijft onaangeraakt door de allocator. P0A is
door reeds geopende routes exploratief en kan alleen verse P0B openen.

## 2026-08-11 — training-optimale laagcache negatief bevestigd

De exacte DP wijst per domein 0–3 slots per laag toe en gebruikt exact 56
slots. General, math en multilingual passeren, maar code-p99 is 657 MiB/token
en instruction p95/p99 is 243/711. Een onafhankelijke implementatie
reproduceert trainingsmisscurves, DP, validationevents en gates: 20/20.
Optimalisatie van gemiddelde training-misses beheerst de validationstaarten
niet; P0B en P1 blijven gesloten.
