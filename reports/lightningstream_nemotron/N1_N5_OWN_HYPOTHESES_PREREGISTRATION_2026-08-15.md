# N1–N5 — vijf eigen hypotheses, gericht op maximale tok/s

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Herkomst: niet uit de packs. Gekozen op de termen die de metingen van deze sessie
als grootst hebben aangewezen, en op één vraag die niemand gesteld heeft.

## Waarom deze vijf

De gemeten opbouw van een token: bij ctx 0 36,05 ms, bij 262K 54,28 ms. S14's
segmenten geven de MoE-stroom op 27,7 ms bij beide diepten; het verschil van
18,2 ms is attention. Van die 27,7 ms is `up` 6,6 en `down_masked` 8,4 —
en Y2-R1 mat dat de GEMV op 81,4 GB/s draait tegen een roofline van orde 250.

Vier van de vijf hypotheses hieronder vallen een term aan die groot én gemeten
is. De vijfde vraagt iets wat in deze hele lijn nog nooit is uitgerekend: **wat
is de fysieke ondergrens van één token?** Zonder dat getal is elke ambitie —
50, 100, 1000 tok/s — een gok.

---

## N1 — De graph-plafondmeting: hoeveel van een token is géén rekenwerk?

Een token vuurt ~600 kernels af. S9 mat een lege launch op 7,02 µs, S14 mat
`host_gap` op 4,7 ms, Y1 mat de per-laag-sync op 6,66 ms. Al die posten zijn
**geen rekenwerk** en verdwijnen in principe als de hele token als één
CUDA-graph draait.

Meting, niet bouw: leg de kernelvolgorde van één token vast met stream-capture —
mét bevroren routes, want capture staat geen synchronisatie toe — en vergelijk
graph-replay tegen eager uitvoering van exact dezelfde reeks. Dezelfde kernels,
dezelfde argumenten, dezelfde bytes; het enige verschil is wie ze aanstuurt.

Semantisch is een replay met bevroren routes na de eerste token onjuist. Als
**tijdmeting** is hij precies goed, en dat is het enige waarvoor hij gebruikt
wordt.

- **G-N1-1:** rapporteer `1 − t_graph / t_eager`. Dit is de **bovengrens** van
  élk ontwerp dat werk van de host naar de GPU verplaatst: megakernel,
  device-side routing, persistente kernels, graph-based decoding. Onder 10% zijn
  die allemaal begrensd op minder dan 10%.
- **G-N1-2:** de graph moet exact dezelfde kernelreeks bevatten als de eager arm
  (zelfde aantal launches), anders vergelijkt hij twee dingen.

---

## N2 — Is het `down`-pad zijn eigen scan en gather waard?

`down_masked_into` is drie kernels: `panel_scan`, `gather_down_sparse`, en de
masked GEMV plus reductie. S14 mat het geheel op 8,393 ms per token — de grootste
enkele post in de MoE-stroom. Niemand heeft de drie gesplitst.

In-lus replicatie (S12-protocol, gebracketeerde basislijnen): repliceer alleen
`panel_scan`, alleen `gather`, en alleen de masked GEMV.

- **G-N2-1:** als `panel_scan` + `gather` samen ≥ 30% van `down_masked` zijn, is
  fusie van de drie stadia in één kernel een bouw waard, en dat wordt als
  vervolgfase vastgelegd — niet in deze fase gebouwd.
- Marginalen zijn ondergrenzen en worden niet naar tok/s omgerekend.

---

## N3 — Exacte ReLU²-prefilter: 91% van het `up`-werk overslaan

Dit is de hypothese met de grootste theoretische opbrengst. S5 mat dat ~91% van
de ReLU²-uitgangen nul is. Die nullen worden **wel volledig berekend**: de
up-GEMV doet alle 1856 rijen. Wie vooraf kan bewijzen dat rij `j` nul wordt, mag
haar hele rij overslaan — 6,6 ms van de 27,7.

C1 probeerde dat met een bit-truncatie-core en faalde omdat de residual-norm te
groot was. Hier een andere decompositie: een **rang-`r` benadering** van de
gewichtsmatrix. Met `W ≈ U_r Σ_r V_rᵀ` is

```
ŷ = (W V_r)(V_rᵀ x)      |y_j − ŷ_j| ≤ ‖w_j − ŵ_j‖₂ · ‖x‖₂
```

en `y_j ≤ 0` is bewezen zodra `ŷ_j + ‖w_j − ŵ_j‖₂‖x‖₂ ≤ 0`. De rijnormen van de
residual zijn `r` getallen per rij en worden één keer vooraf berekend.

Dit is géén low-rank surrogaat als vervanging — dat staat in
`forbidden_hypotheses` en is terecht weerlegd. Het is een **sound bound voor
exact overslaan**: de uitvoer blijft bit-identiek, alleen wordt bewezen nulwerk
niet gedaan.

`r ∈ {8, 16, 32, 64}`, echte experts, echte activaties.

- **G-N3-S1:** nul valse certificaten.
- **G-N3-R1:** ≥ **30%** van de rijen gecertificeerd nul. Daaronder is de
  besparing kleiner dan de kosten van de projectie zelf (`r` extra inproducten
  van lengte 2688 per expert).
- **G-N3-C1:** de projectiekosten `r × 2688` MAC's moeten kleiner zijn dan het
  bespaarde werk `gecertificeerd × 2688` MAC's, dus `r < gecertificeerd × 1856`.

---

## N4 — Wat koopt een goedkopere KV-cache bij lange context?

Bij 262K is attention 18,2 ms van de 54,28 — de grootste enkele post. De KV-cache
is FP8; elke token leest 6 lagen × 2 × 262.144 × 256 B = 805 MB. Een FP4-KV zou
dat halveren.

Vóór er een kernel komt: meet de **bytes-tegen-tijd-helling** van de bestaande
attention-kernel door de contextlengte te variëren, zoals Y2-R1 dat voor de GEMV
deed. Als attention voor 90% byte-gebonden is, halveert FP4-KV hem bijna; is hij
even vast-gebonden als de GEMV (32%), dan levert het veel minder op.

`t = a·bytes + b` over contexten 32K/64K/128K/196K/262K.

- **G-N4-1:** rapporteer de byte-gebonden fractie en wat halvering van de
  KV-bytes bij 262K oplevert.
- **G-N4-2:** de fit moet `R² ≥ 0,98` halen, anders is de helling niet bruikbaar.
- Dit is een **kostenmeting**; FP4-KV is een semantiekwijziging en zou een eigen
  kwaliteitspoort vragen. Die wordt hier niet gehaald of geclaimd.

---

## N5 — De roofline van één token

Nog nooit uitgerekend in deze lijn, en zonder dit getal is elk doel een gok.
Tel de bytes die **elke** correcte forward moet lezen — de aangeraakte
shell-gewichten, de gerouteerde expert-records, de KV — en deel door de
**gemeten** haalbare leesbandbreedte van dit apparaat (niet de specificatie).

- **G-N5-1:** rapporteer `t_floor = bytes / B_measured` bij ctx 0 en 262K, en
  het bijbehorende plafond in tok/s.
- **G-N5-2:** de bandbreedte wordt gemeten met een streaming-leeskernel op
  device-geheugen, niet uit een datasheet overgenomen.
- Dit plafond geldt voor **elke** implementatie die de semantiek behoudt, ook
  voor implementaties die nog niet bestaan. Ligt 1000 tok/s eronder, dan is dat
  geen tegenvaller maar een natuurwet voor dit model op deze GPU.

---

## Wat hier niet gebeurt

Geen van de vijf bouwt een productiepad. N1, N2, N4 en N5 zijn kostenmetingen;
N3 is een numerieke oracle. `runtime.py` verandert niet. Niets wordt naar tok/s
omgerekend behalve N5, waar tok/s de eenheid van de uitkomst zelf is en
uitdrukkelijk als **bovengrens** wordt gepresenteerd.

## Artefacten

`scripts/lightningstream_nemotron/n1n2n4n5_ceilings.py` · `n1n2n4n5_ceilings.json` ·
`scripts/lightningstream_nemotron/n3_relu2_prefilter_oracle.py` ·
`n3_relu2_prefilter_oracle.json` ·
`scripts/lightningstream_nemotron/n1_n5_independent_verify.py` ·
`n1_n5_independent_verification.json` · rapport met claim boundary.

## Claim boundary van dit document

Geen meting, geen resultaat.
