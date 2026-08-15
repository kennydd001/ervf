# NERVF-4 — gatherless down: weerlegd, en de gather blijkt zijn prijs waard

Datum: 2026-08-15
Namespace: `NERVF_NEMOTRON`
Verdict: **Gatherless is fors trager: MoE-blok +8,4 ms en token +7,4 ms bij 262100, ver boven de drift. G-NERVF-4P vroeg ≥6,55 ms winst en meet −5,99 tot −8,45. De hypothese is weerlegd in de richting van zijn eigen teken. Exactheid bleef intact.**
Terminal state: `nervf4_gatherless_refuted_gather_pays_for_itself`

## 1. Wat getest is

`down_masked_into` haalt vandaag de niet-nul kolommen van het panel-major record
met een **gecoalesceerde** warp-per-kolom-gather (S5, 25,05 GB/s geïsoleerd) uit
mapped host naar een device-mirror, waarna de masked GEMV vanaf device leest.
N2 mat die gather in de lus op **8,192 ms/token** met een drift van 0,040.

De gatherless-arm laat de gather weg en laat de masked GEMV het record
**rechtstreeks uit mapped host** lezen — zelfde bytes, zelfde panelwandeling,
zelfde optelvolgorde, alleen de tussenstap verdwijnt. ERVF staat in álle armen
aan, dus de enige variabele is de gather.

## 2. Uitkomst

| context | MoE met gather | gatherless | verschil | drift | token met gather | gatherless | verschil |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21,839 | 27,831 | **−5,992** | 2,027 | 34,543 | 40,241 | **−5,698** |
| 131.072 | 21,434 | 29,883 | **−8,448** | 1,419 | 42,736 | 50,291 | **−7,555** |
| 262.100 | 22,155 | 30,578 | **−8,423** | 0,938 | 50,915 | 58,295 | **−7,380** |

Alle verschillen overschrijden hun drift ruimschoots — conclusief, en negatief.

| poort | eis | gemeten | |
|---|---|---:|:--:|
| **G-NERVF-4C** exact | generatie bit-identiek, ook tegen het anker | identiek in alle armen | ✅ |
| **G-NERVF-4P** | ≥ 6,55 ms winst (80% van N2's 8,192) | **−5,99 tot −8,45** | ❌ |

## 3. Waarom, en wat dat waard is

V1 mat eerder dat dezelfde GEMV zijn codes uit mapped host leest op **6,7 GB/s**
tegen 85,9 vanaf device — een factor 12,8. De reden is het toegangspatroon: de
masked GEMV leest één byte per thread, strided; de gather leest warp-per-kolom
met brede `uchar4`-loads. Over PCIe is dat verschil dodelijk.

De gather kost dus 8,19 ms, maar hij **verdient dat terug**: zonder hem betaalt
de GEMV meer dan hij bespaart. Dat is een expliciete bevestiging van de
ontwerpkeuze die S5 destijds maakte, nu end-to-end gemeten in plaats van
beredeneerd.

**Wat dit sluit:** gatherless door de gather simpelweg weg te laten — de vorm die
in het pack en in Kimi's E2 als eerste kandidaat stond.

**Wat dit niet sluit:** een echte fusie waarin het **gecoalesceerde patroon van
de gather** de GEMV in wordt getrokken — warp-per-kolom brede loads die
onmiddellijk geconsumeerd worden, zonder mirror. Deze meting begrenst precies
waarom zo'n kernel het coalescing-patroon moet behouden: dat patroon, niet de
mirror, is waar de 25 GB/s vandaan komt.

## 4. Meetnotitie — een fout die eerst een verkeerd resultaat gaf

De eerste run van deze fase rapporteerde +3,0 ms winst. Dat was onjuist: een
stille `str.replace` in mijn runnergenerator had de arm-instelling niet
vervangen, waardoor `gatherless_down` nooit werd gezet en de armen in feite
NERVF-3 herhaalden (ERVF aan/uit). De tell was dat de basislijn-arm samenviel met
NERVF-3's **base** in plaats van met zijn **ervf**-arm.

Het artefact is niet overschreven maar apart gezet als
`nervf3r_ervf_replication_MISLABELED.json`; als ERVF-replicatie is het geldig
(+4,08 / +4,45 / +4,54 ms, consistent met NERVF-3's +3,70 / +3,10 / +4,51). De
runner heeft nu `assert`-regels op beide vlaggen, zodat een stille mismatch niet
meer kan.

## 5. Claim boundary

In-lus A/B op deze GPU bij capacity 72, drie armen, ERVF aan in elke arm zodat de
enige variabele de gather is. Het MoE-getal is een component die ook router,
shared expert en up bevat. Exactheid is een harde poort en is gehaald: identieke
generatie tussen de armen én tegen het bevroren V35-anker. Geen tok/s-claim.

## 6. Artefacten

`scripts/nervf_nemotron/nervf4_gatherless_ab.py` · `nervf4_gatherless_ab.json` ·
`nervf3r_ervf_replication_MISLABELED.json` (apart gezet) ·
`src/moe_lab/lightningstream_nemotron/fused_nvfp4.py` (`gatherless_down`, default uit)
