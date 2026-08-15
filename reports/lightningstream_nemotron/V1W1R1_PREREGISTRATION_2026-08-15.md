# V1 / W1-R1 — de router-haalbaarheid en de 0,5 ms bij diepte

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**

## 1. V1 — kan een device-side router überhaupt?

Y1 mat dat de per-laag device→host-sync 6,66 ms per token waard is bij 262K.
W1 pakte daar 0,5 ms van met puur hostwerk; de rest zit in de sync zelf.

De sync bestaat om één reden: bij een **cache-miss** moet de host de
H2D-kopie uitgeven. Een device-side router moet dat zonder host doen, en er is
maar één mechanisme dat dat kan: de GEMV leest de missende expert **rechtstreeks
uit mapped pinned host-geheugen**, via dezelfde UVA-pointer die
`gather_down_sparse` al gebruikt, en schrijft hem onderweg in zijn slot.

Of dat kan hangt op één ongemeten getal: **wat kost een `up_proj`-GEMV die zijn
codes uit mapped host leest, tegen dezelfde GEMV vanaf device?** S5's microbench
mat brede `uchar4`-leesacties op mapped host op 25,05 GB/s, maar dat was een
kopieerkernel, niet deze GEMV met zijn eigen toegangspatroon en schaal-lookups.

Meetopzet: één echt `up_proj`-record, twee armen — codes en scales op device, en
dezelfde bytes via een UVA-pointer in de gepinde bank. Dezelfde kernel, dezelfde
argumenten, 200 aanroepen per sync zoals Y2-R1.

**Poorten:**

- **G-V1-C1 — exactheid.** De host-arm levert **bit-identieke** uitvoer aan de
  device-arm. Zo niet, dan leest de kernel iets anders en zijn de tijden zinloos.
- **G-V1-F1 — haalbaarheid.** Het ontwerp is alleen levensvatbaar als de extra
  kosten van missers kleiner zijn dan de sync die het wegneemt:

  ```
  miss_rate × 138 × (t_host − t_device)  <  6,66 ms
  ```

  met `miss_rate` = 0,1785 (K0's replay bij capacity 72). Dat komt neer op
  `t_host − t_device < 270 µs` per expert-call. Boven die grens is een
  device-side router met host-reads **duurder dan de sync die hij bespaart** en
  wordt hij niet gebouwd.

Dit is een haalbaarheidsmeting, geen bouw. Er wordt geen kernel gewijzigd.

## 2. W1-R1 — de 0,5 ms bij diepte, gepaard gemeten

W1 mat bij 262100 een winst van +0,511 ms tegen een drift tussen twee identieke
armen van 4,520 ms. De ingreep is daarmee niet weerlegd maar **onopgelost**: de
armen duurden minuten en de GPU draait op 86–87 °C.

De opzet verandert, de poort niet. In plaats van drie lange armen worden per
context **tripletten** gemeten op opeenvolgende stappen in dezelfde warme staat:

```
base · fast · base   ×  M
```

Per triplet: `effect = ½(b₁+b₂) − f` en `drift = |b₂ − b₁|`. Naburige stappen
delen hun thermische toestand, dus een trend valt weg uit elk triplet in plaats
van uit één gemiddelde over minuten.

**Poorten:**

- **G-W1R-C1 — semantiek.** Generatie bit-identiek (opnieuw, want het schema is
  anders).
- **G-W1R-R1 — resolutie.** De mediane triplet-drift moet **< 0,5 ms** zijn,
  anders is dit schema niet beter dan W1's en wordt er niets geconcludeerd.
- **G-W1R-E1 — teken.** Het effect heet pas aantoonbaar als de mediaan positief
  is **én** in ≥ 60% van de tripletten `fast < ½(b₁+b₂)`. Dat is een tekentoets
  en geen drempel op de grootte.
- **G-W1-P1 blijft ongewijzigd op ≥ 1,0 ms bij 262100** voor adoptie. Deze fase
  kan die poort halen of niet halen; zij wordt niet verlaagd omdat de meting
  scherper is geworden.

## 3. Wat hier niet gebeurt

Geen device-side router gebouwd, geen kernelwijziging, geen wijziging aan de
routebeslissing. V1 beslist alleen of het ontwerp de moeite van een
preregistratie waard is; W1-R1 beslist alleen of W1's effect bij diepte bestaat.

## 4. Artefacten

`scripts/lightningstream_nemotron/v1w1r1_router_feasibility.py` ·
`v1w1r1_router_feasibility.json` ·
`scripts/lightningstream_nemotron/v1w1r1_independent_verify.py` ·
`v1w1r1_independent_verification.json` · rapport met claim boundary.

## 5. Claim boundary van dit document

Geen meting, geen resultaat.
