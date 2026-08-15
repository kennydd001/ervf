# P13B preregistratie — expliciete-add Exact Virtual-Tile Attention

Datum: 2026-08-12. P13A is geopend en afgekeurd wegens 398 scoreverschillen op
4K en twee BF16-outputverschillen. Geen P13B-output is geopend.

## Reparatie

De virtuele reductie had algebraïsch dezelfde boom, maar gewone `+`-expressies
lieten compilerreassociatie toe. P13B gebruikt voor iedere interne boomstap en
warp-shufflecombinatie expliciet CUDA `__fadd_rn`. Alle overige P13A-code,
inputs, contexten en snelheidspoorten blijven ongewijzigd.

## Gate

Nul score- en outputbitverschillen op alle 48 lagen bij context 128, 512, 1024
en 4096, plus dezelfde P13A-snelheidsgrenzen. Anders wordt de exacte variant
gesloten, ongeacht de grote bijna-exacte winst.

