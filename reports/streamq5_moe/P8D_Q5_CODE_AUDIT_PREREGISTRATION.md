# P8D preregistratie — exact Q5-codehistogram en INT4-overflowgrens

Datum: 2026-08-12

## Hypothese

Een exacte vierbitskern plus verliesloze overflowstaart is alleen plausibel als
de volledige fysieke Q5-bank een voldoende dunne staart buiten `[-7, 7]` heeft.
De audit decodeert daarom alle 28.991.029.248 codes in alle 18.432 fysieke
matrixrecords; steekproeven zijn niet toegestaan.

## Grenzen

- `overflow_fraction < 10%`: kandidaat voor een fysieke P8E-layout- en
  kerneltest.
- `10% <= overflow_fraction <= 20%`: alleen door als de exact berekende
  bytebesparing minstens 15% is na index- en waarde-opslag.
- `overflow_fraction > 20%`: hypothese gefalsificeerd vóór kernelbouw.
- Het histogram moet exact optellen tot 28.991.029.248 codes, per projectie tot
  9.663.676.416 en over precies 48 lagen × 128 experts × 3 matrices.

Voor een conservatieve opslaggrens krijgt iedere overflow een 32-bit index plus
een 5-bit waarde; daarnaast telt de vierbitskern exact 4 bits per gewicht.

