# ERGV-C2 — generated physical-bank performance-autotuner

Datum: 2026-08-12  
Verdict: **PASS tegenover manual P7; parity met manual N1C**

## Kernresultaat

De ERGV-compiler heeft voor widths `4/8/16/32/64` Q8- en Q5-kernels
gegenereerd, alle varianten op de bestaande fysieke modelbanken bitexact
gevalideerd en daarna automatisch een graph per projectiefamilie geselecteerd.

De vooraf geregistreerde C2-poort is gehaald:

| Bank | Manual P7 p50 | Generated p50 | Ratio | Manual P7 p95 | Generated p95 | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Q8 | 9,4516 ms | 8,1272 ms | **0,859878** | 10,5303 ms | 9,0295 ms | **0,857475** |
| Q5 | 7,6686 ms | 7,1094 ms | **0,927079** | 7,8608 ms | 7,3021 ms | **0,928930** |

Dat is tegenover de uniforme handgeschreven P7-width-16-reference een
p50-speedup van `1,16296×` voor Q8 en `1,07866×` voor Q5. Beide families halen
dus afzonderlijk de vooraf vastgelegde `p50 <= 0,98`- en `p95 <= 1,00`-poort;
geen familie regresseert meer dan 2%.

## Automatisch gekozen graph

Validation bevroor vóór test:

```text
Q8: head=16, k=64, o=16, q=16, router=64, v=64
Q5: gate_up=8, down=4
```

Q8 reproduceert exact de eerdere handmatige N1C-keuze. Q5 kiest dezelfde
gate/up-width 8, maar kiest in deze validation-run width 4 voor down in plaats
van N1C-width 8.

## Exactheid

- Alle vijf generated widths waren bitexact tegen P6B voor Q8 én Q5.
- Generated width 16 reproduceerde manual P7 bitexact.
- De generated bevroren N1C-graph reproduceerde manual N1C bitexact.
- De uiteindelijke geselecteerde generated graph had:
  - Q8: 0 verschillen over 502.144 elementen;
  - Q5: 0 verschillen over 1.376.256 elementen;
  - maximale absolute fout 0 en alleen eindige output.

## Parity met N1C — geen nieuwe winst boven N1C

De verplichte AB/BA-paritymeting tegenover manual N1C gaf:

| Bank | Generated/manual N1C p50 | Generated/manual N1C p95 |
|---|---:|---:|
| Q8 | 0,998028 | 1,000354 |
| Q5 | 1,000075 | 0,999831 |

Deze verschillen zijn parity, geen betekenisvolle overwinning op N1C. De
C2-winst tegenover P7 komt doordat de compiler automatisch dezelfde bredere
reductiegeometrieën ontdekt die N1C handmatig testte. Width 4 voor Q5-down
verandert de volledige Q5-plane in de gesloten test niet aantoonbaar tegenover
de manual-N1C-graph.

## Meetprotocol en thermiek

- Compile-only poort: 36 kernels in 15,486 s; geen bank geladen of kernel
  uitgevoerd.
- Validation: 3 warmups en 15 geroteerde/omgekeerde metingen per width en
  projectiefamilie.
- Test: vier onafhankelijke vergelijkingsparen
  (`Q8/Q5 × P7/N1C`), ieder 10 warmups en 120 ronden.
- Even ronden: reference→candidate; oneven ronden: candidate→reference.
- Per paar dus exact 60 AB- en 60 BA-ronden.
- In totaal zijn 960 ruwe eventwaarden bewaard en onafhankelijk herberekend.
- GPU-temperatuur: 58 °C direct vóór de run en 65 °C direct erna.
- De geselecteerde graph is na het openen van test niet aangepast.

## Onafhankelijke verificatie

De afzonderlijke CPU-verifier herberekende hashes, compile-lock,
correctheidsvelden, validationselectie, iedere p50/p95 uit de raw eventarrays,
speedups, AB/BA-rekenwerk en alle gates.

Resultaat: **63/63 verificatiechecks geslaagd**.

## Eerlijk verdict

C2 bewijst nu een belangrijke systems-mijlpaal: de compiler kan uit een
mechanisch gecontroleerde exact-reduction-IR uitvoerbare kernels genereren,
de relevante schedule zoeken en het bestaande P7-resultaat op echte fysieke
Q8/Q5-banken automatisch verbeteren.

Het bewijst nadrukkelijk geen nieuwe performance boven N1C en geen end-to-end
tok/s-winst. Het is ook nog geen industriële of wereldwijde doorbraak: één
modelbank, één activatievector en één Blackwell-laptop-GPU zijn getest.
Tweede-modelvormen, tweede GPU-architectuur, echte runtime-integratie en
equivalente publieke kernelbaselines blijven vereist.

## Artefacten

- Preregistratie:
  `reports/streamq5_moe/ERGV_C2_PERFORMANCE_AUTOTUNER_PREREGISTRATION.md`
- Compile-lock:
  `reports/streamq5_moe/ergv_c2_compile.json`
- Autotuner:
  `scripts/streamq5_moe/ergv_c2_performance_autotuner.py`
- Machineleesbaar resultaat:
  `reports/streamq5_moe/ergv_c2_performance_autotuner.json`
- Onafhankelijke verifier:
  `scripts/streamq5_moe/ergv_c2_independent_verify.py`
- Verificatie-uitvoer:
  `reports/streamq5_moe/ergv_c2_independent_verification.json`
