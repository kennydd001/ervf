# H3 Exact Atomic Expert Oracle — spread-layers en domeinen

## Oordeel

**Alle 12 vooraf vastgelegde laag×domeincellen slagen bij 25% retentie; ook
alle 12 CE-moonshotcellen slagen bij 10%.** De globale exacte bijdragescore is
dus niet alleen een laag-26-artefact en transfereert naar vroege/middenlagen,
WikiText, Nederlandse instructietekst en lokale Pythoncode.

Iedere proef sparseert nog maar één laag tegelijk. Dit resultaat autoriseert
de gelijktijdige 26-laags atom-oracle, maar is op zichzelf nog geen modelbrede
of deploybare Eureka.

## Primaire 25%-matrix

| Laag | Domein | Finale KL | Relatieve CE | Top-1 | Lokale routed rel. L2 |
|---:|---|---:|---:|---:|---:|
| 1 | WikiText-validatie | 0,002304 | +0,0098% | 98,83% | 0,1433 |
| 1 | WikiText-test | 0,002575 | +0,1772% | 98,44% | 0,1400 |
| 1 | lokale instructies | **0,007461** | +0,1481% | 96,88% | 0,1352 |
| 1 | lokale code | 0,002206 | −0,3001% | 98,05% | 0,0999 |
| 13 | WikiText-validatie | 0,001561 | −0,0000% | 98,83% | 0,1286 |
| 13 | WikiText-test | 0,001586 | +0,1278% | 97,66% | 0,1240 |
| 13 | lokale instructies | 0,002104 | −0,0697% | 98,05% | 0,1426 |
| 13 | lokale code | 0,000937 | +0,2251% | 98,05% | 0,1329 |
| 26 | WikiText-validatie | 0,001248 | +0,1908% | 98,44% | 0,1008 |
| 26 | WikiText-test | 0,001586 | +0,0105% | 98,05% | 0,1001 |
| 26 | lokale instructies | 0,001315 | +0,0636% | 96,88% | 0,0651 |
| 26 | lokale code | 0,000917 | −0,0041% | 98,44% | 0,0907 |

Het moeilijkste primaire punt is laag 1 op lokale instructies, maar ook dat
blijft binnen de vooraf vastgelegde KL `≤0,01`, CE `<2%` en top-1 `≥95%`.
Negatieve CE-puntschattingen worden niet als winst geclaimd.

## 10%-moonshot en onzekerheid

Alle CE-delta's bij 10% liggen tussen `−0,2733%` en `+0,3790%`, ruim onder de
3%-gate. De kwetsbaarste kwaliteitscel is laag 1 op lokale instructies: KL
`0,01450`, top-1 `92,58%`, CE `+0,1451%`. Dat slaagt de vooraf gekozen
CE-moonshot maar waarschuwt dat 10% geen uniforme veilige bedrijfspuntclaim is.

Elke cel heeft slechts twee 128-tokenblokken. De 25%-CE-bootstrapintervallen
blijven in alle cellen ruim onder de 2%-grens; zij zijn door hun grofheid niet
gatevormend. De lokale instruction/codecorpora zijn vaste transferchecks, geen
held-out confirmatie.

## Protocol en controles

Lagen 1, 13 en 26 zijn afzonderlijk ingegrepen. Iedere kandidaat gebruikte
dezelfde globale `|p_e a_j| ||d_j||₂`-score en vaste achtpuntscurve, kreeg een
routed delta op de officiële teacherstate en liep daarna door alle resterende
officiële decoderlagen en de volledige LM-head.

- officiële route-ID's en routergewichten zijn op alle drie lagen exact;
- de 100%-controls hebben in alle 12 cellen finale KL/CE `0`, top-1 `1` en
  lokale fout `0`;
- aparte BF16-GEMM-regressies blijven onder NRMSE `1,71×10⁻⁵` en maximum
  `0,0078125`; de exact-control gebruikt dezelfde full-route als deltabasis;
- alle 96 supports per laag/domein zijn lossless bit-packed en hun hashes zijn
  nagecontroleerd;
- het lokale codecorpus bevatte 69 gepinde Pythonbestanden buiten alle
  `craft_moe`-mappen; alle bron- en tokenhashes staan in de JSON;
- totale rekentijd `214,06 s`, piek toegewezen VRAM `3.203.610.624` bytes;
- ruwe JSON `46.499.214` bytes, SHA-256
  `2e49426c58d0132a130e67ffcedb0e616673124198beea7036187a0fd51df66b`.

## Stop/go

**Go:** preregistreer één gelijktijdige full-depth-proef waarin dezelfde
fractie in alle 26 MoE-lagen wordt toegepast en iedere policy haar eigen
natuurlijke route en exacte activaties volgt. **Stop:** nog geen predictor,
packed runtime, snelheid of Eureka-claim; onafhankelijke één-laagresultaten
tonen niet hoe fouten gezamenlijk accumuleren.
