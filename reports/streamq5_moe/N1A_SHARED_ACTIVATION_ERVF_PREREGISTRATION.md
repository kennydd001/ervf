# N1A preregistratie — shared-activation ERVF

Datum: 2026-08-12. Output bij vastlegging: ongeopend.

## Hypothese

De zestien ERVF-subwarps van één block consumeren dezelfde activatievector.
Door die FP32-vector éénmaal per block naar shared memory te laden, blijven de
weightdecode en exacte virtuele reductieboom ongewijzigd terwijl redundante
globale/cache-addressing afneemt.

## Kandidaten en referentie

- referentie: bestaande ERVF-width-16 uit P7B;
- kandidaat: exact dezelfde row-functies, voorafgegaan door coöperatieve staging
  van 2048/4096 FP32-elementen (Q8) of 2048/768 elementen (Q5);
- workload: dezelfde volledige Q8-projectieplane en 48 × acht fysieke Q5-
  expertsets als P7B/P8AB;
- seed: 120819;
- 5 warmups + 30 validationmetingen; alleen een component met validation-p50
  `<=0,97×` opent 10 warmups + 120 testmetingen.

## Gates

- alle Q8- en Q5-uitvoerelementen bitexact waar die component wordt getest;
- test-p50 `<=0,92×` en test-p95 `<=0,95×` van ERVF-16;
- een component die validation niet opent, wordt zonder test als negatief
  gesloten;
- alleen een Q8- én Q5-pass rechtvaardigt de brede shared-activationclaim;
  een afzonderlijke componentpass mag uitsluitend die component openen voor
  end-to-endintegratie.

Geen full-decodersnelheidsclaim volgt uit deze geïsoleerde planeproef.
