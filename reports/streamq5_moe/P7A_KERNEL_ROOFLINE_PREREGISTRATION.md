# STREAMQ5-MoE P7A — fysieke kernel-roofline

Datum: 2026-08-12. Status bij vastlegging: geen P7A-timingoutput geopend.

## Vraag

De P6B-kernels reconstrueren Q5- en Q8-gewichten al rechtstreeks uit codes en
BF16-schalen in registers. Er bestaat geen gedequantiseerde gewichtsscratch.
P7A bepaalt daarom of de resterende tijd primair uit (a) het fysieke
geheugenpad en de row-per-block-geometrie of (b) decode, BF16-rounding en MAC
bestaat.

## Vastgelegde workload

- De geverifieerde P6A Q8-bank en P1D Q5-bank blijven ongewijzigd.
- Q8: alle 241 device-resident projecties, totaal 1.248.931.840 fysieke bytes.
- Q5: experts 0–7 van iedere laag worden vooraf in 384 aaneengesloten GPU-slots
  gezet; de getimede regio bevat nul H2D-verkeer.
- Drie paden worden per bank gemeten:
  1. één coalesced raw scan over de resident bytes;
  2. dezelfde row-per-256-thread-block adressering als P6B, zonder FP-MAC;
  3. de ongewijzigde P6B GEMV-kernel.
- Een gelijk aantal no-op launches wordt apart gemeten. Zowel ruwe als
  no-op-gecorrigeerde CUDA-eventtijden blijven bewaard.
- Vijf warmups en zestig metingen per pad; vaste seed 270812.
- Alle resultaten en omgevingsgegevens gaan naar één JSON-artifact.

## Vooraf vastgelegde interpretatie

Voor iedere bank gebruiken we fysieke unieke payloadbytes als conservatieve
noemer. `pattern/raw < 0,60` is bewijs dat row-geometrie/reductie/launches de
dominante rem zijn. Bij `pattern/raw >= 0,60` en `gemv/pattern < 0,50` is
decode/rounding/MAC dominant. Alle overige gevallen heten gemengd.

Dit is een diagnose, geen snelheidsclaim. De daaropvolgende P7B-kernel moet
apart worden vastgelegd en tegen de bestaande kernel, uitvoer en end-to-end
P6B-runtime worden getest.

## Grenzen

De raw scan is een bovengrens voor deze allocatie, geen officiële
hardwarebandbreedte. Herhaalde schaalloads kunnen uit cache komen; daarom
claimen we geen DRAM-transactietelling. P7A test geen modelkwaliteit en mag op
zichzelf niet als Eureka of SOTA worden gerapporteerd.
