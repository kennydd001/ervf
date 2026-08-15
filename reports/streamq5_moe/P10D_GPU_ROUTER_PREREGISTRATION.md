# P10D preregistratie — GPU-residente router/top-k

Datum: 2026-08-12. Status bij vastlegging: output ongeopend.

## Hypothese

Een enkele GPU-kernel die de 128 routerlogits stabiel sorteert, de top-8
normaliseert, de gewichten naar BF16 afrondt en alleen 64 outputbytes naar de
host kopieert, verkleint de routebarriere ten opzichte van de huidige route:
512 logits naar de host, synchronisatie, NumPy-softmax en stabiele sortering.

## Vastgelegde invoer

- Qwen3-30B-A3B-base, lokaal model.
- De tien reeds verzegelde P0C-testcontexten, 128 tokens per context.
- Per laag de routerlogits van het laatste token vóór die laag: 48 × 10 = 480
  echte logitvectoren. Er wordt niet op uitkomsten geselecteerd.

## Procedure

Vergelijk per vector de ongewijzigde P6A-CPU-route met één CUDA-kernel. Beide
paden starten met logits die al op de GPU staan en eindigen nadat ids en
gewichten op pinned hostgeheugen beschikbaar zijn. Meet 20 herhalingen over de
480 vectoren na drie warmups. Bewaar host-wall-latenties; meet voor de kandidaat
ook CUDA-eventtijd.

## Gates

- top-8-idvolgorde exact gelijk voor alle 480 vectoren;
- alle acht BF16-afgeronde gewichten bitexact gelijk voor alle vectoren;
- somfout van de kandidaat maximaal 0,02;
- kandidaat host-wall p50 en p95 elk hoogstens 90% van de CPU-route.

Een correctheidsfout sluit de integratieclaim, ook als timing beter is. Een
timingfout sluit de snelheidsclaim. Dit experiment bewijst niet dat cachebeleid
of expertkopieën volledig GPU-resident zijn.
