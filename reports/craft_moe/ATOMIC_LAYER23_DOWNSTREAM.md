# H3 Exact Atomic Expert Oracle — laag 23 met exacte downstreamstaart

## Oordeel

**De vooraf geregistreerde downstreamgate slaagt volledig.** Met de reeds op
laag 26 gekozen globale bijdragescore behoudt laag 23 slechts 25% van de routed
atomen en blijft de finale relatieve CE-delta `+0,0539%` op validatie en
`+0,2942%` op test. Gemiddelde finale KL is `0,000906/0,000998` en top-1
`99,22%/99,22%`. Ook de 10%-moonshot repliceert na de officiële lagen 24–26.

Dit is sterker bewijs dan een laat-laags lokale screen: de sparsificatie
overleeft drie routerende downstreamlagen. Het blijft een exacte-activatie-
oracle zonder vroege supportindex of sparse runtime en is nog geen modelbrede
of deploybare Eureka.

## Vaste uitvoering

De officiële decoder voerde lagen 0–22 uit op twee 128-tokenblokken per split.
Op laag 23 zijn de natuurlijke top-6-experts exact ontbonden in 8.448
`p_e*a_j*d_j`-atomen. Alleen de vooraf gekozen globale
`|p_e a_j| ||d_j||₂`-rangschikking en de vaste fractiecurve zijn gebruikt.
Shared experts bleven exact. Iedere kandidaat is als routed delta op de
officiële laag-23-teacherstate gezet en daarna door ongewijzigde lagen 24–26,
final norm en de volledige LM-head gevoerd.

## Volledige curve na lagen 24–26

| Retentie | Val KL | Val rel. CE | Val top-1 | Test KL | Test rel. CE | Test top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 100% | 0 | 0,000% | 100,00% | 0 | 0,000% | 100,00% |
| 75% | 0,000468 | −0,0630% | 99,22% | 0,000517 | +0,1135% | 99,22% |
| 50% | 0,000596 | +0,0086% | 100,00% | 0,000679 | +0,0458% | 98,44% |
| 35% | 0,000772 | +0,0375% | 98,83% | 0,000793 | +0,1014% | 98,83% |
| **25%** | **0,000906** | **+0,0539%** | **99,22%** | **0,000998** | **+0,2942%** | **99,22%** |
| 15% | 0,001433 | +0,1031% | 98,83% | 0,001270 | +0,1523% | 98,83% |
| **10%** | **0,001927** | **+0,2140%** | **99,22%** | **0,001800** | **+0,3250%** | **98,83%** |
| 5% | 0,003016 | −0,0048% | 99,61% | 0,003316 | +0,2659% | 98,83% |

Bij 25% is het 95%-blokbootstrapinterval voor relatieve CE
`−0,0320%–+0,1443%` op validatie en `+0,2894%–+0,2983%` op test. Bij 10% is
dit `+0,1266%–+0,3062%` en `+0,1696%–+0,4597%`. De twee blokken per split
maken dit exploratief; de intervallen waren vooraf niet gatevormend.

Lokale routed relatieve L2 is gemiddeld `0,0894/0,0918` bij 25% en
`0,1908/0,1898` bij 10%. Dat die lokale fout groter is dan de finale KL-schade
onderstreept dat routed-output-NRMSE geen primaire kwaliteitsmaat mag zijn.

## Propagatie door routers

Bij 25% blijft gemiddelde top-6-routeroverlap na lagen 24, 25 en 26 op
validatie respectievelijk `98,96%`, `99,15%`, `99,54%`; op test `99,41%`,
`99,15%`, `99,35%`. Bij 10% is dit nog `97,66–98,76%` op validatie en
`97,66–98,96%` op test. Finale hidden-NRMSE na laag 26 is `0,01070/0,00904`
bij 25% en `0,02116/0,01715` bij 10%.

## Controles en accounting

- officiële en geadapteerde laag-23-route-ID's/-gewichten zijn exact;
- de 100%-delta blijft door alle tail-lagen exact: KL `0`, CE `0`, top-1 `1`;
- de aparte directe BF16-GEMM wijkt sporadisch één afrondingsstap af van de
  handmatige full-route (NRMSE `2,16×10⁻⁵`, maximum `0,0078125`) en staat als
  diagnostiek in de JSON;
- alle supports zijn lossless bit-packed en hun SHA-256 is nagecontroleerd;
- ideale support-known bytes/MACs zijn 25%/10%; tensor-lokale paginadruk blijft
  gemiddeld 50%/~39,5%; geen van deze cijfers is gemeten runtime;
- ruwe JSON SHA-256:
  `398a4bf126f14a4cf583324eac10ac6a55e1128ae512fa84569b4f57b6aee588`.

## Stop/go

**Go:** de vooraf vereiste spread-layers 1, 13 en 26 en lokale
instructie/code-transfer mogen worden geopend met dezelfde selector en
fracties. **Stop:** geen predictor, packed-kernel of modelbrede Eureka-claim
zolang spread-layers, een gelijktijdige full-depth-interventie en hardware-
bruikbare support niet zijn bewezen.
