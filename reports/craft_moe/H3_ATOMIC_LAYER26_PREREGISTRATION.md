# H3 Exact Atomic Expert Oracle — preregistratie laag 26

Vastgelegd vóór H3-code of inspectie van H3-uitkomsten. Deze fase is een
exploratieve laat-laagscreen; zij opent nog geen full-depth- of
confirmatoryclaim.

## Vaste input en exacte decompositie

- gepinde DeepSeek-V2-Lite- en WikiText-commits;
- laag 26, eerst 32 validatietokens, daarna eerste 256 validatie- en eerste 256
  bestaande testtokens uit de vaste componenttrace;
- de zes natuurlijke routed experts, originele ongenormaliseerde gewichten;
- beide shared experts blijven exact en worden niet in de atomfractie geteld;
- per expert exact 1.408 originele SwiGLU-atomen:
  `a_j = silu(gate_j(x))*up_j(x)` en bijdrage
  `v_(e,j)=p_e*a_(e,j)*down_column_(e,j)`.

Kandidaten worden als routed delta op de officiële teacherstate geïnjecteerd:

```text
candidate_hidden = BF16(official_teacher + sparse_routed - manual_full_routed)
```

De 100%-maskercontrol moet exact KL `0`, top-1 `1` en CE-delta `0` geven.

## Vaste fracties en selectors

Retentieniveaus: `{1.0, .75, .50, .35, .25, .15, .10, .05}`.

1. per expert top-`|a_j|`;
2. per expert top-`|a_j| ||d_j||₂`;
3. globaal over alle 8.448 routed atomen top-`|p_e a_j| ||d_j||₂`;
4. globale tegels van 16 neuronen, score is som van gekwadrateerde exacte
   bijdragenormen in de tegel;
5. dezelfde tegelregel met 32 neuronen;
6. dezelfde tegelregel met 64 neuronen.

Per-expertselectie houdt per expert `ceil(f×1408)` neuronen. Globale
neuronselectie houdt `ceil(f×8448)` atomen. Tegelselectie houdt
`ceil(f×aantal_tegelblokken)` volledige expertlokale tegels. Ties gebruiken de
stabiele oorspronkelijke expert-/neuronvolgorde.

Exact sequentieel greedy residual reduction krijgt een afzonderlijke Stage-B-
preregistratie en runtimecheck; zij mag deze vaste Stage-A-curves niet
vervangen. Stage A beantwoordt eerst of de goedkope exacte bijdragescore al een
bruikbaar oracleplafond toont.

## Metrics en accounting

Voor iedere selector/fractie:

- routed-output relatieve L2 per token;
- finale volledige-vocabulaire teacher→candidate-KL, CE en top-1;
- gepaarde sequence-block-bootstrapintervallen;
- werkelijke behouden atom-/tegelcounts;
- ideale support-known BF16-weightbytes en MAC-fractie, lineair met behouden
  gate-, up- en downrijen/kolommen.

Dit is alleen een fysieke bovengrens. Omdat `a_j` eerst gate/up vereist, is de
bytebesparing zonder vroege supportindex niet deploybaar; dat is eventueel H5.

## Gates

De primaire laag-26-screen is `oracle_positive` wanneer de globale
neuronbijdragescore bij 25% retentie op validatie én test een relatieve
CE-toename <2% heeft. De moonshotscreen gebruikt 10% en <3% relatieve CE.

Als veiligheidsdiagnostiek worden KL en top-1 altijd gerapporteerd, maar de
vooraf aangeleverde primaire gate blijft CE. Een negatieve CE-delta telt als
pass, niet als bewezen kwaliteitswinst.

De tile-64-hardwaregate slaagt wanneer bij 25% retentie de gemiddelde KL niet
meer dan 20% hoger is dan de globale neuronoracle op beide splits. Bij vrijwel
nul neuron-KL wordt tevens het absolute verschil gerapporteerd.

Harde stop voor H3 wanneer geen neuronselector bij 25% op validatie de 2%-CE-
gate haalt én de beste KL groter is dan `0,01`. Alleen een positieve primaire
screen opent laag 13, laag 1, lokale instructie/code-transfer en exact-greedy
Stage B. Geen predictor of sparse kernel vóór die gate.

