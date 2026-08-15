# H4 Residual Syndrome Sketch / SketchGate — preregistratie laag 26

Vastgelegd vóór implementatie en vóór inspectie van nieuwe H4-uitkomsten. De
bestaande WikiText-componenttrace is eerder voor andere experimenten gemaakt,
maar H4 gebruikt vaste prefixvensters en geen testgestuurde modelkeuze.

## Hypothese

Een kleine random-projectieschets van de werkelijke Q3→Q4-downresidu kan op
laag 26, zonder teacherlogits of een geleerd semantisch model tijdens selectie,
de schadelijkste routed expertinvocaties goed genoeg herkennen om minstens 80%
van de exacte 25%-oracle-KL-winst terug te winnen.

## Vaste data en interventie

- model: `deepseek-ai/DeepSeek-V2-Lite`, revision
  `604d5664dddd88a0433dbae533b7fe9472482de0`;
- dataset: WikiText-2-raw-v1, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`;
- ontwikkeling: de eerste 256 tokens van de bestaande validatietrace;
- eenmalige evaluatie: de eerste 256 tokens van de bestaande testtrace;
- laag 26, natuurlijke top-6-route, ongenormaliseerde routergewichten;
- shared experts, attention, normen en alle overige gewichten blijven exact;
- kandidaat-hidden states zijn
  `BF16(official_teacher + candidate_routed - natural_BF16_routed)`;
- iedere volledige-Q3, volledige-Q4 en BF16-control wordt opnieuw berekend en
  tegen de bevroren componenttrace gecontroleerd.

Smoke is maximaal 32 validatietokens en adjudiceert geen gate. De volledige run
is exact 256 validatie- plus 256 testtokens, sequence blocks van 128 tokens en
10.000× gepaarde block-bootstrap met seed `20260810`.

## Fase A — matrixattributie

Per expert worden zes uitvoerfamilies gemaakt vanaf dezelfde input:

1. alle matrices Q3;
2. alleen gate Q4;
3. alleen up Q4;
4. alleen down Q4, met de Q3-SwiGLU-activatie;
5. gate+up Q4 en down Q3;
6. alle matrices Q4.

Voor iedere upgradefamilie worden alle 64 maskers over de zes invocaties per
token met volledige-vocabulaire teacher→candidate-KL gemeten. Een exacte
globale dynamic program kiest bij exact 25% budget het beste masker per token.
De down-attributiefractie is

`(KL_Q3 - KL_down_oracle25) / (KL_Q3 - KL_all_matrix_oracle25)`.

De downresidu is promotiewaardig wanneer deze fractie op validatie én test
minstens 70% is. `gate+up` domineert wanneer zijn oraclewinst groter is dan de
down-only-oraclewinst. Falen van de 70%-regel verhindert spreadpromotie, maar
de vooraf vastgelegde layer-26-sketchmeting wordt wel afgemaakt.

## Fase B — vaste sketch

Voor `ΔD = D4 - D3`, Q3-activatie `a` en probes `z` geldt:

`u_k = ΔDᵀ z_k`, `q_hat = p_router² · mean_k[(u_kᵀ a)²]`.

Vast staan:

- dimensies `r ∈ {4, 8, 16, 32, 64}`;
- probeverdelingen Gaussian en Rademacher;
- vijf seeds `20260810..20260814`;
- de eerste `r` rijen van één seedbank, dus geneste dimensies;
- één probe-bank wordt gedeeld over alle experts;
- iedere `u`-rij wordt symmetrisch per rij naar int8 gekwantiseerd met een
  FP16-schaal; selectie gebruikt de gedequantiseerde int8-rijen;
- geen lineaire calibratie in deze primaire proef;
- stabiele globale aflopende rangschikking; ties volgen token- en slotvolgorde.

Voor deployment-accounting telt alleen één gekozen bank: int8 `u`-waarden plus
FP16-schalen voor alle 64 experts. Effectieve metadata-bits worden gedeeld door
alle originele routed gate/up/down-gewichten.

## Fase C — vaste selectie en controles

Iedere methode kiest exact 25% van alle token×6-invocaties. Volledige Q4 wordt
alleen op die invocaties toegepast. We meten minimaal:

- vaste random-25%-controles met vijf seeds;
- routergewicht;
- SketchGate voor alle 50 verdeling×seed×r-configuraties;
- perfecte volledige-matrix-oracle uit fase A;
- alle-Q3, alle-Q4 en exact BF16;
- historische, duidelijk apart gelabelde 1.024-tokenresultaten van de eerdere
  ridge/progressive/quadratic predictors; die tellen niet als same-window gate.

Oracle recovery is

`(KL_Q3 - KL_method) / (KL_Q3 - KL_perfect_oracle25)`.

Een high-damage event behoort tot de hoogste 10% van de positieve exacte
single-invocation-KL-winst bij één volledige Q3→Q4-upgrade. De false-negative
rate is het aandeel van die events dat niet in de 25%-selectie zit. Als minder
dan 10% positieve events bestaan, gebruiken we alle positieve events en wordt
dit expliciet gerapporteerd.

Modelkeuze gebruikt uitsluitend validatie. Een configuratie kwalificeert
wanneer alle vijf seeds validatie-recovery `≥80%` en high-damage-FN `≤1%`
halen. We kiezen eerst de kleinste `r`, bij gelijke `r` Rademacher. Als niets
kwalificeert, kiezen we alleen voor diagnose de hoogste mediane recovery, daarna
laagste maximale FN, kleinere `r`, en Rademacher. De primaire runtimebank is
altijd de eerste vaste seed `20260810`; seed wordt nooit op test gekozen.

## Gates en stop/go

H4-laag-26 is alleen positief als:

1. de down-attributiefractie op validatie én test `≥70%` is;
2. de gekozen configuratie met seed `20260810` op validatie én test
   oracle-recovery `≥80%` en high-damage-FN `≤1%` haalt;
3. alle vijf seeds van de gekozen configuratie op beide splits dezelfde
   recovery- en FN-grenzen halen;
4. metadata `<0,1` effectieve bit per origineel routed expertgewicht is;
5. gemeten batched sketchcompute `<10%` is van de gemeten host→device-tijd
   voor de in het expliciete batch-1 out-of-core-model vermeden vierde-bitbytes;
6. de officiële BF16-deltacontrol exact is.

De hardwaremeting is slechts een gemeten hardwaremodel, geen packed runtime en
geen snelheidsclaim. Slechts een volledig positieve layer-26-gate opent een
nieuwe preregistratie voor lagen 13/23 en OOD. Als gate/up domineert en de
downsketch mist, wordt H4 gefalsificeerd vóór een complexere first-order sketch.
Testresultaten mogen na opening niet worden gebruikt om `r`, verdeling,
quantisatie, score of drempels te veranderen.

## Smoke-addendum vóór de volledige run

De 32-token-smoke toonde dat opnieuw uitgevoerde BF16 tensor-core-GEMM's met
een andere expertbatchvorm niet bitidentiek zijn aan de eerder in één
2.048-tokenrun gemaakte Q3/Q4-componenttrace: Q3 NRMSE `0,000619`, Q4 NRMSE
`0,000684`, en voor beide maximale absolute fout `0,125`. Dit is een
batchvorm-afhankelijke GEMM-regressie, niet de officiële original-control.

Daarom staat vanaf nu, nog vóór opening van de 256-tokenvalidatie en test, de
componentregressie vast op NRMSE `≤0,001` en maximale absolute fout `≤0,25`
voor zowel Q3 als Q4. Bitexactheid wordt daarnaast ongewijzigd gerapporteerd.
De officiële teacher-deltacontrol blijft zonder tolerantie bitexact. Geen
inhoudelijke H4-gate, probe, seed, score, budget of selectieregel is gewijzigd.
