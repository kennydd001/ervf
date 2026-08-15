# FLEQ-MoE onderzoekslog

## 2026-08-11 — Nieuwe onafhankelijke registry geopend

CRAFT en RSIV blijven gesloten. `FLEQ_MOE` test full-rank maar low-entropy
expertgewichten. GSQ/QMoE maken duidelijk dat de brede methode prior art is;
de eerste fase is een onafhankelijke Qwen-reproductie en hardwarehaalbaarheid.

De officiële GSQ-repository is gepind op commit
`03fc16484c369e3127225615d5e03e8d3a6043e3`. De 30B-recipe vraagt vier H200's,
de runtime is niet als native Windows/sm120-pad bewezen en de productie-YAML is
op deze commit inconsistent met zijn eigen configparser. Daarom wordt vóór
enige evaluatie een expert-streamed smoke vastgelegd die de officiële
quantizeroperator behoudt en de laptopresourcegrenzen rechtstreeks meet.

## 2026-08-11 — P1 uitvoeringspoging 001 zonder resultaat gestopt

Laag 0/expert 46 voltooide de initialisatie en GSQ-optimalisatie, maar stopte
bij de eerste metricforward: de gepinde GPTQ/RTN-helper retourneert FP32
gedequantiseerde weights en Qwen voert BF16-activaties aan. PyTorch weigert die
gemengde `F.linear`. Er is geen artifact of metriekrapport geschreven. Poging
001 blijft als JSON behouden. De enige correctie cast de gedequantiseerde
candidate naar de ongewijzigde modeldtype; codes, scales, data, selectie,
epochs en gates veranderen niet.

## 2026-08-11 — P1 protocoladdendum 001 vóór verdere evaluatie

Poging 002 schreef de eerste volledige expertoutput, maar implementeerde tien
epochs als tien totale optimizerstappen. De officiële batchgrootte 64 impliceert
voor expert 46 zestig stappen. Alleen expert 46 was bekeken; alle andere
experts en laag 47 bleven ongeopend. Artifact en rapport van poging 002 zijn
onder nieuwe namen en hashes bewaard. Addendum 001 bevriest batch-64,
upstreamconforme full-batchpermutatie en global-stepannealing voor de
definitieve run.

## 2026-08-11 — P1 protocoladdendum 002 na assemblercontrole

Alle zestien primaire 2-bitresultaten waren geldig en bleven ongewijzigd. De
ternarydiagnostiek faalde echter twee auditcontrols: raw RTN-Q/scalepaarwaarden
lagen bij enkele rijen buiten het ternary grid en beide samengestelde
determinismebooleans waren false zonder uitgesplitste oorzaak. De volledige
diagnostiek en eerste aggregate zijn als poging 003 bewaard. Addendum 002
gebruikt vóór training de officiële `get_hard_weights()`-projectie als
ternarybaseline en splitst weight- en lossdeterminisme. De reeds gefaalde
primaire 2-bitgate kan hierdoor niet worden gewijzigd.

## 2026-08-11 — P1 addendum 003 en definitieve afsluiting

De eerste gecorrigeerde ternary-herhaling liet zien dat de upstream
quantizerconstructor zijn initialisatietensors in-place normaliseert. Die run
is als poging 004 met hashes bewaard. Addendum 003 cloneert uitsluitend de
constructorinput; daarna sluiten gewichten en alle 60 losswaarden bit-exact.

De definitieve 2-bitsmoke is negatief. In laag 0 verbetert GSQ 0/8 experts ten
opzichte van GPTQ (gemiddeld -28,90%); in laag 47 eveneens 0/8 (gemiddeld
-105,84%). Alle zestien experts hebben een slechtere held-out p95. De primaire
gate faalt dus op elk inhoudelijk criterium, terwijl alle uitvoeringscontroles
slagen.

De toegestane ternarydiagnostiek verbetert de geldige hard-RTN-baseline op
16/16 experts (gemiddeld +10,25% in laag 0 en +19,78% in laag 47), maar de
absolute held-out fout ligt veel hoger dan bij 2-bit GPTQ. Bovendien vereist
een gewone 2-bit pack met BF16 group-128 scales nog steeds 2,125 bpp; alleen de
ideale log2(3)-bound komt op 1,710 bpp. Er is geen kernel, full-depth CE,
rollout of runtimebewijs.

De onafhankelijke verifier sluit 18/18 controles en berekent acht held-out
metricsets opnieuw uit de opgeslagen gewichten. De volledige testsuite sluit
op 142/142. P2 is volgens de preregistratie niet geautoriseerd en de
trajectory-QAT-hypothese wordt niet geopend: dit resultaat is geen PTQ-near
miss. De specifieke P1-hypothese is gefalsificeerd; een algemene
onmogelijkheid van low-entropy MoE-quantisatie is niet beweerd of bewezen.
