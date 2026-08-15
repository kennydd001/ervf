# CORETAIL-MoE P2 - definitief kwaliteitsoordeel

**Oordeel: P2 is negatief en de kwaliteitslijn is gesloten.**

De vooraf vastgelegde primaire kandidaat, `GPTQ experts + INT4 trunk`, verhoogt
de teacher-forced cross-entropy met **35,953% op validation** en **42,943% op de
eenmalig geopende held-out testset**. De harde passgrens was voor beide splits
maximaal 2%. De testscore ligt bovendien boven de 10%-grens; daardoor is de
enige voorziene rank-8-repair niet toegestaan en wordt de geïntegreerde
wall-clockfase niet geopend.

| Variant | Validation CE-toename | Test CE-toename | Test top-1 vs. BF16 |
|---|---:|---:|---:|
| GPTQ experts + BF16 trunk | 21,520% | 23,762% | 71,732% |
| BF16 experts + INT4 trunk | 7,938% | 11,087% | 80,709% |
| GPTQ experts + INT4 trunk | **35,953%** | **42,943%** | **61,811%** |
| GPTQ experts + INT8 trunk | 21,603% | 23,758% | 71,890% |

De isolaties zijn eenduidig. De expert-GPTQ alleen overschrijdt de gate ruim;
de INT4-trunk alleen eveneens. INT8 maakt de expertschade praktisch niet
kleiner en mag volgens de preregistratie de geheugengekwalificeerde INT4-
kandidaat niet vervangen. Bij de primaire testvariant faalt elk domein:
code +32,536%, general +49,025%, instruction +27,915%, math +66,140% en
multilingual +50,737%.

De evaluatie omvat alle 48 lagen, vijf domeinen, twee contexten van 128 tokens
per domein en per split, in totaal 1.270 next-tokenlabels per split. Alle
waarden zijn eindig. De test werd pas na de opgeslagen validationrun geopend.

Een afzonderlijke verifier controleerde bron- en artifacthashes, de disjuncte
inputlocks, alle 48 laagrapporten, per-domein- en aggregaatrekenkunde, top-1-
weging en de preregistreerde beslisregel: **55/55 controles geslaagd**.

P0 en P1 blijven geldig als representatie- en microkernelresultaat. P2
falsificeert echter de geregistreerde inzetbare totaalclaim: er is voor deze
exacte kandidaat geen kwaliteitsbehoud en dus geen wetenschappelijke basis om
nog een end-to-end snelheid als Eureka te presenteren.
