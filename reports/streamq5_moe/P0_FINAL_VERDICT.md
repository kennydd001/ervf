# STREAMQ5-MoE P0 - definitief kwaliteitsoordeel

**Oordeel: P0 full-depth kwaliteit geslaagd en onafhankelijk geverifieerd.**

De vaste primaire `Q5 experts + INT8 trunk`-kandidaat verhoogt cross-entropy
met **0,6976% op validation** en **0,9986% op de eenmalig geopende testset**.
Beide liggen onder de vooraf vastgelegde 2%-gate. De top-1-overeenkomst met
BF16 is respectievelijk 93,465% en 93,701%.

De isolaties bevestigen de hypothese. Op test kost Q5-experts met BF16-trunk
0,9069%; de INT8-trunk met BF16-experts is metrisch neutraal (-0,0016%). De
Q5+INT4-controle faalt met +8,8207%, zodat de INT8-trunk essentieel is.

Validation en test gebruiken opnieuw verse, vooraf gelockte contexten in vijf
domeinen. Alle 48 lagen en 1.270 next-tokenlabels per split zijn meegenomen;
alle waarden zijn eindig. Een onafhankelijke audit van hashes, 96
laagrapporten, splitseparatie en metriekrekenkunde slaagde **57/57**.

Dit opent uitsluitend de fysieke bank-, cache- en kernelfasen. P0 bewijst nog
geen fysiek 5-bitformaat, cachehitratio, transferlatentie of end-to-end tok/s.
