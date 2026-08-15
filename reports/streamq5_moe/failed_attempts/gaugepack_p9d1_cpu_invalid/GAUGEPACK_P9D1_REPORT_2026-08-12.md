# GaugePack P9D-1 — codec/oracle

Uitkomst: **gaugepack_p9d1_fail**. De codec bewaart P9B voor laag 0, experts 0–15 met 44,317 BF16-decodemismatches over 75,497,472 gereconstrueerde matrixelementen.

Het bestand is 24,395,840 bytes; ratio tegenover dezelfde volledige P1D-records: **0.502363**. Lineaire full-bankprojectie: 8.725 GiB.

Gate/up-survivorcodes en raw BF16-schalen zijn letterlijk uit P1D overgenomen en onafhankelijk tegen de P9B-reconstructie gecontroleerd. Down is uit bron-BF16 plus het frozen P9B-masker gereconstrueerd, omdat P9B's nulmasker groepsmaxima kan wijzigen en er historisch geen P9B-codebank is opgeslagen.

Bij down verschilden 86,796 van 196,608 groepsschalen van de ongesnoeide P1D-bank; dit bevestigt dat blind P1D-downbytes kopiëren semantisch fout zou zijn.

Claimgrens: één laag en zestien experts; nog geen kernel-, full-bank-, kwaliteits- of throughputbewijs.
