# STREAMQ4-MoE P0 - definitief oordeel

**Oordeel: de vaste RTN-Q4-expert plus INT8-trunkkandidaat is negatief en
gesloten zonder test te openen.**

Op de nieuwe, vooraf gelockte validationcontexten verhoogt de primaire
kandidaat de cross-entropy met **3,0441%**. De vaste progression gate was
maximaal 3%; de overschrijding is 0,0441 procentpunt. De grens is niet achteraf
verruimd.

De isolatie is informatief: RTN-Q4-experts met BF16-trunk kosten 2,7436%,
terwijl BF16-experts met INT8-trunk slechts 0,4405% kosten. De Q4+INT4-controle
kost 11,7062%. De resterende fout zit dus hoofdzakelijk in Q4-experts, niet in
de INT8-trunk.

Alle 48 lagen en vijf domeinen zijn uitgevoerd; alle waarden zijn eindig. De
inputcontexten zijn exact disjunct van CORETAIL P2. Een onafhankelijke audit
slaagde **26/26** en bevestigt dat het gereserveerde testartifact ongeopend
bleef. Fysieke bank-, cache-, kernel- en wall-clockfasen zijn niet geopend.
