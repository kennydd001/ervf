# CORETAIL-MoE P2 — full-depth modelkwaliteit preregistratie

Gelockt op 2026-08-12 na de onafhankelijk bevestigde P1-pass en vóór enige
P2-modeluitvoer is geopend.

## Doel en claimgrens

P2 isoleert modelkwaliteit. CORETAIL reconstrueert volgens P0 exact dezelfde
GPTQ-codes en BF16-schalen als de bronbank; daarom is een gedequantiseerde
full-depth GPTQ-student kwaliteits-equivalent aan de fysieke CORETAIL-runtime.
Deze fase meet geen tokens per seconde.

## Datasetlock

Vijf domeinen: general, code, math, multilingual en instruction. Per domein en
split worden exact twee contexten van 128 tokens gebruikt. Validation en test
zijn disjunct en nooit voor GPTQ-kalibratie gebruikt:

- general: de afzonderlijke WikiText-validatie- en testbestanden;
- overige domeinen: de laatste disjuncte tokenvensters van de lokale bronnen,
  strikt achter de eerder gebruikte HERA/Qwen-GPTQ-kalibratievensters;
- code verdeelt ieder venster gelijk over Python en Java;
- multilingual verdeelt ieder venster gelijk over de acht gelockte talen.

Alle input- en bronhashes worden vastgelegd vóór de eerste forwardpass. Er is
geen dataselectie of drempelaanpassing na validation.

## Verplichte full-depth varianten

1. BF16 teacher;
2. GPTQ experts + BF16 trunk;
3. BF16 experts + INT4 trunk;
4. GPTQ experts + INT4 trunk — primaire inzetbare kandidaat;
5. GPTQ experts + INT8 trunk — isolerende fallbackdiagnostiek.

`trunk` omvat embeddings, LM-head, router en alle niet-expertmatrices in de 48
decoderlagen. RMSNorm-vectoren blijven BF16. INT4/INT8 gebruikt symmetrische
per-row group-128 quantisatie, round-to-nearest-even, codes `[-7,7]` of
`[-127,127]`, en onmiddellijke BF16-dequantisatie. Geen clipping-, group- of
laagselectiesweep is toegestaan.

## Metingen

- teacher-forced next-token cross-entropy en relatieve toename versus BF16;
- top-1 tokenovereenkomst versus BF16;
- per-laag en finale hidden relative-L2/max-abs;
- per domein én totaal;
- piek-VRAM, proces-RSS, runtime en alle artifacthashes.

Validation wordt volledig uitgevoerd en opgeslagen voordat test éénmaal wordt
geopend. Validation kiest niets; dezelfde vaste constructie gaat naar test.

## Beslisregel

De primaire `GPTQ experts + INT4 trunk`-kandidaat beslist:

- validation én test relatieve CE `<=2%`: **P2 pass**, geïntegreerde wall-clock
  wordt geopend;
- test `>2%` en `<=10%`: precies één vooraf geregistreerde rank-8 INT4-repair
  mag worden onderzocht; geen wall-clockclaim vóór die repair opnieuw slaagt;
- test `>10%`: kwaliteitslijn gesloten;
- iedere niet-eindige waarde, bronhashfout of teacher-reproductiefout sluit P2.

INT8-resultaten mogen de INT4-beslisregel niet vervangen, omdat de P0-
geheugenfit voor de INT4-trunk is bewezen.
