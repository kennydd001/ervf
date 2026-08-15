# N3C — INT8/INT4-KV bij 8K/32K: capaciteit gesloten, kwaliteit geblokkeerd

## Rekenkundige uitkomst

De fysieke P13-layout gebruikt 98.304 BF16-KV-bytes per contexttoken. Uit de
gemeten P7C-VRAMgrens (`free_before=7.385.120.768`, Q8-trunk
`1.248.931.840`, reserve `402.653.184`) en Q5-recordgrootte `3.035.136` volgt:

| context / KV | KV-bytes | Q5-slots totaal | minimum per laag | static-20 compatibel |
|---|---:|---:|---:|---|
| 8K BF16 | 805.306.368 | 1.623 | 33 | ja |
| 32K BF16 | 3.221.225.472 | 827 | 17 | nee |
| 32K INT8, zonder schaaloverhead | 1.610.612.736 | 1.358 | 28 | ja |
| 32K INT4, zonder schaaloverhead | 805.306.368 | 1.623 | 33 | ja |

Zelfs met redelijke group-scaleoverhead blijven INT8 en INT4 bij 32K boven de
huidige 20 slots per laag. Compressie kan het capaciteitsprobleem dus oplossen.

## Waarom geen kwaliteits- of snelheidsclaim wordt geopend

De lokale decoder en zijn onafhankelijke kwaliteitssets zijn fysiek begrensd op
4.096 posities; alle attentionindexering en KV-strides zijn daarop vergrendeld.
Een synthetische random-KV-quantisatie zou geen geldige modelkwaliteit bij 8K of
32K meten. Een echte proef vereist tegelijk:

1. een opnieuw gebouwde 32K-runtime/KV-layout;
2. lange, verse modelcontexten en kwaliteitslabels;
3. een vooraf gekozen INT8/INT4-schaalsemantiek;
4. fysieke quantize/dequantize-attentionkernels.

Die verandering is een afzonderlijk model/runtimeproject en valt niet eerlijk
onder een kleine lokale kernelvariant. Daarom is N010 `blocked_scope`, met de
capaciteitshelft positief maar de expliciete kwaliteit/context-wall onbewezen.

Claimgrens: exacte byte-accounting op gemeten fysieke budgetten; geen lossy-KV-
kwaliteit, geen 8K/32K-decode en geen tok/s.
