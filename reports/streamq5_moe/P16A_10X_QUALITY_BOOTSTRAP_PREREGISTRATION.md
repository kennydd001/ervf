# P16A 10× grotere kwaliteitsproef — preregistratie

Datum: 2026-08-12

## Vraag

Blijft de vaste Qwen3-30B-A3B-kandidaat met fysieke Q5-schaalsemantiek voor
routed experts en INT8-trunk binnen de 2%-kwaliteitsgrens op 12.700
next-tokenlabels, tienmaal P0C?

## Vastgelegde data en kandidaat

- bron: de immutable Qwen-GPTQ-supplementbundel;
- per domein `general`, `code`, `math`, `multilingual`, `instruction` exact de
  eerste twintig rijen en de eerste 128 tokens per rij;
- totaal 100 contexten en 12.700 labels;
- BF16 teacher versus één kandidaat: Q5 routed experts + INT8 overige
  tweedimensionale gewichten en INT8 LM-head;
- group-128 RTN, codes gekozen tegen FP32-maxabs-schaal, schaal voor
  reconstructie naar BF16 afgerond;
- officiële volledige 48-laags forward, router en top-8 blijven ongewijzigd.

De supplementdata zijn eerder voor route-/GPTQ-calibratie-onderzoek gebruikt,
maar de RTN-Q5-kandidaat is niet op deze labels gefit. Dit is een grote
corroboratieve kwaliteitsproef, geen volledig onaangeraakte benchmarkset.

## Poorten

1. alle 48 lagen, 100 contexten en 12.700 labels zijn eindig;
2. relatieve geaggregeerde CE-toename `<= 2,0%`;
3. gepaarde contextbootstrap met vaste seed `20260812`, 10.000 trekkingen:
   tweezijdige 95%-bovengrens `<= 2,5%`;
4. top-1-overeenkomst met teacher `>= 90%`.

De uitkomst is uitsluitend een kwaliteitsclaim. Fysieke bytes, tok/s en
endurance komen uit hun afzonderlijke experimenten.

