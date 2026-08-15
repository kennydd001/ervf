# P14A DeepSeek-V2-Lite Q5-replicatie — preregistratie

Datum: 2026-08-12

## Hypothese

De vaste, fysieke-schaalsemantiek `Q5 routed experts + BF16 overige
gewichten` houdt de full-depth next-token-cross-entropy van
DeepSeek-V2-Lite binnen 2% van BF16. Hiermee wordt eerst de kwaliteitsdrempel
op een architectuur met top-6 routing en shared experts getest; een fysieke
STREAMQ5-bank is pas gerechtvaardigd als deze poort slaagt.

## Vastgelegde kandidaat

- lokaal checkpoint `models/deepseek-v2-lite`, 27 decoderlagen waarvan lagen
  1–26 routed MoE bevatten;
- alleen de drie matrices van ieder routed expert worden Q5;
- symmetrisch per rij, codes `[-15, 15]`, codes gekozen tegen de FP32-maxabs-
  schaal, schaal vóór reconstructie afgerond naar BF16;
- attention, router, shared experts, embeddings, normen en LM-head blijven
  BF16;
- officiële modelcode en officiële router-top-k blijven autoritatief.

## Data en poorten

De reeds gehashte BITFLOW-inputen worden gebruikt: 256 validationtokens en
256 testtokens. Die historische testset is eerder door een andere Q4-vraag
geopend; zij is daarom slechts corroboratief en geen nieuwe blinde test.

1. Validation opent de Q5-test alleen bij eindige waarden, alle 26 MoE-lagen
   uitgevoerd en relatieve CE-toename `<= 2%`.
2. De kandidaat krijgt `quality_pass` alleen als validation én test `<= 2%`
   blijven en de mediane officiële route-overlap minstens 95% is.
3. Bij falen stopt de fysieke DeepSeek-bank; bij slagen is uitsluitend een
   tweede-model-kwaliteitsreplicatie bewezen, nog geen snelheid of cachewinst.

