# N2D — exacte greedy LM-head write-elision preregistratie

Vastgelegd vóór enige N2D-timing. De compilefase mag de kernels compileren,
maar laadt de fysieke head niet, start geen kernel en opent geen timingpartition.

## Hypothese

Bij greedy decode zijn de volledige 151.936 logits niet nodig buiten de
LM-head. De bestaande ERVF16-Q8-head kan iedere groep van zestien vocabrijen
bitexact reduceren tot één `(BF16-afgeronde waarde, kleinste rij-index)`-
kandidaat. Een tweede exacte vergelijkingsreductie kiest de globale argmax.
Daarmee verdwijnen de full-logitwrite en de daaropvolgende full-logitread zonder
de Q8-dequantisatie, virtuele reductieboom, BF16-outputafronding of tie-regel te
wijzigen.

## Fysieke bron en drie paden

- Head: het geverifieerde fysieke `lm_head.q8bin` uit de P6-bank,
  151.936 × 2.048, 316.026.880 bytes resident op de GPU.
- Pad A — huidige controle: `q8_ervf16_full` schrijft alle logits en
  `logits_stats_current` berekent logsumexp, targetveld en argmax.
- Pad B — argmax-only controle: dezelfde full-logithead gevolgd door een exacte
  argmaxreductie zonder logsumexp.
- Pad C — kandidaat: `q8_ervf16_block_argmax` schrijft 9.496 waarden en 9.496
  indices; `reduce_block_candidates` kiest exact de globale argmax. Er bestaat
  in dit pad geen full-logitbufferwrite.

Alle vergelijkingen kiezen bij gelijke waarden de kleinste vocabindex, gelijk
aan `numpy.argmax` en de huidige runtime. Pad C gebruikt exact dezelfde
ERVF16-operatievolgorde per vocabrij als A en B.

## Inputs en partitions

- BF16-afgeronde standaardnormale vectors van lengte 2.048.
- Validatieseed `120823`, zestien vectors en daarnaast één nulvector.
- Testseed `120824`, zestien vectors en daarnaast één nulvector.
- De nulvector forceert een 151.936-voudige fysieke logittie; alle paden moeten
  index nul kiezen en exact dezelfde maximale floatbits teruggeven.
- De testseed blijft verzegeld tot de validatie-openingspoort slaagt.

## Timingprotocol

- De head en alle inputs zijn vóór timing resident.
- Vijftien ongetimede warmups per pad.
- CUDA-eventtijd van de volledige padcompositie, dus twee kernels per pad.
- Validatie: 48 ABBA-cycli; test: 96 ABBA-cycli.
- Per cyclus: `A-B-B-A` en `A-C-C-A`; de volgorde van beide paren wisselt per
  cyclus. De input roteert deterministisch door de zestien vectors.
- Rapporteer voor ieder pad mean, p50, p95, p99, minimum, maximum en samples,
  plus kandidaat/control-ratio's en gepaarde ABBA-ratio's.

## Correctheid en bytes

Voor alle zeventien correctheidsinputs moeten gelden:

1. A, B, C en `numpy.argmax` geven dezelfde index;
2. B en C retourneren exact dezelfde float32-bits als de geselecteerde
   BF16-afgeronde full logit;
3. de nulvector kiest index nul;
4. alle resultaten zijn eindig.

Globale head-outputbytes per token, inclusief eindresultaten:

- A: `151936×4 + 2×4 + 1×4 = 607.756` bytes;
- B: `151936×4 + 1×4 + 1×4 = 607.752` bytes;
- C: `9496×4 + 9496×4 + 1×4 + 1×4 = 75.976` bytes.

Pad C moet dus exact 531.780 bytes en minstens 87,49% van A's outputwrites
vermijden.

## Poorten

Validatie opent de test alleen bij:

- alle correctheidsvoorwaarden;
- exact bytecontract;
- C/A-p50-ratio hoogstens `1,02` en C/A-p95-ratio hoogstens `1,05`.

De onafhankelijke test is een fysieke componentpass bij:

- alle correctheidsvoorwaarden;
- C/A-p50-ratio hoogstens `0,98` en C/A-p95-ratio hoogstens `1,00`;
- C/B-p50-ratio hoogstens `0,99` en C/B-p95-ratio hoogstens `1,00`.

Pad B/A is diagnostisch: het scheidt logsumexp-eliminatie van write-elision en
heeft geen eigen acceptatiepoort.

## Claimgrens

Een pass bewijst alleen exacte greedy argmax-write-elision op de residentiële
fysieke Q8-head. Pad C kan niet zonder meer worden gebruikt voor CE, top-k,
top-p, temperature sampling of logitprocessors. Geen volledige decoderwinst,
tok/s, modelkwaliteit of 80B-generalisatie wordt hiermee bewezen.
