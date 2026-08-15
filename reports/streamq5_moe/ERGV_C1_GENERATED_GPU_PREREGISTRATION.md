# ERGV-C1 preregistratie — generated-vs-manual GPU-gate

Datum: 2026-08-12. Status bij vastlegging: de C1-module is nog niet
gecompileerd en er is geen C1-resultaatbestand geopend of aangemaakt.

## Hypothese

De restricted ERGV-codegenerator kan voor width 16 uitvoerbare Q8- en
Q5-row-reducers genereren die op CUDA bit-voor-bit dezelfde BF16-afgeronde
FP32-uitvoer produceren als de handgeschreven P7-ERVF-helpers.

## Bevroren vergelijking

- Referentie: `ERVF_SOURCE` uit
  `scripts/streamq5_moe/run_p7b_ervf_kernel.py`.
- Kandidaat: `generate_cuda_source()` uit
  `src/moe_lab/ergv_compiler.py`.
- Breedte: uitsluitend 16; er vindt geen selectie plaats.
- Q8-shapes: `(rows=137, cols=2048)` en `(rows=65, cols=4096)`.
- Q5: acht synthetische fysieke records met de bestaande P7-layout;
  gate/up gebruikt `cols=2048`, down gebruikt `cols=768`.
- Inputs: vier vaste activatiefamilies met seed `120843`: random, nul,
  alternerende schaal en cancellation.
- Codes en BF16-schalen zijn synthetisch maar fysiek in hetzelfde byteformat
  als de P7-kernels.

H2D, allocatie en compilatie zijn niet getimed. Deze gate bevat bewust geen
prestatiemeting.

## Correctheidspoort

C1 slaagt alleen als:

1. de gecombineerde P6B-helperbron, handmatige P7-bron, gegenereerde bron en
   dunne generated-kernelwrappers binnen 120 seconden compileren;
2. voor iedere inputfamilie alle Q8-, Q5-gate-, Q5-up- en Q5-down-elementen
   eindig zijn;
3. iedere kandidaatoutput bit-voor-bit gelijk is aan de handgeschreven P7;
4. in totaal nul verschillende outputbits worden gevonden.

Een compilefout, timeout, niet-eindige uitvoer of bitverschil sluit de gate. Er
wordt na het zien van outputs geen kernelvariant aangepast.

## Bewijsgrens

Een pass valideert alleen generated width-16-code op één lokale GPU en
synthetische inputs. Het is geen snelheids-, echte-model-, tweede-architectuur-,
externe-baseline- of nieuwheidsclaim. Width 4/8/32/64 blijven voor een latere,
afzonderlijk geregistreerde GPU-uitbreiding.
