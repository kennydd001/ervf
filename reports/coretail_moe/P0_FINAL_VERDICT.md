# CORETAIL-MoE P0 — definitief full-bankoordeel

**Oordeel: P0 geslaagd; de fysieke universele-coreclaim is op Qwen bewezen en P1 is geopend.**

De onafhankelijk geverifieerde bronbank bevat 6.144 experts, 18.432 matrices,
28.991.029.248 GPTQ-codes en 226.492.416 BF16-schalen. Daaruit is het vooraf
vastgelegde row-random-access CORETAIL-formaat werkelijk gebouwd, zonder RTN-
of fixed-widthsubstitutie.

De fysieke core meet 6.317.056.000 bytes (5,883217 GiB; 1,743175 bpp). De tail
meet 908.083.200 bytes (0,845718 GiB; 0,250583 bpp). Samen is dit 1,993759 bpp.
Met INT4-trunk, 4K BF16-KV-cache en de verplichte 0,75-GiB-runtimebuffer komt de
residentformule op 7,725844 GiB tegenover 7,959961 GiB gerapporteerd VRAM.

Een onafhankelijke decoder controleerde ieder record, iedere CRC, offset,
uitlijning en fallbackbyte en reconstrueerde alle codes en alle schaalbits exact:
26/26 controles geslaagd.

Dit is een representatie- en geheugendoorbraak, geen snelheidsclaim. Conform de
preregistratie opent nu uitsluitend P1: een exacte fused-kernel moet minimaal
27,2 miljard routed weight-applicaties/s halen en binnen het 100-ms/tokenbudget
blijven. Modelkwaliteit en end-to-end tokens per seconde zijn eveneens nog niet
door P0 bewezen.
