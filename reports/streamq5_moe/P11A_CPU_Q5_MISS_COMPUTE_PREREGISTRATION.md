# P11A preregistratie — CPU Q5-compute bij acht cachemissers

Datum: 2026-08-12. Status bij vastlegging: geen P11A-output geopend.

## Hypothese

Bij een volledig koude MoE-laag kan zestien-core CPU-compute op acht fysieke
Q5-experts sneller zijn dan acht PCIe-recordkopieën plus de gecachete GPU-ERVF-
compute. De CPU-kern decodeert de werkelijke bank, emuleert exact de 256
virtuele accumulators en reductieboom en voert gate/up, BF16-SwiGLU en down uit.

## Selectie en test

Selecteer op drie validation-iteraties uit 1, 4, 8 en 16 OpenMP-threads. Meet
daarna twintig iteraties met de geselecteerde instelling. Vergelijk met de
fysieke all-cold GPU-schatting:

`8 × (P10B event-seconden / gekopieerde records) + P7B Q5-plane / 48`.

## Gates

- CPU-uitvoer bitexact gelijk aan ERVF-16 voor acht experts.
- CPU p50 en p95 hoogstens 95% van respectievelijk de fysieke mean/p95-GPU-
  all-coldgrens.
- Alle uitgangen eindig.

Een scalar/AVX-compilerkern die faalt bewijst niet dat iedere denkbare CPU-kern
faalt; hij falsificeert de concrete lokaal gebouwde exact-semantische route.

