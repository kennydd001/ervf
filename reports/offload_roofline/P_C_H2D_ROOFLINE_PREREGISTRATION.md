# P-C preregistratie — lokale pinned-H2D-roofline

## Nieuwe meting

- GPU en driver worden tijdens de run vastgelegd.
- Gepinde CPU-bron en vooraf gealloceerde GPU-bestemming.
- Transfergroottes: 64, 256 en 512 MiB.
- Per grootte: 10 warmups en 50 CUDA-eventmetingen van een asynchrone H2D-copy,
  met synchronisatie op het eindevent.
- Rapporteer p05, mediaan, gemiddelde en p95 van effectieve decimale GB/s.
- Primaire lokale bandwidth is de mediaan van de grootste succesvol gemeten
  buffer. De conditionele K3-trunkroofline is `BW / 27,28 GB`.

## Gate en bewijsgrens

De hardwareleg ondersteunt de voorgestelde `≤1 tok/s`-grens alleen als de
primaire bandwidth ≤27,28 GB/s. Dit bewijst de K3-claim niet: de genoemde
27,28 GB actieve trunkbytes is een externe, nog niet lokaal gemeten input. Een
echte 64-token K3-decode blijft vereist. Als allocatie van een grootte faalt,
wordt dat geregistreerd en kiest de grootste succesvolle grootte de primaire
meting; bij geen succesvolle grootte is de hardwareleg geblokkeerd.
