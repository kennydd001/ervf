# PH0X-R1 CPU/OpenCL compile diagnostic preregistration

De eerste PH0X-poging stopte vóór een Intel-kernelstart met OpenCL-buildcode `-11`.
Ook rapporteerde de CPU-only q8→7-sensitivity-witness een digestmismatch.

Deze diagnostiek mag uitsluitend:

1. dezelfde OpenCL-bron compile-only bouwen en de volledige buildlog bewaren;
2. de frozen codes/scales-selector en one-hot-berekening stap voor stap opnieuw
   uitvoeren en alle woorden/digests bewaren;
3. geen kernel starten en NVIDIA niet openen.

De uitkomst is uitsluitend een implementatiediagnose.

