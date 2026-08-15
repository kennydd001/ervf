# P15B — energievergelijking: gedeeltelijk gemeten, vergelijkingsclaim geblokkeerd

P13C logde tijdens 10.000 tokens 40 GPU-powerpunten. Gemiddeld was dit
48,32975 W. Gedeeld door 14,234758 tok/s is de **GPU-only projectie 3,395 J per
token**. Dit is geen systeemenergie: CPU, DRAM, SSD, voedingsefficiëntie en het
scherm ontbreken.

De CPU-baseline leverde geen package-energy- of wandmeting; Windows/WSL stelde
in deze omgeving ook geen bruikbare RAPL/powercap-teller beschikbaar. Daarom is
een same-hardware joule/tokenvergelijking niet berekend. Daarvoor is een
gekalibreerde wandmeter of gelijktijdige CPU-package-telemetrie nodig.

Status: `blocked_measurement`, niet getest als vergelijkende energieclaim.
