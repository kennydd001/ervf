# CORETAIL-MoE P0A onafhankelijke verificatie

Uitkomst: **locked16_mechanics_verified_full_p0_still_blocked** (28/28 controles geslaagd).

Een tweede decoder controleerde headers, offsets, record- en blokchecksums en vergeleek alle 75.497.472 gereconstrueerde codes plus alle BF16-scalebits opnieuw met de 16 canonieke GPTQ-bronbestanden.

De locked16-codec en geheugenrekenkunde zijn bevestigd. De officiële full-bank P0 blijft geblokkeerd omdat 6.128 van de 6.144 vereiste GPTQ-experts ontbreken; P1 blijft gesloten.
