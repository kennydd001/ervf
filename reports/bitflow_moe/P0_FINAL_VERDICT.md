# BITFLOW-MoE P0 — definitief oordeel

**Oordeel: de geregistreerde lineaire BITFLOW-tak is negatief en gesloten.**

De maximale C1/Q4-kandidaat verslechterde de cross-entropy-schade in plaats
van die te herstellen: −645,95% recovery op validation en −395,02% op de
eenmalig geopende testset. Op test was de top-1-overeenkomst 75,00% en de
late-laagexplosieratio 6,4635. Daarmee faalt zowel de 50%-progression gate als
de primaire 70%/1%/97%/geen-explosie-gates.

Een onafhankelijke herberekening uit de bewaarde ruwe logits en 52 BF16
repairmatrices slaagde 23/23. De oorspronkelijke Q4-testbaseline is exact
gereproduceerd.

Conform de vooraf vastgelegde hard-stop zijn C0, C2, Q3, syndrome en runtime
niet uitgevoerd. De conclusie geldt voor deze sequentiële, data-gelimiteerde
dense lineaire equalizer; ze is geen bewijs tegen alle mogelijke niet-lineaire
BITFLOW-architecturen.
