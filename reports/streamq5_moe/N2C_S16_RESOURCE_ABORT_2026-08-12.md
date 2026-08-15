# N2C S=16 — gecontroleerde resource-abort

Datum: 2026-08-12.

De vooraf geregistreerde N2C-sweep produceerde volledige, bitexacte
validationuitkomsten voor S=2, S=4 en S=8. De S=16-arm voltooide de initiële
GPU-fasen niet binnen een praktische tijdgrens en werd gecontroleerd afgebroken.

Er was geen CUDA-crash of out-of-memory-fout. De GPU bleef circa 97–100% bezet.
De compile-time `float partial[16][16]`-staat plus meerdere dynamische
tokenaccumulaties veroorzaakt zeer waarschijnlijk registerdruk en spill naar
local memory. Dat is een inferentie uit de schaalcurve en uitvoeringsduur; een
SASS-/occupancyprofiel is niet beschikbaar in deze lokale omgeving.

Tijdens de lange sweep bereikte de laptop-GPU 85–86 °C en zakte de grafische
klok van eerdere boostniveaus naar circa 1,7–2,0 GHz. De AB/BA-volgorde houdt
referentie en kandidaat binnen ieder S vergelijkbaar, maar absolute tijden
tussen verschillende S-waarden mogen daarom niet als een zuivere schaalcurve
of winnaarselectie worden gebruikt.

Besluit: S=16 krijgt status `blocked_by_resource_spill_timeout`; geen snelheids-
of correctnessclaim. Dit verandert de volledig gemeten negatieve AB/BA-ratio's
van S=2/4/8 niet.
