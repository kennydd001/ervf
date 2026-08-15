# E2GQ-MoE P0 — coverageverdict

Uitkomst: **coverage_negative**, onafhankelijk geverifieerd met 12/12 controles.

De vooraf vastgelegde 32.768 WikiText-train-tokens dekken de volledige expertbank niet voldoende:

- alle **48/48 lagen** bevatten experts met minder dan 128 routed rijen;
- **1,695/6.144** laag-expertparen zitten onder 128;
- **196** laag-expertparen hebben exact nul invocaties;
- de verdeling loopt van 0 tot 23,480 invocaties.

Volgens de preregistratie zijn geen GPTQ-codes voor ondergedekte experts geconstrueerd en is P1 niet geopend. Dit falsifieert de gekozen monolinguale calibratieprocedure, niet de reeds bevestigde 16-expert entropy-precondition en niet entropy-GPTQ in het algemeen.

Een vervolg vereist een nieuwe, vooraf vastgelegde coveragehypothese met een representatieve meertalige/multidomeincalibratie of een principiële activation-agnostic quantizer. De huidige registry mag die keuze niet post-hoc maken.
