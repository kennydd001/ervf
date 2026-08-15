# C1 — CertiPlane-oracle: hoeveel is er te certificeren zonder de staart te lezen?

Datum: 2026-08-15
Status: **bevroren vóór uitvoering.**
Aanleiding: `NEMOTRON_EXACTFLOW_AGENT_PACK` hypothese C (CertiPlane /
Proof-Carrying Precision) en `TREESWEEP_200` H8 (BranchCert). Beide ongemeten.

## 1. Het idee, en waarom het los staat van alles wat al weerlegd is

Elke hypothese die tot nu toe sneuvelde ging over *hoe vaak* je het target
gebruikt (speculatie) of *hoeveel bytes* een gewicht kost (quantisatie). Dit gaat
over iets anders: **welke bytes je überhaupt niet hoeft te lezen**, met een
mechanisch bewijs dat het antwoord er niet van verandert.

Schrijf het NVFP4-gewicht als een lage-bitcore plus een exacte residuallaag:

```
w = w0 + Δw          w0 = E2M1[code & 0b1100] · s      Δw = (E2M1[code] − E2M1[code & 0b1100]) · s
```

Bereken `y0 = w0 · x` uit alleen de core, en begrens de rest conservatief per
groep van 16 (de NVFP4-schaalgroep):

```
|δy_j| ≤ Σ_g ‖Δw_{j,g}‖₂ · ‖x_g‖₂
```

De expert-activatie is **ReLU²**. Voor de preactivatie `y_j` geldt dan: is
`y0_j + B_j ≤ 0`, dan is `ReLU²(y_j)` **exact nul**, ongeacht wat de residual
bits zeggen. Zo'n neuron hoeft zijn staartbytes nooit gelezen te krijgen, en zijn
`down_proj`-kolom doet toch al niet mee (S5).

S5 mat dat ~91% van de ReLU²-uitgangen nul is. De vraag is niet of ze nul zijn —
dat weten we — maar **welk deel daarvan uit de core alleen te bewijzen valt**.

## 2. Meetopzet

Echte experts en echte activaties, opgenomen uit een echte greedy generatie op de
bevroren prompts: per meetstap de genormaliseerde laag-invoer `x` en de zes
officiële expert-ids. Dequantisatie op de host met de bestaande `nvfp4`-code, dus
geen nieuwe kernel en geen benadering in de referentie.

Per (expert, activatie)-paar en per coregrootte `c ∈ {1, 2, 3}` bits:

- `y_full` — de exacte NVFP4-preactivatie;
- `y0` — dezelfde GEMV uit alleen de core;
- `B` — de Cauchy-Schwarz-grens per groep van 16;
- **gecertificeerd nul**: `y0 + B ≤ 0`;
- **vals certificaat**: gecertificeerd nul terwijl `ReLU²(y_full) > 0`.

≥ 200 paren, verdeeld over lagen en stappen.

## 3. Poorten, overgenomen uit het pack

- **G-C1-S1 — soundness.** **Nul** valse certificaten. Eén is genoeg om de
  hele tak te sluiten, want het bewijs is dan geen bewijs.
- **G-C1-R1 — rendement.** ≥ **30%** van de routed staartbytes moet zonder fetch
  certificeerbaar zijn (het pack's doorgangspoort; 50% is hun stretch). Bij een
  core van `c` bits is de besparing per gecertificeerd neuron `(4−c)/4` van zijn
  codebytes.
- **G-C1-B1 — de grens moet iets waard zijn.** Het aandeel gecertificeerde
  nullen moet ≥ 30% zijn van de werkelijke nullen. Certificeert de grens vrijwel
  niets, dan is de conservatieve begrenzing te slap en helpt geen kernel.

Poorten worden na het zien van het resultaat niet verruimd.

## 4. Wat dit niet doet

Geen kernel, geen runtime-wijziging, geen residual-paginaformaat, geen
bound-compute op het kritieke pad. Dit is de oracle die het pack zelf vóór elke
kernel eist: *"Mislukt de oracle, dan geen kernel."*

Ook expliciet: een gecertificeerde besparing in **bytes** is door Y2-R1 al
begrensd op ~0,68 tijd-per-byte, en door S11/S12/X1 is bekend dat bytes op deze
runtime geen tijd voorspellen. Een geslaagde oracle opent dus een kernelvraag,
geen doorvoerclaim.

## 5. Artefacten

`scripts/lightningstream_nemotron/c1_certiplane_oracle.py` ·
`c1_certiplane_oracle.json` ·
`scripts/lightningstream_nemotron/c1_independent_verify.py` ·
`c1_independent_verification.json` · rapport met claim boundary.

## 6. Claim boundary van dit document

Geen meting, geen resultaat.
