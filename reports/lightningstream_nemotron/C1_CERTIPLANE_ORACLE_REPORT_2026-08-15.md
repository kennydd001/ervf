# C1 — CertiPlane: het bewijs klopt, maar het bewijst vrijwel niets

Datum: 2026-08-15
Verdict: **De intervalgrens is aantoonbaar sound — nul valse certificaten over 890.880 beslissingen. Maar hij certificeert 0,33% van de rijen waar 30% de poort was, en 0,37% van de werkelijke nullen. De Cauchy-Schwarz-grens is 9 tot 31× groter dan de preactivatie zelf. ExactFlow hypothese C en TreeSweep H8 zijn hiermee gesloten op hun eigen oracle.**
Terminal state: `c1_certiplane_sound_but_bound_90x_too_loose`
Preregistratie: `C1_CERTIPLANE_ORACLE_PREREGISTRATION_2026-08-15.md`

## 1. Wat er getest is

Splits het 4-bits NVFP4-codewoord in een lage-bitcore (tekenbit plus de bovenste
`c−1` magnitudebits) en een exacte residual. Bereken de preactivatie uit alleen
de core, begrens de bijdrage van de residual conservatief per schaalgroep van 16:

```
|δy_j| ≤ Σ_g ‖Δw_{j,g}‖₂ · ‖x_g‖₂
```

Is `y0_j + B_j ≤ 0`, dan is `ReLU²(y_j)` **exact nul** wat de residual ook zegt,
en hoeven die bits nooit gelezen te worden.

240 (expert, activatie)-paren over 15 MoE-lagen en 108 verschillende experts,
echte gewichten en echte activaties uit een echte greedy generatie, referentie in
float64.

## 2. De uitkomst

| core | gecertificeerd | van de echte nullen | valse certificaten | grens / \|y0\| |
|---:|---:|---:|---:|---:|
| 2 bits | **0,33%** | 0,37% | **0** | 31,09 |
| 3 bits | **0,33%** | 0,37% | **0** | 9,06 |

Werkelijke ReLU²-nullen: **90,64%** — consistent met S5's ~91%.

| poort | vereist | gemeten | |
|---|---|---|:--:|
| **G-C1-S1** soundness | nul valse certificaten | **0 van 890.880** | ✅ |
| **G-C1-R1** rendement | ≥ 30% van de staartbytes | **0,33%** | ❌ |
| **G-C1-B1** grens bruikbaar | ≥ 30% van de echte nullen | **0,37%** | ❌ |

Verifier 18/18, `VERIFIED`. Protected 0 modified / 0 removed.

## 3. Waarom het faalt, en waarom dat niet te repareren is met een betere core

De laatste kolom is de hele verklaring. De grens is **9 tot 31 keer groter dan de
preactivatie waar hij iets over moet zeggen**. Certificeren lukt alleen voor
neuronen die zó ver onder nul liggen dat zelfs een 9× te ruime grens ze niet over
de streep tilt — en dat zijn er 0,33%.

De oorzaak is structureel, niet parametrisch. De som loopt over **168 groepen**,
en Cauchy-Schwarz is per groep scherp maar telt de 168 termen als worst case bij
elkaar op alsof alle residuals in dezelfde richting wijzen als `x`. In
werkelijkheid middelen ze uit: de echte `|δy|` is ordes kleiner dan `B`. Meer
corebits helpt precies zoals verwacht — de grens zakt van 31× naar 9× — maar de
certificatiegraad beweegt niet mee, want bij 9× is hij nog steeds hopeloos ruim.
Een core van 4 bits zou een grens van nul geven en 100% certificeren, maar dan is
er ook geen staart meer om over te slaan.

Wat wél zou werken is een **niet-conservatieve** grens: een geleerde of
statistische schatting van `δy`. Maar dan vervalt precies de eigenschap waarvoor
CertiPlane in het pack staat — *"Dit is geen geleerde risicopredictor. Iedere
overgeslagen page heeft een mechanisch bewijs"* — en het pack verbiedt die ruil
zelf, onder `Verboden`: *"een onveilige learned gate als vervanging van een
gefaalde certificatebranch"*.

Het pack schrijft ook zelf voor wat er nu geldt: *"Mislukt de oracle, dan geen
kernel."*

## 4. Wat dit sluit

- **ExactFlow C (CertiPlane / Proof-Carrying Precision).** Gesloten op zijn eigen
  poort, met een factor 90.
- **TreeSweep H8 (BranchCert).** Zelfde mechanisme, zelfde grens; het idee om
  juist off-path boomknopen goedkoop te certificeren erft deze grens ongewijzigd.

En de meting is bruikbaar buiten deze twee: elke variant die routerbeslissingen,
ReLU²-tekens of afgeronde projecties met **conservatieve intervalrekenkunde**
over 168 groepen wil bewijzen, loopt tegen dezelfde 9–31× ruimte aan.

Wat het níét sluit: certificering met een **scherpere** sound bound. Cauchy-Schwarz
per groep is de simpelste keuze, niet de beste denkbare. Een grens die correlatie
tussen residual en activatie meeneemt zou scherper zijn — maar hij moet sound
blijven, en dat is een wiskundige vraag, geen kernelvraag. Wie die tak wil openen,
moet eerst laten zien dat er een sound bound bestaat die ordes scherper is.

## 5. Claim boundary

Numerieke oracle op echte expert-records en echte activaties, opgenomen uit een
echte greedy generatie. De residualgrens gebruikt de **exacte** per-groep-normen,
die een gebouwd systeem niet gratis kan opslaan — het pack begroot 0,15 bit per
gewicht voor certificaat-metadata — dus de hier gemeten certificatiegraad is een
**bovengrens** op wat een gebouwde versie haalt. De referentie-preactivatie is in
float64 op de host berekend, niet door de float32-kernel van de runtime, dus
neuronen die binnen float32-ruis van nul liggen zijn op de float64-waarde
ingedeeld. Er is geen kernel geschreven en niets hiervan is een tijd- of
doorvoermeting; besparingen in bytes zijn op deze runtime geen besparingen in
tijd (S11, S12, X1, Y2-R1).

## 6. Artefacten

`C1_CERTIPLANE_ORACLE_PREREGISTRATION_2026-08-15.md` ·
`scripts/lightningstream_nemotron/c1_certiplane_oracle.py` ·
`c1_certiplane_oracle.json` ·
`scripts/lightningstream_nemotron/c1_independent_verify.py` ·
`c1_independent_verification.json` · `protected_verification_after_c1.json`
