# RSIV-MoE / GhostWeights — terminaal onderzoeksverdict

**Datum:** 2026-08-11  
**Registry-uitkomst:** `falsified`  
**Mechanistische uitkomst:** `falsified_rank_working_set`  
**Eureka bereikt:** **nee**

## Besluit

De exacte promptopslagbound van GhostWeights blijft wiskundig geldig:

```text
S_layer <= (2d + 3m) * top_k * prompt_tokens
```

De noodzakelijke empirische aanname erachter is echter hard gefalsificeerd op
twee MoE-families. Toekomstige routed input- en SwiGLU-intermediaire activaties
vallen vrijwel nooit in een bruikbare rank-32 promptatlas. De rang groeit op de
gemeten checkpoints nagenoeg tot de observatiebound, in plaats van te
verzadigen. De volledige weights kunnen daardoor niet met de vereiste lage
page-faultrate door deze operator-images worden vervangen.

Dit is geen mislukte implementatie en geen onbesliste meting. Het is een geldige
negatieve uitkomst van de vooraf geregistreerde make-or-break-test.

## Resultaten tegenover de bevroren gates

De primaire P1-gate vereiste bij `rank <= 32` tegelijk minimaal `92%`
double-fast en minimaal `10x` geprojecteerde routed cold-bytereductie. De
hard-falsificatieregel sloot de hypothese wanneer rank 32 op V2 én een
hogere-E-familie onder `80%` double-fast bleef.

| Proef | Prefix → future | Gerapporteerde kandidaat | Test double-fast | Test cold-byte reductie |
|---|---:|---:|---:|---:|
| P1A, DeepSeek-V2-Lite | 96 → 32 | locked `r4, t=0,001` | 0,000% | 1,000x |
| P1B, DeepSeek-V2-Lite | 1.024 → 128 | locked `r32, t=0,10` | 0,434% | 1,007x |
| P1C, Qwen3-30B-A3B | 1.024 → 128 | locked diagnostic `r4, t=0,001` | 0,000% | 1,000x |
| P1C harde diagnose | 1.024 → 128 | `r32, t=0,10` | 1,742% | 1,034x |
| P1C post-lock bovengrens | 1.024 → 128 | `r128, t=0,10` | 5,762% | 1,108x |

Zelfs de niet-inzetbare rank-128 Qwen-diagnose ligt dus ruim onder zowel de
92%-gate als de 80%-falsificatiegrens. De afwijking is tientallen procentpunten,
niet een grensgeval dat met kleine threshold- of numerieke wijzigingen kan
worden gered.

## Waarom meer experts het probleem niet oplosten

Qwen3-30B-A3B verlaagt bij 1.024 prefilltokens de verwachte routerbelasting per
expert van V2's 96 naar 64 invocaties. Dat was de centrale gunstige
schaalvoorspelling. In de echte Qwen-census was voor alle twaalf
context-laagcombinaties de inputrang op checkpoints 128, 512 en 1.024 exact
gelijk aan het aantal observaties. De intermediaire rang deed vrijwel hetzelfde;
alleen laag 23 week op checkpoint 1.024 licht af. De totale exacte
atlasbenutting lag tussen 98,98% en 100% van de analytische observatiebound.

Het expert-count-cancellation theorem beperkt dus correct hoeveel exacte
promptkolommen kunnen bestaan, maar de data benutten bijna de volledige bound.
Het theorem levert geen lage-rankcompressie en voorspelt geen transfer naar
future tokens. Dat onderscheid is nu empirisch beslist.

## Integriteit van de hogere-E-replicatie

- Model: officieel `Qwen/Qwen3-30B-A3B-Base`, gepinde commit
  `1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9`.
- Checkpoint: 16/16 BF16-shards, exact 61.066.575.648 bytes; ieder lokaal
  SHA-256-hash gelijk aan de officiële LFS-hash.
- Lagen: 0, 23 en 47; twee validation- en twee testcontexten van 1.152 tokens.
- Iedere laag bevat exact 18.432 routed invocaties per split.
- Router-ID's, routergewichten en logits sloten exact met de native
  Qwen-forward; alle tensors waren eindig en alle count/bound-controls slaagden.
- Validation selecteerde en vergrendelde één kandidaat voordat test werd
  geopend. Test werd daarna één keer gebruikt.
- Onafhankelijke eindverifier: 4.429/4.429 verplichte controles geslaagd,
  nul fouten, één gedeclareerde verifierwaarschuwing.

De eerste verifierrun gebruikte een te strenge extra tolerantie van `0,002`
voor de som van naar BF16 teruggecastte routergewichten. Addendum 001 verving
alleen die verifiergrens door de analytische BF16-unit-roundoff
`eps/2 = 0,00390625`; de grootste gemeten afwijking was `0,00244140625`.
De mislukte eerste audit, captures, selectie en resultaten zijn onveranderd en
hash-verankerd gebleven.

Belangrijkste SHA-256-ankers:

- Qwen-acquisitierapport: `318980cd6aa634072e97a4c06dfbfd50f7b255cd7f340d00d1f1e6105e6e3daf`.
- Validation-lock: `19038b6a718ebb5b6607dd52c4c8ca59b1542b746711c77ff0568e8a72804f3f`.
- Testcapture: `c36cd1e5c331877329044bc5c2305b79fd03496c3056a7577e85aeb80d5db382`.
- P1C-resultaat: `e3224cd759db6e1e008a7b0295b6d9dfb783c5d13ee6ac2d693d7fb62bb00864`.
- Definitieve verifier-JSON: `d981781b577951f6f257abd4afa125c3ffd4852200c193f96e747a5420fd8719`.

## Waarom P2, runtime, Kimi en V4 niet volgen

P2 en verdere fasen waren expliciet afhankelijk van een positieve P1-screen.
Met 1,742% rank-32 fast path zou minstens 98,258% van de Qwen-invocations nog
een cold path vereisen. Daarmee zijn de beoogde `<=8%` cold-path- en `>=10x`
cold-bytegates al vóór kwaliteit en runtime onmogelijk voor deze constructie.
Een packed kernel kan die causale page-faultverhouding niet herstellen.

De preregistratie bepaalde bovendien dat reproductie onder 80% op V2 en één
hogere-E-model de hypothese terminaal falsificeert. Nu alsnog Kimi of V4 kiezen
zou achteraf modelselectie veranderen en een negatief resultaat wegselecteren.
Die modellen mogen alleen terugkeren onder een nieuwe, mechanistisch
onafhankelijke en vooraf geregistreerde hypothese.

## Claimgrens en vervolg

Niet gemeten voor RSIV zijn full-depth CE, echte cold I/O, 512-tokenrollouts en
batch-1-decodesnelheid. Dat maakt de uitkomst niet inconclusive: P1 was juist
ontworpen om deze dure stappen te blokkeren zodra hun noodzakelijke rank- en
page-faultvoorwaarde faalt.

Het officiële eindverdict is daarom:

> `falsified` — GhostWeights/RSIV is geen Eureka. De algebra is exact, maar de
> gemeten routed activatiewerkset is niet laag-rank genoeg om de volledige
> experts met de vereiste toekomstige hitrate te virtualiseren.

Verder onderzoek moet onder een nieuwe registry-ID beginnen, met een nieuw
mechanisme en nieuwe vooraf geregistreerde gates. CRAFT en RSIV blijven beide
gesloten; hun negatieve resultaten mogen niet worden herlabeld of post-hoc
getuned.
