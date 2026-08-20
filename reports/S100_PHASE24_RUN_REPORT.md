# S100 Phase 24 — Best-of-All Lightning Synthesis

## Component screens

- Attention BF16 M4: `False`, speedup `0.2675214537584212`
- Router FP32 M4: `False`, speedup `0.9086266667924214`
- Shared NVFP4 M4: `False`, speedup `0.22053594478636843`

## Scale-resident synthesis screen

| Arm | Status | H4 ms @1024 | tok/s | Plane MiB |
|---|---:|---:|---:|---:|
| baseline | measured | 78.85544999999999 | 50.725726630182194 | 0.0 |
| synth_k0 | measured | 79.1252 | 50.55279481126114 | 0.0 |
| synth_k4 | measured | 77.5622 | 51.571512927689 | 79.7 |
| synth_k8 | measured | 76.90865 | 52.00975442944325 | 159.4 |
| synth_k12 | measured | 75.2363 | 53.16582553900179 | 248.0 |
| synth_k16 | measured | 75.2546 | 53.15289696576688 | 321.7 |
| synth_k23 | measured | 73.79929999999999 | 54.20105610757827 | 492.4 |

## Adoption and active parent

- Selected: `synth_k23`
- Thermally adopted: `True`
- Active parent: `phase24_best_of_all`
- Context1024 H4: `74.1702` ms
- Target-only: `53.930015019509185` tok/s
- Beats V18 four-token equivalent `78.2916` ms: `True`
- Remaining factor to 40 ms: `1.8542549999999998`

## Prompt/H8 route generalization

| Prompt | H4 repeat | H8 / two-H4 streams |
|---|---:|---:|
| factual | 0.333 | 0.801 |
| code | 0.312 | 0.847 |
| reasoning | 0.396 | 0.828 |
| conversation | 0.292 | 0.822 |
| technical | 0.312 | 0.805 |
| dutch | 0.375 | 0.829 |
| translation | 0.375 | 0.833 |
| json | 0.271 | 0.794 |

- Generalization green: `True`
- H8 build open: `True`

## Final

- Target <=40 ms/H4: `False`
- Drafter shootout open: `False`
- Next route: `BUILD_H8_BEST_OF_ALL_FULL_VERIFIER`
- S100 single achieved: `False`
