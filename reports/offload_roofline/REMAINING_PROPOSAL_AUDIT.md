# Resterende offloadvoorstellen — uitvoerbaarheidsaudit

- **P-A geblokkeerd:** Qwen BF16 is compleet, maar slechts 16/6.144 echte GPTQ-experts en geen packed LFU/async-runtime zijn aanwezig.
- **Volledige P-C geblokkeerd:** geen lokaal K3-checkpoint, geen gemeten actieve trunkbytes en geen 64-token K3-decode.
- **P-D geblokkeerd:** geen K3-target of werkende autoregressieve drafter; acceptatie is niet meetbaar.

De K3-unieformule geeft 120.279427, niet 118,6; naive/uniek is 1.0642×, niet circa 1,08×.

Wel gemeten op de echte Qwen top-8-routes (gemiddelde unieke experts per laag bij diepte 8):
- general: 27.108/64 (0.424); naive/uniek 2.361×.
- code: 27.028/64 (0.422); naive/uniek 2.368×.
- math: 30.398/64 (0.475); naive/uniek 2.105×.
- multilingual: 23.918/64 (0.374); naive/uniek 2.676×.
- instruction: 27.806/64 (0.434); naive/uniek 2.302×.

Deze uniemeting zegt niets over speculative acceptatie; P-D blijft daarom niet geslaagd.
