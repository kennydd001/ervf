# E2GQ-MoE — onafhankelijke audit van de locked precondition

De GPTQ-codehistogrammen zijn opnieuw afgeleid uit de 16 originele safetensors-artifacts; de aangeleverde JSON is niet als meetbron gebruikt.

- Codes: `{'-2': 4713974, '-1': 17846753, '0': 31599966, '1': 21336779}` over `75,497,472` gewichten.
- Code-entropie: `1.782864891374` bpp.
- Inclusief raw BF16 group-128 scales: `1.907864891374` bpp.
- Alle 16 experts onder 2 bpp: `True`.
- Alle 48 matrices onder 2 bpp: `True`.
- Exacte ternary-core + extreme-tail-identiteit: `1.907864891374` bpp.

Dit bevestigt een harde theoretische representatieprecondition. Het is nog geen werkelijk geëncodeerd bestand, full-bankmeting, kwaliteitsbewijs of runtimebewijs.
