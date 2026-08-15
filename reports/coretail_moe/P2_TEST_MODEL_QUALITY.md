# CORETAIL-MoE P2 — test full-depth kwaliteit

Uitkomst: **p2_quality_closed**.

BF16 CE: 1.916881. GPTQ+BF16 CE: 2.372374 (23.762%).
BF16+INT4 CE: 2.129396 (11.087%).
GPTQ+INT4 CE: 2.740037 (42.943%); top-1 61.811%.
GPTQ+INT8 CE: 2.372291 (23.758%).

Deze fase is kwaliteitsisolatie. Geïntegreerde tokens per seconde worden alleen bij een P2-pass geopend.
