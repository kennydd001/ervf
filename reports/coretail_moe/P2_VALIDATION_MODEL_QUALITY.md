# CORETAIL-MoE P2 — validation full-depth kwaliteit

Uitkomst: **validation_complete_test_authorized**.

BF16 CE: 2.027261. GPTQ+BF16 CE: 2.463525 (21.520%).
BF16+INT4 CE: 2.188178 (7.938%).
GPTQ+INT4 CE: 2.756130 (35.953%); top-1 61.024%.
GPTQ+INT8 CE: 2.465208 (21.603%).

Deze fase is kwaliteitsisolatie. Geïntegreerde tokens per seconde worden alleen bij een P2-pass geopend.
