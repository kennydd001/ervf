# STREAMQ4-MoE P0 - validation full-depth kwaliteit

Uitkomst: **p0_validation_closed**.

BF16 CE: 2.098903. Q4+BF16 CE: 2.156489 (2.744%).
BF16+INT8 CE: 2.108149 (0.441%).
Q4+INT8 CE: 2.162796 (3.044%); top-1 89.213%.
Q4+INT4 CE: 2.344606 (11.706%).

Deze fase is alleen kwaliteitsisolatie; fysieke streaming en wall-clock blijven apart geblokkeerd.
