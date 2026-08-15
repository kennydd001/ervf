# STREAMQ5-MoE P0 - test full-depth kwaliteit

Uitkomst: **p0_quality_pass**.

BF16 CE: 2.258431. Q5+BF16 CE: 2.278913 (0.907%).
BF16+INT8 CE: 2.258395 (-0.002%).
Q5+INT8 CE: 2.280984 (0.999%); top-1 93.701%.
Q5+INT4 CE: 2.457640 (8.821%).

Deze fase is alleen kwaliteitsisolatie; fysieke streaming en wall-clock blijven apart geblokkeerd.
