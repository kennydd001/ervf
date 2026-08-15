# STREAMQ5-MoE P0 - validation full-depth kwaliteit

Uitkomst: **p0_validation_pass_test_authorized**.

BF16 CE: 2.087952. Q5+BF16 CE: 2.095832 (0.377%).
BF16+INT8 CE: 2.088073 (0.006%).
Q5+INT8 CE: 2.102519 (0.698%); top-1 93.465%.
Q5+INT4 CE: 2.267869 (8.617%).

Deze fase is alleen kwaliteitsisolatie; fysieke streaming en wall-clock blijven apart geblokkeerd.
