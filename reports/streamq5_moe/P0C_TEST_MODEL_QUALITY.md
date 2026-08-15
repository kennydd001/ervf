# STREAMQ5-MoE P0C - test fysieke-schaal-kwaliteit

Uitkomst: **p0c_quality_pass**.

BF16 CE: 2.259874. Q5+BF16 CE: 2.253688 (-0.274%).
BF16+INT8 CE: 2.267202 (0.324%).
Q5+INT8 CE: 2.249066 (-0.478%); top-1 92.598%.
Q5+INT4 CE: 2.439283 (7.939%).

Deze correctiefase gebruikt exact de BF16-schaalvolgorde van het fysieke formaat; fysieke streaming en wall-clock blijven apart geblokkeerd.
