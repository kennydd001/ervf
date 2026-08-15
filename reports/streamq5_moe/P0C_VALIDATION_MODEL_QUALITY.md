# STREAMQ5-MoE P0C - validation fysieke-schaal-kwaliteit

Uitkomst: **p0c_validation_pass_test_authorized**.

BF16 CE: 1.870477. Q5+BF16 CE: 1.887656 (0.918%).
BF16+INT8 CE: 1.862275 (-0.439%).
Q5+INT8 CE: 1.898118 (1.478%); top-1 92.362%.
Q5+INT4 CE: 2.005714 (7.230%).

Deze correctiefase gebruikt exact de BF16-schaalvolgorde van het fysieke formaat; fysieke streaming en wall-clock blijven apart geblokkeerd.
