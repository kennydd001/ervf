# P-C pinned-H2D-roofline — resultaat

**Uitkomst hardwareleg: hardware_leg_supports_conditional_k3_le_1.**

- 64 MiB: mediaan 26.341 GB/s (p05 25.913, p95 26.381).
- 256 MiB: mediaan 26.367 GB/s (p05 26.194, p95 26.404).
- 512 MiB: mediaan 26.159 GB/s (p05 25.992, p95 26.318).

Conditioneel plafond bij T=27,28 GB: **0.9589 tok/s**.

Dit is geen volledige K3-meting: actieve trunkbytes en de 64-token decode ontbreken lokaal. De uitkomst valideert of falsifieert alleen de busleg onder de externe T-aanname.
