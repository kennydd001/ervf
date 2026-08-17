# S100 phase 6 final build artifact

Date: 2026-08-17

Final one-click artifact:

- ZIP SHA256: `c60085043e806e8dc2294571493e6eef5c9a541ddd242ca4b0a88a6b36ed959b`
- patch SHA256: `7834bf562bfa6530b46474b10b447cc69a04b355ce10cc83d18040cb353ea438`

Final timing catalog isolates the mechanisms:

- `prefix_exact`: new deterministic prefix scan plus existing B3 pipeline;
- `wave2_exact`, `wave3_exact`, `wave6_exact`: legacy exact scan plus WAVE;
- `wave3_prefix_exact`: their composition.

Validation completed before target-GPU execution:

- all Python sources compile;
- PowerShell variable-colon and delimiter scan passes;
- `git apply --check` passes;
- strict whitespace check passes;
- reconstructed phase3 -> phase4 -> phase5 -> phase6 patch chain applies cleanly.

Target-SM120 CUDA compilation, exact token parity and fresh timing remain fail-closed runtime gates.
