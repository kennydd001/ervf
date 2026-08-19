# S100 Phase 14 survivor adjudication — revised with DFlash2

## Decision table

| Track | State entering Phase 14 | New test | Promotion condition |
|---|---|---|---|
| 13A entropy | closed | none | none |
| 13C temporal delta | closed | none | none |
| 13D native BF16 | open, strongest survivor | 14D | component breadth + strict validation + official heldout |
| 13B activation subspace | partially open | 14B2 | output-aware quality plus physical bytes |
| 13E expert basis | naïve form closed | 14E2 | decoded, activation-weighted output quality |
| DFlash2 | untested prior art | 14F0/14F1 | verifier budget + resident memory + transfer signal |

## DFlash2 adjudication

The public technique is worth measuring because it attacks acceptance length,
a variable our Phase-12 perfect-draft verifier intentionally held at 100%.
However, that same perfect-draft result establishes a stronger fact: the current
verifier itself is too slow for S100. DFlash2 therefore enters as a gated future
training route, not as a competing immediate kernel patch.

The bundle separates four claims that must never be conflated:

1. **Published method works on Qwen/H200 or Apple** — external prior art.
2. **Nemotron target trajectories contain transferable suffix/path signal** —
   tested by 14F1.
3. **A resident Nemotron drafter can fit alongside the exact target** — tested
   by 14F0 memory accounting.
4. **The target can verify enough accepted tokens inside a 10 ms/token-equivalent
   budget** — tested only by a measured full-block verifier.

Only all three local gates may open draft training. A trained checkpoint and
end-to-end speculative cycle would be a later phase.

## Claim discipline

- A component speedup is not full-model speed.
- A perfect-draft ceiling is an upper bound, not an achieved drafter result.
- Oracle top-16 coverage is selector headroom, not selector performance.
- A transfer proxy failure closes only the preregistered cheap screen.
- `null` means incomplete evidence, never no-go.
