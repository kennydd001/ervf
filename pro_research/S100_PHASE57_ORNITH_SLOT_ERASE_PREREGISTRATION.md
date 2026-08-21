# S100 Phase57 — Ornith DFlash slot-erase mitigation

## Question

Does llama-server's documented slot erase clear enough persistent DFlash state
to keep the Phase55 `ubatch=256` arithmetic result byte-exact after a preceding
coding request?

## Frozen setup

- Same exact artifacts, build and `ubatch=256`, DFlash K=8 configuration as
  Phase56, with prompt caching disabled.
- Two fresh server replicates.
- In each: coding request, `POST /slots/0?action=erase`, arithmetic request.
- Strict greedy request settings and 64-token limits from Phase56.
- Baseline consensus texts are frozen from Phase56 baseline replicate 1.

## Gates

1. Both processes serve non-empty coding and arithmetic completions.
2. Both coding completions match the frozen baseline.
3. Both slot-erase calls return successfully and report an erase result.
4. Both post-erase arithmetic completions match the frozen baseline.
5. Both replicates expose positive DFlash acceptance.

If gate 3 is green but gate 4 remains red, the public slot-cache reset is not a
sufficient drafter-state reset; process/worker recreation or an upstream code
fix is required for a strict lossless service contract.

## Technical amendment before retry

The first call returned HTTP 501 before the arithmetic request. Because this
build reports slot-save support as disabled unless a save path is configured,
the retry adds an explicit isolated slot-save directory. The 501 result is
archived as technical evidence and no model gate was adjudicated from it.
