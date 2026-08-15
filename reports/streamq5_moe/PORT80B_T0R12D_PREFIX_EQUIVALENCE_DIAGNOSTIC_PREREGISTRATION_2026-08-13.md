# T0-R12-D prefix-equivalence diagnostic

Diagnostic only; scientific pass is impossible. The pinned R12 checkpoint, four prompts, CPU runtime, loader and direct official router capture are unchanged. One process loads layer 0 once. For every prompt it runs whole length 16 and fresh prefixes 1..16 without raising on inequality. It retains raw whole last-position outputs, every prefix final output, whole and prefix-16 conv/recurrent states, hashes and finiteness. For every length it reports BF16 differing-word count, max BF16 ULP, max absolute difference and relative L2.

Prompt 1 gets an additional same-length whole repeat and fresh-prefix-3 repeat. These distinguish same-input nondeterminism from whole-versus-prefix semantics. Direct official whole router tuple and tie evidence are retained. No manual MoE, Q5, bank, P4, GPU or pass claim.
