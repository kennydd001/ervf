# PH1 Intel execution R8P6 — independent frozen source audit

Date: 2026-08-14  
Scope: static/read-only audit before execution. No candidate import, preflight, payload read, compiler, OpenCL, or device call was performed.

## Verdict

**GO for exactly one closed, no-device R8P6 preflight invocation, and for nothing physical.**

The two R8P5 blockers are closed without changing the dual identity, runtime, frozen CPU preparation, controls, stage hashes, resource gates, numerical content, no-device boundary, or claim. This GO authorizes CPU-only reading of the already frozen preparation slice during the one exact preflight. It does not authorize a model forward, compiler, OpenCL, CUDA, GPU, other device action, independent verifier run, or physical R8 execution.

## Frozen identities and topology

| Artifact | SHA-256 | Status |
|---|---|---|
| preflight | `f67030aaf9cfeac5266bb3a59971b8589692a6b96781413ed56b599a9975c6e5` | exact handoff/lock |
| independent verifier | `74173dbc62c0b8c0b956a89010270e9534bbaa35b4b23eaa84109549255308f0` | exact handoff/lock |
| preregistration | `73de179a3e7dd6a53191a62ec73ff5de18d92a11d4705ffff0d8acf0a59064a5` | exact handoff/lock |
| closed lock | `e705521cfd4d1e0f086d92d0f830a861d17b142d843d2bf01cf8002dfe504fe8` | closed/PENDING |

All eight direct bindings match. The 64-key lock retains all 53 inherited bindings, including R8P5 audit SHA `c431578cc6a1edefa0d3843ca0fdd26ec5d07b9e592d5a758a4ed0f40e36d608`. The current lower-case R8 family contains only the seven R8 through R8P6 locks. Every R8P6 result, manifest, commit, independent-verification, failure, quarantine and `.inprogress.*` path is absent.

## Current production transaction closure

The R8P6 success path and its TEMP simulator call the same current functions:

- `atomic_create()` at lines 47–54;
- `quarantine_core()` at lines 61–67;
- `verify_bundle()` at lines 69–71;
- `publish()` at lines 73–80;
- `transaction_simulation()` at lines 82–107;
- production publication at line 184.

The seven exact, nonempty transaction outcomes cover a clean three-file commit, unchanged repeat rejection, stale-temp cleanup, pre-link failure cleanup, post-link interruption recovery, partial-publication quarantine, and immutable committed bytes. `main()` requires the exact key set and every value true (`preflight`, line 182).

The independent verifier implements its own `atomic_create()` and cleanup instead of importing the candidate writer. Its six-outcome TEMP suite covers clean output creation, repeat preservation, stale cleanup, pre-link cleanup, post-link recovery, and immutable bytes (`verifier`, lines 51–81). The final verifier output uses that same writer at line 168. Result adjudication requires the exact seven production transaction keys, and mutations reject empty, missing, extra and false dictionaries (`verifier`, lines 151–166).

## Typed CPU-slice state closure

- `state_bits()` proves `isinstance(state, str)` before membership and permits only `not_started`, `started_not_completed`, or `completed`; the validator cleanly catches invalid values (`preflight`, lines 27–34; verifier, lines 31–38).
- `main()` retains the correct transitions immediately around `preparation_summary()`: initial `not_started`, then `started_not_completed`, then `completed` only after return (`preflight`, lines 169–181).
- The exact Boolean mapping remains false/false, true/false and true/true. The current production and independent failure simulations cover all three states and preserve the complete R8P4 writer lifecycle (`preflight`, lines 109–155; verifier, lines 83–124).
- Unknown string, list, dictionary, integer, null, wrong started/completed Boolean, missing state and extra state are all rejected without throwing from `state_valid()` (`preflight`, lines 133–155; verifier, lines 103–124 and 155–166).

## Retained boundaries

- The Windows venv-launcher/base-interpreter dual identity and full native/original/application argv contracts are unchanged.
- Runtime, psutil/NumPy RECORD identities, 16-GiB start-RAM gate, frozen CPU preparation, 22 controls, five BF16 stage hashes, R7D1 failure evidence and R8P1 protocol-negative provenance remain transitively locked.
- Current and inherited AST/callgraph gates remain CPU-only and forbid model, compiler, OpenCL, CUDA and device entrypoints.
- Invalid application argv/token returns before filesystem mutation. Correct-token failures retain bounded create-new evidence with truthful CPU-slice state and preserve the primary exception.

## Exact authorized action

Run exactly once through the frozen venv launcher with `-I -B`, the absolute R8P6 preflight path, and ACK:

`PH1_INTEL_EXECUTION_R8P6_CONSOLIDATED_CURRENT_WRITER_CLOSED`

The full command vector must match the preregistration. Any invocation, topology, resource, preparation, transaction, failure-state or static-boundary mismatch remains a clean negative and must not be bypassed or retried within this freeze.

