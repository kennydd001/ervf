# PORT80B-D10A1 component resource-stop report

Verdict: **component not started — hard RAM stop**.

The authorized component command reached its first preregistered resource gate
and stopped before any physical action. Available system RAM was
50,252,566,528 bytes (46.801 GiB), below the required 53,687,091,200 bytes
(50 GiB) by 3,434,524,672 bytes (3.199 GiB). GPU memory was not the blocker:
7,899 of 8,151 MiB was free.

No host range was registered, no large device buffer allocated, no CUDA kernel
launched, no bank page scanned, and no endurance phase entered. Consequently
there were zero ranges to unregister and cleanup is trivially clean. The
hash-locked preregistration, runner, and passing compile/preflight evidence were
not changed after the attempted start.

The D10A1 component can be retried only when the same immediate pre-registration
check observes at least 53,687,091,200 available bytes. Its locked device
envelope remains 4,521,569,280 bytes (4.211 GiB) plus a 512 MiB reserve; if it
opens, it will register exactly 48 × 499 records (45.228 GiB).

Claim boundary: this is resource-stop evidence only. It establishes no
component correctness, first-touch, validation timing, endurance, or model
result.
