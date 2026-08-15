# PORT80B-D10A1-R conservative resource-stop report

Verdict: **component not started — conservative hard RAM stop**.

The final conservative compile/preflight passed, including both independent
resource-audit locks and the exact 52,652,163,072-byte start formula. Immediately
before launching the runner process, psutil reported 52,800,036,864 bytes, only
147,873,792 bytes (141.023 MiB) above the gate. After importing the runner and
CUDA context, its authoritative immediate check reported 52,026,593,280 bytes,
which is 625,569,792 bytes (596.590 MiB) below the gate. It therefore stopped.

No host range was registered, no large device buffer allocated, no kernel
launched, no bank page scanned, and no endurance phase entered. There were zero
ranges to unregister and cleanup is trivially clean. The compile, runner,
preregistration, source audit and conservative budget audit remain hash-locked.

A retry needs materially more than 52,652,163,072 bytes in the outer check so
that interpreter/CUDA-import overhead cannot consume the entire margin before
the authoritative in-runner check. This report does not authorize a retry.

Claim boundary: resource-stop evidence only; no component correctness,
first-touch, performance, endurance, or model result.
