# PORT80B-D10B held-out 10,000-step CPU-only preflight report

Verdict: **compile_preflight_fail**.

Statically expected CUDA symbols: **0**. Device request for the separately authorized component run: **4.211 GiB**, plus a 512 MiB reserve. Registered host prefix in that run: **45.228 GiB**.

This CPU-only phase initialized no CUDA context, invoked no NVRTC compiler, launched no kernel, registered no host range, allocated no large device buffer, and did not scan the bank. The exact 10,000-step executor is implemented but remains closed until a separate explicit GPU authorization.

Claim boundary: Compile/read-only preflight only; no component execution or endurance authorization.
