# PORT80B-D10A2-R2 CPU-only preflight report

Verdict: **compile_preflight_pass_endurance_closed**.

Statically expected CUDA symbols: **16**. Device request for the separately authorized component run: **4.211 GiB**, plus a 512 MiB reserve. Registered host prefix in that run: **45.228 GiB**.

This CPU-only phase initialized no CUDA context, invoked no NVRTC compiler, launched no kernel, registered no host range, allocated no large device buffer, and did not scan the bank. The executable D10A2-R2 component gate is implemented; the 10,000-step executor remains closed until a clean component result and separate authorization.

Claim boundary: Compile/read-only preflight only; no component execution or endurance authorization.
