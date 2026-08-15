# PORT80B-D10A1 component/composition compile report

Verdict: **compile_preflight_pass_endurance_closed**.

Resolved CUDA symbols: **16**. Device request for the separately authorized component run: **4.211 GiB**, plus a 512 MiB reserve. Registered host prefix in that run: **45.228 GiB**.

This phase compiled only. It launched no kernel, registered no host range, allocated no large device buffer, and did not scan the bank. The executable D10A1 component gate is implemented; the 10,000-step executor remains closed until a clean component result and separate authorization.

Claim boundary: Compile/read-only preflight only; no component execution or endurance authorization.
