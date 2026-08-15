# HET-NEXT-L0 PH1 NVIDIA NC10 compile-only preregistration

Status: immutable design-only; implementation, preflight, NVRTC, compiler, payload and device execution are closed. NC10 supersedes NC9 only for the three bounded defects in independent audit SHA `39bb2df92aab90272a915b9fbc9fef40791ac6bb1db79b3654a8127f8b860bad`.

Fresh topology requires exact absence of `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc9_static_preflight_failures`, `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc9_static_preflight_quarantine`, and `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc9_durability_adjudication`. None is an allowed terminal. File/directory shapes are rejected; NC9 durability also rejects canonical-file, directory, orphan and over-cap shapes. Historical NC8 durability absence remains independently frozen.

The future shared NC10 contract adds `load_fixture_manifest_bounded`. It stats before open, rejects zero and sizes above `8388608` without opening, reads exactly the stat size once, then performs one UTF-8 JSON parse. Fixtures are exact: 0 bytes rejects with 0 bytes read; 8,388,607 and 8,388,608 bytes are a valid literal JSON prefix followed only by ASCII spaces, are read completely and accepted; 8,388,609 rejects with 0 bytes read. Their exact SHA-256 values and counters are normative in the manifest.

All NC9 environment, history, shared-function, transaction, output-cap and compile-only requirements remain unchanged. No run is authorized.
