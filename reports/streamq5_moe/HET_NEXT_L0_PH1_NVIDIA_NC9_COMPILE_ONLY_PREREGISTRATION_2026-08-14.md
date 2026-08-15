# HET-NEXT-L0 PH1 NVIDIA NC9 compile-only preregistration

Status: immutable design-only. Implementation, preflight, NVRTC, compiler, payload and device execution remain closed. NC9 supersedes NC8 only for the three bounded defects in independent audit SHA `25fb25546e6c56c8d3bc96b79112c45be4c751371703c8b25af5016781e63c8c`.

The normative fixture-manifest cap is exactly 8 MiB (`8388608` bytes). This supersedes only the former 4 MiB fixture-manifest cap. Source, log, PTX, CUBIN, JSON, result, failure and total-bundle caps remain unchanged. Freeze requires the complete NC9 manifest to be nonempty and at most 8 MiB.

Inherited topology explicitly requires `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc8_durability_adjudication` to be absent. It is neither an allowed terminal nor recoverable output. File, directory and temporary-prefix appearances are separate rejection fixtures.

The sole future shared module is `scripts/streamq5_moe/het_next_l0_ph1_nvidia_nc9_compile_contract.py`, stdlib-only and import-inert. It exports exactly `capture_environment`, `apply_private_environment`, `restore_environment`, `classify_topology`, `recover_inprogress`, `publish_transaction`, `write_incidental_failure`, `adjudicate_terminal`. Runner and preflight import identical function objects with identical `__code__` identities from the exact bound module. Copies and monkeypatches are forbidden.

The call order is capture, apply, ten NVRTC calls, restore, publish. Publication before complete reverse restoration is forbidden. Static preflight must inject mappings into these production functions and execute every one of the 50 NC8 environment fixtures, including partial and secondary restoration failures, plus copy, monkeypatch, order and early-publish mutations.

All NC8 cache, history-digest, artifact, ABI, ledger, terminal and transaction requirements otherwise remain exact. No run is authorized.
