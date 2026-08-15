# PH1 Intel execution R7D — final authorization-only revision

Date: 2026-08-14

R7D changes authorization only. It reuses the immutable R7A physical computation and the independently audited R7C2 delegated-return/failure lifecycle. Before any filesystem mutation, recovery, payload read, backend construction, OpenCL load, allocation, or launch, R7D requires:

- exact R7C2 closed-preflight SHA `de8745c02cd0b2951adbb04338cf350704608023530edd91a260b73880ebcd8c`, exact schema, exact nine named checks all true, PASS 9/9, all nine delegated cases true, all three device-state cases true, exact eight rejected mutations, exact clean-state evidence, and no payload/compiler/device;
- exact R7A authorization-preflight SHA `a5b8e70cd40e241e16a250347cf06258a6540100f40423bc7216cb3639191265` and PASS 7/7 contract;
- exact R7P SHA `e10c513fdbecb27e08319c462ba1d1020b1c94c4ff5d9199047ae513197dd959` and full PASS 18/18/sentinel/mutation contract;
- exact R7D open lock/token and full R7C2→R0 source/evidence hash chain;
- current absence of all R7A/R7D output, failure, quarantine, verification, and matching in-progress paths.

The retained physical result includes the complete R7D authorization evidence. The standalone R7D verifier independently reconstructs this contract before importing the hash-bound frozen R7A numerical verifier. R7C2 failure summaries are stored under R7D failure/quarantine paths without semantic changes.

R7D remains NO-GO for physical execution until an independent audit explicitly authorizes exactly one attempt. Claim remains one real expert/input Intel correctness component only.
