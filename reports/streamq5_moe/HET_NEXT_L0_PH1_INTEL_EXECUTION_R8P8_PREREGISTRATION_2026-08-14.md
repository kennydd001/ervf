# PH1 Intel execution R8P8 — explicit topology repair

Date: 2026-08-14  
Status: immutable closed preregistration; no execution authorization.

R8P8 supersedes only the R8P7 topology-depth defect identified by independent audit SHA-256 `00de2f823af2a2c1f10dc8aa2239ccdc7de20ffca1cfe07a6bfe6bccf98e03fc`. R8P7 was never executed and remains immutable. Its local-entry repair is retained unchanged.

The R8P8 preflight imports the frozen R8P1 through R8P6 modules by their exact full names. One canonical explicit topology comprises: six R8 base paths from R8P1; six paths each for R8P1 through R8P5; five absent R8P6 paths excluding its permitted failure root; and six R8P7 paths. These 47 ancestor paths must be pairwise distinct. Six fresh R8P8 result/verifier/failure/quarantine paths are separately absent. The only allowed negative is the exact one-file R8P6 failure SHA-256 `03e48ed76dd848f0c1e993f8452245917115b1b8fb22596871dd933e4758b372`.

Static/TEMP contract tests remove and duplicate a path from every R8P1–R8P7 revision group and substitute a wrong chain-depth path; every mutation must be rejected. Module identities, group names, cardinalities, union size, and path identities are exact. The independent verifier constructs the same expected topology from literal paths without importing the R8P8 candidate.

R8P7 local current-module identity primitives, mutations, current R8P8 writer/TX/failure functions, typed CPU-slice state, dual runtime identities, RECORDs, 16-GiB RAM, frozen CPU preparation, controls, hashes, R7D1/R8P1 provenance, and no-device boundary are unchanged. R8P8 is closed/PENDING. No model, compiler, OpenCL, CUDA, payload expansion, or device action is authorized.
