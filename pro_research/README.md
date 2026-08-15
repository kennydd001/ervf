# PRO research pack

Additive breakthrough probes for the current Nemotron 3.5 Lightning / ERVF
runtime. This branch starts from commit
`96811c4e381bf788f9133f5d1fc025e6885cf78f` and does not overwrite existing
reports, registries, runtime files, or protected manifests.

The source pack is stored as a SHA-256 checked split Base64 payload because the
GitHub connector cannot upload a binary archive directly. Installation expands
the complete, human-readable source tree into this directory.

## Install

From the repository root in PowerShell:

```powershell
git fetch origin
git switch pro-research
.\pro_research\INSTALL_AND_RUN.ps1 -Mode install
```

## Run

Technical smoke:

```powershell
.\pro_research\INSTALL_AND_RUN.ps1 -Mode smoke
```

Full dependency-aware sequence:

```powershell
.\pro_research\INSTALL_AND_RUN.ps1 -Mode full
```

Individual tracks:

```powershell
.\pro_research\INSTALL_AND_RUN.ps1 -Mode graph
.\pro_research\INSTALL_AND_RUN.ps1 -Mode dense
.\pro_research\INSTALL_AND_RUN.ps1 -Mode epoch
```

All generated data stays under:

```text
pro_research/results/
```

Previous result files are archived automatically in
`pro_research/results/history/`.

## What it tests

1. The already-built but unmeasured E1F22 full-token CUDA graph.
2. A missed generalisation of ERVF to the resident BF16 and FP8-tensor GEMVs.
3. End-to-end integration of generalized ERVF when the exact microbenchmark
   passes.
4. Exact K-token epoch graphs that amortize graph launch/readback without any
   speculative model or changed target semantics.

Read `PRO_HYPOTHESES.md` before running. After installation, the expanded
`README.md` contains the exact commands, frozen gates, and interpretation rules.

No performance claim is made by this commit. The code was syntax-checked and
its CPU reduction-tree selftest passed; the GPU experiments must be run on the
target laptop.
