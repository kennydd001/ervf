# PORT80B-T0Y-R canonical router include-path revision

Date: 2026-08-13

This revision inherits every T0Y hypothesis, input, arithmetic operation, gate, resource limit, no-retry rule and claim boundary. The first T0Y attempt stopped during NVRTC compilation before a kernel launch and before creating any output because CuPy did not automatically locate CUDA headers.

The sole execution repair is adding the existing bundled directory `.venv/Lib/site-packages/nvidia/cu13/include` as an explicit NVRTC `--include-path`. This is the same local CUDA include source already used by the independently verified D5-D10 runners. T0Y-R uses create-new source, lock and output paths. No arithmetic source, block shape, selector, threshold or input changes.
