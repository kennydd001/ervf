# S100 Phase 13H — native datatype block screen

This screen adds native datatype checks that were not covered by the BF16
Mamba test: BF16 attention Q/O, one FP8 Mamba matrix, and one NVFP4 Mamba
matrix. The runtime custom kernel is measured as the baseline. PyTorch native
matrix multiplication is used only where the installed CUDA/PyTorch stack
supports the datatype.

This is a capability/component screen, not an end-to-end speculative runtime:
there is no heldout quality, routing-margin gate, or full B=4/B=8 generation
test. Promotion remains closed regardless of native capability.

Native BF16 attention Q and O both ran successfully and measured about 6.24x
over the custom row-wise kernel at B=4. This checkpoint had no FP8 Mamba
matrix in the selected runtime population, so no FP8 case was available. The
NVFP4 path remained packed (`x2`) and PyTorch could not consume it as a normal
matrix; its native capability is therefore unproven, while the existing fused
NVFP4 baseline did run.
