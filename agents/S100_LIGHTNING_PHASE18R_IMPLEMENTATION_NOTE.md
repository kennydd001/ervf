# Phase 18R implementation note

The runtime's `step_graph()` launches the graph object currently stored in `rt._graph`; therefore parent and oracle graphs can coexist on one runtime by preserving the original graph object and switching only that Python reference between prompt and target phases. The production parent graph keeps its captured MoE/cache pointers. The target graph is re-captured after installing the replay/surgical wrapper on the same buffers.

Phase 18R additionally removes per-token replay-index staging. The record/replay CUDA kernels read the existing device `_pos_dev` and a single per-prompt `table_offset_dev`, so target row selection is graph-native and cannot race a host staging slot.

No snapshot-based prompt restore is allowed in measured roles. `_reset_exact_state` is followed by the untouched parent graph over the whole prompt, which rebuilds the canonical expert-cache/LRU state before switching to a target graph.
