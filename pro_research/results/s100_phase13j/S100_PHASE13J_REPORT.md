# S100 Phase 13J — cross-layer activation basis

This test compares separate origin-SVD bases for attention and MoE normalized
inputs with one shared basis over both families. It uses real calibration and
validation activations from the same checkpoint.

The screen measures only input residual energy. It does not compile separate
`W U` projections, measure amortized projection cost, or run a fused kernel;
promotion remains closed.

At rank 256, the shared basis raised attention residual energy from 0.329 to
0.429 and left MoE residual energy essentially unchanged (0.700 versus
0.700). At rank 512, attention was still worse with one shared basis (0.345
versus 0.267). The basis-sharing idea therefore saves projection setup only
at the cost of weaker input capture in this screen; no promotion signal was
found.
