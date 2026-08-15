# Agent prompt — ERGV exact reduction graph compiler

Open `ERGV_COMPILER` independently from the model-port work.

## Goal

Compile an exact logical floating-point reduction DAG into alternative CUDA
thread topologies without changing any ordered arithmetic operation.

## Required IR

Represent:

- logical accumulator IDs;
- ordered add edges;
- cast/round points;
- FMA policy;
- source load order;
- virtual-to-physical lane mapping.

## Search space

- subwarp widths 4/8/16/32/64;
- rows per block;
- virtual accumulators per lane;
- vectorized code loads;
- scale broadcast;
- activation staging;
- register cap and occupancy.

## Correctness

First prove graph isomorphism mechanically. Then require bit equality on
random, adversarial and real model inputs.

## Baselines

- original P6B kernel;
- manual P7 ERVF;
- N1C autotuner;
- equivalent GemLite/CUTLASS/QUICK kernels where semantics match.

## Gates

- reproduce manual P7;
- beat manual P7 on at least one matrix family;
- exact on Q5 and Q8;
- second model shape;
- second GPU architecture;
- report generated code, search cost and a predictive performance model.

No novelty claim before the external-baseline and second-architecture gates.
