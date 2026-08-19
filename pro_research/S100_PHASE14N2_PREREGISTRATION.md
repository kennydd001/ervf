# Phase 14N2 preregistration

Reference branch: `origin/pro-s100-nativefp4-c2b`.

Required evidence:
1. checkpoint format audit is exact group-16 packed;
2. lossless C1 repack evidence is present on the reference branch;
3. C2D rerun is measured on target SM120;
4. all four format-preserving shapes have >=4x-L2 rotation;
5. M4/M1 <=1.30 for at least 3/4 shapes;
6. M8/M1 <=1.45 for at least 3/4 shapes.

Only then:
`NATIVE_NVFP4_C3_RUNTIME_BUILD_OPEN=true`.

C3 means: consume real repacked Lightning weights and real activations through
the native FP4 path and run numerical/official quality. N2 itself is not C3 and
cannot claim S100.
