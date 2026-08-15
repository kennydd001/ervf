# PH1 Intel execution R7 — independent verifier row-write correction

Status: closed/PENDING. No preflight, payload, compiler, OpenCL load, or device
call is authorized.

R7 makes one scientific-code correction: in the standalone independent CPU
verifier, `out[row]=rb(lanes[0])` is inside the `for row` loop. The reduction
tree, integer FMA, BF16 rounding, codec and all production implementation remain
unchanged.

The static preflight must exercise both production linear shapes with 512 and
2048 nonzero rows. Row `i` alternates exact BF16 words `0x3f80` and `0x4000` in
column zero, input word zero is `0x3f80`, and all other words are zero. Each
output must equal its row's selected word. Two complete evaluations per shape
must be byte-identical and both SHA-256 digests must equal the independently
constructed expected-byte digest. The real full-shape verifier baseline and all
28 named mutations remain mandatory.

Runner changes are mechanical provenance/output-path/kind changes only. R6
backend and common files are reused byte-identically. R7 binds the R6P1 15/16
result, R6P1 source audit and negative diagnosis, as well as the complete prior
package chain.
