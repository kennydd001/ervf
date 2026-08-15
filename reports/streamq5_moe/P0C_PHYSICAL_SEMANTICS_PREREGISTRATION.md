# STREAMQ5-MoE P0C - physical-scale semantics corrective preregistration

Locked on 2026-08-12 after the scale-semantics audit and before any P0C output.

## Reason for correction

P0 implemented code selection and dequantization with an FP32 temporary scale,
then rounded the dequantized weight to BF16. The declared physical format stores
BF16 scales. A locked five-matrix audit found 1,090,972/7,864,320 resulting BF16
values differ when the scale is rounded before dequantization. P1B was stopped
before producing any bank record.

## Corrected candidate

For Q5 experts and INT8/INT4 trunks, calculate `max(abs(group))/qmax` in FP32,
select codes by round-to-nearest-even against that FP32 scale, store the scale
as BF16, and materialize the candidate weight as `code * float(BF16_scale)`
rounded to BF16. This exactly matches the intended physical decoder. All other
P0 architecture, variants and layer coverage remain fixed.

Validation/test each use two new 128-token contexts in five domains and must be
disjunct from CORETAIL P2, STREAMQ4 P0 and original STREAMQ5 P0. Hashes and the
corrected evaluator are locked before output.

Validation progression requires primary `Q5 experts + INT8 trunk` relative CE
`<=2.5%`; final pass requires validation and once-only test both `<=2%`. No
repair or quantizer sweep is allowed. A pass supersedes original P0 for the
physical-bank claim and forces fresh candidate routes before cache confirmation.
