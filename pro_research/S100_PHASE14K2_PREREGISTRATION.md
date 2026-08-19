# Phase 14K2 preregistration

Candidate: same parent checkpoint with native BF16 substitutions.
Splits: calibration `_01`, validation `_02`; heldout untouched.

For each teacher-forced token:
- candidate top-64 list;
- candidate top1/top2 normalized margin;
- exact parent top1;
- exact parent score restricted to candidate shortlist.

K = 8/16/32/64.

Primary gate:
- validation K=16 exact-parent-top1 inclusion = 1.000.

Margin selection:
- candidate margin threshold selected on calibration only from q=0.50/0.75/0.90;
- choose largest fast fraction with calibration candidate-top1 agreement
  >=0.999 and K16 inclusion=1.0;
- validation fast fraction >=0.10;
- validation fast candidate-top1 agreement >=0.999;
- validation fast K16 inclusion=1.0.

No heldout is read here. Full-model native-BF16 heldout remains Phase14D2's job.
