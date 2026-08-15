# PORT80B-D10A2-R2 report erratum

Date: 2026-08-13

The immutable Markdown component report (SHA-256
`a8388d54b4e0b5e76e2eec6a28f9d7cd3ef7e9e858d2051a8cdcd2cf104e601b`)
prints `Endurance evidence-authorized: True` because its report formatter
interpolated the component `overall_pass` boolean. That sentence is incorrect.

The canonical raw JSON (SHA-256
`cd4486221dae9073a14a7e0d617c803120f7f3e094580559c81d9035111063b1`)
stores `endurance_authorized_by_evidence=false`; the locked runner makes its
endurance phase unconditionally raise; and the preregistration requires a new
post-component preregistration and runner. Therefore the corrected reading is:

> Component pass: True. D10A2-R2 endurance authorization: False/closed. The
> component evidence is eligible only as an input to a new separately
> preregistered endurance arm.

The independent CPU verification confirms this boundary with 26/26 checks:

- verification JSON SHA-256:
  `409c379600b733bc466b21c981f75342d6087612f3c60f6e7c4889f31828ab6d`;
- verification report SHA-256:
  `a02e1caa978e27a6cf717629128ec6d57292232d36a351934c48e6dba81efbaa`.

No locked raw result, runner, preregistration, preflight or original Markdown
report was edited.

