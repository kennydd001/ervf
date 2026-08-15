# HET-NEXT-L0-PH0-R3 single real projection — preregistration

Date: 2026-08-13  
State: **final immutable design; implementation, preflight and execution closed**

This revision incorporates PH0-R2 in full by immutable reference. PH0-R2 preregistration SHA is `5454638b227fcabee6924cc545604d7ff4e1150f090ab26621cf557fc3b5b70b`; PH0-R2 design SHA is `f5e2b6336d16843912c96f9239bb077d406795870bbf595ac131a687c49cf5b7`. Every R2 clause remains normative except the checker order and wrong-identity evidence below. No scope, payload, arithmetic, threshold, device, resource, call count, mutation selector, synthetic witness or claim changes.

## Sole semantic repair: safe-checker order

The one safe dispatch checker has exactly this global order:

`exact byte size → header structural schema → CRC → exhaustive field scan (all fields <=30) → frozen source identity plus pristine codes digest plus pristine scales digest → requested identity → frozen input identity/digest → canonical full-record digest → dispatch`.

Clarifications:

- “header structural schema” validates magic/version/layer/projection/bits/rows/columns/group/code bytes/scale bytes/reserved bytes. It parses expert but does not compare expert to the request.
- CRC covers codes/scales and therefore does not change when only the expert header field changes.
- Source/codes/scales gates bind the official expert-50 source and its pristine payload, independently of header expert.
- Requested identity is the exact tuple `(layer=0, expert=50, projection=gate, shape=[512,2048])`.
- Canonical full-record digest is checked only after requested identity and input identity. There is no digest override or per-control expected-record substitution.
- Field31 with recomputed CRC therefore rejects at field scan before payload digests; payload corruption with recomputed CRC rejects at codes/scales digests; wrong identity with pristine payload rejects at requested identity before the inevitably different full-record digest.

## Frozen wrong-identity control

Starting from the pristine R2 record, mutate only the header expert field `50→51`; keep the pristine codes, scales, padding and CRC `1,976,639,022` unchanged. The exact mutated evidence is:

- header SHA-256 `378a33921294284be5dc5c632bac8da8203a7f9678440d8c3d60a3081b2a754f`;
- full record SHA-256 `90847ea93476c19b9e1dc934e892055f2744e655930a6caa693136c7c3fe4758`;
- codes SHA-256 remains `20399f2cabbc0adc1e4c02866e0894df2642342b95dc5c63e9b971d58c19ed6b`;
- scales SHA-256 remains `658d43f3085c4b98ac4a64ede92143068ce13f91ebd30693e43e7945ddfd53e8`.

It must pass exact size, structural schema, CRC, exhaustive field scan, source identity and pristine codes/scales digests; it must then reject specifically `requested_identity` with both device submission counters still zero. The canonical full-record check and dispatch must not execute for this control.

All other R2 controls retain their exact bytes, hashes, ordering outcomes and deterministic selector/witness. The positive conjunction and narrow validation-only claim remain exactly R2.
