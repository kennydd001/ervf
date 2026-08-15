# PORT80B-D10B — held-out 10,000-step endurance preregistration

Date: 2026-08-13  
Current state: design frozen; preflight and physical execution prohibited until
the named independent audit explicitly clears this source.

## Immutable authorization input

D10B is downstream only of the clean D10A2-R2 component pass. The complete
D10A2-R2 chain is immutable:

- runner `8bce44334d9d416ad53e7f8499b676133e21248964dee79bccd16cc04f65cf8c`;
- preregistration `46cffb32de3228b30ff6b45003bdc288c39d11d02857a00409334e29a980022a`;
- conv unit test `02dfa87cee8ac58b54a8b71656d109a55d96298db659228c2726f447214f3650`;
- conv unit JSON `ba7c398facaaa88b46ad95ec020bd031fed324755ed0bf7550af0c63ba9941c1`;
- CPU preflight JSON `eba995a770b8671f05fdea9c4fd593c9eacf04f05f26cb82ba343b5a0afb160c`;
- CPU preflight report `f7b2ceb9edc4ae06f2741fffc4010211348945c0fb5fc6bb0f7510e447d6349a`;
- component JSON `cd4486221dae9073a14a7e0d617c803120f7f3e094580559c81d9035111063b1`;
- component report `a8388d54b4e0b5e76e2eec6a28f9d7cd3ef7e9e858d2051a8cdcd2cf104e601b`.

The component result must remain `overall_pass=true`, all 28 gates true,
`error=null`, 48/48 clean unregister, and
`endurance_authorized_by_evidence=false`. This D10B preregistration is the new
authorization boundary; it does not broaden the component claim.

## Frozen held-out route stream

The route label remains `p4d_shaped_synthetic_proxy`. D10B uses only the frozen
`endurance_source=[768,1024)` route partition, disjoint from correctness
`[0,8)` and validation `[512,576)`. Cases are generated in this exact nested
order:

1. epochs 0 through 7;
2. domains `general`, `code`, `math`, `multilingual`, `instruction`;
3. source tokens 768 through 1023;
4. the existing frozen `lift(route, domain_index, token, epoch)` transform;
5. truncate after exactly 10,000 emitted cases.

The resulting route SHA-256 must be
`85f12fb0020bb8568dfc3683662e8251b29bf83684beb296dbb6d8734f5ffd20`;
maximum cold records per step must be 31, within the frozen 32-slot cold escape.
Eight first-touch warm-ups use the first eight held-out cases, after which the
full exact 10,000-case stream is measured from step 0. Warm-ups are not counted.

## Frozen physical mechanism

D10B copies the locked D10A2-R2 data path and numerical kernels byte-for-byte:
499 registered records per layer, 13 possible cold experts outside the prefix,
exact Q5 routed and shared compute, attention, GDN, dense shell and composed
state. A single explicit non-blocking stream owns all allocation,
initialization, upload, reset, table/counter construction, kernels and timing.
Host reads occur only after stream synchronization.

The runner stores exactly 48 registration-attempt rows and 48
unregister-attempt rows. Every row must be attempted, successful and error-free.
All physical cleanup runs in `finally` after a stream synchronization attempt.

## Frozen resource and emergency gates

- available RAM before allocation/registration: at least 52,652,163,072 bytes;
- exact device request: 4,521,569,280 bytes;
- free VRAM after allocation and throughout all measured steps: at least
  536,870,912 bytes;
- available RAM after registration and after eight first-touch warm-ups: at
  least 2,147,483,648 bytes;
- emergency check before every measured step: at least 1,610,612,736 bytes;
- measured-run available-RAM loss, first telemetry row minus last: at most
  1,073,741,824 bytes;
- every post-warm-up hard-page-read sample: at most 2,048 reads/s;
- null CUDA/runner error and zero unregister failures.

Any hard-stop failure makes the run negative and leaves no partial pass claim.

## Exact timing, telemetry and state evidence

The runner records raw inclusive wall latency and CUDA-event latency for every
one of the 10,000 measured steps. Hard gates:

- exactly 10,000 finite positive values in each latency vector;
- wall p95 at most 150 ms and wall p99 at most 200 ms;
- last-1,000 wall p95 divided by first-1,000 wall p95 at most 1.20;
- all 10,000 telemetry rows contain available RAM, free VRAM and process memory;
- finite composed state at every step (checked on-device through the existing
  output path and at every digest checkpoint); no required poison sentinel.

After the measured end event and outside that step's wall timing, state/output
SHA-256 evidence is captured at steps 0, 99, 199, ..., 9,999: exactly 101
checkpoints. Each checkpoint stores shape, dtype, finite flag, poison count and
SHA-256 for exactly these nine arrays: `routed_capture`, `routed_down`,
`shared_down`, `attention`, `delta`, `kv_state`, `recurrent_state`,
`conv_state`, `composed_state`. Thus exactly 909 array-evidence records are
required. All must be finite, have a 64-hex digest and have zero poison count.
All 101 composed-state digests must be unique.

Checkpoint copies are explicitly outside recorded latency but remain part of
wall-clock endurance and resource telemetry. No checkpoint cadence or gate may
be changed after preflight.

## Staged authorization

1. This turn may create only this preregistration and the runner source.
2. No Python preflight, CUDA initialization, NVRTC compile, host registration,
   allocation, kernel launch or bank scan may occur until an independent
   `new_pack_audit` explicitly accepts the D10B design/source.
3. After audit acceptance, a separate CPU-only preflight may lock the runner,
   preregistration, audit and immutable inputs. It must still perform no CUDA or
   bank action.
4. A further explicit component/endurance GPU go is required after that clean
   preflight. No automatic execution is authorized.

## Planned evidence paths

- runner: `scripts/streamq5_moe/run_port80b_d10b_heldout_10000_endurance.py`;
- future CPU preflight JSON:
  `reports/streamq5_moe/port80b_d10b_heldout_10000_endurance_preflight.json`;
- future preflight report:
  `reports/streamq5_moe/PORT80B_D10B_HELDOUT_10000_ENDURANCE_PREFLIGHT_REPORT_2026-08-13.md`;
- future raw result:
  `reports/streamq5_moe/port80b_d10b_heldout_10000_endurance.json`;
- future report:
  `reports/streamq5_moe/PORT80B_D10B_HELDOUT_10000_ENDURANCE_REPORT_2026-08-13.md`.

## Claim boundary

Even a pass proves only 10,000-step stability of this synthetic,
shape-informed component/composition mechanism on held-out P4D-shaped proxy
routes and uniform synthetic Q5 payloads. It is not a real checkpoint, natural
route trace, output-quality proof, production throughput result or industrial
breakthrough.

