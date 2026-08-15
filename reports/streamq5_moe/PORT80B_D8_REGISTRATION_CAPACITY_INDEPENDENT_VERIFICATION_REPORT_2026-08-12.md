# PORT80B-D8 independent CPU protocol audit

Verdict: **largest_clean_prefix_499_entropy_pin_below_capacity_raw_512_claim_invalid**. All replayable checks: **True**. No GPU code was executed.

## Corrected clean capacity

The preregistration requires immediate unregister after every successful registration. Applying that full lifecycle definition, the largest clean row is **499/512 experts per layer**, **48,563,159,040 bytes = 45.227966309 GiB (97.4609%)**. The raw JSON's **46.406250-GiB** largest-success claim is invalid because the 512 row contains **44**, not zero, raw unregister failures.

Important precision correction: the existing erratum report says all 48 unregister calls failed, but the immutable raw JSON contains exactly **44** failure strings. Four unregister calls therefore returned without a recorded exception. This does not rescue the row: any unregister failure violates the frozen clean-lifecycle requirement.

## EntropyPin arithmetic

The theoretical **41.441 GiB** size is below the clean observed capacity by **3.786966 GiB**. This establishes only arithmetic capacity plausibility. No compressed-bank artifact, decoder memory, decode time or working-set headroom was measured.

## Cumulative RAM caveat

Available RAM fell from **49.920 GiB** before the first row to **4.340 GiB** after the clean 499 unregister, a cumulative drop of **45.580 GiB**. The first four rows chain directly from the prior row's post-unregister state. This is not a set of independent cold-start capacity trials; page residency/cache/OS state accumulated. Therefore 499 is the largest clean point observed in this run, not a stable monotone knee or endurance guarantee.

## Provenance limitation

The raw result pins evaluator `fb011bfc98e0b61ecad53e7c472924cf5ca3bd8e20efc75ac1e4ae4bbfdc5a9e`, while the current repaired evaluator hashes to `5d8539bfb785c3baa8bb5ea7b22fd3d34510d0c1ca915ab28354ee7a44eb0332`. The original hashed evaluator is no longer present at that path. The preregistration, manifest, bank size and independently recomputed full-bank SHA-256 do match.
