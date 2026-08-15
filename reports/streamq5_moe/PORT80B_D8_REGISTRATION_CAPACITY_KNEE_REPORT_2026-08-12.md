# PORT80B-D8 — registration-capacity knee (erratum)

The raw evaluator marked a prefix `success=true` immediately after 48
`hostRegister` calls and only attached unregister errors afterward. That is too
weak for the preregistered success definition.

The largest **cleanly registered and unregistered** prefix is:

```text
499 / 512 experts per layer
45.227966 GiB
97.4609%
```

The 100% row is **not a pass**: 44 of the 48 unregister operations returned
`cudaErrorMemoryAllocation`. Any unregister failure violates the frozen clean
lifecycle gate, so its apparent 46.406-GiB success must not be cited.

The theoretical 41.441-GiB EntropyPin size is below the largest clean prefix,
so capacity is plausible. This does not test a compressed bank, entropy decode,
or stable long-duration registration. Available RAM also fell cumulatively
during the sweep, which limits cross-row independence.
