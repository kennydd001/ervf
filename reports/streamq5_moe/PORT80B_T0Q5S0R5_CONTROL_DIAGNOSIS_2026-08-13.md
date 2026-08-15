# T0Q5-S0-R5 control diagnosis

The independent verifier reproduced every scientific and provenance component of S0-R5 except the conjunctive control gate. Exactly one of twenty expanded control assertions was false.

- `p0`, position `8`, shared-down selected field `(row=0,column=0)`: packed q changed `6 -> 5`; original and presented digests differ; the metadata/digest checker rejected; independent replay equals the retained unsafe arrays; nevertheless the BF16 shared-raw and gated outputs changed in `0/2048` words because this one contribution rounded away.
- The identical mutation at `p0`, position `15` changed one shared-raw and one gated BF16 word.
- All sixteen wrong-expert/projection-swap controls changed between `2035` and `2047` BF16 words.

Therefore R5 remains formally verifier-negative under its frozen control gate. This is not a Q5 quality negative or corrupted result. The selection predicate (`q != 0 and activation != 0`) did not guarantee an observable BF16 output change. A new preregistered control-only sentinel is required; the 759-matrix numerical-quality computation need not be rerun.
