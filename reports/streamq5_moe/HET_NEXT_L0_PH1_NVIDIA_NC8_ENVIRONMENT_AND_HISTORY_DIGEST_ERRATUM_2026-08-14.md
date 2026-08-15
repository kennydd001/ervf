# NC8 environment and history-digest erratum

## Normative manifest

`reports/streamq5_moe/het_next_l0_ph1_nvidia_nc8_fixture_manifest.json` is normative. It directly binds the future NC8 shared contract and retains the NC7 literal cache sentinel, tree entries, twelve stages and over-cap full-stream evidence unchanged.

The environment protocol has exactly six fields, one ordered capture, one ordered replacement, and one reverse restoration. Scalar replacements are `CUDA_CACHE_DISABLE=1` and `CUDA_CACHE_MAXSIZE=0`. Path replacements are normalized absolute descendants of the create-new staging root and may not alias. Original absence is represented only by `present=false,value=null`; original presence is `present=true,value=<exact string>`. Publication occurs only after restoration completes. The manifest includes, per field, original-absent, original-present, wrong replacement, alias, set-failure and both absence/presence restoration-failure fixtures, plus swap, outside, missing, extra, preauthorization, partial and secondary-error fixtures.

Each cache fixture stores the digest of its exact twelve-row history. Canonical JSON recursively sorts object keys, preserves array order, uses compact comma/colon separators, UTF-8, no ASCII escaping and rejects NaN. The frozen nominal digest is recorded in the schema. Executable future fixtures mutate row order/content and digest missing/value/type/extra independently; the production shared contract must reject each.

NC7 rules for four distinct precreated directories, no nominal file, sentinel `NC7_CACHE_SENTINEL\0`, cache file/symlink/traversal/extra-entry rejection, complete postcommit evidence, transaction durability, terminal classification and over-cap `full_bytes/full_sha256` remain normative without expansion.
