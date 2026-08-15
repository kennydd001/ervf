# NC7 cache-history and mutation erratum

The NC7 manifest supersedes NC6 only for cache/history and embedded-prefix records. It directly binds:

- five nominal directory entries, ordinal order and canonical tree-digest derivation;
- twelve exact snapshots: one before load, ten after the named NVRTC operations, one after release;
- literal environment map with four different destinations and exact restoration semantics;
- exact sentinel bytes/path/base64/SHA;
- executable cases for empty nominal tree; sentinel file; external write; missing root and each of four required directories; entry order/path/type/size/mtime/SHA; tree digest; extra entry; traversal; symlink; extra directory; history order;
- exact full-size/full-SHA and independently decoded 4,096-byte prefixes for log/PTX/CUBIN over-cap cases.

`filesystem_observation.cache_tree` has exactly `private_root,environment,environment_original,environment_restored,history,history_digest`. `history_digest` is SHA-256 of canonical compact UTF-8 JSON for the ordered twelve rows. Each row's `tree_digest` is SHA-256 over its ordinal entry lines `path NUL type NUL decimal_size NUL decimal_mtime_ns NUL sha256-or-empty LF`. The manifest's fake directory mtimes are zero; physical mtimes are retained exact integers and independently recomputed, not compared to zero.

Positive and negative manifests contain no cache file because nominal permits none. The exact empty directories are retained as directory topology and the complete cache observation lives in result/negative JSON. Any sentinel or other file causes incidental failure; that failure JSON retains the offending entry metadata and, if <=4 MiB, raw file bytes/base64/SHA. External paths are never opened to collect bytes after the guard detects them.

The shared import-inert `compile_contract.py` remains mandatory. Preflight calls its actual topology, recovery, transaction, failure and terminal functions with injected filesystem/fake API. Each cache case mutates one production input and must produce the literal manifest disposition. A copied/toy predicate is rejected by function module/path/SHA/qualified-name/code-digest identity.

All NC6 schemas, caps, ABI, authorization, no-payload/no-Driver boundary and postcommit durability adjudication remain directly bound. NC7 authorizes no execution.

