# PH1 NVIDIA NC7 compile-only preregistration

Status: **design-only; implementation and execution closed**.

NC7 retains the complete NC6 compile-only contract and changes only cache evidence and embedded over-cap evidence. The normative matrix is `reports/streamq5_moe/het_next_l0_ph1_nvidia_nc7_fixture_manifest.json`, with 314 unique cases.

After authorization, the fresh private root is exactly `private_tree`. Before compiler load it contains exactly four distinct precreated empty directories: `cuda_cache`, `tmp`, `temp`, `nvrtc_cache`; no alias, reparse point or file exists. Environment mapping is exactly `CUDA_CACHE_PATH=private_tree/cuda_cache`, `TMP=private_tree/tmp`, `TEMP=private_tree/temp`, `NVRTC_CACHE_PATH=private_tree/nvrtc_cache`; `CUDA_CACHE_DISABLE=1`, `CUDA_CACHE_MAXSIZE=0`. Original presence/value for all six variables is captured before mutation and restored exactly in `finally` (originally absent means delete; present means exact restore). Restore failure is terminal incidental failure.

Every snapshot includes sorted entries for root `.` then `cuda_cache`, `nvrtc_cache`, `temp`, `tmp`. Entry fields are exactly `path,type,size,mtime_ns,sha256`; directories use normative size 0 and SHA null while retaining observed mtime. Snapshot history is exactly 12 ordered rows: `pre_load`, after each of the ten NVRTC operations in ledger order, `post_release`. Every row contains `index,stage,entries,tree_digest`. Nominal requires only the five directory entries and no files. Any file, symlink/reparse, traversal, extra/missing directory, alias, external write, order/field/digest/history drift is incidental-invalid.

The only file sentinel is raw ASCII `NC7_CACHE_SENTINEL` followed by one NUL: 19 bytes (18 non-NUL ASCII plus terminal NUL), base64 `TkM3X0NBQ0hFX1NFTlRJTkVMAA==`, SHA-256 `5d7bfc7021fa3b29532e4cb32c29eb6fc5f6ad165d602eb76afda433c29d916f`, relative path `private_tree/nvrtc_cache/cache.bin`. It is mutation input only and must be rejected, never nominal evidence.

Over-cap embedded prefix objects now contain exactly `source,offset=0,length=min(4096,full_bytes),base64,sha256,derivation,full_bytes,full_sha256`. `full_bytes` is the applicable cap plus one. The full stream is the exact baseline raw artifact repeated cyclically to that length; verifier reconstructs full SHA and the first 4,096 bytes independently. No prefix file or null file field exists.

NC6 terminal/debris/postcommit/shared-contract rules are unchanged. No implementation, preflight or compiler/device call is authorized.

