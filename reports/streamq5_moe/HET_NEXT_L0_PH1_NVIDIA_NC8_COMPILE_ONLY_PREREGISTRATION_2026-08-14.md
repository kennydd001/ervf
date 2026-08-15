# HET-NEXT-L0 PH1 NVIDIA NC8 compile-only preregistration

Status: immutable design-only; implementation, preflight, NVRTC, compiler, payload and device execution are closed. NC8 supersedes NC7 only for the three defects in the NC7 independent audit SHA `6392429f119ef0d16a6aebb3601e274b449bfa73c52e48f7bd98f1e5c6fb6b80`. Every other NC7/NC6 scientific, ABI, ledger, artifact, lifecycle and terminal rule is unchanged.

The sole future shared stdlib-only, import-inert production contract is exactly `scripts/streamq5_moe/het_next_l0_ph1_nvidia_nc8_compile_contract.py`. Runner and static preflight must invoke the same hash-bound classifier and transaction function objects. No NC6/NC7 contract path is permitted.

After exact authorization, capture in order `CUDA_CACHE_DISABLE,CUDA_CACHE_MAXSIZE,CUDA_CACHE_PATH,TMP,TEMP,NVRTC_CACHE_PATH`, retaining `{name,present,value}`. Set in that order to `1`, `0`, and four normalized absolute, distinct descendants of `${INPROGRESS_ABS}/private_tree/{cuda_cache,tmp,temp,nvrtc_cache}`. Before publication restore in exact reverse order: delete originally absent variables and set originally present variables to their exact original strings, including empty strings. Continue restoration after a failure; retain ordered secondary errors. Alias, swap, outside path, missing/extra key, premature set, partial restoration or restoration failure is terminal incidental failure.

The private tree history contains exactly twelve ordered snapshots: `pre_load`, one after each of the ten NVRTC operations, and `post_release`. Every cache fixture carries `history_digest`, lower-case SHA-256 over canonical compact UTF-8 JSON of its exact history with `sort_keys=true,separators=(',',':'),ensure_ascii=false,allow_nan=false`. Missing, wrong, non-string or extra digest fields are rejected. The manifest contains the literal field/history mutation matrix.

No run is authorized by this preregistration.
