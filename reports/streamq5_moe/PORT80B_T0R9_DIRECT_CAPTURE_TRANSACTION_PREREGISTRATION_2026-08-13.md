# PORT80B T0-R9 direct-capture transaction preregistration

R9 supersedes unexecuted R8 and changes only its audited blockers. It preserves direct official router tuple capture, official IDs/weights for manual MoE, exact second-router-call comparison, tie retention, prefix ladder, whole/prefix16 cache equality, BF16/meta/source/resource gates, raw manifests, differentiated real-Q5 bank semantics, failure evidence and the closed R4 cross-backend negative.

The canonical four-prompt lock is emitted before outputs by the frozen deterministic generator. Execution lockcheck rehashes and re-executes that generator and compares canonical prompt rows exactly; the generator has one declared candidate/domain and no rejection/output-dependent filtering.

Capture1 and capture2 are distinct commands and processes. Their pre-model create-new ledgers bind capture index, PID, parent PID, process-create time, UTC/performance start, exact argv, nonce and runner hash. Capture2 requires PID, process-create time and nonce all distinct. Capture2 may read capture1 ledger plus the committed bank transaction (bank, manifest, completion marker) for byte reconstruction; it must not read capture1 raw/result. Only independent compare may read both.

Before model load, any capture-specific raw/result/failure/partial/journal target is fatal; capture1 also rejects any bank/manifest/marker. All retained raw tensors are finite and manifested before bank construction. Bank and manifest use create-new temp files, fsync, prepared journal, ordered promotion, full hash verification and a completion marker; failures quarantine partial/journal/orphan artifacts. No retry/retuning.

No preflight, model forward, bank build or GPU is authorized until independent source audit.
