# PORT80B T0-R10 transaction-state repair preregistration

R10 preserves all R9 scientific semantics and changes only the bank transaction state machine. Capture1 rejects any final or partial transaction. It writes/fsyncs create-new bank and manifest temporaries, writes/fsyncs a prepared journal, promotes bank and manifest, verifies hashes, writes/fsyncs a temporary completion marker, promotes it, re-verifies the committed triple, then removes the journal. Capture2 requires and validates the exact completion-marker schema/state plus bank/manifest sizes and hashes; a valid completion marker is not treated as stale.

The no-model preflight must run the actual transaction validation logic on a temporary mini-payload: clean capture1 state, temp/fsync/commit marker, capture2 validation, read-only byte reconstruction, overwrite refusal and failure/quarantine. This simulation cannot build the real Q5 bank or load/execute a model.

All R9 direct-router, cache, prompt, process, raw, failure and claim boundaries remain unchanged. No preflight or capture is opened until independent source audit.
