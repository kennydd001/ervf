# HET-NEXT-CAP0-R4 ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â static preflight design

This is a standalone capability-only design. Static preflight imports standard library only and makes zero device/runtime calls. It opens no D2, shard, checkpoint or model artifact because none belongs to CAP0-R4.

It must:

1. bind prereg, runner, verifier, capability source, shared protocol, both locks and itself;
2. AST-reject model/Q5/D2/shard/safetensors/transformers/torch and performance/statistics code;
3. independently reconstruct the seed input, both 1024-word outputs and their frozen SHA fixtures;
4. execute the exact shared protocol functions with mock Win32 handles for three epochs, then inject stale ack, wrong epoch, timeout, premature read, release failure and partial-output atomic failure/quarantine;
5. assert actual runner topology constants LP0/2/4/6, three repetitions, exact allocation bytes/flags/counter paths and closed source/validation;
6. AST-check every OpenCL/CUDA/PDH/Win32 function used has explicit ABI declaration or high-level runtime binding, and forbid Intel cl_mem/read/write/migrate paths;
7. ensure independent verifier imports neither runner, protocol nor capability source and requires nonempty exact cardinalities;
8. require output absence and all execution locks false/PENDING.

Static PASS cannot authorize capability. An authorization-only lock revision and independent audit are required before one physical CAP0-R4 attempt.








