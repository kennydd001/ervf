# Phase 18R run gate

The hardware runner must stop before all performance claims if any of the following fail:

1. parent baseline manifest is finite;
2. exact record graph matches baseline on all ten prompts;
3. replay table is completely finite and SHA-stable;
4. E dual-graph smoke matches generated IDs, final logits, hidden/recurrent state, used KV bytes and device position on all ten prompts;
5. E timing reproduces the Phase-17 corrected E saving within 0.75 ms/token.

Only then may the D/PD/UPD/RD/S/EMPTY_E/interactions and E_L* layer atlas execute. A failure in these gates is a harness/technical result, never a negative MoE result.
