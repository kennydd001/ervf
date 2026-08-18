# S100 phase 9 repair

The original phase-9 summary is not a negative scientific result.

Valid evidence retained:

- complete 8,192-token route trace;
- measured route-slot miss fraction;
- five real six-miss expert captures.

Invalid/missing evidence in the original run:

1. the multiple-choice cache DP indexed an integer as a list;
2. capacity profiles were therefore never generated;
3. the RTX miss probe omitted `repo/src` from `sys.path`;
4. DirectHost and Arc miss economics were never executed;
5. the summary converted missing evidence into false promotion flags.

The repair runner reuses the valid trace and NPZ captures, fixes the source,
runs the oracle/capacity/miss experiments, and marks any remaining missing
evidence as incomplete rather than as a no-go.
