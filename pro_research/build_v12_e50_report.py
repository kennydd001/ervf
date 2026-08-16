"""Build a conservative human-readable report from V12/V12B/V12C raw results."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "pro_research" / "results" / "v12_async"
OUT = DIR / "V12_E50_REPORT.md"


def load(name: str):
    p = DIR / name
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(x, digits=4):
    return "—" if x is None else f"{float(x):.{digits}f}"


def main() -> int:
    v12 = load("PRO_V12_ASYNC_HARVEST.json")
    b = load("PRO_V12B_CREDIT_STREAM.json")
    bv = load("PRO_V12B_VERIFY.json")
    c = load("PRO_V12C_EVENT_WAIT.json")
    cv = load("PRO_V12C_VERIFY.json")
    ad = load("DIAG_ADDNORM_LATE_DIVERGENCE.json")

    lines = ["# V12 E50 scheduler report", "",
             "This report separates blocking token latency, queued single-sequence generation throughput, and individually host-delivered streaming throughput.", ""]
    if v12:
        lines += ["## V12 fixed-K baseline", "",
                  f"Status: **{v12.get('status')}**", "",
                  f"SYNC A p50: {fmt(v12.get('sync',{}).get('A',{}).get('p50'))} ms",  
                  f"SYNC B p50: {fmt(v12.get('sync',{}).get('B',{}).get('p50'))} ms",
                  f"SYNC drift: {fmt(v12.get('sync',{}).get('p50_drift_ms'))} ms", ""]
        for label, key in (("Best queued", "best_queued"), ("Best fixed-K event stream", "best_event_stream")):
            rec = v12.get(key)
            if rec:
                lines.append(f"- {label}: {fmt(rec.get('tok_s'),3)} tok/s")
        lines.append("")
    else:
        lines += ["## V12 fixed-K baseline", "", "Not run.", ""]

    if b:
        best = b.get("best_credit") or {}
        lines += ["## V12B rolling busy-query credit", "",
                  f"Status: **{b.get('status')}**",
                  f"Best window: {best.get('window','—')}",
                  f"Best throughput: {fmt(best.get('tok_s'),3)} tok/s",
                  f"Worst prompt steady p50 gap: {fmt(best.get('max_prompt_steady_p50_gap_ms'))} ms",
                  f"Baseline drift: {fmt(b.get('sync',{}).get('p50_drift_ms'))} ms",
                  f"Independent verifier: **{(bv or {}).get('status','missing')}**", ""]
    else:
        lines += ["## V12B rolling busy-query credit", "", "Not run.", ""]

    if c:
        best = c.get("best_event_wait") or {}
        lines += ["## V12C rolling blocking-event credit", "",
                  f"Status: **{c.get('status')}**",
                  f"Best window: {best.get('window','—')}",
                  f"Best throughput: {fmt(best.get('tok_s'),3)} tok/s",
                  f"Worst prompt steady p50 gap: {fmt(best.get('max_prompt_steady_p50_gap_ms'))} ms",
                  f"Baseline drift: {fmt(c.get('sync',{}).get('p50_drift_ms'))} ms",
                  f"Independent verifier: **{(cv or {}).get('status','missing')}**", ""]
    else:
        lines += ["## V12C rolling blocking-event credit", "", "Not run.", ""]

    verified_stream = False
    if bv and bv.get("E50_streamed_credit_verified_any_recomputed") is True and bv.get("status") == "pass":
        verified_stream = True
    if cv and cv.get("E50_event_wait_verified_any_recomputed") is True and cv.get("status") == "pass":
        verified_stream = True

    sync_verified = False
    for rec in (b, c):
        if rec and rec.get("mode") == "full" and rec.get("gates",{}).get("E50_sync") is True and rec.get("gates",{}).get("baseline_drift_le_1ms") is True:
            sync_verified = True

    queued_signal = bool(v12 and v12.get("gates",{}).get("queued_E50_any"))
    lines += ["## Conservative verdict", ""]
    if sync_verified:
        lines.append("- **Synchronous E50 verified:** blocking per-token p50 <=20 ms under a stable A/B bracket.")
    else:
        lines.append("- Synchronous E50 is not verified by the available stable full result.")
    if verified_stream:
        lines.append("- **Streaming E50 verified:** exact tokens are individually host-visible at >=50 tok/s with the frozen delivery-gap gate.")
    else:
        lines.append("- Streaming E50 is not yet independently verified.")
    if queued_signal:
        lines.append("- Exact queued E50 signal exists; this is generation throughput, not arbitrary host-in-the-loop latency.")
    if not any((sync_verified, verified_stream, queued_signal)):
        lines.append("- No E50 subclaim is currently supported.")
    lines.append("")

    if ad:
        lines += ["## AddNorm late-divergence diagnostic", "",
                  f"Status: **{ad.get('status')}**",
                  f"Conclusion: `{ad.get('conclusion','—')}`",
                  f"First direct bit mismatch: `{ad.get('first_addnorm_bit_mismatch')}`",
                  f"Manual reference vs captured graph first divergence: `{ad.get('manual_vs_graph_first_divergence')}`", ""]

    lines += ["## Next decision", "",
              "1. If V12C verifies streamed E50, freeze that scheduler as the new serving baseline and run 10k-token + thermal validation before optimizing further.",
              "2. If queued E50 passes but both streaming arms miss, attribute the gap using enqueue CPU time, event wait/query time and delivery-gap tails before touching model kernels.",
              "3. If scheduler E50 fails under stable clocks, re-open the exact QKV candidate with an interleaved steady-state harness; its correctness already survived the prior full run.", ""]

    DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
