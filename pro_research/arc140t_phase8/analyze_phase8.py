from __future__ import annotations
import argparse,json,math
from pathlib import Path

def load(p):
    try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except Exception:return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',required=True)
    a=ap.parse_args();r=Path(a.dir)
    route=load(r/'route_miss_census.json') or {}
    ov=load(r/'openvino_distinct_experts.json') or {}
    nv=load(r/'arc_real_nvfp4_down.json') or {}
    bridge=load(r/'shared_pinned_bridge.json') or {}
    inter=load(r/'qfast_arc_interference.json') or {}
    d3d=load(r/'d3d12_cross_adapter.json') or {}

    def ovrec(kind,dtype,n):
        for x in ov.get('records',[]):
            if x.get('kind')==kind and x.get('dtype')==dtype and x.get('experts_or_rows')==n and x.get('status')=='measured':
                return x
    def nvbest(n,fast=False):
        rows=[x for x in nv.get('records',[]) if x.get('nexperts')==n and x.get('fast_math')==fast and x.get('correctness',{}).get('finite')]
        if not rows:return None
        return min(rows,key=lambda x:x['best']['wall_median_ms'])
    bridge55=None
    if bridge.get('rows'):
        bridge55=min(bridge['rows'],key=lambda x:abs(int(x['bytes'])-(6*1856*4+2688*4)))
    same=ovrec('same_weight_batch','i8',6);distinct=ovrec('distinct_down_weighted_sum','i8',6)
    strict6=nvbest(6,False)
    corr_ok=bool(strict6 and strict6['correctness']['cosine']>=.999 and strict6['correctness']['nrmse']<=.02)
    total=None
    if strict6 and bridge55:total=float(strict6['best']['wall_median_ms'])+float(bridge55['median_ms'])
    reg=inter.get('regression_fraction')
    down_promote=bool(corr_ok and total is not None and total<=.25 and (reg is None or reg<=.05))
    miss=route.get('actual_up_cache',{})
    payload={
      "kind":"s100_phase8_summary",
      "qfast_reference":{"ms":18.75165,"tok_s":53.32864041297699},
      "shape_contract":route.get('shape_contract'),
      "actual_up_cache":miss,
      "same_weight_M6_i8_ms":((same or {}).get('standard') or {}).get('median_ms'),
      "distinct_six_i8_ms":((distinct or {}).get('standard') or {}).get('median_ms'),
      "real_nvfp4_six_strict":strict6,
      "bridge_near_55KiB":bridge55,
      "qfast_arc_interference":inter,
      "d3d12_cross_adapter":d3d,
      "arc_down_engine_total_component_ms":total,
      "arc_down_engine_integration_candidate":down_promote,
      "interpretation":[
        "M=6 same-weight is batch amortization, not top-6 distinct experts.",
        "The primary single-stream Arc target is six distinct routed down projections plus route-weighted reduction.",
        "A positive result opens end-to-end ADE integration; a negative result demotes Arc to cache-miss, coalescer, draft, or long-context roles."
      ]
    }
    (r/'S100_PHASE8_SUMMARY.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=["S100 PHASE 8 — ARC ROUTED-DOWN ENGINE",
           f"Shape: {payload['shape_contract']}",
           f"Actual up-cache misses: {miss}",
           f"Same-weight M6 INT8: {payload['same_weight_M6_i8_ms']} ms",
           f"Six DISTINCT INT8 down: {payload['distinct_six_i8_ms']} ms",
           f"Real NVFP4 six-expert strict: {strict6}",
           f"Pinned bridge ~55 KiB: {bridge55}",
           f"QFAST interference: {inter}",
           f"D3D12 cross-adapter: {d3d}",
           f"Combined Arc-down component estimate: {total} ms",
           f"ARC DOWN ENGINE INTEGRATION CANDIDATE: {down_promote}",
           "",
           "NEXT:",
           "If candidate=True, build end-to-end QFAST ADE with RTX routed-up + Arc real NVFP4 routed-down.",
           "If false, use the route census to test Arc only for up-cache misses and retain RTX downflow."]
    (r/'S100_PHASE8_SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__':main()
