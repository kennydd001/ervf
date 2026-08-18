from __future__ import annotations
import argparse,json
from pathlib import Path
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--result",type=Path,default=Path("pro_research/results/s100_phase13k/S100_PHASE13K_TOPK_WITNESS.json")); args=ap.parse_args(); r=json.loads(args.result.read_text(encoding="utf-8")); f=[]
    if r.get("status")!="measured": f.append("status is not measured")
    if r.get("tokens",0)<100: f.append("too few witness rows")
    if len(r.get("aggregates",[]))!=len(r.get("noise_scales",[]))*len(r.get("k_values",[])): f.append("incomplete aggregates")
    if r.get("gates",{}).get("promotion_open") is not False: f.append("promotion must remain closed")
    p={"kind":"verify_s100_phase13k_exact_topk_witness","status":"PASS" if not f else "FAIL","failures":f,"promotion_open":False}; args.result.with_name("S100_PHASE13K_TOPK_WITNESS_VERIFY.json").write_text(json.dumps(p,indent=2)+"\n"); print(json.dumps(p,indent=2)); return 0 if not f else 2
if __name__=="__main__": raise SystemExit(main())
