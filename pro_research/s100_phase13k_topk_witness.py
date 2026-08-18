from __future__ import annotations

import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
NOISE = (0.01, 0.05, 0.10)
KS = (16, 32, 64, 128, 256, 512, 1024)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",default="models/nemotron_3_5_lightning"); ap.add_argument("--tokens-per-prompt",type=int,default=16); ap.add_argument("--output",type=Path,default=Path("pro_research/results/s100_phase13k/S100_PHASE13K_TOPK_WITNESS.json")); args=ap.parse_args()
    sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"pro_research")); os.environ["LS_MODEL_DIR"]=str(Path(args.model_dir).resolve())
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
    from s100_phase13b_activation_census import prompts
    _, val=prompts(REPO); rt=LightningRuntime(Path(args.model_dir).resolve(),contexts_max=4096,embed_on_host=True,fp8_kv=True,verbose=False); rt.enable_cache(72); rt.load_routed_bank(); rt.deterministic_accum=True
    rng=np.random.default_rng(1311); rows=[]
    for p in val:
        rt.reset(); nxt=None
        for token in p["prompt_ids"]: nxt=rt.step(int(token))
        for step in range(args.tokens_per_prompt):
            nxt=rt.step(int(nxt)); exact=rt.cp.asnumpy(rt.logits).astype(np.float32,copy=True); exact_id=int(np.argmax(exact)); std=float(np.std(exact));
            for noise in NOISE:
                approx=exact+rng.normal(0,noise*max(std,1e-12),size=exact.shape).astype(np.float32)
                for k in KS:
                    shortlist=np.argpartition(approx,-k)[-k:]; included=bool(np.any(shortlist==exact_id)); witness=int(shortlist[np.argmax(exact[shortlist])])
                    rows.append({"prompt":p["id"],"step":step,"noise":noise,"k":k,"true_top1_in_shortlist":included,"witness_exact_top1":witness==exact_id})
        print(f"measured witness {p['id']}",flush=True)
    aggregates=[]
    for noise in NOISE:
        for k in KS:
            r=[x for x in rows if x["noise"]==noise and x["k"]==k]; aggregates.append({"noise":noise,"k":k,"shortlist_inclusion":float(np.mean([x["true_top1_in_shortlist"] for x in r])),"witness_exact_top1_rate":float(np.mean([x["witness_exact_top1"] for x in r]))})
    result={"kind":"s100_phase13k_exact_topk_witness","status":"measured","created_utc":datetime.now(timezone.utc).isoformat(),"model_dir":str(Path(args.model_dir).resolve()),"claim_boundary":"controlled shortlist/witness screen on real logits; no compressed lm_head or GPU rerank kernel","tokens":len(rows),"noise_scales":list(NOISE),"k_values":list(KS),"aggregates":aggregates,"gates":{"witness_kernel_green":False,"promotion_open":False}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"status":result["status"],"rows":len(rows),"promotion_open":False},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
