from __future__ import annotations
import json,traceback
from common import require_model_dir,write_json_atomic,utc_now
from s100_phase16_common import RESULTS

OUT=RESULTS/"S100_PHASE16R_PREFLIGHT.json"

def main():
    payload={"kind":"s100_phase16r_preflight","status":"started","started_utc":utc_now()}
    try:
        import torch
        import cupy as cp
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
        rt=LightningRuntime(require_model_dir(),contexts_max=256,embed_on_host=True,fp8_kv=True,verbose=False)

        # First attention Q D2 clone -> transpose -> contiguous contract.
        layer=int(rt.attn_layers[0]);d=rt.layer[layer]
        W=d["q_proj"];rows=int(rt.n_heads*rt.head_dim);cols=int(rt.hidden)
        raw=(torch.utils.dlpack.from_dlpack(W).view(torch.bfloat16)
             .reshape(rows,cols).clone())
        wtt=raw.t().contiguous()
        x=torch.randn((1,cols),device="cuda",dtype=torch.float32).to(torch.bfloat16)
        y=torch.mm(x,wtt).float()
        torch.cuda.synchronize()
        q_ok=bool(torch.isfinite(y).all().item()) and tuple(y.shape)==(1,rows)

        # State ABI sizes from actual runtime allocations.
        m=int(rt.mamba_layers[0])
        expected=int(rt.m_heads*rt.m_hdim*rt.n_state)
        state_size=int(rt.ssm[m].size)
        y_size=int(rt.y.size)
        state_ok=state_size==expected and y_size==int(rt.m_heads*rt.m_hdim)

        payload.update({"status":"measured","q_proj":{"layer":layer,"rows":rows,"cols":cols,
          "d2_clone_mm_finite":q_ok,"wtt_shape":list(wtt.shape)},
          "ssm":{"layer":m,"m_heads":int(rt.m_heads),"m_hdim":int(rt.m_hdim),
                 "n_state":int(rt.n_state),"state_size":state_size,"expected":expected,
                 "y_size":y_size,"state_shape_ok":state_ok},
          "PREFLIGHT_GREEN":bool(q_ok and state_ok),"completed_utc":utc_now()})
    except Exception as e:
        payload.update({"status":"technical_failure","error":{"type":type(e).__name__,
          "message":str(e),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    RESULTS.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps(payload,indent=2));return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
