from __future__ import annotations
import json,traceback,types
import numpy as np
from common import require_model_dir,write_json_atomic,utc_now
from s100_phase16_common import RESULTS,release_runtime

OUT=RESULTS/"S100_PHASE16E_MAMBA_AFFINE_SCAN.json"
H=8
FORMULA_TOL=1e-4
SCAN_TOL=5e-5

def bf16_to_f32(u16):
    a=np.asarray(u16,np.uint16)
    return (a.astype(np.uint32)<<np.uint32(16)).view(np.float32)

def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30))

def capture_layer(rt,layer,tokens):
    """Capture the REAL GPUKernels.ssm_step ABI.

    Runtime call order is:
      ssm_step(y, state, x, B, C, dt, Alog, D, H, P, N, hpg)

    Phase16 accidentally treated y as state; y has exactly d_inner=4096
    elements, explaining the failed 4096 -> (64,64,128) reshape.
    """
    import cupy as cp
    records=[];marker={"layer":None,"enabled":False}
    orig_mamba=rt._mamba;orig_ssm=rt.k.ssm_step

    def mw(self,i,out):
        marker["layer"]=int(i)
        try:return orig_mamba(i,out)
        finally:marker["layer"]=None

    def sw(y,state,x,Bv,Cv,dt,Alog,Dv,heads,hdim,nstate,hpg):
        take=marker["enabled"] and marker["layer"]==int(layer)
        rec=None
        if take:
            expected=int(heads)*int(hdim)*int(nstate)
            if int(state.size)!=expected:
                raise RuntimeError(
                    f"SSM ABI/state mismatch: state.size={int(state.size)} "
                    f"but H*P*N={expected} ({heads}*{hdim}*{nstate})"
                )
            if int(y.size)!=int(heads)*int(hdim):
                raise RuntimeError(
                    f"SSM y size mismatch: y.size={int(y.size)} "
                    f"expected={int(heads)*int(hdim)}"
                )
            rec={"state_pre":cp.asnumpy(state).astype(np.float32,copy=True),
                 "x":cp.asnumpy(x).astype(np.float32,copy=True),
                 "B":cp.asnumpy(Bv).astype(np.float32,copy=True),
                 "C":cp.asnumpy(Cv).astype(np.float32,copy=True),
                 "dt":cp.asnumpy(dt).astype(np.float32,copy=True),
                 "Alog":cp.asnumpy(Alog).astype(np.float32,copy=True),
                 "D_raw":cp.asnumpy(Dv).astype(np.uint16,copy=True),
                 "heads":int(heads),"hdim":int(hdim),
                 "nstate":int(nstate),"hpg":int(hpg)}
        ret=orig_ssm(y,state,x,Bv,Cv,dt,Alog,Dv,heads,hdim,nstate,hpg)
        if take:
            cp.cuda.get_current_stream().synchronize()
            rec["state_post"]=cp.asnumpy(state).astype(np.float32,copy=True)
            rec["y"]=cp.asnumpy(y).astype(np.float32,copy=True)
            records.append(rec)
        return ret

    rt._mamba=types.MethodType(mw,rt);rt.k.ssm_step=sw
    try:
        marker["enabled"]=True
        for token in tokens:
            rt.step(int(token))
            if len(records)>=H:break
        marker["enabled"]=False
    finally:
        rt._mamba=orig_mamba;rt.k.ssm_step=orig_ssm
    if len(records)!=H:
        raise RuntimeError(f"layer {layer}: captured {len(records)} != {H}")
    return records

def affine_terms(records):
    r0=records[0];HH,P,N=r0["heads"],r0["hdim"],r0["nstate"];hpg=r0["hpg"]
    a=[];b_formula=[];b_observed=[];state_pre=[];state_post=[]
    formula_err=[]
    for rec in records:
        pre=rec["state_pre"].reshape(HH,P,N)
        post=rec["state_post"].reshape(HH,P,N)
        x=rec["x"].reshape(HH,P);B=rec["B"].reshape(HH//hpg,N)
        dt=rec["dt"].reshape(HH);Alog=rec["Alog"].reshape(HH)
        decay=np.exp(-np.exp(Alog.astype(np.float32))*dt.astype(np.float32)).astype(np.float32)
        bf=np.empty_like(pre)
        for h in range(HH):
            g=h//hpg
            bf[h]=(dt[h]*x[h])[:,None]*B[g][None,:]
        bo=(post-decay[:,None,None]*pre).astype(np.float32)
        a.append(decay);b_formula.append(bf.astype(np.float32));b_observed.append(bo)
        state_pre.append(pre);state_post.append(post)
        formula_err.append(nrmse(bf,bo))
    return (np.stack(a),np.stack(b_formula),np.stack(b_observed),
            np.stack(state_pre),np.stack(state_post),formula_err)

def prefix_scan(a,b,state0):
    A=a.astype(np.float32).copy();B=b.astype(np.float32).copy()
    off=1
    while off<len(A):
        oa=A.copy();ob=B.copy()
        for t in range(off,len(A)):
            A[t]=(oa[t]*oa[t-off]).astype(np.float32)
            B[t]=(ob[t]+oa[t][:,None,None]*ob[t-off]).astype(np.float32)
        off<<=1
    return (A[:,:,None,None]*state0[None,...]+B).astype(np.float32)

def y_from_states(records,states):
    r0=records[0];HH,P,N=r0["heads"],r0["hdim"],r0["nstate"];hpg=r0["hpg"];ys=[]
    for t,rec in enumerate(records):
        C=rec["C"].reshape(HH//hpg,N);x=rec["x"].reshape(HH,P)
        D=bf16_to_f32(rec["D_raw"]).reshape(HH);y=np.empty((HH,P),np.float32)
        for h in range(HH):
            g=h//hpg
            y[h]=(np.sum(states[t,h]*C[g][None,:],axis=1,dtype=np.float32)+D[h]*x[h]).astype(np.float32)
        ys.append(y.reshape(-1))
    return np.stack(ys)

def main():
    payload={"kind":"s100_phase16e_mamba_affine_scan","status":"started",
      "horizon":H,"formula_tolerance_nrmse":FORMULA_TOL,
      "scan_tolerance_nrmse":SCAN_TOL,"started_utc":utc_now(),
      "claim_boundary":"repaired real SSM ABI; affine scan proof only"}
    try:
        import torch
        import cupy as cp
        from transformers import AutoTokenizer
        from moe_lab.lightningstream_nemotron.runtime import LightningRuntime
        rt=LightningRuntime(require_model_dir(),contexts_max=512,embed_on_host=True,fp8_kv=True,verbose=False)
        rt.load_routed_bank();rt.deterministic_accum=True
        layers=[int(x) for x in rt.mamba_layers]
        chosen=sorted({layers[0],layers[len(layers)//2],layers[-1]})
        tok=AutoTokenizer.from_pretrained(str(require_model_dir()),local_files_only=True,
                                          trust_remote_code=True,use_fast=True)
        prompt=tok.encode("The history of computing and artificial intelligence",
                          add_special_tokens=False)
        results=[]
        for layer in chosen:
            rt.reset();nxt=None
            for t in prompt:nxt=int(rt.step(int(t)))
            continuation=[];cur=int(nxt)
            for _ in range(H):
                continuation.append(cur);cur=int(rt.step(cur))
            rt.reset()
            for t in prompt:rt.step(int(t))
            records=capture_layer(rt,layer,continuation)

            a,bf,bo,pre,post,formula_err=affine_terms(records)
            scan_formula=prefix_scan(a,bf,pre[0])
            scan_observed=prefix_scan(a,bo,pre[0])
            y_formula=y_from_states(records,scan_formula)
            y_observed=y_from_states(records,scan_observed)
            gpu_y=np.stack([r["y"] for r in records])

            rec={"layer":layer,
              "abi_state_elements":int(records[0]["state_pre"].size),
              "abi_expected_state_elements":int(records[0]["heads"]*records[0]["hdim"]*records[0]["nstate"]),
              "max_formula_b_nrmse":float(max(formula_err)),
              "scan_formula_vs_gpu_state_all_nrmse":nrmse(scan_formula,post),
              "scan_formula_vs_gpu_final_state_nrmse":nrmse(scan_formula[-1],post[-1]),
              "scan_formula_vs_gpu_y_nrmse":nrmse(y_formula,gpu_y),
              "scan_observed_vs_gpu_state_all_nrmse":nrmse(scan_observed,post),
              "scan_observed_vs_gpu_y_nrmse":nrmse(y_observed,gpu_y)}
            rec["pass"]=bool(
                rec["abi_state_elements"]==rec["abi_expected_state_elements"]
                and rec["max_formula_b_nrmse"]<=FORMULA_TOL
                and rec["scan_formula_vs_gpu_final_state_nrmse"]<=SCAN_TOL
                and rec["scan_formula_vs_gpu_y_nrmse"]<=SCAN_TOL
            )
            results.append(rec);print(f"16R-E layer {layer}: {rec}",flush=True)
        payload.update({"status":"measured","layers":chosen,"results":results,
          "MAMBA_AFFINE_SCAN_BUILD_OPEN":all(x["pass"] for x in results),
          "completed_utc":utc_now()})
        release_runtime(rt)
    except Exception as e:
        payload.update({"status":"technical_failure","error":{"type":type(e).__name__,
          "message":str(e),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    RESULTS.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),
      "MAMBA_AFFINE_SCAN_BUILD_OPEN":payload.get("MAMBA_AFFINE_SCAN_BUILD_OPEN"),
      "results":payload.get("results"),"error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
