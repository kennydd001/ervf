from __future__ import annotations

import gc,json,math,struct,traceback,types
from pathlib import Path
import numpy as np

from common import REPO, require_model_dir, utc_now, write_json_atomic

OUT=REPO/"pro_research"/"results"/"s100_phase20s"/"S100_PHASE20S_LAYER_ORACLE.json"
PROMPT="The history of computing changed science and engineering."
TOKENS=10

E2=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float32)

def e4(raw):
    a=np.asarray(raw,np.uint8)
    s=((a>>7)&1).astype(np.float32)
    E=((a>>3)&15).astype(np.int32)
    m=(a&7).astype(np.float32)
    v=np.where(E==0,m*np.float32(2.0**-9),
               (8.0+m)*np.exp2(E.astype(np.float32)-10.0))
    # E=15,m=7 is NaN in E4M3FN; checkpoints/scales should not contain it.
    v=np.where((E==15)&(m==7),np.nan,v)
    return np.where(s>0,-v,v).astype(np.float32)

def bf16(raw):
    u=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)
    return (u<<np.uint32(16)).view(np.float32)

class Reader:
    def __init__(self,root):
        self.root=Path(root);self.e={}
        idx=json.loads((self.root/"model.safetensors.index.json").read_text(encoding="utf-8"))
        for shard in sorted(set(idx["weight_map"].values())):
            p=self.root/shard
            with p.open("rb") as f:
                n=struct.unpack("<Q",f.read(8))[0]
                hdr=json.loads(f.read(n).decode("utf-8"))
            for name,m in hdr.items():
                if name=="__metadata__":continue
                self.e[name]=(shard,n,m["dtype"],tuple(m["shape"]),
                              int(m["data_offsets"][0]),int(m["data_offsets"][1]))
        self.cfg=json.loads((self.root/"config.json").read_text(encoding="utf-8"))
    def raw_bytes(self,name):
        shard,n,dtype,shape,a,b=self.e[name]
        with (self.root/shard).open("rb") as f:
            f.seek(8+n+a);data=f.read(b-a)
        if len(data)!=b-a:raise IOError(name)
        return data
    def arr(self,name):
        shard,n,dtype,shape,a,b=self.e[name]
        raw=self.raw_bytes(name)
        if dtype=="BF16":v=bf16(raw)
        elif dtype=="F32":v=np.frombuffer(raw,dtype="<f4").astype(np.float32)
        elif dtype=="F16":v=np.frombuffer(raw,dtype="<f2").astype(np.float32)
        elif dtype in ("F8_E4M3","F8_E4M3FN"):v=e4(np.frombuffer(raw,dtype=np.uint8))
        elif dtype=="U8":v=np.frombuffer(raw,dtype=np.uint8)
        else:raise ValueError(f"{name}: dtype={dtype}")
        return v.reshape(shape) if shape else v.reshape(())
    def scalar(self,name):return float(np.asarray(self.arr(name)).reshape(-1)[0])
    def kind(self,prefix):
        if prefix+".weight_scale_2" in self.e:return "nvfp4"
        if prefix+".weight_scale" in self.e:return "fp8"
        return "bf16"
    def linear(self,prefix):
        kind=self.kind(prefix)
        if kind=="bf16":
            return self.arr(prefix+".weight").astype(np.float32)
        if kind=="fp8":
            return self.arr(prefix+".weight").astype(np.float32)*np.float32(self.scalar(prefix+".weight_scale"))
        codes=self.arr(prefix+".weight").astype(np.uint8)
        scales=self.arr(prefix+".weight_scale").astype(np.float32)
        g=np.float32(self.scalar(prefix+".weight_scale_2"))
        rows,packed=codes.shape;cols=packed*2
        nib=np.empty((rows,cols),np.uint8)
        nib[:,0::2]=codes&15;nib[:,1::2]=codes>>4
        return E2[nib.astype(np.int32)]*np.repeat(scales,16,axis=1)*g
    def linear_nvfp4_stream(self,prefix,x,chunk=1024):
        codes=self.arr(prefix+".weight").astype(np.uint8)
        scales=self.arr(prefix+".weight_scale").astype(np.float32)
        g=np.float32(self.scalar(prefix+".weight_scale_2"))
        rows,packed=codes.shape;cols=packed*2
        out=np.empty(rows,np.float32)
        for a in range(0,rows,chunk):
            b=min(rows,a+chunk);c=codes[a:b]
            nib=np.empty((b-a,cols),np.uint8)
            nib[:,0::2]=c&15;nib[:,1::2]=c>>4
            W=E2[nib.astype(np.int32)]*np.repeat(scales[a:b],16,axis=1)*g
            out[a:b]=(W@x.astype(np.float32)).astype(np.float32)
        return out

def rms(x,w,eps):
    x=x.astype(np.float32);w=w.astype(np.float32)
    s=np.float32(1.0)/np.sqrt(np.mean(x*x,dtype=np.float32)+np.float32(eps))
    return (x*s*w).astype(np.float32)

def silu(x):return (x/(1.0+np.exp(-x))).astype(np.float32)

def nrmse(a,b):
    aa=np.asarray(a,np.float64);bb=np.asarray(b,np.float64)
    return float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-30))

def softmax(x):
    z=x-np.max(x);e=np.exp(z);return e/e.sum()

def capture():
    import cupy as cp
    from transformers import AutoTokenizer
    from moe_lab.lightningstream_nemotron.runtime import LightningRuntime

    model=require_model_dir()
    rt=LightningRuntime(model,contexts_max=512,embed_on_host=True,fp8_kv=False,verbose=False)
    rt.enable_cache(48);rt.load_routed_bank();rt.device_cache=True;rt.deterministic_accum=True
    ml=[int(x) for x in rt.mamba_layers];el=[int(x) for x in rt.moe_layers];al=[int(x) for x in rt.attn_layers]
    msel=sorted({ml[0],ml[len(ml)//2],ml[-1]})
    esel=sorted({el[0],el[len(el)//2],el[-1]})
    data={"mamba":{i:[] for i in msel},"moe":{i:[] for i in esel},
          "attn":{i:[] for i in al}}
    active={"v":False}
    om,oe,oa=rt._mamba,rt._moe,rt._attention

    def wm(self,i,out):
        i=int(i);take=active["v"] and i in data["mamba"]
        if take:
            rec={"h":cp.asnumpy(self.h).astype(np.float32,copy=True),
                 "normed":cp.asnumpy(self.normed).astype(np.float32,copy=True),
                 "conv_pre":cp.asnumpy(self.conv[i]).astype(np.float32,copy=True),
                 "ssm_pre":cp.asnumpy(self.ssm[i]).astype(np.float32,copy=True)}
        ret=om(i,out)
        if take:
            rec.update({"out":cp.asnumpy(out).astype(np.float32,copy=True),
                        "conv_post":cp.asnumpy(self.conv[i]).astype(np.float32,copy=True),
                        "ssm_post":cp.asnumpy(self.ssm[i]).astype(np.float32,copy=True)})
            data["mamba"][i].append(rec)
        return ret
    def we(self,i,out):
        i=int(i);take=active["v"] and i in data["moe"]
        if take:rec={"h":cp.asnumpy(self.h).astype(np.float32,copy=True),
                     "normed":cp.asnumpy(self.normed).astype(np.float32,copy=True)}
        ret=oe(i,out)
        if take:
            rec["out"]=cp.asnumpy(out).astype(np.float32,copy=True)
            data["moe"][i].append(rec)
        return ret
    def wa(self,i,out):
        i=int(i);take=active["v"] and i in data["attn"]
        if take:rec={"h":cp.asnumpy(self.h).astype(np.float32,copy=True),
                     "normed":cp.asnumpy(self.normed).astype(np.float32,copy=True)}
        ret=oa(i,out)
        if take:
            rec["out"]=cp.asnumpy(out).astype(np.float32,copy=True)
            data["attn"][i].append(rec)
        return ret

    rt._mamba=types.MethodType(wm,rt);rt._moe=types.MethodType(we,rt);rt._attention=types.MethodType(wa,rt)
    tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=True,use_fast=True)
    ids=tok.encode(PROMPT,add_special_tokens=False)
    rt.reset();active["v"]=True
    for t in ids[:TOKENS]:rt.step(int(t))
    active["v"]=False;cp.cuda.get_current_stream().synchronize()
    final={"h":cp.asnumpy(rt.h).astype(np.float32,copy=True),
           "normed":cp.asnumpy(rt.normed).astype(np.float32,copy=True),
           "logits":cp.asnumpy(rt.logits).astype(np.float32,copy=True),
           "last_token":int(ids[min(TOKENS,len(ids))-1])}
    rt.bank={};rt.cache={};rt._dev_cache={}
    del rt;gc.collect();cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    return data,final,msel,esel,al

def oracle_mamba(R,i,rec):
    c=R.cfg;p=f"backbone.layers.{i}";m=p+".mixer"
    x=rec["normed"];H=int(c["mamba_num_heads"]);P=int(c["mamba_head_dim"])
    N=int(c["ssm_state_size"]);G=int(c["n_groups"]);hpg=H//G;di=H*P
    W=R.linear(m+".in_proj");proj=(W@x).astype(np.float32)
    z=proj[:di];conv_dim=di+2*G*N
    xbc=proj[di:di+conv_dim];dtr=proj[di+conv_dim:]
    st=rec["conv_pre"].reshape(conv_dim,-1).copy()
    cw=R.arr(m+".conv1d.weight").astype(np.float32).reshape(conv_dim,-1)
    cb=R.arr(m+".conv1d.bias").astype(np.float32).reshape(-1)
    st[:,:-1]=st[:,1:];st[:,-1]=xbc
    convo=silu(np.sum(cw*st,axis=1,dtype=np.float32)+cb)
    xv=convo[:di].reshape(H,P);B=convo[di:di+G*N].reshape(G,N)
    C=convo[di+G*N:].reshape(G,N)
    dtb=R.arr(m+".dt_bias").astype(np.float32).reshape(H)
    v=dtr+dtb;dt=np.where(v>20,v,np.log1p(np.exp(v))).astype(np.float32)
    Alog=R.arr(m+".A_log").astype(np.float32).reshape(H)
    D=R.arr(m+".D").astype(np.float32).reshape(H)
    ss=rec["ssm_pre"].reshape(H,P,N).copy();y=np.empty((H,P),np.float32)
    for h in range(H):
        g=h//hpg;dec=np.float32(np.exp(-np.exp(np.float32(Alog[h]))*np.float32(dt[h])))
        ss[h]=(dec*ss[h]+(dt[h]*xv[h])[:,None]*B[g][None,:]).astype(np.float32)
        y[h]=(np.sum(ss[h]*C[g][None,:],axis=1,dtype=np.float32)+D[h]*xv[h]).astype(np.float32)
    gated=y.reshape(-1)*silu(z)
    nw=R.arr(m+".norm.weight").astype(np.float32).reshape(-1)
    gs=di//G;gn=np.empty(di,np.float32)
    eps=np.float32(c["layer_norm_epsilon"])
    for g in range(G):
        a=g*gs;b=(g+1)*gs;u=gated[a:b]
        gn[a:b]=(u/np.sqrt(np.mean(u*u,dtype=np.float32)+eps)*nw[a:b]).astype(np.float32)
    out=(R.linear(m+".out_proj")@gn).astype(np.float32)
    return out,st.reshape(-1),ss.reshape(-1)

def oracle_attention(R,i,recs):
    c=R.cfg;m=f"backbone.layers.{i}.mixer"
    Wq=R.linear(m+".q_proj");Wk=R.linear(m+".k_proj")
    Wv=R.linear(m+".v_proj");Wo=R.linear(m+".o_proj")
    nh=int(c["num_attention_heads"]);nkv=int(c["num_key_value_heads"])
    hd=int(c["head_dim"]);groups=nh//nkv;scale=1.0/math.sqrt(hd)
    xs=[r["normed"] for r in recs];K=[];V=[];outs=[]
    for t,x in enumerate(xs):
        q=(Wq@x).reshape(nh,hd);k=(Wk@x).reshape(nkv,hd);v=(Wv@x).reshape(nkv,hd)
        K.append(k);V.append(v);ctx=np.empty((nh,hd),np.float32)
        for h in range(nh):
            g=h//groups
            scores=np.asarray([np.dot(q[h],K[j][g])*scale for j in range(t+1)],np.float64)
            pr=softmax(scores)
            ctx[h]=sum(np.float32(pr[j])*V[j][g] for j in range(t+1))
        outs.append((Wo@ctx.reshape(-1)).astype(np.float32))
    return np.stack(outs)

def relu2(x):
    r=np.maximum(x,0);return (r*r).astype(np.float32)

def oracle_moe(R,i,x):
    c=R.cfg;m=f"backbone.layers.{i}.mixer"
    gate=R.arr(m+".gate.weight").astype(np.float32)
    bias=R.arr(m+".gate.e_score_correction_bias").astype(np.float32).reshape(-1)
    logits=gate@x;scores=(1.0/(1.0+np.exp(-logits))).astype(np.float32)
    choice=scores+bias
    ids=np.argsort(-choice,kind="stable")[:int(c["num_experts_per_tok"])]
    w=scores[ids].astype(np.float64);w=w/(w.sum()+1e-20)*float(c["routed_scaling_factor"])
    sh=m+".shared_experts"
    out=(R.linear(sh+".down_proj")@relu2(R.linear(sh+".up_proj")@x)).astype(np.float32)
    for slot,e in enumerate(ids):
        pre=f"{m}.experts.{int(e)}"
        u=relu2(R.linear(pre+".up_proj")@x)
        d=(R.linear(pre+".down_proj")@u).astype(np.float32)
        out=(out+d*np.float32(w[slot])).astype(np.float32)
    return out,ids.tolist(),w.tolist()

def main():
    payload={"kind":"s100_phase20s_layer_oracle","status":"started","started_utc":utc_now(),
             "oracle_independence":"fresh safetensors reader + NumPy math; candidate runtime only captures inputs/outputs"}
    try:
        R=Reader(require_model_dir())
        if R.cfg.get("moe_latent_size") not in (None,0):
            raise RuntimeError("latent MoE present; direct-expert oracle intentionally refuses this schema")
        data,final,msel,esel,al=capture()
        results={"norm":[],"mamba":[],"attention":[],"moe":[]}

        for kind,layers in (("mamba",msel),("moe",esel),("attn",al)):
            for i in layers:
                recs=data["attn" if kind=="attn" else kind][i]
                for rec in (recs[-1:],):
                    r=rec[0];w=R.arr(f"backbone.layers.{i}.norm.weight").astype(np.float32)
                    pred=rms(r["h"],w,float(R.cfg["layer_norm_epsilon"]))
                    results["norm"].append({"layer":i,"type":kind,"nrmse":nrmse(pred,r["normed"])})

        for i in msel:
            r=data["mamba"][i][-1];o,cv,ss=oracle_mamba(R,i,r)
            results["mamba"].append({"layer":i,"output_nrmse":nrmse(o,r["out"]),
                "conv_state_nrmse":nrmse(cv,r["conv_post"]),
                "ssm_state_nrmse":nrmse(ss,r["ssm_post"])})

        for i in al:
            pred=oracle_attention(R,i,data["attn"][i])
            got=np.stack([r["out"] for r in data["attn"][i]])
            results["attention"].append({"layer":i,"tokens":len(got),"output_nrmse":nrmse(pred,got)})

        for i in esel:
            r=data["moe"][i][-1];pred,ids,w=oracle_moe(R,i,r["normed"])
            results["moe"].append({"layer":i,"output_nrmse":nrmse(pred,r["out"]),
                                   "oracle_experts":ids,"oracle_weights":w})

        fnw=R.arr("backbone.norm_f.weight").astype(np.float32)
        fn=rms(final["h"],fnw,float(R.cfg["layer_norm_epsilon"]))
        final_norm_err=nrmse(fn,final["normed"])
        if R.kind("lm_head")=="nvfp4":
            logits=R.linear_nvfp4_stream("lm_head",fn)
        else:
            logits=(R.linear("lm_head")@fn).astype(np.float32)
        logits_err=nrmse(logits,final["logits"])
        top1=int(np.argmax(logits))==int(np.argmax(final["logits"]))

        gates={
            "norm":max(x["nrmse"] for x in results["norm"])<=2e-6,
            "mamba":max(max(x["output_nrmse"],x["conv_state_nrmse"],x["ssm_state_nrmse"])
                        for x in results["mamba"])<=1e-4,
            "attention":max(x["output_nrmse"] for x in results["attention"])<=1e-4,
            "moe":max(x["output_nrmse"] for x in results["moe"])<=5e-4,
            "final_norm":final_norm_err<=2e-6,
            "final_logits":logits_err<=5e-4,
            "final_top1":top1,
        }
        payload.update({"status":"measured","sampled":{"mamba":msel,"moe":esel,"attention":al},
            "results":results,"final":{"norm_nrmse":final_norm_err,"logits_nrmse":logits_err,
                                      "top1_match":top1},
            "gates":gates,"INDEPENDENT_LAYER_ORACLE_GREEN":all(gates.values()),
            "completed_utc":utc_now()})
    except Exception as exc:
        payload.update({"status":"technical_failure","error":{"type":type(exc).__name__,
            "message":str(exc),"traceback":traceback.format_exc()},"completed_utc":utc_now()})
    OUT.parent.mkdir(parents=True,exist_ok=True);write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({"status":payload.get("status"),"sampled":payload.get("sampled"),
      "gates":payload.get("gates"),"final":payload.get("final"),
      "INDEPENDENT_LAYER_ORACLE_GREEN":payload.get("INDEPENDENT_LAYER_ORACLE_GREEN"),
      "error":(payload.get("error") or {}).get("message"),"output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
