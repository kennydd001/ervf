"""V18 H-SCALE+B3 with per-layer K and fused down-activation thresholding."""
from __future__ import annotations
import types
import numpy as np
from moe_dev_batched import DOWN_PANEL_BYTES,UP_CODE,UP_SCALE


def install_phase5_combined(rt,batch_kernels,up_kernels,sres,threshold_kernels,config):
    cp=rt.cp
    max_k=int(rt.top_k)
    if max_k != 6:
        raise RuntimeError(f"phase5 expects QFAST top_k=6, got {max_k}")
    inter=rt.moe_inter; hidden=rt.hidden; npanel=inter//16; nchunks=rt.fused.nchunks
    orig=rt._moe_dev; state={}
    mirrors=[rt.mstate["mirror"],cp.zeros(DOWN_PANEL_BYTES,dtype=cp.uint8)]
    gather_stream=cp.cuda.Stream(non_blocking=True)
    g_done=[cp.cuda.Event(block=False,disable_timing=True) for _ in range(max_k+1)]
    m_done=[cp.cuda.Event(block=False,disable_timing=True) for _ in range(max_k)]
    gather_blocks=(inter*32+255)//256

    def alloc():
        return {"act":cp.zeros(max_k*inter,dtype=cp.float32),
                "masks":cp.zeros(max_k*npanel,dtype=cp.uint32),
                "plist":cp.zeros(max_k*npanel,dtype=cp.int32),
                "pcount":cp.zeros(max_k,dtype=cp.int32),
                "nz":cp.zeros(max_k*inter,dtype=cp.int32),
                "nzc":cp.zeros(max_k,dtype=cp.int32),
                "partials":cp.zeros(max_k*nchunks*hidden,dtype=cp.float32),
                "max_act":cp.zeros(max_k,dtype=cp.float32)}

    def moe(self,i,out):
        cp2,k,d,fused=self.cp,self.k,self.layer[i],self.fused
        bank,c=self.bank[i],self.cache[i]
        ki=int(config["layer_k"].get(int(i),max_k))
        if ki not in (4,5,6): raise RuntimeError(f"bad K={ki} for layer {i}")
        alpha=float(config.get("alpha",0.0))
        if not hasattr(self,"_dev_cache"): self._dev_cache={}
        if i not in self._dev_cache:
            self._dev_cache[i]=fused.alloc_device_cache(self.n_experts,c["cap"],max_k,bank["globals"])
        dev=self._dev_cache[i]
        if i not in state: state[i]=alloc()
        bs=state[i]
        if i not in sres.planes: sres.alloc_planes(i,int(c["cap"]))
        planes=sres.planes[i]

        k.mv_f32(self.rlog,d["gate_w"],self.normed,self.n_experts,self.hidden)
        fused.route_topk(self.rlog,d["gate_b"],dev["ids"],dev["w"],self.n_experts,ki,self.scaling,bad_pick=self._bad_pick)
        fused.cache_assign(dev,dev["ids"],c["cap"],ki)
        self.evt[0].record()
        with self.copy_stream:
            self.copy_stream.wait_event(self.evt[0])
            fused.cache_fetch(bank["up_codes"].ctypes.data,bank["up_scales"].ctypes.data,c["codes"],c["scales"],dev,UP_CODE,UP_SCALE,ki)
            sres.fetch_planes(bank["down_base_ptr"],planes,dev,ki)
            self.evt[1].record(self.copy_stream)

        out.fill(0)
        fused.gemv_into(self._act_shared,d["sh_up_c"],d["sh_up_s"],self.normed,d["sh_up_g"],self.shared_inter,self.hidden,apply_relu2=True)
        fused.gemv_into(out,d["sh_dn_c"],d["sh_dn_s"],self._act_shared,d["sh_dn_g"],self.hidden,self.shared_inter)
        main=cp2.cuda.get_current_stream(); main.wait_event(self.evt[1])

        up_kernels.run_batched(bs["act"],c["codes"],c["scales"],dev["slots"],dev["ids"],dev["globals"],1,fused.e2m1,fused.e4m3,self.normed,self.moe_inter,self.hidden,True,UP_CODE,UP_SCALE,ki)
        if alpha <= 0.0:
            # Preserve the exact V18/QFAST scan path when thresholding is off.
            batch_kernels.panel_scan_batched(
                (ki,), (256,),
                (bs["act"], np.int32(inter), bs["masks"], bs["plist"],
                 bs["pcount"], bs["nz"], bs["nzc"]))
        else:
            threshold_kernels.panel_scan_threshold_batched(
                (ki,), (256,),
                (bs["act"], np.int32(inter), np.float32(alpha),
                 bs["masks"], bs["plist"], bs["pcount"], bs["nz"],
                 bs["nzc"], bs["max_act"]))

        grid_dm=((hidden+127)//128,nchunks)
        def issue(s):
            sres.gather_cols(gather_blocks,bank["down_base_ptr"],dev["ids"][s:],mirrors[s&1],bs["nz"][s*inter:(s+1)*inter],bs["nzc"][s:s+1],hidden)

        main.record(g_done[max_k]); gather_stream.wait_event(g_done[max_k])
        with gather_stream:
            issue(0); g_done[0].record(gather_stream)
        for s in range(ki):
            if s+1<ki:
                with gather_stream:
                    if s>=1: gather_stream.wait_event(m_done[s-1])
                    issue(s+1); g_done[s+1].record(gather_stream)
            main.wait_event(g_done[s])
            sres.down_masked_sres(grid_dm,mirrors[s&1],planes,dev["slots"][s:],dev["ids"][s:],dev["globals"],bs["act"][s*inter:(s+1)*inter],bs["plist"][s*npanel:(s+1)*npanel],bs["masks"][s*npanel:(s+1)*npanel],bs["pcount"][s:s+1],fused.e2m1,fused.e4m3,bs["partials"][s*nchunks*hidden:(s+1)*nchunks*hidden],hidden,inter)
            m_done[s].record(main)

        blocks=(hidden+255)//256
        batch_kernels.reduce_partials_batched((blocks,ki),(256,),(bs["partials"],self.contrib,np.int32(hidden),np.int32(nchunks)))
        batch_kernels.run_accumulate_batched(out,self.contrib,dev["w"],self.hidden,ki)
        return None,None

    rt._moe_dev=types.MethodType(moe,rt)
    def restore():
        rt._moe_dev=orig; sres.planes.clear()
    return restore
