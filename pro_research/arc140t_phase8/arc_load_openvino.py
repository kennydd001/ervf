from __future__ import annotations
import argparse,time
import numpy as np
import openvino as ov
from openvino import opset13 as ops

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shape',required=True);ap.add_argument('--seconds',type=float,default=45)
    a=ap.parse_args()
    import json
    d=json.loads(open(a.shape,encoding='utf-8').read());shape=d.get('shape_contract',d)
    h=int(shape['hidden']);inter=int(shape['moe_inter']);n=6
    core=ov.Core();gpu=next(x for x in core.available_devices if x.upper().startswith('GPU'))
    rng=np.random.default_rng(8);params=[];feeds={};summed=None
    route=np.ones(n,dtype=np.float32)/n
    for s in range(n):
        W=rng.integers(-7,8,size=(inter,h),dtype=np.int8)
        x=rng.integers(-7,8,size=(1,inter),dtype=np.int8)
        A=ops.parameter([1,inter],np.int8,name=f'A{s}')
        y=ops.convert(ops.matmul(A,ops.constant(W),False,False),np.float32)
        term=ops.multiply(y,ops.constant(np.float32(route[s])))
        summed=term if summed is None else ops.add(summed,term)
        params.append(A);feeds[s]=x
    comp=core.compile_model(ov.Model([summed],params,'arc_load'),gpu,{"PERFORMANCE_HINT":"LATENCY"})
    req=comp.create_infer_request()
    end=time.perf_counter()+a.seconds;count=0
    while time.perf_counter()<end:
        req.infer(feeds,share_inputs=True,share_outputs=True);count+=1
    print(f"arc_load_calls={count} seconds={a.seconds}")
if __name__=='__main__':main()
