from __future__ import annotations
import json, statistics, time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1]/"results"/"arc140t_phase7"/"ram_bandwidth.json"

def main():
    mib=512
    n=mib*1024*1024
    a=np.empty(n,dtype=np.uint8); b=np.empty_like(a); a.fill(7)
    vals=[]
    for _ in range(3): np.copyto(b,a)
    for _ in range(9):
        t=time.perf_counter();np.copyto(b,a);dt=time.perf_counter()-t
        vals.append((2*n/1e9)/dt) # read + write traffic
    payload={"kind":"arc140t_phase7_ram_copy","bytes":n,"gb_s_read_plus_write_median":statistics.median(vals),"samples":vals}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
