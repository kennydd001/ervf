from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
R=ROOT/"reports/streamq5_moe"; ROUTES=ROOT/"reports/runs/streamq5_moe/p4d_routes"
CAPTURE=R/"p4d_route_capture_result.json"; N2A=R/"n2a_temporal_ervf_oracle.json"
PREREG=R/"N2AU_ROUTE_UNION_PREREGISTRATION.md";OUTPUT=R/"n2au_route_union.json"
DOMAINS=("general","code","math","multilingual","instruction");SIZES=(2,4,8)

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def summary(x):
    a=np.asarray(x,dtype=np.float64);return{"mean":float(a.mean()),"p5":float(np.percentile(a,5)),"p50":float(np.percentile(a,50)),"p95":float(np.percentile(a,95)),"min":float(a.min()),"max":float(a.max()),"samples":int(a.size)}

def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    capture=json.loads(CAPTURE.read_text(encoding="utf-8"));n2a=json.loads(N2A.read_text(encoding="utf-8"))
    if not n2a["overall_pass"]:raise RuntimeError("N2A kernel pass required")
    values={str(s):[] for s in SIZES};per_domain={d:{str(s):[] for s in SIZES}for d in DOMAINS}
    hashes={}
    for layer in range(48):
        path=ROUTES/f"layer_{layer:02d}.safetensors"; hashes[str(layer)]=sha256(path)
        if hashes[str(layer)]!=capture["manifests"][str(layer)]["artifact_sha256"]:raise ValueError("route hash mismatch")
        with safe_open(path,framework="numpy") as handle:
            for domain in DOMAINS:
                route=handle.get_tensor(f"{domain}_router_ids").astype(np.int64)
                for s in SIZES:
                    for start in range(0,route.shape[0]-s+1,s):
                        count=int(np.unique(route[start:start+s]).size);values[str(s)].append(count);per_domain[domain][str(s)].append(count)
    aggregate={s:summary(v) for s,v in values.items()};domain_summary={d:{s:summary(v)for s,v in sizes.items()}for d,sizes in per_domain.items()}
    base=n2a["test"];u=aggregate["4"]["mean"]
    projection={"s4_mean_union":u,"sequential_q8_q5_p50_ms":base["q8"]["sequential"]["p50"]+base["q5"]["sequential"]["p50"],
                "temporal_q8_p50_ms":base["q8"]["temporal"]["p50"],"same8_temporal_q5_p50_ms":base["q5"]["temporal"]["p50"],
                "byte_linear_q5_p50_ms":base["q5"]["temporal"]["p50"]*u/8.0}
    projection["byte_linear_combined_ratio"]=(projection["temporal_q8_p50_ms"]+projection["byte_linear_q5_p50_ms"])/projection["sequential_q8_q5_p50_ms"]
    gates={"all_48_layers":len(hashes)==48,"s4_mean_le_25_6":aggregate["4"]["mean"]<=25.6,"s4_p95_le_30":aggregate["4"]["p95"]<=30}
    result={"kind":"streamq5_moe_n2au_route_union","completed_utc":datetime.now(timezone.utc).isoformat(),"inputs":{"preregistration_sha256":sha256(PREREG),"capture_sha256":sha256(CAPTURE),"n2a_sha256":sha256(N2A),"route_hashes":hashes},"aggregate":aggregate,"per_domain":domain_summary,"projection":projection,"gates":gates,"overall_pass":all(gates.values()),"claim_boundary":"Route-union statistics and a pessimistic byte-linear timing extrapolation only; no sparse temporal kernel, acceptance or end-to-end result."}
    OUTPUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps({"aggregate":aggregate,"projection":projection,"gates":gates,"overall_pass":result["overall_pass"]},indent=2))
if __name__=="__main__":main()
