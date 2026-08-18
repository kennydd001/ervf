from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
ACT_D2H_PER_BYTE=0.0117/44544.0
OUT_H2D_PER_BYTE=0.0487/10752.0
HIDDEN=2688;INTER=1856
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dir',required=True);a=ap.parse_args();d=Path(a.dir);rows=[]
 for rp in sorted(d.glob('RTX_UPMISS_LAYER_*.json')):
  layer=rp.stem.split('_')[-1];apath=d/f'ARC_UPMISS_LAYER_{layer}.json'
  if not apath.exists():continue
  r=json.loads(rp.read_text());x=json.loads(apath.read_text());
  if r.get('status')!='measured' or x.get('status')!='measured':continue
  for rr in r['rows']:
   n=rr['nexperts'];ar=next((z for z in x['rows'] if z['nexperts']==n),None)
   if not ar:continue
   bridge=HIDDEN*4*ACT_D2H_PER_BYTE+n*INTER*4*OUT_H2D_PER_BYTE;st=float(rr['staged_fetch_plus_up']['median_ms']);dh=float(rr['direct_host_up']['median_ms']);aw=float(ar['best']['wall_median_ms'])+bridge;ok=ar['correctness']['cosine']>=.999 and ar['correctness']['nrmse']<=.02 and ar['correctness']['finite']
   rows.append({'layer':int(layer),'nexperts':n,'staged_rtx_ms':st,'direct_host_rtx_ms':dh,'direct_bitexact':rr['direct_bitexact'],'arc_wall_plus_bridge_ms':aw,'arc_kernel_wall_ms':ar['best']['wall_median_ms'],'bridge_estimate_ms':bridge,'arc_correct':ok})
 def med(n,key):
  v=[x[key] for x in rows if x['nexperts']==n and isinstance(x.get(key),(int,float))];return statistics.median(v) if v else None
 summary={}
 for n in (1,2,3):
  s,d0,a0=med(n,'staged_rtx_ms'),med(n,'direct_host_rtx_ms'),med(n,'arc_wall_plus_bridge_ms');summary[str(n)]={'staged_rtx_ms':s,'direct_host_rtx_ms':d0,'arc_wall_plus_bridge_ms':a0,'direct_gain_fraction':(s-d0)/s if s and d0 else None,'arc_gain_fraction':(s-a0)/s if s and a0 else None}
 promote_direct=all(x['direct_bitexact'] for x in rows) and all(summary[str(n)].get('direct_gain_fraction') is not None and summary[str(n)]['direct_gain_fraction']>=.10 for n in (1,2));promote_arc=all(x['arc_correct'] for x in rows) and all(summary[str(n)].get('arc_gain_fraction') is not None and summary[str(n)]['arc_gain_fraction']>=.10 for n in (1,2))
 out={'kind':'s100_phase9_miss_economics','status':'measured','rows':rows,'median_by_n':summary,'DIRECTHOST_PROMOTE':promote_direct,'ARC_MISS_PROMOTE':promote_arc};(d/'S100_PHASE9_MISS_ECONOMICS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
