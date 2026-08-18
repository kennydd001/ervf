from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path
import numpy as np

CURRENT_REDUCE={38,10,40,20,43,13};CURRENT_BOOST={1,3,51,6}
UP_CODE=2494464;UP_SCALE=311808;DOWN_PLANE=311808;SLOT_BYTES=UP_CODE+UP_SCALE+DOWN_PLANE
PCIE_GBS=26.1686

def cmap_current(layers):return {int(l):(52 if int(l) in CURRENT_REDUCE else 102 if int(l) in CURRENT_BOOST else 72) for l in layers}

def lru_layer(ids,cnt,ses,cap,session_filter=None):
    hits=miss=0;cur=None;c=collections.OrderedDict()
    for t in range(len(ids)):
        if session_filter is not None and not session_filter(int(ses[t])):continue
        if cur!=int(ses[t]):cur=int(ses[t]);c=collections.OrderedDict()
        for e in ids[t]:
            e=int(e);hit=e in c
            if cnt[t]:hits+=int(hit);miss+=int(not hit)
            if hit:c.move_to_end(e)
            else:
                c[e]=None
                if len(c)>cap:c.popitem(last=False)
    return hits,miss

def static_layer(ids,cnt,ses,cap,train_filter,test_filter):
    f=collections.Counter()
    for t in range(len(ids)):
        if train_filter(int(ses[t])):
            for e in ids[t]:f[int(e)]+=1
    hot=set(e for e,_ in f.most_common(cap));h=m=0
    for t in range(len(ids)):
        if test_filter(int(ses[t])) and cnt[t]:
            for e in ids[t]:h+=int(int(e) in hot);m+=int(int(e) not in hot)
    return h,m

def belady_layer(ids,cnt,ses,cap,session_filter):
    hits=miss=0
    for sid in sorted(set(int(x) for x in ses if session_filter(int(x)))):
        ts=np.flatnonzero(ses==sid);flat=[];countflat=[]
        for t in ts:
            for e in ids[t]:flat.append(int(e));countflat.append(bool(cnt[t]))
        future=collections.defaultdict(collections.deque)
        for pos,e in enumerate(flat):future[e].append(pos)
        cache=set()
        for pos,(e,co) in enumerate(zip(flat,countflat)):
            q=future[e]
            if q and q[0]==pos:q.popleft()
            hit=e in cache
            if co:hits+=int(hit);miss+=int(not hit)
            if hit:continue
            if len(cache)>=cap:
                victim=max(cache,key=lambda x:(future[x][0] if future[x] else 10**18))
                cache.remove(victim)
            cache.add(e)
    return hits,miss

def optimize_caps(layers,train_miss,budget,caps):
    dp={0:(0,[])}
    for li,l in enumerate(layers):
        nd={}
        for used,(cost,chosen) in dp.items():
            for c in caps:
                nu=used+c
                if nu>budget:continue
                nc=cost+train_miss[int(l)][c]
                if nu not in nd or nc<nd[nu][0]:nd[nu]=(nc,chosen+[c])
        dp=nd
    used,best=min(dp.items(),key=lambda kv:(kv[1][0],-kv[0]))
    return {int(l):int(c) for l,c in zip(layers,best[1][1])},used,best[1][0]

def eval_map(ids,cnt,ses,layers,capmap,filt):
    h=m=0
    for j,l in enumerate(layers):
        a,b=lru_layer(ids[:,j,:],cnt,ses,capmap[int(l)],filt);h+=a;m+=b
    return {'hits':h,'misses':m,'miss_fraction':m/(h+m),'misses_per_layer_token':m/(cnt[[filt(int(s)) for s in ses]].sum()*len(layers)) if len(layers) else None}

def markov_prefetch(ids,cnt,ses,layers,capmap,train_filter,test_filter,budget):
    L=len(layers);nexp=128;trans=np.ones((L,nexp,nexp),dtype=np.float64)*0.01
    prev={}
    for t in range(len(ids)):
        sid=int(ses[t])
        if not train_filter(sid):continue
        if sid in prev:
            for j in range(L):
                for a in prev[sid][j]:
                    for b in ids[t,j]:trans[j,int(a),int(b)]+=1
        prev[sid]=ids[t].copy()
    caches=[collections.OrderedDict() for _ in range(L)];pf=[set() for _ in range(L)];cur=None;prevroute=None
    hits=miss=pfhits=pftotal=pfused=0
    for t in range(len(ids)):
        sid=int(ses[t])
        if not test_filter(sid):continue
        if cur!=sid:caches=[collections.OrderedDict() for _ in range(L)];pf=[set() for _ in range(L)];cur=sid;prevroute=None
        for j,l in enumerate(layers):
            c=caches[j]
            for e0 in ids[t,j]:
                e=int(e0);main=e in c;pref=e in pf[j]
                if cnt[t]:
                    if main:hits+=1
                    elif pref:pfhits+=1
                    else:miss+=1
                if main:c.move_to_end(e)
                else:
                    c[e]=None
                    if len(c)>capmap[int(l)]:c.popitem(last=False)
                if pref and cnt[t]:pfused+=1
        # predict next token after consuming current token
        cand=[]
        for j,l in enumerate(layers):
            score=trans[j,ids[t,j].astype(int)].sum(axis=0);den=float(score.sum())
            for e in np.argsort(score)[-3:][::-1]:
                e=int(e)
                if e not in caches[j]:cand.append((float(score[e]/den),j,e))
        cand.sort(reverse=True);newpf=[set() for _ in range(L)]
        next_counted = (t + 1 < len(ids) and int(ses[t + 1]) == sid and bool(cnt[t + 1]))
        for sc,j,e in cand:
            if sum(len(x) for x in newpf)>=budget:break
            if e not in newpf[j]:
                newpf[j].add(e)
                pftotal += int(next_counted)
        pf=newpf
    total=hits+pfhits+miss
    return {'budget_records_per_token':budget,'demand_miss_fraction':miss/total,'main_hits':hits,'prefetch_hits':pfhits,'demand_misses':miss,'prefetches':pftotal,'prefetch_used':pfused,'prefetch_precision':pfused/pftotal if pftotal else 0.0,'bytes_prefetched_per_counted_token':pftotal*SLOT_BYTES/max(1,int(cnt[[test_filter(int(s)) for s in ses]].sum()))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trace',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args();od=Path(a.outdir);od.mkdir(parents=True,exist_ok=True)
    z=np.load(a.trace);ids=z['ids'].astype(np.int16);cnt=z['counted'].astype(bool);ses=z['session'].astype(np.int16);layers=[int(x) for x in z['layers']]
    train=lambda s:s%2==0;test=lambda s:s%2==1;cur=cmap_current(layers);caps=list(range(32,129,2));actual_need=z['need'][cnt].sum()/z['need'][cnt].size
    cur_all=eval_map(ids,cnt,ses,layers,cur,lambda s:True);sim_gate=abs(cur_all['miss_fraction']-actual_need)<=0.015
    train_miss={};test_miss={};curves={}
    for j,l in enumerate(layers):
        train_miss[l]={};test_miss[l]={};curves[str(l)]={}
        for c in caps:
            ht,mt=lru_layer(ids[:,j,:],cnt,ses,c,train);hv,mv=lru_layer(ids[:,j,:],cnt,ses,c,test)
            train_miss[l][c]=mt;test_miss[l][c]=mv;curves[str(l)][str(c)]={'train_misses':mt,'test_misses':mv}
    budgets=[1656,1784,1912,2035];profiles={'current':{str(k):v for k,v in cur.items()}};optrows={}
    for B in budgets:
        mp,used,cost=optimize_caps(layers,train_miss,B,caps);ev=eval_map(ids,cnt,ses,layers,mp,test);name='budget_neutral' if B==1656 else f'plus_{B-1656}'
        profiles[name]={str(k):v for k,v in mp.items()};optrows[name]={'slot_budget':B,'slots_used':used,'train_misses':cost,'test':ev,'extra_vram_mib_estimate':max(0,used-1656)*SLOT_BYTES/1024**2}
    static_h=static_m=0;bel_h=bel_m=0
    for j,l in enumerate(layers):
        h,m=static_layer(ids[:,j,:],cnt,ses,cur[l],train,test);static_h+=h;static_m+=m
        h,m=belady_layer(ids[:,j,:],cnt,ses,cur[l],test);bel_h+=h;bel_m+=m
    pref=[markov_prefetch(ids,cnt,ses,layers,cur,train,test,b) for b in (4,8,12)]
    test_current=eval_map(ids,cnt,ses,layers,cur,test)
    out={'kind':'s100_phase9_cache_oracle','status':'measured','simulation_gate':sim_gate,'measured_miss_fraction':float(actual_need),'simulated_current_all':cur_all,'test_current':test_current,'static_train_frequency_test':{'miss_fraction':static_m/(static_h+static_m)},'belady_current_map_test':{'miss_fraction':bel_m/(bel_h+bel_m)},'optimized_profiles':optrows,'prefetch':pref,'slot_bytes':SLOT_BYTES,'pcie_gbs_anchor':PCIE_GBS,'theoretical_current_up_fetch_serial_ms':cur_all['misses_per_layer_token']*len(layers)*(UP_CODE+UP_SCALE)/(PCIE_GBS*1e9)*1e3,'profiles_path':'S100_PHASE9_CAPACITY_PROFILES.json'}
    (od/'S100_PHASE9_CACHE_ORACLE.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8');(od/'S100_PHASE9_CAPACITY_PROFILES.json').write_text(json.dumps({'profiles':profiles,'oracle':optrows},indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2));return 0 if sim_gate else 2
if __name__=='__main__':raise SystemExit(main())
