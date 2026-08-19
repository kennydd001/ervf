from __future__ import annotations
import json,traceback
import numpy as np
from common import write_json_atomic,utc_now
from s100_phase21_common import (
    identity_gate,load_trace,prefill_to,expected_for_block,release,
)
from s100_phase22_common import (
    make_v6,eager_verifier,selected_head_mode,capture_state,compare_states,
)
from s100_phase23_common import GraphH4VerifierGrouped

OUT=__import__("pathlib").Path(
    __import__("common").REPO/"pro_research"/"results"/"s100_phase23"/
    "S100_PHASE23_STATE_CHECK.json"
)
CTX=1024

def main():
    payload={"kind":"s100_phase23_state_check","status":"started",
      "context":CTX,"started_utc":utc_now(),
      "claim_boundary":"V6 device-row parent versus GPU grouped H4"}
    rt=None
    try:
        import cupy as cp
        identity_gate();tr=load_trace();tokens=tr["tokens"];mode=selected_head_mode()
        rt,keep=make_v6(CTX)
        grouped=GraphH4VerifierGrouped(rt,mode)
        cap=grouped.setup_graph()
        parent=eager_verifier(rt,mode)

        # Parent from canonical prefix.
        rt.reset();prefill_to(rt,tokens,CTX)
        draft,expected=expected_for_block(tokens,CTX)
        pids,_=parent.block(draft.tolist(),False)
        cp.cuda.get_current_stream().synchronize()
        parent_state=capture_state(rt,parent,CTX+4)

        # Candidate eager with census, identical prefix.
        rt.reset();prefill_to(rt,tokens,CTX)
        cids,census=grouped.v.block(draft.tolist(),True)
        cp.cuda.get_current_stream().synchronize()
        cand_state=capture_state(rt,grouped.v,CTX+4)
        cmp_parent=compare_states(parent_state,cand_state)

        # Candidate graph, identical prefix.
        rt.reset();prefill_to(rt,tokens,CTX)
        grouped.set_pos_from_host()
        gids=grouped.launch(draft.tolist())
        graph_state=capture_state(rt,grouped.v,CTX+4)
        cmp_graph=compare_states(cand_state,graph_state)

        # Determinism.
        rt.reset();prefill_to(rt,tokens,CTX)
        grouped.set_pos_from_host()
        gids2=grouped.launch(draft.tolist())

        hist={str(m):0 for m in (1,2,3,4)}
        down_parent=0;down_group=0;up_parent=0;up_group=0
        repeats=[]
        for row in census:
            if row is None:continue
            for m,n in row["m_histogram"].items():
                hist[str(m)]+=int(n)
            down_parent+=int(row["down_sparse_bytes_parent_est"])
            down_group+=int(row["down_sparse_bytes_grouped"])
            up_parent+=int(row["up_weight_streams_parent"])
            up_group+=int(row["up_weight_streams_grouped"])
            repeats.append(float(row["repeat_rate"]))

        gates={
          "parent_ids_exact":bool(np.array_equal(pids,expected)),
          "candidate_ids_exact":bool(np.array_equal(cids,expected)),
          "graph_ids_exact":bool(np.array_equal(gids,expected)),
          "graph_deterministic":bool(np.array_equal(gids,gids2)),
          "parent_candidate_ssm":cmp_parent["max_ssm_nrmse"]<=5e-5,
          "parent_candidate_conv":cmp_parent["max_conv_nrmse"]<=1e-5,
          "parent_candidate_kv":cmp_parent["max_kv_nrmse"]<=5e-6,
          "parent_candidate_logits":cmp_parent["logits_nrmse"]<=5e-4,
          "candidate_graph_ssm":cmp_graph["max_ssm_nrmse"]<=5e-5,
          "candidate_graph_conv":cmp_graph["max_conv_nrmse"]<=1e-5,
          "candidate_graph_kv":cmp_graph["max_kv_nrmse"]<=5e-6,
          "candidate_graph_logits":cmp_graph["logits_nrmse"]<=5e-4,
          "group_counts_valid":all(int(k) in (1,2,3,4) for k,v in hist.items() if int(v)>0),
          "finite":bool(all(np.isfinite(x) for x in list(cmp_parent.values())+list(cmp_graph.values()))),
        }
        payload.update({
          "status":"measured","head_mode":mode,"capture_info":cap,
          "expected":expected.tolist(),
          "parent_ids":np.asarray(pids).tolist(),
          "candidate_ids":np.asarray(cids).tolist(),
          "graph_ids":np.asarray(gids).tolist(),
          "graph_ids_repeat":np.asarray(gids2).tolist(),
          "parent_vs_candidate":cmp_parent,
          "candidate_vs_graph":cmp_graph,
          "census":{
            "moe_layers":len(census),
            "m_histogram":hist,
            "m1_fraction":(
              hist["1"]/float(max(sum(hist.values()),1))
            ),
            "median_repeat_rate":float(np.median(repeats)) if repeats else None,
            "up_weight_stream_reduction_fraction":(
              1.0-up_group/float(up_parent) if up_parent else None
            ),
            "down_sparse_byte_reduction_fraction":(
              1.0-down_group/float(down_parent) if down_parent else None
            ),
            "up_parent_streams":up_parent,"up_grouped_streams":up_group,
            "down_parent_bytes_est":down_parent,"down_grouped_bytes":down_group,
          },
          "gates":gates,
          "GPU_GROUPED_CORRECTNESS_GREEN":all(gates.values()),
          "completed_utc":utc_now(),
        })
    except Exception as exc:
        payload.update({"status":"technical_failure",
          "error":{"type":type(exc).__name__,"message":str(exc),
                   "traceback":traceback.format_exc()},
          "completed_utc":utc_now()})
    finally:
        if rt is not None:
            try:release(rt)
            except Exception:pass
    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(OUT,payload,archive=True)
    print(json.dumps({
      "status":payload.get("status"),"parent_vs_candidate":payload.get("parent_vs_candidate"),
      "candidate_vs_graph":payload.get("candidate_vs_graph"),
      "census":payload.get("census"),"gates":payload.get("gates"),
      "GPU_GROUPED_CORRECTNESS_GREEN":payload.get("GPU_GROUPED_CORRECTNESS_GREEN"),
      "error":(payload.get("error") or {}).get("message"),"output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2
if __name__=="__main__":raise SystemExit(main())
