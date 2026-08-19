from __future__ import annotations
import json,traceback
import numpy as np
from common import write_json_atomic,utc_now
from s100_phase22_common import (
    RESULTS,identity_gate,load_trace,make_v6,selected_head_mode,
    GraphH4Verifier,prefill_to,expected_for_block,capture_state,
    compare_states,release,
)

OUT=RESULTS/"S100_PHASE22_GRAPH_STATE_CHECK.json"
CTX=1024

def main():
    payload={"kind":"s100_phase22_graph_state_check","status":"started",
             "context":CTX,"started_utc":utc_now(),
             "claim_boundary":"direct eager-repaired V6 versus H4 graph state parity"}
    rt=None
    try:
        import cupy as cp
        identity_gate();tr=load_trace();tokens=tr["tokens"]
        mode=selected_head_mode()
        rt,keep=make_v6(CTX)
        g=GraphH4Verifier(rt,mode)
        cap=g.setup_graph()

        # Eager repaired reference from exact canonical prefill.
        prefill_to(rt,tokens,CTX)
        draft,expected=expected_for_block(tokens,CTX)
        got,_=g.v.block(draft.tolist(),False)
        cp.cuda.get_current_stream().synchronize()
        if not np.array_equal(got,expected):
            raise RuntimeError("eager repaired reference block diverged")
        eager_ids=np.asarray(got,np.int32)
        eager=capture_state(rt,g.v,CTX+4)

        # Graph from the identical canonical prefix.
        rt.reset()
        prefill_to(rt,tokens,CTX)
        g.set_pos_from_host()
        graph_ids=g.launch(draft.tolist())
        graph=capture_state(rt,g.v,CTX+4)
        cmp=compare_states(eager,graph)

        # Determinism: same prefix, same graph, same ids.
        rt.reset()
        prefill_to(rt,tokens,CTX)
        g.set_pos_from_host()
        graph_ids2=g.launch(draft.tolist())

        gates={
          "eager_ids_exact":bool(np.array_equal(eager_ids,expected)),
          "graph_ids_exact":bool(np.array_equal(graph_ids,expected)),
          "graph_deterministic_ids":bool(np.array_equal(graph_ids,graph_ids2)),
          "ssm":cmp["max_ssm_nrmse"]<=5e-5,
          "conv":cmp["max_conv_nrmse"]<=1e-5,
          "kv":cmp["max_kv_nrmse"]<=5e-6,
          "logits":cmp["logits_nrmse"]<=5e-4,
          "finite":bool(all(np.isfinite(x) for x in cmp.values())),
        }
        payload.update({
          "status":"measured",
          "head_mode":mode,
          "capture_info":cap,
          "expected":expected.tolist(),
          "eager_ids":eager_ids.tolist(),
          "graph_ids":graph_ids.tolist(),
          "graph_ids_repeat":graph_ids2.tolist(),
          "state":cmp,
          "gates":gates,
          "GRAPH_CORRECTNESS_GREEN":all(gates.values()),
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
      "status":payload.get("status"),
      "head_mode":payload.get("head_mode"),
      "capture_info":payload.get("capture_info"),
      "state":payload.get("state"),"gates":payload.get("gates"),
      "GRAPH_CORRECTNESS_GREEN":payload.get("GRAPH_CORRECTNESS_GREEN"),
      "error":(payload.get("error") or {}).get("message"),
      "output":str(OUT)},indent=2))
    return 0 if payload.get("status")=="measured" else 2

if __name__=="__main__":raise SystemExit(main())
