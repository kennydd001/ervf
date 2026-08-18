from __future__ import annotations
import argparse,json,traceback
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    out=Path(a.out); p={'kind':'s100_p8_overnight_opencl_caps','status':'started'}
    try:
        import pyopencl as cl
        rows=[]
        for plat in cl.get_platforms():
            for d in plat.get_devices():
                if not (d.type & cl.device_type.GPU): continue
                ext=set((d.extensions or '').split())
                rows.append({'platform':plat.name,'name':d.name,'vendor':d.vendor,
                    'version':d.version,'driver':d.driver_version,
                    'compute_units':int(d.max_compute_units),
                    'max_clock_mhz':int(d.max_clock_frequency),
                    'max_work_group_size':int(d.max_work_group_size),
                    'global_mem':int(d.global_mem_size),'local_mem':int(d.local_mem_size),
                    'extensions':sorted(ext),
                    'interesting':{k:(k in ext) for k in [
                        'cl_khr_subgroups','cl_khr_integer_dot_product','cl_khr_command_buffer',
                        'cl_khr_external_memory','cl_khr_external_memory_win32','cl_khr_semaphore',
                        'cl_khr_external_semaphore','cl_khr_external_semaphore_win32',
                        'cl_khr_external_semaphore_dx_fence','cl_intel_unified_shared_memory',
                        'cl_intel_subgroups']}})
        p.update({'status':'measured','devices':rows})
    except Exception as e:
        p.update({'status':'technical_failure','error':{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(p,indent=2,allow_nan=False));return 0 if p['status']=='measured' else 2
if __name__=='__main__':raise SystemExit(main())
