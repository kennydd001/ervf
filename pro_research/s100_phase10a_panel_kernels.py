from __future__ import annotations
import numpy as np
HIDDEN=2688;INTER=1856;NPANEL=116;ROWHALF=1344;PANEL_STRIDE=24192
DOWN_PANEL_BYTES=2806272;CODE_PANEL_BYTES=21504;PLANE_BYTES=311808
CUDA=r"""
extern "C" __global__ void gather_cols_pc(
 const unsigned char* down_base,const int* id_ptr,const int* pmap,
 const size_t panel_bytes,unsigned char* mirror,const int* nz,const int* nzc,
 const int rows,const int npanel){
 int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;
 if(warp>=*nzc)return;int j=nz[warp],p=j>>4,c=j&15,e=*id_ptr;
 if(pmap[e*npanel+p]>=0)return;int rh=rows>>1;
 size_t stride=(size_t)rows+16u*(size_t)rh;
 size_t off=(size_t)e*panel_bytes+(size_t)p*stride+rows+(size_t)c*rh;
 const uchar4* s=(const uchar4*)(down_base+off);
 uchar4* d=(uchar4*)(mirror+(size_t)p*stride+rows+(size_t)c*rh);
 for(int k=lane;k<rh/4;k+=32)d[k]=s[k];
}
extern "C" __global__ void down_pc(
 const unsigned char* mirror,const unsigned char* planes,const int* slot_ptr,
 const int* id_ptr,const float* globals,const int* pmap,
 const unsigned char* pdata,const float* act,const int* plist,
 const unsigned int* masks,const int* pcount,const float* e2,const float* e4,
 float* partials,const size_t plane_bytes,const int rows,const int inter,
 const int npanel,const int code_panel_bytes){
 int chunk=blockIdx.y,row=blockIdx.x*blockDim.x+threadIdx.x;
 if(row>=rows)return;__shared__ float se2[16],se4[256];
 if(threadIdx.x<16)se2[threadIdx.x]=e2[threadIdx.x];
 if(threadIdx.x<256)se4[threadIdx.x]=e4[threadIdx.x];__syncthreads();
 int e=*id_ptr,slot=*slot_ptr,hb=row>>1,hi=row&1,rh=rows>>1;
 float g=globals[e*2];size_t stride=(size_t)rows+16u*(size_t)rh;
 const unsigned char* plane=planes+(size_t)slot*plane_bytes;float acc=0.0f;
 for(int pi=chunk;pi<*pcount;pi+=gridDim.y){
  int p=plist[pi];float sc=se4[plane[(size_t)p*rows+row]]*g;
  int cs=pmap[e*npanel+p];
  const unsigned char* pc=cs>=0?pdata+(size_t)cs*code_panel_bytes:
      mirror+(size_t)p*stride+rows;
  unsigned int m=masks[p];
  while(m){int c=__ffs(m)-1;m&=m-1;
   unsigned char by=pc[(size_t)c*rh+hb];
   float ww=se2[hi?(by>>4):(by&15)]*sc;
   acc=fmaf(ww,act[(p<<4)+c],acc);
  }
 }
 partials[(size_t)chunk*rows+row]=acc;
}
"""
class PanelCacheKernels:
 def __init__(self):
  import cupy as cp
  self.cp=cp;self.mod=cp.RawModule(code=CUDA,options=("-std=c++14",))
  self.g=self.mod.get_function("gather_cols_pc");self.d=self.mod.get_function("down_pc")
 def gather(self,blocks,down,idp,pmap,mirror,nz,nzc,rows):
  self.g((blocks,),(256,),(np.uint64(down),idp,pmap,np.uint64(DOWN_PANEL_BYTES),mirror,nz,nzc,np.int32(rows),np.int32(NPANEL)))
 def down(self,grid,mirror,planes,slot,idp,glob,pmap,pdata,act,plist,masks,pcount,e2,e4,partials,rows,inter):
  self.d(grid,(128,),(mirror,planes,slot,idp,glob,pmap,pdata,act,plist,masks,pcount,e2,e4,partials,np.uint64(PLANE_BYTES),np.int32(rows),np.int32(inter),np.int32(NPANEL),np.int32(CODE_PANEL_BYTES)))
