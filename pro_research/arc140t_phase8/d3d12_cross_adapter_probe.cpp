#include <windows.h>
#include <wrl.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <iostream>
#include <vector>
using Microsoft::WRL::ComPtr;

static ComPtr<IDXGIAdapter1> find_adapter(IDXGIFactory6* f, UINT vendor) {
    for (UINT i=0;;++i) {
        ComPtr<IDXGIAdapter1> a;
        if (f->EnumAdapters1(i,&a)==DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d{}; a->GetDesc1(&d);
        if (d.VendorId==vendor) return a;
    }
    return {};
}
int wmain() {
    ComPtr<IDXGIFactory6> fac;
    HRESULT hr=CreateDXGIFactory2(0,IID_PPV_ARGS(&fac));
    if(FAILED(hr)){std::cout<<"{\"status\":\"factory_failed\",\"hr\":"<<(long)hr<<"}\n";return 2;}
    auto nv=find_adapter(fac.Get(),0x10DE); auto intel=find_adapter(fac.Get(),0x8086);
    if(!nv||!intel){std::cout<<"{\"status\":\"adapter_missing\",\"nvidia\":"<<(nv?1:0)<<",\"intel\":"<<(intel?1:0)<<"}\n";return 0;}
    ComPtr<ID3D12Device> dn,di;
    hr=D3D12CreateDevice(nv.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&dn));
    if(FAILED(hr)){std::cout<<"{\"status\":\"nvidia_device_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    hr=D3D12CreateDevice(intel.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&di));
    if(FAILED(hr)){std::cout<<"{\"status\":\"intel_device_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    D3D12_HEAP_DESC hd{}; hd.SizeInBytes=65536;hd.Properties.Type=D3D12_HEAP_TYPE_DEFAULT;
    hd.Flags=(D3D12_HEAP_FLAGS)(D3D12_HEAP_FLAG_SHARED|D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER);
    ComPtr<ID3D12Heap> hn,hi;
    hr=dn->CreateHeap(&hd,IID_PPV_ARGS(&hn));
    if(FAILED(hr)){std::cout<<"{\"status\":\"create_cross_heap_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    HANDLE hh=nullptr;hr=dn->CreateSharedHandle(hn.Get(),nullptr,GENERIC_ALL,nullptr,&hh);
    if(FAILED(hr)){std::cout<<"{\"status\":\"heap_handle_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    hr=di->OpenSharedHandle(hh,IID_PPV_ARGS(&hi));CloseHandle(hh);
    if(FAILED(hr)){std::cout<<"{\"status\":\"intel_open_heap_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    D3D12_RESOURCE_DESC rd{};rd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;rd.Width=65536;rd.Height=1;rd.DepthOrArraySize=1;rd.MipLevels=1;rd.SampleDesc.Count=1;rd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;rd.Flags=D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
    ComPtr<ID3D12Resource> rn,ri;
    hr=dn->CreatePlacedResource(hn.Get(),0,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&rn));
    if(FAILED(hr)){std::cout<<"{\"status\":\"nvidia_resource_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    hr=di->CreatePlacedResource(hi.Get(),0,&rd,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&ri));
    if(FAILED(hr)){std::cout<<"{\"status\":\"intel_resource_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    ComPtr<ID3D12Fence> fn,fi;hr=dn->CreateFence(0,(D3D12_FENCE_FLAGS)(D3D12_FENCE_FLAG_SHARED|D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER),IID_PPV_ARGS(&fn));
    if(FAILED(hr)){std::cout<<"{\"status\":\"cross_fence_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    HANDLE fh=nullptr;hr=dn->CreateSharedHandle(fn.Get(),nullptr,GENERIC_ALL,nullptr,&fh);
    if(FAILED(hr)){std::cout<<"{\"status\":\"fence_handle_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    hr=di->OpenSharedHandle(fh,IID_PPV_ARGS(&fi));CloseHandle(fh);
    if(FAILED(hr)){std::cout<<"{\"status\":\"intel_open_fence_failed\",\"hr\":"<<(long)hr<<"}\n";return 0;}
    std::cout<<"{\"status\":\"cross_adapter_heap_fence_resource_pass\",\"bytes\":65536}\n";
    return 0;
}
