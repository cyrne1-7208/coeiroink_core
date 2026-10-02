// CUDA/OpenCL間の構文差だけを吸収し、補正の数値式は共通ソースで管理する。
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#define CLK_LOCAL_MEM_FENCE 0
#define KERNEL extern "C" __global__
#define GLOBAL
#define LOCAL __shared__
#define LOCAL_PTR
#define DEVICE __device__
#define barrier(flags) __syncthreads()
#define M_PI_F 3.14159265358979323846f
DEVICE int get_global_id(int axis) { return axis ? blockIdx.y*blockDim.y+threadIdx.y : blockIdx.x*blockDim.x+threadIdx.x; }
DEVICE int get_local_id(int axis) { return axis ? threadIdx.y : threadIdx.x; }
DEVICE int get_group_id(int axis) { return axis ? blockIdx.y : blockIdx.x; }
DEVICE float2 v2(float x) { return make_float2(x,x); }
DEVICE float2 v2(float x,float y) { return make_float2(x,y); }
DEVICE float3 v3(float x,float y,float z) { return make_float3(x,y,z); }
DEVICE float4 v4(float x,float y,float z,float w) { return make_float4(x,y,z,w); }
DEVICE int2 i2(int x,int y) { return make_int2(x,y); }
DEVICE int4 i4(int x,int y,int z,int w) { return make_int4(x,y,z,w); }
DEVICE float2 operator+(float2 a,float2 b) { return v2(a.x+b.x,a.y+b.y); }
DEVICE float2 operator-(float2 a,float2 b) { return v2(a.x-b.x,a.y-b.y); }
DEVICE float2 operator*(float2 a,float b) { return v2(a.x*b,a.y*b); }
DEVICE float2 operator/(float2 a,float b) { return v2(a.x/b,a.y/b); }
DEVICE float2& operator+=(float2& a,float2 b) { a=a+b; return a; }
DEVICE float2 fmax(float2 a,float2 b) { return v2(fmaxf(a.x,b.x),fmaxf(a.y,b.y)); }
DEVICE int clamp(int x,int a,int b) { return min(b,max(a,x)); }
DEVICE float clamp(float x,float a,float b) { return fminf(b,fmaxf(a,x)); }
DEVICE float mix(float a,float b,float t) { return a+(b-a)*t; }
#define sinpi sinpif
#define cospi cospif
#else
#define KERNEL __kernel
#define GLOBAL __global
#define LOCAL __local
#define LOCAL_PTR __local
#define DEVICE static inline
#define v2(...) (float2)(__VA_ARGS__)
#define v3(...) (float3)(__VA_ARGS__)
#define v4(...) (float4)(__VA_ARGS__)
#define i2(...) (int2)(__VA_ARGS__)
#define i4(...) (int4)(__VA_ARGS__)
#endif
