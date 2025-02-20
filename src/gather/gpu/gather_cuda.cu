#include <cuda.h>
#include <cub/cub.cuh>

// 最暴力的写法
template<typename T, typename Tind>
__global__ void gather_kernel(const T* data, const Tind* indices, T* output, 
                        long long len_indices, long long stride, long long len_output, long long len_axis) 
{
    long long id = 1ll * blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= len_output) return;
    long long x = id / stride;
    long long pre_axis = x / len_indices;
    long long in_axis = x % len_indices;
    long long suf_axis = id % stride;
    long long ptr = pre_axis * stride * len_axis + indices[in_axis] * stride + suf_axis;
    output[id] = data[ptr];
}

template<typename T, typename Tind>
void gather_launch(const void* data, const void* indices, void* output, 
                        long long len_indices, long long stride, long long len_output, long long len_axis)
{
    gather_kernel<T, Tind><<<(len_output + 1023) / 1024, 1024>>>(
        (T*)data, (Tind*)indices, (T*)output, len_indices, stride, len_output, len_axis
    );
}



extern "C" void gather_nv_f32(const void* data, const void* indices, void* output, 
                        long long len_indices, long long stride, long long len_output, long long len_axis)
{
    gather_launch<float, size_t>(data, indices, output, len_indices, stride, len_output, len_axis);
}
extern "C" void gather_nv_f16(const void* data, const void* indices, void* output, 
                        long long len_indices, long long stride, long long len_output, long long len_axis)
{
    gather_launch<half, size_t>(data, indices, output, len_indices, stride, len_output, len_axis);
}



