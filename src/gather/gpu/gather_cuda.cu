#include <cuda.h>
#include <stdio.h> 
#include <cub/cub.cuh>

const int block_size = 128;

template<typename T, typename Tind>
__global__ void gather_kernel(const T* data, const Tind* indices, T* output, 
                        long long stride, long long len_axis) 
{
    long long o_offset = (blockIdx.y * gridDim.x + blockIdx.x) * stride;
    long long in_offset = (blockIdx.y * len_axis + indices[blockIdx.x]) * stride;
    long long tid = threadIdx.x;
    for( ; tid < stride ; tid += block_size) 
    {
        output[o_offset + tid] = data[in_offset + tid];
    }    
}

template<typename T, typename Tind>
void gather_launch(const void* data, const void* indices, void* output, 
                        long long len_indices, long long stride, long long len_output, long long len_axis)
{
    long long grid_dim_x = len_indices;
    long long delta = len_indices * stride;
    long long grid_dim_y = (len_output + delta - 1) / delta;
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    gather_kernel<T, Tind><<<grid_dim, block_size>>>(
        (T*)data, (Tind*)indices, (T*)output, stride, len_axis
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


