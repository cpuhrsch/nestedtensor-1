#include <nestedtensor/csrc/cuda/functions.h>

#include <cooperative_groups.h>
#include <cuda.h>
// #include <cuda_fp16.h>
// #include <curand_kernel.h>
// #include <stdio.h>
// #include <stdlib.h>
#include <limits>
// #include <stdexcept>
#include <algorithm>

#define FINAL_MASK 0xffffffff
#define MAX_THREADS 1024
// Maximum sequence-length support based on the number of threads (2048) allowed
// in each block and this MAX is 8K For higher sequence length we need to use
// higher Max, like for 64K : 32
#define MAX_THREAD_ITERATIONS 8 // Maximum 8K
#define MAX_THREAD_STRIDE 32
#define MAX_WARP_NUM 32
#define THREADS 256
#define TILE_DIM 32
#define minus_infinity -1 * std::numeric_limits<float>::infinity()

#define WARP_SIZE 32

// #define CUDA_CHECK(callstr)                                                 \
//   {                                                                         \
//     cudaError_t error_code = callstr;                                       \
//     if (error_code != cudaSuccess) {                                        \
//       std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" \
//                 << __LINE__;                                                \
//       assert(0);                                                            \
//     }                                                                       \
//   }

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                             \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);   \
       i += blockDim.x * gridDim.x)                                 \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m); \
         j += blockDim.y * gridDim.y)

#define DS_CUDA_NUM_THREADS 512
#define DS_MAXIMUM_NUM_BLOCKS 4096

inline int DS_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + DS_CUDA_NUM_THREADS - 1) / DS_CUDA_NUM_THREADS,
          DS_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

// From deepspeed https://github.com/microsoft/DeepSpeed/blob/e549be607c0f85fc3eb91b3ce977f1d063d65f3c/csrc/transformer/softmax_kernels.cu

namespace cg = cooperative_groups;

// Fused attention + softmax
template <int tbSize, int blockStride, int tbSeq>
__global__ void attn_softmax(float* vals,
                             const float* attn_mask,
                             int heads,
                             int seq_length,
                             int iterations)
{
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int batch = blockIdx.x;
    int row = blockIdx.y;
    int max_threads_in_sequence = std::max(seq_length, tbSeq);
    int seq_lane = threadIdx.x % max_threads_in_sequence;

    int data_offset = batch * (gridDim.y * block_width) + row * block_width +
                      (threadIdx.x / max_threads_in_sequence) * seq_length;
    int mask_offset = batch * seq_length;

    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;

    float4* val_cast = reinterpret_cast<float4*>(vals);
    const float4* attn_mask_cast = reinterpret_cast<const float4*>(attn_mask);

    float4 data[MAX_THREAD_ITERATIONS];

    float max_val = minus_infinity;

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) {
            float4 mask = attn_mask_cast[mask_offset + data_id];
            data[i] = val_cast[data_offset + data_id];

            data[i].x += mask.x;
            data[i].y += mask.y;
            data[i].z += mask.z;
            data[i].w += mask.w;

            max_val = (data[i].x > max_val ? data[i].x : max_val);
            max_val = (data[i].y > max_val ? data[i].y : max_val);
            max_val = (data[i].z > max_val ? data[i].z : max_val);
            max_val = (data[i].w > max_val ? data[i].w : max_val);
        } else {
            data[i].x = minus_infinity;
            data[i].y = minus_infinity;
            data[i].z = minus_infinity;
            data[i].w = minus_infinity;
        }
    }

    for (int i = 1; i < tbSize; i *= 2) {
        auto temp = g.shfl_xor(max_val, i);
        max_val = (temp > max_val ? temp : max_val);
    }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = max_val;
        b.sync();

        if (lane < warp_num) max_val = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

        for (int i = 1; i < iters; i *= 2) {
            auto temp = g.shfl_xor(max_val, i);
            max_val = (temp > max_val ? temp : max_val);
        }

        max_val = g.shfl(max_val, threadIdx.x / tbSize);
    }

    float sum = 0;
    for (int i = 0; i < iterations; i++) {
        data[i].x = __expf(data[i].x - max_val);
        data[i].y = __expf(data[i].y - max_val);
        data[i].z = __expf(data[i].z - max_val);
        data[i].w = __expf(data[i].w - max_val);

        sum += (data[i].x + data[i].y + data[i].z + data[i].w);
    }

    for (int i = 1; i < tbSize; i *= 2) { sum += g.shfl_xor(sum, i); }

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = sum;
        b.sync();

        if (lane < warp_num) sum = partialSum[lane];

#ifndef __STOCHASTIC_MODE__
        b.sync();
#endif

        int iters = warp_num;
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

        for (int i = 1; i < iters; i *= 2) { sum += g.shfl_xor(sum, i); }

        sum = g.shfl(sum, threadIdx.x / tbSize);
    }

    sum += 1e-6;

    for (int i = 0; i < iterations; i++) {
        data[i].x /= sum;
        data[i].y /= sum;
        data[i].z /= sum;
        data[i].w /= sum;

        int data_id = i * iteration_stride + seq_lane;
        if (data_id < seq_length) val_cast[data_offset + data_id] = data[i];
    }
}

template <typename T>
void launch_attn_softmax(T*, const T*, int, int, int, cudaStream_t, bool);

template <>
void launch_attn_softmax<float>(float* vals,
                                const float* attn_mask,
                                int batch_size,
                                int heads,
                                int sequence_length,
                                cudaStream_t stream)
{
    const int threads = 128;
    int seq_length4 = sequence_length / 4;
    int seq2 = sequence_length * seq_length4;

    int block_compute_size =
        (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
    dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

    int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

    dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                            subblock_max_workload * threads)
                                         : threads);
    int iterations =
        (sequence_length < subblock_max_workload ? (seq_length4 + threads - 1) / threads
                                                 : MAX_THREAD_ITERATIONS);

    if (sequence_length <= 8)
        attn_softmax<2, (threads / 2), 2>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 16)
        attn_softmax<4, (threads / 4), 4>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 32)
        attn_softmax<8, (threads / 8), 8>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 64)
        attn_softmax<16, (threads / 16), 16>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 128)
        attn_softmax<32, (threads / 32), 32>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else if (sequence_length <= 256)
        attn_softmax<32, (threads / 64), 64>
            <<<grid_dim, block_dim, 0, stream>>>(vals, attn_mask, heads, seq_length4, iterations);
    else {
        const int threads = 256;
        block_compute_size =
            (seq_length4 < threads ? ((threads / seq_length4) * seq_length4) : seq_length4);
        dim3 grid_dim(batch_size, heads * seq2 / block_compute_size);

        int subblock_max_workload = MAX_THREAD_ITERATIONS * 4 * threads;

        dim3 block_dim(seq_length4 > threads ? ((sequence_length + subblock_max_workload - 1) /
                                                subblock_max_workload * threads)
                                             : threads);

        if (sequence_length <= 512)
            attn_softmax<32, (threads / 128), 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        else if (sequence_length < (MAX_THREADS * MAX_THREAD_ITERATIONS * 4))
            attn_softmax<32, 1, 128><<<grid_dim, block_dim, 0, stream>>>(
                vals, attn_mask, heads, seq_length4, iterations);
        //else
        //    exit(1);
        //    // throw std::runtime_error(
        //    //     "Unsupport Seq_Length! Check the restriction of the max_threads and "
        //    //     "max_thread_iterations!");
    }
}

template <typename T, int tbSize, int blockStride>
__global__ void softmax_backward_kernel(T* out_grad, const T* soft_inp, int seq_length)
{
    __shared__ float partialSum[MAX_WARP_NUM];

    int warp_num = blockDim.x >> 5;  // warp-count = num_threads / WARP_SIZE (32)

    int iteration_stride = blockDim.x;
    int block_width = blockStride * seq_length;

    int iterations = (seq_length < (MAX_THREAD_ITERATIONS * iteration_stride)
                          ? (seq_length + iteration_stride - 1) / iteration_stride
                          : MAX_THREAD_ITERATIONS);

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<tbSize> g = cg::tiled_partition<tbSize>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;

    int wid = id >> 5;
    int lane = id & 0x1f;

    T val_reg[MAX_THREAD_ITERATIONS];
    T soft_reg[MAX_THREAD_ITERATIONS];
    float grad_reg = 0.0f;

#pragma unroll
    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + id;
        if (data_id < block_width) {
            val_reg[i] = out_grad[row * block_width + data_id];
            soft_reg[i] = soft_inp[row * block_width + data_id];

            grad_reg += ((float)val_reg[i] *
                         (float)soft_reg[i]);  // if done in half, the multiplication, we may lose
                                               // 2% of accuracy in computation!!
        }
    }
    for (int i = 1; i < tbSize; i *= 2) grad_reg += g.shfl_xor(grad_reg, i);

    if (seq_length > tbSize) {
        if (lane == 0) partialSum[wid] = grad_reg;
        b.sync();

        if (lane < warp_num) grad_reg = partialSum[lane];

        int iters = warp_num;
        if (seq_length < iteration_stride) iters = warp_num / (iteration_stride / seq_length);

        for (int i = 1; i < iters; i *= 2) grad_reg += g.shfl_xor(grad_reg, i);

        grad_reg = g.shfl(grad_reg, id / tbSize);
    }

    for (int i = 0; i < iterations; i++) {
        int data_id = i * iteration_stride + id;
        if (data_id < block_width) {
            float temp = (float)soft_reg[i] * ((float)val_reg[i] - grad_reg);
            out_grad[row * block_width + data_id] = (T)temp;
        }
    }
}

template <typename T, int ITERATIONS>
__global__ void softmax_backward_kernel_v2(T* grad /* input & output*/,
                                           const T* output,
                                           int softmax_length)
{
    int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int offset = batch_idx * softmax_length + threadIdx.x;

    grad += offset;
    output += offset;

    T grad_reg[ITERATIONS];
    T output_reg[ITERATIONS];
    float sum = 0.0;

#pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int curr_idx = threadIdx.x + i * WARP_SIZE;
        if (curr_idx < softmax_length) {
            grad_reg[i] = grad[i * WARP_SIZE];
            output_reg[i] = output[i * WARP_SIZE];
            sum += (float)grad_reg[i] * (float)output_reg[i];
        }
    }

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

    for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_xor(sum, i);

#pragma unroll
    for (int i = 0; i < ITERATIONS; ++i) {
        int curr_idx = threadIdx.x + i * WARP_SIZE;
        if (curr_idx < softmax_length)
            grad[i * WARP_SIZE] = (float)output_reg[i] * ((float)grad_reg[i] - sum);
    }
}

template <typename T>
void launch_attn_softmax_backward_v2(T* out_grad,
                                     const T* soft_inp,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     cudaStream_t stream)
{
    // if ((seq_length % WARP_SIZE) != 0 || seq_length > 2048)
    //     throw std::runtime_error("Invalid sequence length found in softmax backward.");

    const int warps_per_block = 4;
    dim3 grid_dim(batch_size * heads * seq_length / warps_per_block);
    dim3 block_dim(WARP_SIZE, warps_per_block);

    switch (seq_length) {
        case 32:
            softmax_backward_kernel_v2<T, 1>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 64:
            softmax_backward_kernel_v2<T, 2>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 128:
            softmax_backward_kernel_v2<T, 4>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 256:
            softmax_backward_kernel_v2<T, 8>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 384:
            softmax_backward_kernel_v2<T, 12>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 512:
            softmax_backward_kernel_v2<T, 16>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 768:
            softmax_backward_kernel_v2<T, 24>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 1024:
            softmax_backward_kernel_v2<T, 32>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        case 2048:
            softmax_backward_kernel_v2<T, 64>
                <<<grid_dim, block_dim, 0, stream>>>(out_grad, soft_inp, seq_length);
            break;
        // default:
        //     throw std::runtime_error(
        //         std::string("Special sequence length found in softmax backward, seq_length: ") +
        //         std::to_string(seq_length));
    }
}

template void launch_attn_softmax_backward_v2<float>(float* out_grad,
                                                     const float* soft_inp,
                                                     int batch_size,
                                                     int heads,
                                                     int seq_length,
                                                     cudaStream_t stream);



void launch_softmax_backward(
    at::Tensor vals,
    int batch_size,
    int heads,
    int sequence_length) {
    return vals;
}

void launch_softmax(
    at::Tensor vals,
    int batch_size,
    int heads,
    int sequence_length) {
    return vals;
}
