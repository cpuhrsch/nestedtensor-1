#pragma once
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <stdexcept>

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

#define CUDA_CHECK(callstr)                                                 \
  {                                                                         \
    cudaError_t error_code = callstr;                                       \
    if (error_code != cudaSuccess) {                                        \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" \
                << __LINE__;                                                \
      assert(0);                                                            \
    }                                                                       \
  }

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

template <typename T>
void launch_attn_softmax_backward(
    T* out_grad,
    const T* soft_inp,
    int batch_size,
    int heads,
    int seq_length,
    cudaStream_t stream);

template <typename T>
void launch_attn_softmax_backward_v2(
    T* out_grad,
    const T* soft_inp,
    int batch_size,
    int heads,
    int seq_length,
    cudaStream_t stream);

// Custom softmax with scaling and attention mask addition
template <typename T>
void launch_attn_softmax(
    T* vals,
    const T* attn_mask,
    int batch_size,
    int heads,
    int sequence_length,
    cudaStream_t stream);
