#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<typename T, int num_threads_sqrt>
__global__
void transpose_nchw_nhwc(
    T* input,
    T* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int num_channel,
    const int* image_numel)
{
  __shared__ T tile[num_threads_sqrt][num_threads_sqrt + 1];
  const int block_id  = blockIdx.x;
  const int tid2 = threadIdx.x / 32;
  const int tid3 = threadIdx.x % 32;
  int batch_id = threadIdx.x % 32;
  bool found = false;
  while (batch_id < batch_size) {
    if (block_offsets[batch_id] <= block_id && 
        block_id < block_offsets[batch_id + 1]) {
      found = true;
      break;
    }
    batch_id += 32;
  }
  if (!found) {
    batch_id = 0;
  }
  // TODO: Parameterize on warp size instead of assuming 32.
  for (int warp_offset = 16; warp_offset > 0; warp_offset /= 2)
      batch_id = batch_id | __shfl_down_sync(0xFFFFFFFF, batch_id, warp_offset);
  batch_id = __shfl_sync(0xFFFFFFFF, batch_id, 0, 32);

  const int grain_size = num_threads_sqrt;
  const int size2 = num_channel;
  const int size3 = image_numel[batch_id];
  const int block_offset = block_offsets[batch_id];
  const int offset = offsets[batch_id];

  const int num_chunks_3 = (size3  + grain_size - 1) / grain_size;
  const int current_block = block_id - block_offset;
  const int current_block_mod = (current_block % num_chunks_3) * grain_size;
  const int current_block_div = (current_block / num_chunks_3) * grain_size;
  const int offset1_tid2 = (current_block_mod) + tid2;
  const int offset2_tid2 = (current_block_div) + tid2;
  const int offset1_tid3 = (current_block_mod) + tid3;
  const int offset2_tid3 = (current_block_div) + tid3;
  const int ii3 = offset1_tid3;
#pragma unroll
  for (int sub = 0; sub < 4; sub++) {
    const int ii2 = offset2_tid2 + sub * 8;
    if (ii2 < size2 && ii3 < size3) {
      const int ii = ii2 * size3 + ii3;
      tile[tid2 + sub * 8][tid3] = input[offset + ii];
    }
  }

  __syncthreads();

  const int ii21 = offset2_tid3;
#pragma unroll
  for (int sub = 0; sub < 4; sub++) {
    const int ii31 = offset1_tid2 + sub * 8;
    if (ii21 < size2 && ii31 < size3) {
      const int ii1 = ii21 * size3 + ii31;
      const int j = (ii1 % size3) * size2;
      const int i = (ii1 / size3);
      output[offset + j + i] = tile[tid3][tid2 + sub * 8];
    }
  }
}

template <typename T>
void transpose_nchw_nhwc_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const int* image_numel,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = block_numel;

  transpose_nchw_nhwc<T, 32><<<grid, 256, 0, stream>>>(
      input,
      output,
      block_offsets,
      offsets,
      batch_size,
      num_channel,
      image_numel);
}

template void transpose_nchw_nhwc_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const int* image_numel,
    const cudaStream_t stream);

template void transpose_nchw_nhwc_kernelLauncher<float>(
    float* input,
    float* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const int* image_numel,
    const cudaStream_t stream);

template<typename T, int num_threads_sqrt>
__global__
void transpose_nhwc_nchw(
    T* input,
    T* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int num_channel,
    const int* image_numel)
{
  __shared__ T tile[num_threads_sqrt][num_threads_sqrt + 1];
  const int block_id  = blockIdx.x;
  const int tid2 = threadIdx.x / 32;
  const int tid3 = threadIdx.x % 32;
  int batch_id = threadIdx.x % 32;
  bool found = false;
  while (batch_id < batch_size) {
    if (block_offsets[batch_id] <= block_id && 
        block_id < block_offsets[batch_id + 1]) {
      found = true;
      break;
    }
    batch_id += 32;
  }
  if (!found) {
    batch_id = 0;
  }
  // TODO: Parameterize on warp size instead of assuming 32.
  for (int warp_offset = 16; warp_offset > 0; warp_offset /= 2)
      batch_id = batch_id | __shfl_down_sync(0xFFFFFFFF, batch_id, warp_offset);
  batch_id = __shfl_sync(0xFFFFFFFF, batch_id, 0, 32);

  const int grain_size = num_threads_sqrt;
  const int size2 = image_numel[batch_id];
  const int size3 = num_channel;
  const int block_offset = block_offsets[batch_id];
  const int offset = offsets[batch_id];

  const int num_chunks_3 = (size3  + grain_size - 1) / grain_size;
  const int current_block = block_id - block_offset;
  const int current_block_mod = (current_block % num_chunks_3) * grain_size;
  const int current_block_div = (current_block / num_chunks_3) * grain_size;
  const int offset1_tid2 = (current_block_mod) + tid2;
  const int offset2_tid2 = (current_block_div) + tid2;
  const int offset1_tid3 = (current_block_mod) + tid3;
  const int offset2_tid3 = (current_block_div) + tid3;
  const int ii3 = offset1_tid3;
  if (ii3 < size3) {
#pragma unroll
    for (int sub = 0; sub < 4; sub++) {
      const int ii2 = offset2_tid2 + sub * 8;
      if (ii2 < size2) {
        const int ii = ii2 * size3 + ii3;
        tile[tid2 + sub * 8][tid3] = input[offset + ii];
      }
    }
  }

  __syncthreads();

  const int ii21 = offset2_tid3;
  if (ii21 < size2) {
#pragma unroll
    for (int sub = 0; sub < 4; sub++) {
      const int ii31 = offset1_tid2 + sub * 8;
      if (ii31 < size3) {
        const int ii1 = ii21 * size3 + ii31;
        const int j = (ii1 % size3) * size2;
        const int i = (ii1 / size3);
        output[offset + j + i] = tile[tid3][tid2 + sub * 8];
      }
    }
  }
}

template <typename T>
void transpose_nhwc_nchw_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const int* image_numel,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = block_numel;

  transpose_nhwc_nchw<T, 32><<<grid, 256, 0, stream>>>(
      input,
      output,
      block_offsets,
      offsets,
      batch_size,
      num_channel,
      image_numel);
}

template void transpose_nhwc_nchw_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const int* image_numel,
    const cudaStream_t stream);

template void transpose_nhwc_nchw_kernelLauncher<float>(
    float* input,
    float* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const int* image_numel,
    const cudaStream_t stream);

}
} // namespace nested_tensor
