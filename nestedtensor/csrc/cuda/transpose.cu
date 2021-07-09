#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<typename T, int grain_size>
__global__
void transpose_nchw_nhwc(
    T* input,
    T* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int num_channel)
{
  __shared__ T tile[grain_size][grain_size + 1];
  const int batch_id = blockIdx.y;
  const int tid2 = threadIdx.x / 32;
  const int tid3 = threadIdx.x % 32;
  const int block_offset = block_offsets[batch_id];
  const int next_block_offset = block_offsets[batch_id + 1];
  const int offset = offsets[batch_id];
  input = input + offset;
  output = output + offset;
  const int next_offset = offsets[batch_id + 1];
  const int size2 = num_channel;
  const int size3 = (next_offset - offset) / num_channel;
  const int num_chunks_3 = (size3  + grain_size - 1) / grain_size;
  for (int current_block = blockIdx.x; current_block < (next_block_offset - block_offset);
           current_block += 256) {

  const int current_block_mod = (current_block % num_chunks_3) * grain_size;
  const int current_block_div = (current_block / num_chunks_3) * grain_size;
  const int offset1_tid2 = (current_block_mod) + tid2;
  const int offset2_tid2 = (current_block_div) + tid2;
  const int ii3 = (current_block_mod) + tid3;
  const int offset2_tid3 = (current_block_div) + tid3;

  int ii2 = offset2_tid2;
  int ii = ii3 + ii2 * size3;
#pragma unroll
  for (int sub = 0; sub < 4; sub++) {
    bool valid = ii3 < size3 && ii2 < size2;
    tile[tid2 + sub * 8][tid3] = valid ? input[ii] : T(0);
    ii2 += 8;
    ii += 8 * size3;
  }

  __syncthreads();

  const int ii21 = offset2_tid3;
  int ii31 = offset1_tid2;
  int ii1 = ii21 * size3 + ii31;
#pragma unroll
  for (int sub = 0; sub < 4; sub++) {
    const int j = (ii1 % size3) * size2;
    const int i = (ii1 / size3);
    if (ii21 < size2 && ii31 < size3) {
      output[j + i] = tile[tid3][tid2 + sub * 8];
    }
    ii31 += 8;
    ii1 += 8;
  }

  __syncthreads();

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
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = 256;
  grid.y = batch_size;

  transpose_nchw_nhwc<T, 32><<<grid, 256, 0, stream>>>(
      input,
      output,
      block_offsets,
      offsets,
      batch_size,
      num_channel);
}

template void transpose_nchw_nhwc_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const cudaStream_t stream);

template void transpose_nchw_nhwc_kernelLauncher<float>(
    float* input,
    float* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
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
    const int num_chunks)
{
  __shared__ T tile[num_threads_sqrt][num_threads_sqrt + 1];
  const int batch_id  = blockIdx.y;
  const int tid2 = threadIdx.x / 32;
  const int tid3 = threadIdx.x % 32;

  const int block_offset = block_offsets[batch_id];
  const int next_block_offset = block_offsets[batch_id + 1];
  const int offset = offsets[batch_id];
  const int next_offset = offsets[batch_id + 1];
  const int image_numel = next_offset - offset;
  const int size2 = image_numel / num_channel;
  input = input + offset;
  output = output + offset;

  for (int block_id = blockIdx.x + block_offset;
           block_id < next_block_offset;
           block_id += 256) {
  const int current_block = block_id - block_offset;
  const int current_block_mod = (current_block % num_chunks) * num_threads_sqrt;
  const int current_block_div = (current_block / num_chunks) * num_threads_sqrt;
  const int offset1_tid2 = (current_block_mod) + tid2;
  const int offset2_tid3 = (current_block_div) + tid3;

  int ii = (current_block / num_chunks) * num_threads_sqrt * num_channel + tid2 * num_channel + (current_block_mod) + tid3;
#pragma unroll
  for (int sub = 0; sub < 4; sub++) {
    bool valid = ii < next_offset;
    tile[tid2 + sub * 8][tid3] = valid ? input[ii] : T(0);
    ii += 8 * num_channel;
  }

  __syncthreads();

  int ii21 = offset2_tid3;
  if (ii21 < size2) {
    ii21 = ii21 * num_channel;
    int ii31 = offset1_tid2;
    int ii1 = ii21 + ii31;
#pragma unroll
    for (int sub = 0; sub < 4; sub++) {
      if (ii31 < num_channel) {
        const int j = (ii1 % num_channel) * size2;
        const int i = (ii1 / num_channel);
        output[j + i] = tile[tid3][tid2 + sub * 8];
      }
      ii31 += 8;
      ii1 += 8;
    }
  }

  __syncthreads();
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
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = 256;
  grid.y = batch_size;

  const int num_chunks = (num_channel + 32 - 1) / 32;
  transpose_nhwc_nchw<T, 32><<<grid, 256, 0, stream>>>(
      input,
      output,
      block_offsets,
      offsets,
      batch_size,
      num_channel,
      num_chunks);
}

template void transpose_nhwc_nchw_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const cudaStream_t stream);

template void transpose_nhwc_nchw_kernelLauncher<float>(
    float* input,
    float* output,
    const int* block_offsets,
    const int* offsets,
    const int batch_size,
    const int block_numel,
    const int num_channel,
    const cudaStream_t stream);

}
} // namespace nested_tensor
