#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/padding.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

template<typename T>
__global__
void add_padding_1(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size)
{
  const int batch_id  = blockIdx.x;
  const int grid_id  = blockIdx.y;
  const int tid = threadIdx.x + grid_id * 256;
  const int grainsize = 16 * 256;
  const int batch_input_offset = offsets[batch_id];
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int numel_i = sizes_i[0];
  const int batch_output_offset = batch_id * output_sizes[1];
  for (int ii = 0; ii < (output_sizes[1] / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int output_offset = batch_output_offset + i;
    if (i < sizes_i[0]) {
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
  const int i = (output_sizes[1] / grainsize) * grainsize + tid;
  if (i < output_sizes[1]) {
    const int output_offset = batch_output_offset + i;
    if (i < sizes_i[0]) {
      output[output_offset] = input[batch_input_offset + i];
    } else {
      output[output_offset] = padding_value;
    }
  }
}

template<typename T>
__global__
void add_padding_2(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size)
{
  const int batch_id  = blockIdx.x;
  const int grid_id  = blockIdx.y;
  const int tid = threadIdx.x + grid_id * 256;
  const int grainsize = 16 * 256;
  const int offset = offsets[batch_id];
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int numel_i = sizes_i[0] * sizes_i[1];
  const int output_offset = batch_id * output_sizes[1] * output_sizes[2];
  const int output_numel = output_sizes[1] * output_sizes[2];
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes[2]);
    const int i1 = i % output_sizes[2];
    if (i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes[2]);
    const int i1 = i % output_sizes[2];
    if (i0 < sizes_i[0] && i1 < sizes_i[1]) {
      const int input_offset = offset + i0 * sizes_i[1] + i1;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template<typename T>
__global__
void add_padding_3(
    const T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size)
{
  const int batch_id  = blockIdx.x;
  const int grid_id  = blockIdx.y;
  const int tid = threadIdx.x + grid_id * 256;
  const int grainsize = 16 * 256;
  const int offset = offsets[batch_id];
  const int* sizes_i = input_sizes + batch_id * input_dim;
  const int numel_i = sizes_i[0] * sizes_i[1] * sizes_i[2];
  const int output_offset = batch_id * output_sizes[1] * output_sizes[2] * output_sizes[3];
  const int output_numel = output_sizes[1] * output_sizes[2] * output_sizes[3];
  for (int ii = 0; ii < (output_numel / grainsize); ii++) {
    const int i = ii * grainsize + tid;
    const int i0 = i / (output_sizes[2] * output_sizes[3]);
    const int i1 = (i % (output_sizes[2] * output_sizes[3])) / output_sizes[3];
    const int i2 = i % output_sizes[3];
    if (i0 < sizes_i[0] && i1 < sizes_i[1] && i2 < sizes_i[2]) {
      const int input_offset = offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
  const int i = (output_numel / grainsize) * grainsize + tid;
  if (i < output_numel) {
    const int i0 = i / (output_sizes[2] * output_sizes[3]);
    const int i1 = (i % (output_sizes[2] * output_sizes[3])) / output_sizes[3];
    const int i2 = i % output_sizes[3];
    if (i0 < sizes_i[0] && i1 < sizes_i[1] && i2 < sizes_i[2]) {
      const int input_offset = offset + i0 * (sizes_i[1] * sizes_i[2]) + i1 * sizes_i[2] + i2;
      output[output_offset + i] = input[input_offset];
    } else {
      output[output_offset + i] = padding_value;
    }
  }
}

template<typename T>
void add_padding_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;
  grid.y = 16;
  if (input_dim == 1) {
    add_padding_1<T><<<grid, 256, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes,
        batch_size);
  }
  if (input_dim == 2) {
    add_padding_2<T><<<grid, 256, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes,
        batch_size);
  }
  if (input_dim == 3) {
    add_padding_3<T><<<grid, 256, 0, stream>>>(
        input,
        output,
        padding_value,
        offsets,
        input_sizes,
        input_dim,
        output_sizes,
        batch_size);
  }
}

template void add_padding_kernelLauncher<float>(
    float* input,
    float* output,
    float padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream);

template void add_padding_kernelLauncher<c10::Half>(
    c10::Half* input,
    c10::Half* output,
    c10::Half padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const int* output_sizes,
    const int batch_size,
    const cudaStream_t stream);

template<typename T>
__global__
void add_padding_mask(
    const T* input,
    T* output,
    int* output_mask,
    const int* offsets,
    const int batch_size,
    const int mask_stride,
    const int output_stride,
    const int inner_size)
{
  const int batch_id  = blockIdx.x;
  for (int i = 0; i < (offsets[batch_id + 1] - offsets[batch_id]); i++) {
    output_mask[batch_id*mask_stride + i] = 1;
  }
  for (int i = 0; i < (offsets[batch_id + 1] - offsets[batch_id]) * inner_size; i++) {
    output[batch_id * output_stride + i] = input[offsets[batch_id] * inner_size + i];
  }
}

template<typename T>
void add_padding_mask_kernelLauncher(
    T* input, // [batch_size x None]
    T* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    int* output_mask, // [batch_size x max(input.nested_size(1))]
    const int* offsets, // [batch_size]
    const int batch_size,
    const int mask_stride,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  add_padding_mask<float><<<grid, 1, 0, stream>>>(
      input,
      output,
      output_mask,
      offsets,
      batch_size,
      mask_stride,
      output_stride,
      inner_size);
}

template void add_padding_mask_kernelLauncher<float>(
    float* input,
    float* output,
    int* output_mask,
    const int* offsets,
    const int batch_size,
    const int mask_stride,
    const int output_stride,
    const int inner_size,
    const cudaStream_t stream);

template<typename T, int num_threads, int grid_y>
__global__
void remove_padding(
    const T* input,
    T* output,
    const int* offsets,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const int input_sizes_1_2_3,
    const int input_sizes_2_3,
    const int input_sizes_3)
{
  const int batch_id  = blockIdx.x / grid_y;
  const int grid_id  = blockIdx.x % grid_y;
  const int tid = threadIdx.x + grid_id * num_threads;
  const int offset = offsets[batch_id];
  const int* sizes_i = output_sizes + batch_id * output_dim;
  const int size_0 = sizes_i[0];
  const int size_1 = sizes_i[1];
  const int size_2 = sizes_i[2];
  const int size_1_2 = size_1 * size_2;
  const int numel_i = size_0 * size_1_2;
  int input_offset = batch_id * input_sizes_1_2_3;
  int num_chunks = input_sizes_1_2_3 / (grid_y * num_threads);
  for (int ii = 0; ii < num_chunks; ii++) {
    const int i = ii * grid_y * num_threads + tid;
    const int i0 = i / (input_sizes_2_3);
    const int i1 = (i % (input_sizes_2_3)) / input_sizes_3;
    const int i2 = i % input_sizes_3;
    if (i0 < size_0 && i1 < size_1 && i2 < size_2) {
      output[offset + i0 * size_1_2 + i1 * size_2 + i2] = input[input_offset + i];
    }
  }
  const int i = (input_sizes_1_2_3 / (grid_y * num_threads)) * (grid_y * num_threads) + tid;
  if (i < input_sizes_1_2_3) {
    const int i0 = i / (input_sizes_2_3);
    const int i1 = (i % (input_sizes_2_3)) / input_sizes_3;
    const int i2 = i % input_sizes_3;
    if (i0 < size_0 && i1 < size_1 && i2 < size_2) {
      output[offset + i0 * size_1_2 + i1 * size_2 + i2] = input[input_offset + i];
    }
  }
}

template<typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size * 16;
  const int input_sizes_1_2_3 = input_sizes[1] * input_sizes[2] * input_sizes[3];
  const int input_sizes_2_3 = input_sizes[2] * input_sizes[3];
  const int input_sizes_3 = input_sizes[3];

  remove_padding<T, 256, 16><<<grid, 256, 0, stream>>>(
    input,
    output,
    offsets,
    output_sizes,
    output_dim,
    batch_size,
    input_sizes_1_2_3,
    input_sizes_2_3,
    input_sizes_3);
}

template void remove_padding_kernelLauncher<float>(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const cudaStream_t stream);

template void remove_padding_kernelLauncher<c10::Half>(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size,
    const cudaStream_t stream);
}
}
