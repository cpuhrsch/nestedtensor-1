#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <nestedtensor/csrc/cuda/transpose.h>
#include <stdio.h>

namespace nested_tensor {
namespace cuda {

// template<int num_threads>
// __global__
// void transpose(
//     const c10::Half* input,
//     c10::Half* output,
//     const int* offsets,
//     const int* sizes_dim2,
//     const int* sizes_dim3,
//     const int batch_size,
//     const int numel)
// {
//   const int tid = threadIdx.x;
//   const int ii  = (blockIdx.x * num_threads) + tid;
//   if (ii < numel) {
//     int batch_id = 0;
//     for (; batch_id < batch_size; batch_id++) {
//       if (offsets[batch_id + 1] > ii) {
//         break;
//       }
//     }
//     const int size2 = sizes_dim2[batch_id];
//     const int size3 = sizes_dim3[batch_id];
//     const int offset = offsets[batch_id];
//     const int j = ((ii - offset) % size3) * size2;
//     const int i = ((ii - offset) / size3);
//     output[offset + j + i] = __ldg(reinterpret_cast<const __half*>(input) + ii);
//   }
// }

template<int num_threads_sqrt>
__global__
void transpose(
    c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* sizes_dim2,
    const int* sizes_dim3,
    const int batch_size)
{
  const int batch_id  = blockIdx.x;
  const int grain_size = num_threads_sqrt;
  const int tid2 = threadIdx.x;
  const int tid3 = threadIdx.y;
  const int size2 = sizes_dim2[batch_id];
  const int size3 = sizes_dim3[batch_id];
  const int num_chunks_2 = size2 / grain_size;
  const int num_chunks_3 = size3 / grain_size;
  for (int id2 = 0; id2 < num_chunks_2; id2++) {
    for (int id3 = 0; id3 < num_chunks_3; id3++) {
      int ii2 = id2 * grain_size + tid2;
      int ii3 = id3 * grain_size + tid3;
      int ii = ii2 * size3 + ii3;
      const int j = (ii % size3) * size2;
      const int i = (ii / size3);
      output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
    }
  }
  for (int id2 = 0; id2 < num_chunks_2; id2++) {
    const int leftover3 = num_chunks_3 * grain_size;
    if (leftover3 + tid3 < size3) {
      int ii2 = id2 * grain_size + tid2;
      int ii3 = leftover3 + tid3;
      int ii = ii2 * size3 + ii3;
      const int j = (ii % size3) * size2;
      const int i = (ii / size3);
      output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
    }
  }
  const int leftover2 = num_chunks_2 * grain_size;
  if (leftover2 + tid2 < size2) {
    for (int id3 = 0; id3 < num_chunks_3; id3++) {
      int ii2 = leftover2 + tid2;
      int ii3 = id3 * grain_size + tid3;
      int ii = ii2 * size3 + ii3;
      const int j = (ii % size3) * size2;
      const int i = (ii / size3);
      output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
    }
  }
  if (leftover2 + tid2 < size2) {
    const int leftover3 = num_chunks_3 * grain_size;
    if (leftover3 + tid3 < size3) {
      int ii2 = leftover2 + tid2;
      int ii3 = leftover3 + tid3;
      int ii = ii2 * size3 + ii3;
      const int j = (ii % size3) * size2;
      const int i = (ii / size3);
      output[offsets[batch_id] + j + i] = input[offsets[batch_id] + ii];
    }
  }
}

void transpose_kernelLauncher(
    c10::Half* input, // [batch_size x None]
    c10::Half* output, // [batch_size x max(input.nested_size(1)) x inner_size]
    const int* offsets, // [batch_size]
    const int* sizes_dim2,
    const int* sizes_dim3,
    const int batch_size,
    const int numel,
    const cudaStream_t stream)
{
  dim3 grid;
  grid.x = batch_size;

  transpose<16><<<grid, dim3(16, 16), 0, stream>>>(
      input,
      output,
      offsets,
      sizes_dim2,
      sizes_dim3,
      batch_size);
}

}
} // namespace nested_tensor
