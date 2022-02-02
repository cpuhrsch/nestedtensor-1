/*
* Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

Changes in comparison to original.
- Removed any code unrelated to softmax

*/

#include <nestedtensor/csrc/cuda/softmax.h>
#include <stdio.h>
#include <c10/util/Half.h>

namespace fastertransformer 
{

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
    #pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
    __inline__ __device__
T blockReduceSum(T val)
{
    static __shared__ T shared[32]; 
    int lane = threadIdx.x & 0x1f; 
    int wid = threadIdx.x >> 5;  

    val = warpReduceSum<T>(val);

    if(lane == 0)
    shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);
                                
    return val;
}

template <typename T>
    __inline__ __device__
T warpReduceMax(T val)
{
    #pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
    __inline__ __device__
T blockReduceMax(T val)
{
    static __shared__ T shared[32]; 
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;  // warp idx

    val = warpReduceMax(val); // get maxx in each warp

    if(lane == 0) // record in-warp maxx by warp Idx
    shared[wid] = val;

    __syncthreads();


    val = (threadIdx.x < (blockDim.x >> 5 )) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

__inline__ __device__
int target_index(int id1, int id2, int id3, int id4, int dim_1, int dim_2, int dim_3, int dim_4)
{
  return id1 * (dim_2 * dim_3 * dim_4) + id3 * (dim_2 * dim_4) + id2 * dim_4 + id4;
}

template <typename T>
__global__
void softmax_kernel(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, 
  const T scalar)
{
    int batch_id = blockIdx.x / head_num;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    for(int i = 0; i < seq_len; ++i)
    {
      float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
      float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
      mask_val = (1.0f - mask_val) * -10000.0f;

      float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val): -1e20f;

      float max_val = blockReduceMax<float>(tmp);

      if(threadIdx.x == 0)
        s_max = max_val;
      __syncthreads();

      qk = threadIdx.x < seq_len ? __expf(tmp - s_max) : 0.0f;

      float sum_val = blockReduceSum<float>(qk);

      if(threadIdx.x == 0)
      {
        s_sum = sum_val + 1e-6f;
      }
      __syncthreads();

      if(threadIdx.x < seq_len)
        qk_buf_[threadIdx.x + qk_offset] = (T)(qk / s_sum);

      qk_offset += seq_len;
      mask_offset += seq_len;
    }
}


template <typename T>
__global__
void softmax_kernel_v2(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, 
  const int seq_len, const float scalar)
{
    int batch_id = blockIdx.x / head_num / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int qk_offset = blockIdx.x * seq_len;
    int mask_offset = batch_id * seq_len * seq_len + seq_id * seq_len;

    __shared__ float s_sum, s_max;

    float qk = threadIdx.x < seq_len ? (float)qk_buf_[threadIdx.x + qk_offset] : 0.0f;
    float mask_val = threadIdx.x < seq_len ? (float)attr_mask[threadIdx.x + mask_offset] : 0.0f;
      
    mask_val = (1.0f - mask_val) * -10000.0f;

    float tmp = threadIdx.x < seq_len ? (float)(qk * (float)scalar + mask_val) : -1e20f;
    float max_val = blockReduceMax<float>(tmp);
    if(threadIdx.x == 0)
      s_max = max_val;
    __syncthreads();

    float qk_tmp = threadIdx.x < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);

    if(threadIdx.x == 0)
    {
      s_sum = sum_val + 1e-6f;
    }
    __syncthreads();

    if(threadIdx.x < seq_len)
      qk_buf_[threadIdx.x + qk_offset] = (T)(qk_tmp / s_sum);
}

//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len + 31)/32*32)
template <typename T>
__global__
void softmax_kernel_v3(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
    
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    float tmp = -1e20f;
    int qk_offset;
    __shared__ float s_mean, s_max;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = qk * static_cast<float>(scalar) + mask_val;
    }

    float max_val = blockReduceMax<float>(tmp);
    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();
    
    float qk_tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = blockReduceSum<float>(qk_tmp);
    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();
    
    if(qual)
      qk_buf_[qk_offset] = (T)(qk_tmp * s_mean);
  }
}  


//grid = (seq_len/word_per_thread, batch_size, head_num)
//block.x = max(32, (seq_len/2 + 31)/32*32)
//seq_len % 2 == 0
template <>
__global__
void softmax_kernel_v3(half* qk_buf_, const half* attr_mask, 
                      const int batch_size, const int head_num, 
                      const int seq_len, const half scalar)
{
  int threadIdx2 = threadIdx.x << 1;
  bool qual = threadIdx2 < seq_len;
  half2* qk_buf_half2Ptr = (half2*) qk_buf_;
  const half2* attr_mask_half2Ptr = (const half2*) attr_mask;
  __shared__ float s_mean, s_max;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    half2 tmp = __float2half2_rn(0.0f);

    float max_val = -1e20f;
    half2 qk;
    if (qual){ 
      qk_offset = ((((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len) >> 1) + threadIdx.x;
      int mask_offset = (((blockIdx.y * seq_len + seq_id) * seq_len) >> 1) + threadIdx.x;

      qk = qk_buf_half2Ptr[qk_offset];
      half2 mask_val = __ldg(&attr_mask_half2Ptr[mask_offset]);
      half2 mask_val_tmp = __hmul2(__hsub2(__float2half2_rn(1.0f), mask_val), __float2half2_rn(-10000.0f));
      tmp = __hadd2(__hmul2(__half2half2(scalar), qk), mask_val_tmp);
      max_val = fmax((float)tmp.x, (float)tmp.y);
    }
    
    max_val = blockDim.x <= 32 ? warpReduceMax(max_val) : blockReduceMax<float>(max_val);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();
    
    if (qual){
      tmp = h2exp(__hsub2(tmp, __float2half2_rn(s_max)));
    }
    float sum_val = blockDim.x <= 32 ? warpReduceSum((float)(tmp.x + tmp.y)) : blockReduceSum<float>((float)(tmp.x + tmp.y));

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual){
      qk = __hmul2(tmp, __float2half2_rn(s_mean));
      qk_buf_half2Ptr[qk_offset] = qk;
    }
  }
}

template <typename T>
__global__
void softmax_kernel_v3_LE32(T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
  bool qual = threadIdx.x < seq_len;
  for (int seq_id = blockIdx.x ; seq_id < seq_len ; seq_id += gridDim.x){
    int qk_offset;
    __shared__ float s_mean, s_max;
    float tmp = -1e20f;
    if (qual){
      qk_offset = ((blockIdx.y*head_num + blockIdx.z)*seq_len + seq_id) *seq_len + threadIdx.x;
      int mask_offset = (blockIdx.y * seq_len + seq_id) * seq_len + threadIdx.x;

      float qk = static_cast<float>(qk_buf_[qk_offset]);
      float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

      mask_val = (1.0f - mask_val) * -10000.0f;

      tmp = static_cast<float>(qk) * static_cast<float>(scalar) + mask_val;
    }
    float max_val = warpReduceMax<float>(tmp);

    if (threadIdx.x == 0){
      s_max = max_val;
    }
    __syncthreads();

    tmp = qual ? __expf(tmp - s_max) : 0.0f;
    float sum_val = warpReduceSum<float>(tmp);

    if (threadIdx.x == 0){
      s_mean = sum_val + 1e-6f;
      s_mean = __fdividef(1.0f, s_mean);
    }
    __syncthreads();

    if(qual)
      qk_buf_[qk_offset] = (T)(tmp * s_mean);
  }
}

template<typename T>
void attn_softmax_kernelLauncher(
  T* buffer,
  const T* attr_mask,
  const int batch_size,
  const int seq_len,
  const int head_num,
  const T scalar,
  cudaStream_t stream)
{
  dim3 grid, block;
  //deal with odd seq_len
  if (seq_len % 2 != 0){
    if(seq_len <= 32)
      block.x = 32;
    else if(seq_len > 32 && seq_len <= 64)
      block.x = 64;
    else if(seq_len > 64 && seq_len <= 128)
      block.x = 128;
    else if(seq_len > 128 && seq_len <= 256)
      block.x = 256;
    else if(seq_len > 256 && seq_len <= 512)
      block.x = 512;
    else
      block.x = 1024;

    if(batch_size * head_num <= 120)
    {
      grid.x = batch_size * head_num * seq_len;
      softmax_kernel_v2<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    }
    else
    {
      grid.x = batch_size * head_num;
      softmax_kernel<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    }
  }
  //deal with even seq_len 
  else{
    grid.x = seq_len;
    if (batch_size * head_num > 360)
      grid.x = ceil(float(seq_len)/32.0f);
    grid.y = batch_size;
    grid.z = head_num;
    if (seq_len <= 32){
      block.x = 32;
      softmax_kernel_v3_LE32<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
    }
    else{
      if (sizeof(T) == 2){
        block.x = (seq_len/2 + 31)/32*32;
        softmax_kernel_v3<<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
      }
      else{
        block.x = (seq_len + 31)/32*32;
        softmax_kernel_v3<T><<<grid, block, 0, stream>>>(buffer, attr_mask, batch_size, head_num, seq_len, scalar);
      }
    }
    grid.x = grid.y = grid.z = 1;
  }
}

template void attn_softmax_kernelLauncher(
    float* buffer,
    const float* attr_mask,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const float scalar,
    cudaStream_t stream);
    
template void attn_softmax_kernelLauncher(
    half* buffer,
    const half* attr_mask,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const half scalar,
    cudaStream_t stream);

template void attn_softmax_kernelLauncher(
    c10::Half* buffer,
    const c10::Half* attr_mask,
    const int batch_size,
    const int seq_len,
    const int head_num,
    const c10::Half scalar,
    cudaStream_t stream);
      
} // namespace fastertransformer
