#pragma once
#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {
namespace impl {

inline EfficientSizeNode _cont_stride(const EfficientSizeNode& nested_size) {
  auto nested_stride = map_efficient_size(
      [](int64_t* size_ptr, int64_t size) {
        auto cont_stride = _cont_stride(size_ptr, size);
        for (int64_t i = 0; i < size; i++) {
          size_ptr[i] = cont_stride[i];
        }
      }, nested_size);
  return nested_stride;
}

inline std::tuple<TensorNode, at::Tensor> build_structure(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size_,
    const EfficientSizeNode& nested_stride_) {
  TORCH_CHECK(
      buffer.dim() == 1, "Given buffer must be vector, i.e. dim 1 Tensor.");
  std::vector<int64_t> split_sizes;
  split_sizes.reserve(nested_size_.degree());
  map_efficient_size([&split_sizes] (int64_t* sizes_ptr0, int64_t* sizes_ptr1, int64_t size) {
      split_sizes.push_back(num_memory(sizes_ptr0, sizes_ptr1, size));
      }, nested_size_, nested_stride_);
  std::vector<int64_t> nonzero_split_sizes;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      nonzero_split_sizes.push_back(split_sizes[i]);
    }
  }
  std::vector<at::Tensor> buffers_;
  if (nonzero_split_sizes.size() > 0) {
    buffers_ =
        at::split_with_sizes(buffer, c10::IntArrayRef(nonzero_split_sizes), 0);
  }
  std::vector<at::Tensor> buffers;
  int64_t index = 0;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      buffers.push_back(buffers_[index]);
      index++;
    } else {
      buffers.push_back(at::empty({}, buffer.options()));
    }
  }
  std::vector<TensorNode> result_tensors;
  index = 0;
  map_efficient_size([&buffers, &result_tensors, &index](
        int64_t* size_ptr, int64_t* stride_ptr, int64_t size) {
      std::vector<int64_t> sizes(size_ptr, size_ptr + size);
      std::vector<int64_t> strides(stride_ptr, stride_ptr + size);
      result_tensors.push_back(TensorNode(at::as_strided(
            buffers[index], c10::IntArrayRef(sizes), c10::IntArrayRef(strides))));
      index++;
      }, nested_size_, nested_stride_);
  return std::make_tuple(TensorNode(std::move(result_tensors)), buffer);
}

inline std::tuple<TensorNode, at::Tensor> build_structure(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size) {
  TORCH_CHECK(
      buffer.dim() == 1, "Given buffer must be vector, i.e. dim 1 Tensor.");
  EfficientSizeNode nested_stride = _cont_stride(nested_size);
  return build_structure(buffer, nested_size, nested_stride);
}
}

enum NestedTensorStorageKind { packed, channellastpacked };

struct NestedTensorStorage {
  virtual ~NestedTensorStorage() = default;
  virtual int64_t dim() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual TensorNode get_structure() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const caffe2::TypeMeta dtype() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual c10::Device device() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual bool is_pinned() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const EfficientSizeNode& nested_size() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const EfficientSizeNode& nested_stride() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual const std::vector<c10::optional<int64_t>> opt_sizes() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual NestedTensorStorageKind kind() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual bool is_contiguous() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual bool is_cuda() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
  virtual int64_t numel() const {
    TORCH_CHECK(false, "Not Implemented.");
  }
};
} // namespace nested_tensor
} // namespace torch
