#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/storage/StorageBase.h>

namespace torch {
namespace nested_tensor {

struct PaddedStorage : public NestedTensorStorage {
  explicit PaddedStorage(
      at::Tensor&& padded,
      EfficientSizeNode nested_size)
      : _padded(padded),
        _nested_size(nested_size) {
    TORCH_CHECK(
        _nested_size.height(),
        "PaddedStorage must be given NestedSize of at least height 1.");
  }
  int64_t dim() const override {
    return _nested_size.dim();
  }
  TensorNode get_structure() const override {
    TORCH_CHECK(false, "get_structure not implemented for PaddedStorage");
  }
  at::Tensor& get_padded() {
    return _padded;
  }
  const at::Tensor& get_padded() const {
    return _padded;
  }
  const caffe2::TypeMeta dtype() const override {
    return _padded.dtype();
  }
  c10::Device device() const override {
    return _padded.device();
  }
  bool is_pinned() const override {
    return _padded.is_pinned();
  }
  const EfficientSizeNode& nested_size() const override {
    return _nested_size;
  }
  const EfficientSizeNode& nested_stride() const override {
    TORCH_CHECK(false, "nested_stride not implemented for PaddedStorage");
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const override {
    return _nested_size.opt_sizes();
  }
  NestedTensorStorageKind kind() const override {
    return NestedTensorStorageKind::padded;
  }
  bool is_contiguous() const override {
    return _padded.is_contiguous();
  }
  bool is_cuda() const override {
    return _buffer.is_cuda();
  }
  int64_t numel() const override {
    return _nested_size.numel();
  }

 private:
  at::Tensor _padded;
  EfficientSizeNode _nested_size;
};

} // namespace nested_tensor
} // namespace torch
