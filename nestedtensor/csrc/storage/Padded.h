#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/storage/StorageBase.h>

namespace torch {
namespace nested_tensor {
namespace impl {

inline std::vector<int64_t> _get_max_size(const SizeNode& size_node) {
  std::vector<int64_t> result;
  if (size_node.is_leaf()) {
    for (const auto& size : size_node.payload()) {
      result.push_back(size);
    }
    return result;
  }
  if (size_node.degree() > 0) {
    std::vector<int64_t> first_size = _get_max_size(size_node.children(0));
    for (const auto& size : first_size) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      std::vector<int64_t> ith_size = _get_max_size(size_node.children(i));
      for (size_t j = 0; j < ith_size.size(); j++) {
        result[j] = result[j] > ith_size[j] ? result[j] : ith_size[j];
      }
    }
  }
  return result;
}

inline std::vector<int64_t> get_max_size(EfficientSizeNode nested_size) {
  if (nested_size.height() == 1){
    auto nt_opt_sizes = nested_size.opt_sizes();
    if (nt_opt_sizes.size() > 0 && *nt_opt_sizes[0] > 0) {
      auto sizes = nested_size.sizes();
      auto max_sizes = std::get<0>(sizes.max(0));
      std::vector<int64_t> result;
      for (int64_t i = 0; i < max_sizes.size(0); i++) {
        result.push_back(max_sizes[i].item<int64_t>());
      }
      return result;
    }
  }
  return impl::_get_max_size(nested_size.to_size_node());
}

}

struct PaddedStorage : public NestedTensorStorage {
  explicit PaddedStorage(
      at::Tensor&& padded,
      EfficientSizeNode nested_size)
      : _nested_size(nested_size),
        _padded_size(impl::get_max_size(_nested_size)),
        _padded(padded.reshape(_padded_size)) {
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
  at::Tensor get_padded() {
    return _padded;
  }
  const at::Tensor& get_padded() const {
    return _padded;
  }
  std::vector<int64_t> padded_size() const {
    return _padded_size;
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
    return _padded.is_cuda();
  }
  int64_t numel() const override {
    return _nested_size.numel();
  }

 private:
  const EfficientSizeNode _nested_size;
  const std::vector<int64_t> _padded_size;
  at::Tensor _padded;
};

} // namespace nested_tensor
} // namespace torch
