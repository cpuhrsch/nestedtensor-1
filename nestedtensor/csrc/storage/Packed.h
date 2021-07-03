#pragma once
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <nestedtensor/csrc/storage/StorageBase.h>
#include <nestedtensor/csrc/utils/nested_node.h>

namespace torch {
namespace nested_tensor {
namespace impl {

inline at::Tensor pack(const TensorNode& structure) {
  TORCH_CHECK(structure.height() == 1, "Expected structure of height 1, got ", structure.height(), " instead.");
  if (structure.degree() == 0) {
    return at::ones({0});
  }
  auto tensor_nodes = structure.unbind();
  std::vector<at::Tensor> tensors;
  tensors.resize(structure.degree());
  int64_t full_numel = 0;
  for (size_t i = 0; i < tensors.size(); i++) {
    tensors[i] = tensor_nodes[i].payload();
    full_numel = full_numel + tensors[i].numel();
  }
  at::Tensor result_buffer = empty({full_numel}, tensors[0].options());
  int64_t index = 0;
  for (size_t i = 0; i < tensors.size(); i++) {
    at::Tensor narrowed_result_buffer = 
      result_buffer.narrow(0, index, tensors[i].numel());
    narrowed_result_buffer = narrowed_result_buffer.reshape(tensors[i].sizes());
    narrowed_result_buffer.copy_(tensors[i], true);
    index = index + tensors[i].numel();
  }
  return result_buffer;
}

inline bool storage_is_contiguous(
    const at::Tensor& buffer,
    const EfficientSizeNode& nested_size,
    const EfficientSizeNode& nested_stride) {
  if (!buffer.is_contiguous()) {
    return false;
  }
  if (buffer.numel() == 0) {
    return true;
  }
  const at::Tensor& sizes_sizes = nested_size.sizes();
  const at::Tensor& strides_sizes = nested_stride.sizes();
  int64_t* sizes_sizes_ptr = sizes_sizes.data_ptr<int64_t>();
  int64_t* strides_sizes_ptr = strides_sizes.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes_sizes.size(0); i++) {
    if (!_is_cont_stride(
            sizes_sizes_ptr + i * sizes_sizes.size(1),
            strides_sizes_ptr + i * strides_sizes.size(1),
            sizes_sizes.size(1))) {
      return false;
    }
  }
  return true;
}

} // namespace impl

struct PackedStorage : public NestedTensorStorage {
  explicit PackedStorage(
      at::Tensor&& buffer,
      EfficientSizeNode nested_size,
      EfficientSizeNode nested_stride)
      : _buffer(buffer.view(-1)),
        _nested_size(nested_size),
        _nested_stride(nested_stride),
        _data_type(buffer.dtype()),
        _device(buffer.device()),
        _is_pinned(buffer.is_pinned()),
        _is_contiguous(impl::storage_is_contiguous(
            _buffer,
            _nested_size,
            _nested_stride)) {
    TORCH_CHECK(
        _nested_size.height(),
        "PackedStorage must be given NestedSize of at least height 1.");
    TORCH_CHECK(
        _nested_stride.height(),
        "PackedStorage must be given NestedStride of at least height 1.");
  }

  explicit PackedStorage(
      at::Tensor&& buffer,
      EfficientSizeNode nested_size)
      : PackedStorage(std::move(buffer),
                      nested_size,
                      impl::_cont_stride(nested_size)) {}

  explicit PackedStorage(
      at::Tensor&& buffer,
      SizeNode nested_size,
      SizeNode nested_stride)
      : PackedStorage(
            std::move(buffer),
            EfficientSizeNode(nested_size),
            EfficientSizeNode(nested_stride)) {}

  explicit PackedStorage(at::Tensor&& buffer, SizeNode nested_size)
      : PackedStorage(
            std::move(buffer),
            EfficientSizeNode(nested_size)) {}

  explicit PackedStorage(TensorNode structure)
      : PackedStorage(
            impl::pack(structure),
            EfficientSizeNode(
              map([](at::Tensor tensor) { return tensor.sizes().vec(); },
                structure))) {}

  int64_t dim() const override {
    return _nested_size.dim();
  }
  TensorNode get_structure() const override {
    return std::get<0>(impl::build_structure(
        _buffer.view({-1}),
        _nested_size,
        _nested_stride));
  }
  at::Tensor& get_buffer() {
    return _buffer;
  }
  const at::Tensor& get_buffer() const {
    return _buffer;
  }
  const caffe2::TypeMeta dtype() const override {
    return _data_type;
  }
  c10::Device device() const override {
    return _device;
  }
  bool is_pinned() const override {
    return _is_pinned;
  }
  const EfficientSizeNode& nested_size() const override {
    return _nested_size;
  }
  const EfficientSizeNode& nested_stride() const override {
    return _nested_stride;
  }
  const std::vector<c10::optional<int64_t>> opt_sizes() const override {
    return _nested_size.opt_sizes();
  }
  NestedTensorStorageKind kind() const override {
    return NestedTensorStorageKind::packed;
  }
  bool is_contiguous() const override {
    return _is_contiguous;
  }
  bool is_cuda() const override {
    return _buffer.is_cuda();
  }
  int64_t numel() const override {
    return _nested_size.numel();
  }

 private:
  at::Tensor _buffer;
  EfficientSizeNode _nested_size;
  EfficientSizeNode _nested_stride;
  const caffe2::TypeMeta _data_type;
  c10::Device _device;
  bool _is_pinned;
  const bool _is_contiguous;
};

} // namespace nested_tensor
} // namespace torch
