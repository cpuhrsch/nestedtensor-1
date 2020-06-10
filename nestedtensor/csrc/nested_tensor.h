#pragma once
#include <nestedtensor/csrc/utils/nested_node.h>

namespace torch {
namespace nested_tensor {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using SizeNode = NestedNode<c10::List<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

// TODO: Eventually allow construction from a list of _BufferNestedTensors.
struct NestedTensor {
  NestedTensor() = delete;
  NestedTensor(TensorNode&& structure);
  NestedTensor(at::Tensor&& buffer, TensorNode&& structure);
  NestedTensor(at::Tensor&& buffer, SizeNode nested_size);
  NestedTensor(at::Tensor&& buffer, SizeNode nested_size, SizeNode nested_stride);
  at::Tensor& get_buffer() {
    return _buffer;
  }
  const at::Tensor& get_buffer() const {
    return _buffer;
  }
  std::vector<c10::optional<int64_t>> sizes() const;
  caffe2::TypeMeta dtype() const {
    return _buffer.dtype();
  }
  int64_t element_size() const {
    return _buffer.element_size();
  }
  // This is a C++ representation of a nested list of torch.Sizes
  //
  // It can never be a list of just numbers, because torch.Size
  // is always a list and NestedTensors represent lists of torch.Tensors
  //
  // Noteworthy cases:
  //
  // This is an empty list of lists if we construct
  // nested_tensor([])
  // which is of nested_dim 1, dim 1 and tensor_dim 0
  //
  // This is a list of empty lists if we construct e.g.
  // nested_tensor([torch.tensor(0), torch.tensor(1), ...])
  // which is of nested_dim 1, dim 1 and tensor_dim 0
  //
  // This is a list of list of numbers if we construct e.g.
  // nested_tensor([torch.tensor([1]), torch.tensor([2]), ...])
  // which is of nested_dim 1, dim 2 and tensor_dim 1
  //
  // That means, if the list is not empty it is either a list of
  // lists of numbers or a list of empty lists.
  SizeNode nested_size() const {
    return _nested_size;
  }
  SizeNode nested_stride() const {
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.strides()); },
        get_structure());
  }
  NestedTensor pin_memory() {
    // NOTE: The assumption here is that pin_memory will materialize
    // the views that _structure contains when NestedTensor is contiguous.
    return NestedTensor(
        map([](at::Tensor tensor) { return tensor.pin_memory(); },
            get_structure()));
  }
  NestedTensor grad() {
    // TORCH_CHECK(
    //     _buffer.grad().contiguous(), "Gradient of buffer is not contiguous.");
    return NestedTensor(std::move(_buffer.grad()), _nested_size);
  }
  NestedTensor detach() {
    // NOTE: For the contiguous case the tensors in _structure are views
    // of parts of _buffer and the returned detached views will still
    // modify that buffer if using in-place methods etc.
    return NestedTensor(std::move(_buffer.grad()), _nested_size, _nested_stride);
  }
  NestedTensor requires_grad_(bool requires_grad) {
    _buffer.set_requires_grad(requires_grad);
    return *this;
  }
  void backward(NestedTensor gradient, bool retain_graph, bool create_graph) {
    _buffer.backward(gradient.get_buffer(), retain_graph, create_graph);
  }
  int64_t __len__() const {
    return _nested_size.degree();
  }
  at::Tensor to_tensor();
  NestedTensor to_nested_tensor(c10::optional<int64_t> dim);
  int64_t nested_dim() const {
    return _nested_size.height();
  }
  at::ScalarType scalar_type() const {
    return _buffer.scalar_type();
  }
  at::Backend backend() const {
    return options().backend();
  }
  at::Layout layout() const {
    return _buffer.layout();
  }
  at::Device device() const {
    return _buffer.device();
  }
  at::TensorOptions options() const {
    return _buffer.options();
  }
  bool requires_grad() const {
    return _buffer.requires_grad();
  }
  int64_t dim() const {
    return _buffer.dim() + nested_dim();
  }
  int64_t numel() const {
    auto fn = [](at::Tensor leaf, int64_t input) {
      return input + leaf.numel();
    };
    return reduce<decltype(fn), int64_t, at::Tensor>(get_structure(), fn, 0);
  }
  bool is_pinned() const {
    return _buffer.is_pinned();
  }
  bool is_contiguous() const {
    // NOTE: The Tensors themselves might not be contiguous even if there is a
    // buffer. For this to be contiguous not only the individuals Tensors have
    // to be but also the buffer.
    auto fn = [](at::Tensor leaf, bool input) {
      return input && leaf.is_contiguous();
    };
    return _buffer.is_contiguous() &&
        reduce<decltype(fn), bool, at::Tensor>(get_structure(), fn, true);
  }
  NestedTensor contiguous() const;
  const TensorNode get_structure() const;

  // torch.Tensor methods
  NestedTensor copy_(const NestedTensor& source, bool non_blocking = false);
  NestedTensor squeeze_(c10::optional<int64_t> dim);

 private:
  at::Tensor _buffer;
  SizeNode _nested_size;
  SizeNode _nested_stride;
};

} // namespace nested_tensor
} // namespace torch
