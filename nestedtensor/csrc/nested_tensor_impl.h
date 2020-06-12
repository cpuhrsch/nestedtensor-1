#pragma once

#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor.h>

namespace at {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using SizeNode = NestedNode<c10::List<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

constexpr auto NestedTensorKey = DispatchKey::PrivateUse1_PreAutograd;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensor(at::Tensor&& buffer, SizeNode nested_size, SizeNode nested_stride);
  explicit NestedTensor(at::Tensor&& buffer, SizeNode nested_size);
  explicit NestedTensor(const TensorNode structure);

  int64_t dim() const override {
    return _data.dim();
  }
  int64_t numel() const override {
    return _data.numel();
  }
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    return _data.is_contiguous();
  }

  IntArrayRef sizes() const override;
  int64_t size(int64_t dim) const override;
  IntArrayRef strides() const override;

  const TensorNode get_structure() const;

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
    auto flattened = flatten(_nested_size);
    if (flattened.size() > 0) { 
      return flattened.get(0).size() + nested_dim();
    }
    return nested_dim();
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
  NestedTensor NestedTensor::contiguous() const {
    if (is_contiguous()) {
      return *this;
    }
    return NestedTensor(_buffer.contiguous(), _nested_size, _nested_stride);
  }
  const TensorNode get_structure() const;

  // torch.Tensor methods
  NestedTensor copy_(const NestedTensor& source, bool non_blocking = false);
  NestedTensor squeeze_(c10::optional<int64_t> dim);

 private:
  std::vector<int64_t> _sizes;
  at::Tensor _buffer;
  SizeNode _nested_size;
  SizeNode _nested_stride;
};

inline bool is_nested_tensor_impl(const at::Tensor tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
}

inline at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor) {
  if (!is_nested_tensor_impl(tensor)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline torch::nested_tensor::NestedTensor get_nested_tensor(
    const at::Tensor tensor) {
  return get_nested_tensor_impl(tensor)->_data;
}

inline torch::nested_tensor::TensorNode get_nested_tensor_structure(
    const at::Tensor tensor) {
  return get_nested_tensor(tensor).get_structure();
}

inline at::Tensor get_nested_tensor_buffer(
    const at::Tensor tensor) {
  return get_nested_tensor(tensor).get_buffer();
}

inline torch::nested_tensor::SizeNode get_nested_size(
    const at::Tensor tensor) {
  return get_nested_tensor(tensor).nested_size();
}

inline bool is_tensor_shape(const at::Tensor tensor) {
  auto nt = get_nested_tensor(tensor);
  for (const auto& size : nt.sizes()) {
    if (!size) {
      return false;
    }
  }
  return true;
}

inline at::Tensor wrap_nested_tensor(
    torch::nested_tensor::NestedTensor&& result) {
  return at::detail::make_tensor<NestedTensorImpl>(std::move(result));
}

inline at::Tensor wrap_buffer(
    at::Tensor&& buffer,
    torch::nested_tensor::SizeNode&& nested_size) {
  auto nt = torch::nested_tensor::NestedTensor(
      std::move(buffer), std::move(nested_size));
  return at::detail::make_tensor<NestedTensorImpl>(std::move(nt));
}

inline at::Tensor wrap_tensor_node(
    const torch::nested_tensor::TensorNode result) {
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(result)));
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_);

inline std::ostream& operator<<(
    std::ostream& out,
    const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor._data.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

} // namespace at
