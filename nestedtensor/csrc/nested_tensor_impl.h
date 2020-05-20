#pragma once

#include <ATen/ATen.h>
#include <nestedtensor/csrc/nested_tensor.h>
#include <torch/csrc/autograd/variable.h>

namespace at {

namespace impl {

using namespace torch::autograd;

struct TORCH_API MyAutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;

  std::vector<std::shared_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list;

  // Only meaningful on leaf variables (must be false otherwise)
  bool requires_grad_;

  // Only meaningful on non-leaf variables (must be false otherwise)
  bool retains_grad_;

  bool is_view_;

  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  uint32_t output_nr_;

  // Mutex to ensure that concurrent read operations that modify internal
  // state are still thread-safe. Used by grad_fn() and
  // grad_accumulator().
  std::mutex mutex_;

  /// Sets the `requires_grad` property of `Variable`. This should be true for
  /// leaf variables that want to accumulate gradients, and false for all other
  /// variables.
  void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl)
      override;

  bool requires_grad() const override {
    std::cout << "ADADA1" << std::endl;
    return requires_grad_ || grad_fn_;
  }

  /// Accesses the gradient `Variable` of this `Variable`.
  Variable& grad() override {
    std::cout << "ADADA2" << std::endl;
    return grad_;
  }

  const Variable& grad() const override {
    std::cout << "ADADA3" << std::endl;
    return grad_;
  }

  MyAutogradMeta(
      at::TensorImpl* self_impl = nullptr,
      bool requires_grad = false,
      Edge gradient_edge = Edge()) {
    std::cout << "ADADA4" << std::endl;
    grad_fn_ = std::move(gradient_edge.function);
    requires_grad_ = false;
    retains_grad_ = false;
    is_view_ = false;
    output_nr_ = gradient_edge.input_nr;

    // set_requires_grad also checks error conditions.
    if (requires_grad) {
      TORCH_INTERNAL_ASSERT(self_impl);
      set_requires_grad(requires_grad, self_impl);
    }
    TORCH_CHECK(
        !grad_fn_ || !requires_grad_,
        "requires_grad should be false if grad_fn is set");
  }
};

} // namespace impl

constexpr auto NestedTensorKey = DispatchKey::PrivateUse1_PreAutograd;

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(torch::nested_tensor::NestedTensor&& data)
      : TensorImpl(
            c10::DispatchKeySet(NestedTensorKey),
            data.dtype(),
            data.device()),
        _data(std::move(data)) {
    set_autograd_meta(std::make_unique<impl::MyAutogradMeta>());
  }

  int64_t dim() const override {
    return _data.dim();
  }
  int64_t numel() const override {
    return _data.numel();
  }
  bool is_contiguous(
      at::MemoryFormat memory_format) const override {
    return _data.is_contiguous();
  }

  IntArrayRef sizes() const override;
  int64_t size(int64_t dim) const override;
  IntArrayRef strides() const override;

  torch::nested_tensor::NestedTensor _data;

};

inline torch::nested_tensor::NestedTensor get_nested_tensor(
    const at::Tensor tensor) {
  if (!tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  auto nt_impl =
      static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
  return nt_impl->_data;
}

inline at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor) {
  if (!tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey)) {
    throw std::runtime_error("Function requires NestedTensorImpl");
  }
  return static_cast<at::NestedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline at::Tensor wrap_nested_tensor(
    torch::nested_tensor::NestedTensor&& result) {
  return at::detail::make_tensor<NestedTensorImpl>(std::move(result));
}

inline at::Tensor wrap_tensor_node(
    torch::nested_tensor::TensorNode&& result) {
  return at::detail::make_tensor<NestedTensorImpl>(
      torch::nested_tensor::NestedTensor(std::move(result)));
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_);

inline std::ostream& operator<<(std::ostream& out, const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor._data.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

}
