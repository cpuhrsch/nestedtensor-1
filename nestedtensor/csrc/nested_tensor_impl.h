#pragma once
#include <ATen/ATen.h>
#include <c10/util/Metaprogramming.h>
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace torch {
namespace nested_tensor {

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;
using SizeNode = NestedNode<c10::List<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

} // namespace nested_tensor
} // namespace torch

namespace at {

using namespace torch::nested_tensor;

constexpr auto NestedTensorKey = DispatchKey::PrivateUse1_PreAutograd;

struct NestedTensorImpl;

template <class A>
bool is_nested_tensor_impl(A tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(at::NestedTensorKey);
}

template <class A, class B>
bool is_nested_tensor_impl(A first, B other) {
  return is_nested_tensor_impl(first) && is_nested_tensor_impl(other);
}

template <class A, class B, class... C>
bool is_nested_tensor_impl(A first, B second, C... other) {
  return is_nested_tensor_impl(first, second) &&
      is_nested_tensor_impl(other...);
}

template <class A>
inline bool tensor_shape_matches(A a) {
  TORCH_CHECK(
      is_nested_tensor_impl(a), "Can only compare shapes of NestedTensors.");
  return true;
}

template <class A, class B>
inline bool tensor_shape_matches(A a, B b) {
  TORCH_CHECK(
      is_nested_tensor_impl(a, b), "Can only compare shapes of NestedTensors.");
  return shape_matches(
      get_nested_tensor_structure(a), get_nested_tensor_structure(b));
}

template <class A, class B, class... C>
inline bool tensor_shape_matches(A a, B b, C... c) {
  TORCH_CHECK(
      is_nested_tensor_impl(a, b, c...),
      "Can only compare shapes of NestedTensors.");
  return shape_matches(
             get_nested_tensor_structure(a), get_nested_tensor_structure(b)) &&
      tensor_shape_matches(b, c...);
}

template <class... A>
inline void torch_check_tensor_shape_matches(A... a) {
  TORCH_CHECK(
      is_nested_tensor_impl(a...), "Can only check shapes of NestedTensors.");
  TORCH_CHECK(tensor_shape_matches(a...), "NestedTensor shapes don't match.");
}

template <class F, class... A>
static inline void apply_nested_tensor(F&& fn, A... a) {
  torch_check_tensor_shape_matches(a...);
  apply(std::move(fn), get_nested_tensor_structure(a)...);
}

at::NestedTensorImpl* get_nested_tensor_impl(const at::Tensor tensor);
torch::nested_tensor::TensorNode get_nested_tensor_structure(
    const at::Tensor tensor);

at::Tensor wrap_tensor_node(NestedTensorImpl);
at::Tensor wrap_tensor_node(TensorNode&&);
std::vector<at::Tensor> wrap_tensor_node(std::vector<TensorNode>);

template <class F, class... A>
static inline at::Tensor map_nested_tensor(F&& fn, A... a) {
  torch_check_tensor_shape_matches(a...);
  return wrap_tensor_node(
      map(std::move(fn), get_nested_tensor_structure(a)...));
}

struct NestedTensorImpl : public c10::TensorImpl {
  explicit NestedTensorImpl(TensorNode structure);

  int64_t dim() const override {
    return _first_variable.dim() + nested_dim();
  }
  int64_t numel() const override {
    auto fn = [](at::Tensor leaf, int64_t input) {
      return input + leaf.numel();
    };
    return reduce<decltype(fn), int64_t, at::Tensor>(get_structure(), fn, 0);
  }
  bool is_contiguous(at::MemoryFormat memory_format) const override {
    // NOTE: The Tensors themselves might not be contiguous even if there is a
    // buffer. For this to be contiguous not only the individuals Tensors have
    // to be but also the buffer.
    auto fn = [](at::Tensor leaf, bool input) {
      return input && leaf.is_contiguous();
    };
    return reduce<decltype(fn), bool, at::Tensor>(get_structure(), fn, true);
  }
  TensorNode& get_structure() {
    return _structure;
  }
  const TensorNode& get_structure() const {
    return _structure;
  }
  void backward(Tensor gradient, bool retain_graph, bool create_graph) {
    apply(
        [retain_graph, create_graph](at::Tensor tensor1, at::Tensor tensor2)
            -> void { tensor1.backward(tensor2, retain_graph, create_graph); },
        get_structure(),
        get_nested_tensor_impl(gradient)->get_structure());
  }
  int64_t nested_dim() const {
    return get_structure().height();
  }
  Tensor to_nested_tensor(c10::optional<int64_t> dim);
  Tensor grad() {
    auto fn = [](at::Tensor leaf, bool input) {
      return input && leaf.grad().defined();
    };
    if (!reduce<decltype(fn), bool, at::Tensor>(get_structure(), fn, true)) {
      throw std::runtime_error("Grad is undefined");
    }
    return wrap_tensor_node(
        map([](at::Tensor tensor) { return tensor.grad(); }, get_structure()));
  }
  Tensor requires_grad_(bool requires_grad) {
    apply(
        [requires_grad](at::Tensor& tensor) -> void {
          tensor.set_requires_grad(requires_grad);
        },
        get_structure());
    return at::detail::make_tensor<NestedTensorImpl>(_structure);
  }
  bool requires_grad() const {
    return _first_variable.requires_grad();
  }
  bool is_pinned() const {
    return _first_variable.is_pinned();
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
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
        get_structure());
  }
  SizeNode nested_stride() const {
    return map(
        [](at::Tensor tensor) { return c10::List<int64_t>(tensor.strides()); },
        get_structure());
  }
  at::Tensor to_tensor();

  std::vector<c10::optional<int64_t>> opt_sizes() const;
  IntArrayRef sizes() const override {
    return IntArrayRef(_sizes);
  }
  int64_t size(int64_t dim) const override;
  IntArrayRef strides() const override;

 private:
  TensorNode _structure;
  at::Tensor _first_variable;
  SizeNode _nested_size;
  std::vector<int64_t> _sizes;
};

template <class F, F func, class R, class TypeList>
struct _Function_no_bw
    : public torch::autograd::Function<_Function_no_bw<F, func, R, TypeList>> {
};

template <class F, F func, class R, class... Args>
struct _Function_no_bw<F, func, R, c10::guts::typelist::typelist<Args...>>
    : public torch::autograd::Function<_Function_no_bw<F, func, R, Args...>> {
  static typename c10::guts::infer_function_traits<F>::type::return_type forward(
      torch::autograd::AutogradContext* ctx,
      Args... args) {
    return (*func)(args...);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output_) {
    TORCH_CHECK(false, "Backward not implemented for ", typeid(F).name());
    return {};
  }
};

// #define NO_BW(FUNC) \
// #define impl_UNBOXED_NO_BW(NAME< FUNC)
// template <class F, F func, class R, class... Args>
// struct _Function_no_bw<F, func, R, c10::guts::typelist::typelist<Args...>>
//     : public torch::autograd::Function<_Function_no_bw<F, func, R, Args...>> {
//   static typename c10::guts::infer_function_traits<F>::type::return_type forward(
//       torch::autograd::AutogradContext* ctx,
//       Args... args) {
//     return (*func)(args...);
//   }
//   static torch::autograd::variable_list backward(
//       torch::autograd::AutogradContext* ctx,
//       torch::autograd::variable_list grad_output_) {
//     TORCH_CHECK(false, "Backward not implemented for ", typeid(F).name());
//     return {};
//   }
// };
//   static_cast<decltype(FUNC)>( \
//   _Function_no_bw<decltype(&FUNC),&FUNC,typename c10::guts::infer_function_traits<decltype(&FUNC)>::type::return_type,\ 
//       typename c10::guts::infer_function_traits<decltype(&FUNC)>::type::parameter_types>::apply)

// template <class F, F fn, class R, class TypeList>
// struct __Function_no_bw;
//
// template <class F, F fn, class R, class... Args>
// struct __Function_no_bw<F, fn, R, c10::guts::typelist::typelist<Args...>> {
//  public:
//   static R function(Args... args) {
//     std::cout << "WTFFF" << std::endl;
//     return fn(args...);
//   }
// };
//
// template <class F, F fn>
// struct _Function_no_bw {
//   static auto f = __Function_no_bw<
//       F,
//       fn,
//       typename c10::guts::infer_function_traits<F>::type::return_type,
//       typename
//       c10::guts::infer_function_traits<F>::type::parameter_types>::function;
// };

// template <typename Fn, Fn func>
// typename std::result_of<Fn(Args...)>::type callfunc(Args&&... args) {
//   // do something else here
//   std::cout << "WTFFFF" << std::endl;
//   apply(args...);
// }

//  public:
//   static typename c10::guts::infer_function_traits<F>::type::return_type
//   function(typename c10::guts::infer_function_traits<
//            F>::type::parameter_types... args) {
//     std::cout << "WTF" << std::endl;
//     return fn(args...);
//   }
// };

// template <class F, F fn>
// static inline NestedNode<
//     typename c10::guts::infer_function_traits<F>::type::return_type>
//   public:
//   static auto _helper =  _Function_no_bw<
//       F,
//     fn,
//       typename c10::guts::infer_function_traits<F>::type::return_type,
//       typename
//       c10::guts::infer_function_traits<F>::type::parameter_types>::
//       function;

inline bool is_tensor_shape(const at::Tensor tensor) {
  auto nt = get_nested_tensor_impl(tensor);
  for (const auto& size : nt->opt_sizes()) {
    if (!size) {
      return false;
    }
  }
  return true;
}

Tensor NestedTensor_to_tensor(Tensor tensor, c10::optional<int64_t> dim_);

inline std::ostream& operator<<(
    std::ostream& out,
    const NestedTensorImpl& batch_tensor) {
  auto node = batch_tensor.get_structure();
  out << "NESTED_TENSOR";
  apply([&out](at::Tensor tensor) { out << tensor << std::endl; }, node);
  out << std::endl;
  return out;
}

} // namespace at
