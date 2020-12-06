#include <nestedtensor/csrc/BinaryOps.h>

namespace at {

using namespace torch::nested_tensor;

Tensor& NestedTensor_sub_(Tensor& self, const Tensor& other, Scalar alpha) {
  check_binary_shape(self, other);
  if (is_nested_tensor_impl(self, other)) {
    torch_check_tensor_shape_matches(self, other);
    apply_nested_tensor(
        [&alpha](Tensor& tensor, Tensor& other) {
          at::native::sub_(tensor, other, alpha);
        },
        self,
        other);
    return self;
  }
  if (is_nested_tensor_impl(self)) {
    torch_check_tensor_shape_matches(self);
    apply_nested_tensor(
        [&other, &alpha](Tensor& self) {
          at::native::sub_(self, other, alpha);
        },
        self);
    return self;
  }
  torch_check_tensor_shape_matches(other);
  apply_nested_tensor(
      [&self, &alpha](Tensor& other) { at::native::sub_(self, other, alpha); },
      other);
  return self;
}

Tensor& NestedTensor_sub_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  TORCH_CHECK(
      is_nested_tensor_impl(result),
      "NT binary out variant requires NT as result argument.");
  check_binary_shape(self, other);
  is_nested_tensor_impl(result, self, other);
  apply_nested_tensor(
      [&alpha](Tensor& result, Tensor& tensor, Tensor& other) {
        return at::sub_out(result, tensor, other, alpha);
      },
      result,
      self,
      other);
  return result;
}

Tensor& NestedTensor_pow_out_1(
    Tensor& result,
    const Tensor& base,
    const Tensor& exp) {
  TORCH_CHECK(
      is_nested_tensor_impl(result),
      "NT binary out variant requires NT as result argument.");
  check_binary_shape(base, exp);
  if (is_nested_tensor_impl(result, base, exp)) {
    torch_check_tensor_shape_matches(result, base, exp);
    apply_nested_tensor(
        [](Tensor& result, Tensor& base, Tensor& exp) {
          at::pow_out(result, base, exp);
        },
        result,
        base,
        exp);
    return result;
  }
  if (is_nested_tensor_impl(result, base)) {
    torch_check_tensor_shape_matches(result, base);
    apply_nested_tensor(
        [&exp](Tensor& result, Tensor& base) {
          at::pow_out(result, base, exp);
        },
        result,
        base);
    return result;
  }
  TORCH_CHECK(
      is_nested_tensor_impl(result, exp),
      "At least one of base or exp needs to be a NestedTensor");
  torch_check_tensor_shape_matches(result, exp);
  apply_nested_tensor(
      [&exp](Tensor& result, Tensor& base) { at::pow_out(result, base, exp); },
      result,
      base);
  return result;
}

Tensor& NestedTensor_pow__1(Tensor& base, const Tensor& other) {
  check_binary_shape(base, other);
  return NestedTensor_pow_out_1(base, base, other);
}

Tensor& NestedTensor_pow_out_2(Tensor& result, const Tensor& base, Scalar exp) {
  apply_nested_tensor(
      [&exp](Tensor& result, Tensor& base) {
        return at::pow_out(result, base, exp);
      },
      result,
      base);
  return result;
}

Tensor NestedTensor_pow_2(const Tensor& base, Scalar exp) {
  return map_nested_tensor(
      [exp](Tensor base) { return at::pow(base, exp); }, base);
}

Tensor& NestedTensor_pow_out_3(Tensor& result, Scalar base, const Tensor& exp) {
  apply_nested_tensor(
      [&base](Tensor& result, Tensor& exp) {
        return at::pow_out(result, base, exp);
      },
      result,
      exp);
  return result;
}

Tensor NestedTensor_pow_3(Scalar base, const Tensor& exp) {
  return map_nested_tensor(
      [&base](Tensor exp) { return at::pow(base, exp); }, exp);
}

template <Tensor& (*func)(Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_(Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  apply_nested_tensor(
      [](Tensor& tensor, const Tensor other) { func(tensor, other); },
      self,
      other);
  return self_;
}

template <Tensor (*func)(const Tensor&, Scalar)>
Tensor NestedTensor_binary_scalar(const Tensor& self, Scalar other) {
  return map_nested_tensor(
      [&other](Tensor self) { return func(self, other); }, self);
}

template <Tensor (*func)(const Tensor&, const Tensor&)>
Tensor NestedTensor_binary(const Tensor& self_, const Tensor& other_) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  return map_nested_tensor(
      [](Tensor s, Tensor o) { return func(s, o); }, self, other);
}

template <typename S, Tensor (*func)(const Tensor&, const Tensor&, S)>
Tensor NestedTensor_binary(
    const Tensor& self_,
    const Tensor& other_,
    S scalar) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  std::cout << "is_nested_tensor_impl(self): " << is_nested_tensor_impl(self) << std::endl;
  std::cout << "is_nested_tensor_impl(other): " << is_nested_tensor_impl(other) << std::endl;
  std::cout << "get_nested_tensor_impl(self)->nested_dim(): " << get_nested_tensor_impl(self)->nested_dim() << std::endl;
  std::cout << "get_nested_tensor_impl(other)->nested_dim(): " << get_nested_tensor_impl(other)->nested_dim() << std::endl;
  return map_nested_tensor(
      [&scalar](Tensor self, Tensor other) {
        return func(self, other, scalar);
      },
      self,
      other);
}

template <Tensor& (*func)(Tensor&, const Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  // at::Tensor self;
  // at::Tensor other;
  // std::tie(self, other) = _expand_other_as(self_, other_);
  TORCH_CHECK(
      is_nested_tensor_impl(result),
      "NT binary out variant requires NT as result argument.");
  TORCH_CHECK(
      is_nested_tensor_impl(result, self, other),
      "binary_out doesn't support non-NT arguments.")
  apply_nested_tensor(
      [](Tensor& result, Tensor& tensor, Tensor& other) {
        return func(result, tensor, other);
      },
      result,
      self,
      other);
  return result;
}

Tensor NestedTensor_add(
    const Tensor& self_,
    const Tensor& other_,
    Scalar alpha) {
  at::Tensor self;
  at::Tensor other;
  std::tie(self, other) = _expand_other_as(self_, other_);
  if (is_packed(self, other) &&
      nested_size_matches(get_nested_size(self), get_nested_size(other))) {
#ifdef TRACEPACKED
    std::cout << "calling packed add" << std::endl;
#endif
    return wrap_tensor_node(torch::nested_tensor::impl::build_structure(
        at::add(get_buffer(self), get_buffer(other)),
        get_nested_tensor_impl(self)->nested_size()));
  }
  return map_nested_tensor(
      [&alpha](at::Tensor s, at::Tensor o) { return at::add(s, o, alpha); },
      self,
      other);
}

Tensor& NestedTensor_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  // at::Tensor self;
  // at::Tensor other;
  // std::tie(self, other) = _expand_other_as(self_, other_);
  check_binary_shape(self, other);
  apply_nested_tensor(
      [&](at::Tensor& s, at::Tensor o) { at::native::add_(s, o, alpha); },
      self,
      other);
  return self;
}

#define BINARY_OP(NAME)                                                    \
  nt_impl(m, #NAME ".Tensor", NestedTensor_binary<at::NAME>);              \
  nt_impl(m, #NAME ".Scalar", NestedTensor_binary_scalar<at::NAME>);       \
  nt_impl(m, #NAME "_.Tensor", NestedTensor_binary_<at::native::NAME##_>); \
  nt_impl(m, #NAME ".out", NestedTensor_binary_out<at::NAME##_out>);

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "sub_.Tensor", NestedTensor_sub_);
  nt_impl(m, "sub.out", NestedTensor_sub_out);

  nt_impl(m, "pow.Tensor_Tensor_out", NestedTensor_pow_out_1);
  nt_impl(m, "pow_.Tensor", NestedTensor_pow__1);
  nt_impl(m, "pow.Tensor_Scalar_out", NestedTensor_pow_out_2);
  nt_impl(m, "pow.Tensor_Scalar", NestedTensor_pow_2);
  nt_impl(m, "pow.Scalar_out", NestedTensor_pow_out_3);
  nt_impl(m, "pow.Scalar", NestedTensor_pow_3);

  nt_impl(m, "add.Tensor", NestedTensor_add);
  nt_impl(m, "add_.Tensor", NestedTensor_add_);
  BINARY_OP(div)
  BINARY_OP(mul)
  BINARY_OP(remainder)

  // floor_divide has an inconsistent signature
  nt_impl(m, "floor_divide", NestedTensor_binary<at::floor_divide>);
  nt_impl(
      m,
      "floor_divide_.Tensor",
      NestedTensor_binary_<at::native::floor_divide_>);
  nt_impl(m, "floor_divide.out", NestedTensor_binary_out<at::floor_divide_out>);

  nt_impl(m, "eq.Tensor", NestedTensor_binary<at::eq>);
  nt_impl(m, "eq.Scalar", NestedTensor_binary_scalar<at::eq>);
  nt_impl(m, "ne.Tensor", NestedTensor_binary<at::ne>);
  nt_impl(m, "ne.Scalar", NestedTensor_binary_scalar<at::ne>);
  nt_impl(m, "ge.Tensor", NestedTensor_binary<at::ge>);
  nt_impl(m, "ge.Scalar", NestedTensor_binary_scalar<at::ge>);

  nt_impl(m, "atan2", NestedTensor_binary<at::atan2>);
  nt_impl(m, "atan2_", NestedTensor_binary_<at::native::atan2_>);
  nt_impl(m, "atan2.out", NestedTensor_binary_out<at::atan2_out>);

  nt_impl(m, "logical_and", NestedTensor_binary<at::logical_and>);
  nt_impl(m, "logical_and_", NestedTensor_binary_<at::native::logical_and_>);
  nt_impl(m, "logical_and.out", NestedTensor_binary_out<at::logical_and_out>);

  nt_impl(m, "logical_or", NestedTensor_binary<at::logical_or>);
  nt_impl(m, "logical_or_", NestedTensor_binary_<at::native::logical_or_>);
  nt_impl(m, "logical_or.out", NestedTensor_binary_out<at::logical_or_out>);

  nt_impl(m, "sub.Tensor", (NestedTensor_binary<Scalar, at::sub>));
  nt_impl(m, "pow.Tensor_Tensor", NestedTensor_binary<at::pow>);
}
} // namespace at
