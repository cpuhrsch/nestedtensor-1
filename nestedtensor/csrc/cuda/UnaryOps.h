#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/library.h>
#include "multi_tensor_apply.cuh"

namespace at {

template <Tensor& (*func)(Tensor&, const Tensor&)>
Tensor& NestedTensor_binary_cuda_(Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(other)) {
    apply(
        [](Tensor& tensor, const Tensor other) { func(tensor, other); },
        get_nested_tensor_structure(self),
        get_nested_tensor_structure(other));
    return self;
  }
  apply(
      [&other](Tensor& tensor) { func(tensor, other); },
      get_nested_tensor_structure(self));
  return self;
}

}
