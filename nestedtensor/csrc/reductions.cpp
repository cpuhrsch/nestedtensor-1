#include <reductions.h>

namespace torch {
namespace nested_tensor {

at::Tensor sum(const NestedTensor& self, c10::optional<ScalarType> dtype) {
  NestedTensor cont = self.contiguous();
  return at::native::sum(*cont.get_buffer(), dtype);
}

NestedTensor& sum_out(
    NestedTensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  std::cout << "HHH" << std::endl;
  return result;
}

NestedTensor sum(
    const NestedTensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  std::cout << "FFF" << std::endl;
  return self;
}

} // namespace nested_tensor
} // namespace torch
