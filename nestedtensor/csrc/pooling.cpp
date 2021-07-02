#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <nestedtensor/csrc/transpose.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_adaptive_avg_pool2d(
    at::Tensor const& input,
    IntArrayRef output_size) {
  return map_nested_tensor(
      [&output_size](at::Tensor input) {
        return at::native::adaptive_avg_pool2d(input, output_size);
      },
      input);
}

Tensor NestedTensor_adaptive_avg_pool2d_backward(
    const Tensor& gradInput,
    const Tensor& input) {
  return map_nested_tensor(
      [](at::Tensor gradInput, at::Tensor input) {
        return at::_adaptive_avg_pool2d_backward(gradInput, input);
      },
      gradInput,
      input);
}

Tensor NestedTensor_max_pool2d(
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor self = self_;
  bool got_channel_last = false;
  // std::cout << "0 self.dtype(): " << self.dtype() << std::endl;
  if (get_is_channel_last(self)) {
    // std::cout << "1 self.dtype(): " << self.dtype() << std::endl;
    self = transpose_nhwc_nchw(self);
    got_channel_last = true;
  }
  Tensor result = map_nested_tensor(
      [&](at::Tensor t) {
      // std::cout << "t.dtype(): " << t.dtype() << std::endl;
        Tensor r =  at::max_pool2d(
                   t.unsqueeze(0),
                   kernel_size,
                   stride,
                   padding,
                   dilation,
                   ceil_mode)
            .squeeze(0);
      // std::cout << "r.dtype(): " << r.dtype() << std::endl;
      return r;
      },
      self);
  if (got_channel_last) {
    return transpose_nchw_nhwc(result);
  }
  return result;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "max_pool2d", NestedTensor_max_pool2d);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "adaptive_avg_pool2d", NestedTensor_adaptive_avg_pool2d);
  nt_impl(
      m,
      "adaptive_avg_pool2d_backward",
      NestedTensor_adaptive_avg_pool2d_backward);
}

} // namespace at
