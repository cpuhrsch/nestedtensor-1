#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_conv2d(
    const Tensor& input_,
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups) {
  Tensor input = input_;
  if (is_nested_tensor_impl(input) && !is_nested_tensor_impl(weight)) {
    if (get_dim(input) == 4 && !bias && weight.size(2) == 1 && weight.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 &&
        padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 &&
        groups == 1
      ) {
      input = input.transpose(1, 3);
      input = NestedTensor_contiguous(input);
      at::Tensor input_buffer = get_buffer(input);
      input_buffer = input_buffer.reshape({-1, weight.size(1)});
      at::Tensor result_buffer = at::matmul(input_buffer, 
          weight.reshape({weight.size(0), weight.size(1)}).transpose(0, 1));
      at::Tensor result = wrap_buffer(result_buffer.reshape(-1),
          map([&weight](std::vector<int64_t> size) {
          size[2] = weight.size(0);
          return size;
          }, get_nested_size(input)));
      result = result.transpose(1, 3);
      result = NestedTensor_contiguous(result);
      return result;
    }
  }
    // std::cout << "weight.sizes(): " << weight.sizes() << std::endl;
    // std::cout << "stride: " << stride << std::endl;
    // std::cout << "padding: " << padding << std::endl;
    // std::cout << "dilation: " << dilation << std::endl;
    // std::cout << "groups: " << groups << std::endl;
  if (bias) {
      return map_nested_tensor(
          [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight, at::Tensor bias) {
            return at::conv2d(input.unsqueeze(0), weight, bias, stride, padding, dilation, groups).squeeze(0);
            // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
          },
          input,
          weight,
          *bias);
  }
  return map_nested_tensor(
      [&stride, &padding, &dilation, &groups](at::Tensor input, at::Tensor weight) {
        return at::conv2d(input.unsqueeze(0), weight, c10::nullopt, stride, padding, dilation, groups).squeeze(0);
        // return at::conv2d(input, self, c10::nullopt, stride, padding, dilation, groups);
      },
      input,
      weight);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "conv2d", NestedTensor_conv2d);
}
} // namespace at
