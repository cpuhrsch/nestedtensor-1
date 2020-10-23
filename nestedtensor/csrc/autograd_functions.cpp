#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_dropout(const Tensor& input, double p, bool train) {
  return autograd_map_nested_tensor(
      [&](const at::Tensor t) { return at::dropout(t, p, train); }, input);
}

Tensor NestedTensor_upsample_bilinear2d(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  return autograd_map_nested_tensor(
      [&](at::Tensor t) {
        return at::upsample_bilinear2d(
                   t.unsqueeze(0),
                   output_size,
                   align_corners,
                   scales_h,
                   scales_w)
            .squeeze(0);
      },
      input);
}

Tensor NestedTensor_clone(
    const Tensor& src,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  return autograd_map_nested_tensor(
      [&optional_memory_format](Tensor a) {
        return at::clone(a, optional_memory_format);
      },
      src);
}

namespace {
void check_dims_match_num_input_features(
    const char* arg_name,
    int64_t expected,
    int64_t actual) {
  TORCH_CHECK(
      actual == expected,
      arg_name,
      " should contain ",
      expected,
      " elements not ",
      actual);
}
} // namespace

Tensor NestedTensor_batch_norm(
    const Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<Tensor>& bias,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled) {
  TORCH_CHECK(!running_mean, "Not supported mean.");
  TORCH_CHECK(!running_var, "Not supported var.");
  auto input_impl = get_nested_tensor_impl(input);
  TORCH_CHECK(
      input_impl->opt_sizes()[1],
      "Input must have non-variable number of channels.");
  int64_t num_features = *input_impl->opt_sizes()[1];
  if (running_mean) {
    check_dims_match_num_input_features(
        "running_mean", num_features, (*running_mean).numel());
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var) {
    check_dims_match_num_input_features(
        "running_var", num_features, (*running_var).numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight) {
    check_dims_match_num_input_features(
        "weight", num_features, (*weight).numel());
  }
  if (bias) {
    check_dims_match_num_input_features("bias", num_features, (*bias).numel());
  }
  std::cout << "ASDF" << std::endl;
  at::Tensor result = input;
  if (weight) {
    result = result * (*weight);
  }
  if (bias) {
    result = result + (*bias);
  }
  return result;
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  // nt_impl(m, "upsample_bilinear2d", NestedTensor_upsample_bilinear2d);
  nt_impl(m, "clone", NestedTensor_clone);
  nt_impl(m, "dropout", NestedTensor_dropout);
  nt_impl(m, "batch_norm", NestedTensor_batch_norm);
}

} // namespace at
