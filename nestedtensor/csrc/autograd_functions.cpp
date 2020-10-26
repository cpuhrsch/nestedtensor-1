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

std::vector<int64_t> make_reduce_dims(int64_t input_dim) {
  std::vector<int64_t> result;
  result.push_back(0);
  for (int64_t i = 2; i < input_dim; i++) {
    result.push_back(i);
  }
  return result;
}

std::vector<int64_t> make_scalar_shape(int64_t input_dim, int64_t n_input) {
  std::vector<int64_t> result;
  result.push_back(1);
  result.push_back(n_input);
  for (int64_t i = 2; i < input_dim; i++) {
    result.push_back(1);
  }
  return result;
}

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
  if (input.numel()==0){
    //don't return view of input, don't return empty tensor because it will break gradient chain
    auto out = input.clone();
    if (weight.defined()) out = out * weight[0];
    if (bias.defined()) out = out + bias[0];
    return out;
  }

  auto num_features = input.sizes()[1];
  if (running_mean.defined()) {
    check_dims_match_num_input_features("running_mean", num_features, running_mean.numel());
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features("running_var", num_features, running_var.numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", num_features, bias.numel());
  }

  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  int64_t n_input = input.size(1);
  auto scalar_shape = make_scalar_shape(input.dim(), n_input);

  at::Tensor mean;
  at::Tensor invstd;
  at::Tensor save_mean;
  at::Tensor save_invstd;

  if (training) {
    auto reduce_dims = make_reduce_dims(input.dim());
    save_mean = at::mean(input, IntArrayRef(reduce_dims));

    if (running_mean.defined()) {
      at::Tensor running_mean_(running_mean.getIntrusivePtr());
      running_mean_ = running_mean_.detach();
      running_mean_.copy_(momentum * save_mean + (1 - momentum) * running_mean);
    }

    if (running_var.defined()) {
      Tensor unbiased_var = at::var(input, IntArrayRef(reduce_dims));
      at::Tensor running_var_(running_var.getIntrusivePtr());
      running_var_ = running_var_.detach();
      running_var_.copy_(momentum * unbiased_var + (1 - momentum) * running_var);
    }

    mean = save_mean;
    invstd = at::sqrt(at::var(input, IntArrayRef(reduce_dims), false) + eps);
  } else {
    mean = running_mean;
    invstd = at::sqrt(running_var + eps);
  }

  output = input;
  output = output - mean.reshape(IntArrayRef(scalar_shape));
  output = output / invstd.reshape(IntArrayRef(scalar_shape));

  if (weight.defined()) {
    output = output * weight.reshape(IntArrayRef(scalar_shape));
  }
  if (bias.defined()) {
    output = output + bias.reshape(IntArrayRef(scalar_shape));
  }
  return output;
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  // nt_impl(m, "upsample_bilinear2d", NestedTensor_upsample_bilinear2d);
  nt_impl(m, "clone", NestedTensor_clone);
  nt_impl(m, "dropout", NestedTensor_dropout);
  nt_impl(m, "batch_norm", NestedTensor_batch_norm);
}

} // namespace at
