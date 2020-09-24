#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

struct NestedTensorFunction_embedding_bag
    : torch::autograd::Function<NestedTensorFunction_embedding_bag> {
  static std::vector<at::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& weight,
      const Tensor& indices,
      const Tensor& offsets,
      const bool scale_grad_by_freq,
      const int64_t mode,
      bool sparse,
      const c10::optional<Tensor>& per_sample_weights,
      bool include_last_offset) {
    NestedNode<std::vector<Tensor>> result_tuples = map(
        [&](Tensor i) {
          std::tuple<Tensor, Tensor, Tensor, Tensor> result_i = embedding_bag(
              weight,
              i,
              offsets,
              scale_grad_by_freq,
              mode,
              sparse,
              per_sample_weights,
              include_last_offset);
          std::vector<at::Tensor> result_v({std::get<0>(result_i),
                                            std::get<1>(result_i),
                                            std::get<2>(result_i),
                                            std::get<3>(result_i)});
          return result_v;
        },
        get_nested_tensor_structure(indices));
    std::vector<NestedNode<Tensor>> result_nodes = unzip(result_tuples);
    std::tuple<Tensor, Tensor, Tensor, Tensor> result = std::make_tuple(
        wrap_tensor_node(std::move(result_nodes[0])),
        wrap_tensor_node(std::move(result_nodes[1])),
        wrap_tensor_node(std::move(result_nodes[2])),
        wrap_tensor_node(std::move(result_nodes[3])));
    ctx->save_for_backward({std::get<0>(result),
                            std::get<1>(result),
                            std::get<2>(result),
                            std::get<3>(result)});
    return std::vector<Tensor>({std::get<0>(result),
                                std::get<1>(result),
                                std::get<2>(result),
                                std::get<3>(result)});
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      // TODO: To prevent double backward (for now) check that grad_output
      // doesn't require gradients.
      torch::autograd::variable_list grad_output) {
    std::cout << "grad_output.size(): " << grad_output.size() << std::endl;
    at::Tensor undef;
    return {undef};
  }
};

std::tuple<Tensor, Tensor, Tensor, Tensor> NestedTensor_embedding_bag(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const c10::optional<Tensor>& per_sample_weights,
    bool include_last_offset) {
  std::cout << "Calling into NestedTensor_embedding_bag" << std::endl;
  TORCH_CHECK(indices.dim() == 2, "embedding_bag requires 2d indices as input");
  TORCH_CHECK(
      is_nested_tensor_impl(indices) && !is_nested_tensor_impl(weight),
      "expected a Tensor as a weight, NestedTensor as indices and a Tensor as offsets.");
  std::vector<Tensor> result = NestedTensorFunction_embedding_bag::apply(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset);
  TORCH_CHECK(result.size() == 4, "Expected result to be of size 4.");
  return std::make_tuple(result[0], result[1], result[2], result[3]);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  nt_impl(m, "embedding_bag", NestedTensor_embedding_bag);
}

} // namespace at
