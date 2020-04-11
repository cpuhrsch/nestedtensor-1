#include <reductions.h>

namespace torch {
namespace nested_tensor {

at::Tensor sum(const NestedTensor& self, c10::optional<ScalarType> dtype) {
  NestedTensor cont = self.contiguous();
  return at::native::sum(*cont.get_buffer(), dtype);
}

NestedTensor& sum_out(
    NestedTensor& result,
    const NestedTensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  std::cout << "HHH" << std::endl;
  return result;
}

inline TensorNode _node_sum_keep_dim(
    TensorNode self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  if (dim == 0) {
    NestedNode<std::vector<at::Tensor>> zipped = zip(self.unbind());
    TensorNode reduced = map(
        [&dtype](std::vector<at::Tensor> data_) {
          auto data = at::TensorList(data_);
          at::Tensor stacked_data = at::stack(data);
          return at::sum(stacked_data, 0, false, dtype);
        },
        zipped);
    return TensorNode(std::vector<TensorNode>{reduced});
  }
  std::vector<TensorNode> result_nodes;
  for (const auto& node : self.unbind()) {
    result_nodes.push_back(_node_sum_keep_dim(node, dim - 1, dtype));
  }
  return TensorNode(std::move(result_nodes));
}

// TODO: check dim for each intarrayref and sort + dedup
NestedTensor sum(
    const NestedTensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  TensorNode result = self.get_structure();
  for (int64_t dim_i_ : dim) {
    int64_t dim_i = at::maybe_wrap_dim(dim_i_, self.dim());
    if (dim_i < self.nested_dim()) {
      std::cout << "AAAdim: " << dim_i << std::endl;
      result = _node_sum_keep_dim(result, dim_i, dtype);
    } else {
      std::cout << "FFFdim: " << dim_i << std::endl;
      dim_i = dim_i - self.nested_dim();
      result = map(
          [&dim_i, &dtype](at::Tensor data) {
            return at::sum(data, dim_i, true, dtype);
          },
          result);
    }
  }
  // TODO: Squeeze for keepdim support
  std::cout << "FFF" << std::endl;
  return NestedTensor(std::move(result));
}

} // namespace nested_tensor
} // namespace torch
