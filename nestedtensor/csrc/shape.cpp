#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/InferSize.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_view(const Tensor& self, IntArrayRef size) {
  auto self_data = get_nested_tensor_impl(self);
  TORCH_CHECK(
      int64_t(size.size()) > self_data->nested_dim(),
      "view cannot be exclusive to nested dimensions.");
  for (int64_t i = 0; i < self_data->nested_dim(); i++) {
    if (size[i] >= 0) {
      throw std::runtime_error(
          "Cannot view explicitly along irregular dimension " +
          std::to_string(i) + ". Please use -1 as a placeholder.");
    }
  }
  int64_t nested_dim = self_data->nested_dim();
  std::vector<int64_t> target_shape;
  for (int64_t i = nested_dim; i < int64_t(size.size()); i++) {
    target_shape.push_back(size[i]);
  }
  // TODO: Potential use for packed view, but requires custom backward.
  return map_nested_tensor(
      [target_shape](const at::Tensor t) {
        return at::native::view(t, IntArrayRef(target_shape));
      },
      self);
}

Tensor NestedTensor_reshape(const Tensor& self, IntArrayRef size) {
  //TODO: Implement reshape for padded storage and use it in mha to fix it.
  auto self_data = get_nested_tensor_impl(self);
  TORCH_CHECK(
      int64_t(size.size()) > self_data->nested_dim(),
      "Reshape cannot be exclusive to nested dimensions.");
  for (int64_t i = 0; i < self_data->nested_dim(); i++) {
    if (size[i] >= 0) {
      throw std::runtime_error(
          "Cannot reshape explicitly along nested dimension " +
          std::to_string(i) + ". Please use -1 as a placeholder.");
    }
  }
  int64_t nested_dim = self_data->nested_dim();
  std::vector<int64_t> target_shape;
  for (int64_t i = nested_dim; i < int64_t(size.size()); i++) {
    target_shape.push_back(size[i]);
  }
  std::cout << "target_shape: " << IntArrayRef(target_shape) << std::endl;
  if (get_raw_padded_storage(self)) {
    PaddedStorage* padded_storage = *get_raw_padded_storage(self);
    auto new_size = map([&target_shape](std::vector<int64_t> sizes) {
          std::cout << "0sizes: " << IntArrayRef(sizes) << std::endl;
          int64_t numel = torch::nested_tensor::impl::count_numel(sizes);
          std::cout << "0numel: " << numel << std::endl;
          at::DimVector new_sizes_dv = at::infer_size_dv(target_shape, numel);
          std::vector<int64_t> new_sizes(new_sizes_dv.begin(), new_sizes_dv.end());
          std::cout << "1sizes: " << IntArrayRef(new_sizes) << std::endl;
          return new_sizes;
        }, get_nested_size(self));
    return wrap_padded(padded_storage->get_padded(), new_size);
  }
  // TODO: Potential use for packed reshape, but requires custom backward.
  return map_nested_tensor(
      [target_shape](const at::Tensor t) {
        return at::reshape(t, IntArrayRef(target_shape));
      },
      self);
}

Tensor NestedTensor_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto self_data = get_nested_tensor_impl(self);
  auto ndims = get_dim(self);
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  if (dim0 == dim1) {
    return self;
  }
  int64_t nested_dim = self_data->nested_dim();
  TORCH_CHECK(
      dim0 >= nested_dim && dim1 >= nested_dim,
      "Transposition involving nested dimensions is not implemented yet.");
  if (get_storage_kind(self) == NestedTensorStorageKind::padded) {
    PaddedStorage* pas = dynamic_cast<PaddedStorage*>(get_storage(self).get());
    std::cout << "pas->get_padded().sizes(): " << pas->get_padded().sizes() << std::endl;
    return wrap_padded(pas->get_padded().transpose(dim0, dim1),
                       map([dim0, dim1, nested_dim](std::vector<int64_t> sizes) {
                         std::vector<int64_t> new_sizes = sizes;
                         int64_t tmp = new_sizes[dim0];
                         new_sizes[dim0] = new_sizes[dim1];
                         new_sizes[dim1] = tmp;
                         return new_sizes;
                         }, get_nested_size(self)));
  }
  // TODO: Potential use for packed transpose, but requires custom backward.
  return map_nested_tensor(
      [dim0, dim1, nested_dim](const at::Tensor t) {
        return at::transpose(t, dim0 - nested_dim, dim1 - nested_dim);
      },
      self);
}

std::vector<Tensor> NestedTensor_chunk(const Tensor& self_, int64_t chunks, int64_t dim) {
  at::Tensor self = NestedTensor_contiguous(self_);
  TORCH_CHECK(dim == -1, "chunk: dim must be -1.");
  auto opt_sizes = get_opt_sizes(self);
  TORCH_CHECK(opt_sizes[opt_sizes.size() - 1], "chunk: last dimension must be regular.");
  at::Tensor packed_buf = get_buffer(self).contiguous().reshape({-1, *opt_sizes[opt_sizes.size() - 1]});
  std::vector<at::Tensor> packed_chunks = packed_buf.chunk(chunks, -1);
  TORCH_CHECK(packed_chunks.size() > 0, "Resulting chunks less than 0.");
  std::vector<at::Tensor> result;
  for (const auto& packed_chunk : packed_chunks) {
    TORCH_CHECK(packed_chunk.dim() == 2, "Expected chunk to be 2dim.");
    int64_t last_size = packed_chunk.size(1);
    result.push_back(wrap_buffer(packed_chunk.reshape({-1}),
                     map([&last_size](std::vector<int64_t> sizes) {
                       sizes[sizes.size() - 1] = last_size;
                       return sizes; }, get_nested_size(self))));
  }
  return result;
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "reshape", NestedTensor_reshape);
  nt_impl(m, "view", NestedTensor_view);
  nt_impl(m, "transpose.int", NestedTensor_transpose);
  nt_impl(m, "chunk", NestedTensor_chunk);
}

} // namespace at
