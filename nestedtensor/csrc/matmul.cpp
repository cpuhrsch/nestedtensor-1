#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other) {
  if (is_nested_tensor_impl(self) && is_nested_tensor_impl(other)) {
    if (get_dim(self) == get_dim(other) &&
        get_nested_dim(self) == get_nested_dim(other) &&
        get_nested_dim(self) == 1 &&
        get_dim(self) > 2) {
      if (get_storage_kind(self) == NestedTensorStorageKind::padded &&
          get_storage_kind(other) == NestedTensorStorageKind::padded) {
        auto new_nested_size = map([](std::vector<int64_t> sizes_self,
              std::vector<int64_t> sizes_other) {
              std::vector<int64_t> sizes_new = sizes_self;
              sizes_new[sizes_self.size() - 1] = sizes_other[sizes_other.size() - 1];
              return sizes_new;
            }, get_nested_size(self), get_nested_size(other));
        at::Tensor self_padded = get_padded(self);
        at::Tensor other_padded = get_padded(other);
        std::cout << "self_padded.sizes(): " << self_padded.sizes() << std::endl;
        std::cout << "other_padded.sizes(): " << other_padded.sizes() << std::endl;
        return wrap_padded(at::matmul(self_padded, other_padded), new_nested_size);
      }
    }
  }
  if (is_nested_tensor_impl(self) && !is_nested_tensor_impl(other)) {
    if (get_dim(self) == 3 && get_dim(other) == 2) {
      auto self_opt_sizes = get_opt_sizes(self);
      if (self_opt_sizes[2]) {
        if (*self_opt_sizes[2] == other.size(0)) {
          at::Tensor self_cont = NestedTensor_contiguous(self);
          int64_t other_size_1 = other.size(1);
          EfficientSizeNode new_nested_size =
              get_efficient_nested_size(self_cont).clone();
          EfficientSizeNode new_nested_stride =
              get_efficient_nested_stride(self_cont).clone();
          apply_efficient_size(
              [other_size_1](
                  int64_t* size_ptr,
                  int64_t size_size,
                  int64_t* stride_ptr,
                  int64_t stride_size) {
                size_ptr[1] = other_size_1;
                stride_ptr[1] = 1;
                stride_ptr[0] = other_size_1;
              },
              new_nested_size,
              new_nested_stride);
          if (get_storage_kind(self_cont) == NestedTensorStorageKind::packed) {
            Tensor self_buffer = get_buffer(self_cont);
            Tensor result_buffer =
                at::matmul(self_buffer.reshape({-1, other.size(0)}), other);
            result_buffer = result_buffer.reshape({-1});
            return wrap_buffer(
                std::move(result_buffer), new_nested_size, new_nested_stride);
          }
          if (get_storage_kind(self_cont) == NestedTensorStorageKind::padded) {
            Tensor self_padded = get_padded(self_cont);
            Tensor result_padded = at::matmul(self_padded, other);
            return wrap_padded(std::move(result_padded), new_nested_size);
          }
        }
      }
    }
  }
  return map_nested_tensor(
      [](at::Tensor self, at::Tensor other) { return at::matmul(self, other); },
      self,
      other);
}

TORCH_LIBRARY_IMPL(aten, NestedTensor, m) {
  nt_impl(m, "matmul", NestedTensor_matmul);
}
} // namespace at
