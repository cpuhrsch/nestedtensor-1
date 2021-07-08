#pragma once
#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/python_functions.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>
#include <nestedtensor/csrc/storage/EfficientSizeNode.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/extension.h>

namespace at {

inline Tensor _collapse_two_dims(Tensor input, int64_t dim1, int64_t dim2) {
  TORCH_CHECK(dim2 - 1 == dim1, "dim2 must be one more than dim1.")
  TORCH_CHECK(dim1 == 0 || dim1 == 1 || dim1 == 2, "dim1 must be 0, 1 or 2.")
  TORCH_CHECK(get_dim(input) == 4, "Expected input to be 4 dim.");
  TORCH_CHECK(get_is_contiguous(input), "Expected input to be contiguous.");
  auto input_esizes = get_efficient_nested_size(input);
  Tensor nt_sizes = input_esizes.sizes();

  Tensor sizes_dim1 = at::native::narrow(nt_sizes, 1, 0, 1).contiguous();
  Tensor sizes_dim2 = at::native::narrow(nt_sizes, 1, 1, 1).contiguous();
  Tensor sizes_dim3 = at::native::narrow(nt_sizes, 1, 2, 1).contiguous();

  if (dim1 == 0) {
    auto nt_opt_size = get_opt_sizes(input);
    TORCH_CHECK(nt_opt_size[1], "Cannot collapse dim 0 and 1 if dim 1 is not regular.");
    // std::cout << "nt_sizes: " << nt_sizes << std::endl;
    Tensor collapsed_sizes = at::native::narrow(nt_sizes, 1, 1, 2).contiguous();
    Tensor new_nt_sizes = collapsed_sizes.repeat_interleave(sizes_dim1[0].item<int64_t>(), 0);
    // std::cout << "new_nt_sizes: " << new_nt_sizes << std::endl;
    auto new_structure = input_esizes.structure();
    new_structure = new_structure * sizes_dim1.numel();
    auto new_esizes = torch::nested_tensor::EfficientSizeNode(1, new_structure, new_nt_sizes);
    Tensor result = wrap_buffer(get_buffer(input), new_esizes);
    TORCH_CHECK(get_dim(result) == 3, "Expected result to be 3 dimensional.");
    return result;
  }

  Tensor new_nt_sizes;
  if (dim1 == 1) {
    Tensor collapsed_sizes = sizes_dim1 * sizes_dim2;
    new_nt_sizes = at::cat({collapsed_sizes, sizes_dim3}, 1);
  } else if (dim1 == 2) {
    Tensor collapsed_sizes = sizes_dim2 * sizes_dim3;
    new_nt_sizes = at::cat({sizes_dim1, collapsed_sizes}, 1);
  }
  auto new_esizes = torch::nested_tensor::EfficientSizeNode(1, input_esizes.structure(), new_nt_sizes);
  Tensor result = wrap_buffer(get_buffer(input), new_esizes);
  TORCH_CHECK(get_dim(result) == 3, "Expected result to be 3 dimensional.");
  return result;
}

}

std::tuple<at::Tensor, at::Tensor> to_tensor_mask(
    at::Tensor nt,
    c10::optional<int64_t> mask_dim);

at::Tensor to_mask(
    at::Tensor nt,
    c10::optional<int64_t> mask_dim);

at::Tensor to_padded_tensor(
    at::Tensor nt,
    double padding);

at::Tensor from_padded_tensor(
    at::Tensor nt,
    torch::nested_tensor::EfficientSizeNode target_size,
    torch::nested_tensor::EfficientSizeNode target_stride);

at::Tensor from_padded_tensor(
    at::Tensor nt,
    torch::nested_tensor::EfficientSizeNode target_size);

c10::optional<at::Tensor> nt_from_tensor_mask(
    at::Tensor tensor,
    at::Tensor mask,
    int64_t nested_dim);
