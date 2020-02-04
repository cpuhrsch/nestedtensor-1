#include <nested_node.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/extension.h>

#include <cstring>

namespace torch {
namespace nested_tensor {

using namespace torch::jit;
using namespace torch::autograd::utils;

c10::optional<IValue> py_obj_to_ivalue(py::object py_obj) {
  auto inferred_type = tryToInferType(py_obj);
  if (!inferred_type.success()) {
    return c10::nullopt;
  }
  auto payload = toIValue(py_obj, inferred_type.type());
  return payload;
}

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  if (size.size() == 0) {
    return 0;
  }
  return size[0] * stride[0];
}

int64_t size_node_memory(
    const SizeNode& nested_size,
    const SizeNode& nested_stride) {
  auto fn = [](c10::List<int64_t> size,
               c10::List<int64_t> stride,
               int64_t input) { return num_memory(size, stride) + input; };
  return reduce<decltype(fn), int64_t, c10::List<int64_t>, c10::List<int64_t>>(
      nested_size, nested_stride, fn, 0);
}

bool _verify_shape(const TensorNode& nested_node) {
  if (nested_node.degree() == 0) {
    return true;
  }
  int64_t first_height = nested_node.children(0).height();
  for (size_t i = 0; i < nested_node.degree(); i++) {
    if (first_height != nested_node.children(i).height()) {
      return false;
    }
  }
  for (size_t i = 0; i < nested_node.degree(); i++) {
    if (!_verify_shape(nested_node.children(i))) {
      return false;
    }
  }
  return true;
}

// TODO: Verify that the height of each child of a given node is the same
// That means each entry is either a list or a node carrying a payload.
bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const TensorNode& nested_node) {
  // The attributes must match across all constiuents
  //
  // The NestedTensor's attributes then become that of its
  // constiuents.
  //
  // data must be a list of Tensors or NestedTensors
  //
  // Attributes:
  //     dim()
  //     layout
  //     device
  //     dtype
  //     requires_grad
  //     is_pinned()
  if (nested_node.height() == 0) {
    return false;
  }
  if (!_verify_shape(nested_node)) {
    return false;
  }
  auto fn = [first_variable](at::Tensor variable, bool valid) {
    // TODO: Add more checks?
    valid = valid && (variable.dim() == first_variable.dim());
    valid = valid && (variable.layout() == first_variable.layout());
    valid = valid && (variable.device() == first_variable.device());
    valid = valid && (variable.dtype() == first_variable.dtype());
    valid =
        valid && (variable.requires_grad() == first_variable.requires_grad());
    return valid;
    // NOTE: This is a very costly check! For now we'll let this to be
    // enabled manually. valid = valid && (variable_.is_pinned() ==
    // first_variable.is_pinned());
  };
  return reduce<decltype(fn), bool, at::Tensor>(nested_node, fn, true);
}

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node) {
  std::vector<c10::optional<int64_t>> start;
  auto maybe_first = get_first_leaf(size_node);
  if (!maybe_first) {
    return start;
  }
  c10::List<int64_t> first = *maybe_first;
  for (size_t i = 0; i < first.size(); i++) {
    start.push_back(first[i]);
  }
  auto fn = [](c10::List<int64_t> size,
               std::vector<c10::optional<int64_t>> result) {
    for (size_t i = 0; i < size.size(); i++) {
      if (!result[i]) {
        continue;
      }
      if (*result[i] != size[i]) {
        result[i] = c10::nullopt;
      }
    }
    return result;
  };
  auto result = reduce<
      decltype(fn),
      std::vector<c10::optional<int64_t>>,
      c10::List<int64_t>>(size_node, fn, start);
  std::vector<c10::optional<int64_t>> tmp(size_node.height() + 1, -1);
  walk(
      [&tmp](SizeNode n) {
        if (!tmp[n.height()]) {
          return;
        }
        if (*tmp[n.height()] == -1) {
          tmp[n.height()] = n.degree();
        }
        if (*tmp[n.height()] != n.degree()) {
          tmp[n.height()] = c10::nullopt;
        }
      },
      size_node);
  for (size_t i = 0; i < result.size(); i++) {
    tmp.push_back(result[i]);
  }

  return tmp;
}

} // namespace nested_tensor
} // namespace torch
