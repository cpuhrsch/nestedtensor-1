#pragma once
#include <nestedtensor/csrc/utils/nested_node.h>
#include <nestedtensor/csrc/py_utils.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace nested_tensor {

void register_python_nested_node(pybind11::module m);

} // namespace nested_tensor
} // namespace torch
