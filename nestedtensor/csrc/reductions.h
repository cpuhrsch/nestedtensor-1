#pragma once
#include <nested_tensor.h>
#include <torch/extension.h>

// TODO: Support for named dimensions.

namespace torch {
namespace nested_tensor {

Tensor sum(const NestedTensor& self, c10::optional<ScalarType> dtype);

NestedTensor& sum_out(
    NestedTensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> opt_dtype);

NestedTensor sum(
    const NestedTensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype);
}
} // namespace torch
