#include <creation.h>
#include <nested_node.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

// NOTE: py::sequence object asserts that this isn't just a Tensor.
// TODO: Support for simple list of Tensors.
NestedNode<c10::IValue> _get_structure(const py::object& py_obj) {
  if (py::isinstance<py::sequence>(py_obj)) {
    std::vector<const NestedNode<c10::IValue>> result;
    auto py_seq = py::sequence(py_obj);
    for (size_t i = 0; i < py_seq.size(); i++) {
      result.emplace_back(_get_structure(py_seq[i]));
    }
    return NestedNode<c10::IValue>(std::move(result));
  } else {
    return NestedNode<c10::IValue>(py_obj_to_ivalue(py_obj));
  }
}

THPNestedTensor as_nested_tensor(py::sequence list) {
  return THPNestedTensor(_ListNestedTensor(std::move(
      map([](c10::IValue a) { return a.toTensor(); }, _get_structure(list)))));
}

_BufferNestedTensor make_contiguous(const TensorNode structure) {
  c10::List<at::Tensor> _tensors = flatten(structure);
  c10::List<at::Tensor> tensors;
  for (const at::Tensor& tensor : _tensors) {
    tensors.emplace_back(tensor.reshape({-1}));
  }
  at::Tensor buffer;
  if (tensors.size() == 0) {
    buffer = torch::ones({});
  } else {
    buffer = at::cat(tensors.vec(), 0);
  }
  return _BufferNestedTensor(
      std::move(buffer),
      std::move(map(
          [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
          structure)));
}

} // namespace nested_tensor
} // namespace torch
