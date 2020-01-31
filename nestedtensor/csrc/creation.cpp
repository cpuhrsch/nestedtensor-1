#include <creation.h>
#include <nested_node.h>
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

// NOTE: py::sequence object asserts that this isn't just a Tensor.
// TODO: Support for simple list of Tensors.
NestedNode<c10::IValue> _get_structure(const py::sequence& py_obj) {
  std::vector<NestedNode<c10::IValue>> result;
  for (size_t i = 0; i < py_obj.size(); i++) {
    if (py::is_instance<py::sequence>(py_object)) {
      result.push_back(_get_structure(py_obj[i]));
    } else {
      result.push_back(py_obj_to_ivalue(py_obj[i]));
    }
  }
  return NestedNode<c10::IValue>(result);
}

THPNestedTensor as_nested_tensor(py::sequence list) {
  return THPNestedTensor(_ListNestedTensor(
      map([](c10::IValue a) { return a.toTensor(); }, _get_structure(list))));
}

_BufferNestedTensor make_contiguous(TensorNode structure) {
  c10::List<at::Tensor> tensors;
  for (const at::Tensor& tensor : flatten(structure)) {
    tensors.emplace_back(tensor.reshape({-1}));
  }
  at::Tensor buffer;
  if (tensors.size() == 0) {
    buffer = torch::ones({});
  } else {
    buffer = at::cat(tensors.vec(), 0);
  }
  return _BufferNestedTensor(
      buffer,
      map([](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
          structure));
}

} // namespace nested_tensor
} // namespace torch
