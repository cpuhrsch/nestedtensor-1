#include <python_nested_tensor.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/pybind_utils.h>

namespace py = pybind11;

namespace torch {
namespace nested_tensor {

py::object _get_nested_size(THPNestedTensor self) {
  return wrap_nested_node(data_map<SizeNode>(
      self.data(), [](auto data) { return data.nested_size(); }));
}

py::object nested_size() {
  return _get_nested_size(*this);
}
py::object nested_stride() {
  return wrap_nested_node(data_map<SizeNode>(
      _data, [](auto data) { return data.nested_stride(); }));
}

py::object THPNestedTensor::getDtype() {
  return data_map<py::object>(_data, [](auto data) {
    return py::reinterpret_steal<py::object>(
        torch::autograd::utils::wrap(torch::getDtype(data.scalar_type())));
  });
}

py::object THPNestedTensor::getLayout() {
  return data_map<py::object>(_data, [](auto data) {
    return py::reinterpret_steal<py::object>(
        torch::autograd::utils::wrap(torch::getLayout(data.backend())));
  });
}

py::object THPNestedTensor::getDevice() {
  return data_map<py::object>(
      _data, [](auto data) { return torch::jit::toPyObject(data.device()); });
}

} // namespace nested_tensor
} // namespace torch
