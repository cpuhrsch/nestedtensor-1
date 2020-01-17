#pragma once
#include <buffer_nested_tensor.h>
#include <list_nested_tensor.h>
#include <torch/custom_class.h>

// NOTE: Causes linktime error for requested symbol as_function
// #include <torch/csrc/jit/script/python_sugared_value.h>
// NOTE: torch/csrc/tensor/python_tensor.h can't be found and will raise compile
// error
// TODO: enable "to" by fixing this.
// #include <torch/csrc/autograd/utils/python_arg_parsing.h>

namespace torch {
namespace nested_tensor {

struct JITTHPSizeNode : torch::jit::CustomClassHolder {
  // JITTHPSizeNode() : _size_node(), _name("asdf") {}
  JITTHPSizeNode() : JITTHPSizeNode(SizeNode(), "asdf") {}
  JITTHPSizeNode(std::string name) : JITTHPSizeNode(SizeNode(), name) {}
  JITTHPSizeNode(SizeNode size_node, std::string name)
      : _size_node(size_node), _name(name) {}
  int64_t len() {
    if (_size_node.is_leaf()) {
      return _size_node.size();
    } else {
      return _size_node.degree();
    }
  }
  std::string str() {
    return SizeNode___str__(_size_node, _name);
  }
  const SizeNode& get_size_node() {
    return _size_node;
  }
  std::string get_name() {
    return _name;
  }

 private:
  SizeNode _size_node;
  std::string _name;
};
} // namespace nested_tensor
} // namespace torch

namespace torch {
namespace jit {
namespace nestedtensor {

static auto my_jit_class =
    torch::jit::class_<torch::nested_tensor::JITTHPSizeNode>("JITSizeNode")
        // .def(torch::jit::init<>())
        .def(torch::jit::init<std::string>())
        .def("__str__", &torch::nested_tensor::JITTHPSizeNode::str)
        // .def(
        //     "__iter__",
        //     [](torch::nested_tensor::JITTHPSizeNode& self) {
        //       return py::make_iterator(
        //           self.get_elements().data(),
        //           self.get_elements().data() + self.get_elements().size());
        //     },
        //     py::keep_alive<0, 1>())
        // .def(
        //     "__eq__",
        //     [](torch::nested_tensor::JITTHPSizeNode& a,
        //        torch::nested_tensor::JITTHPSizeNode& b) {
        //       return a.get_size_node() == b.get_size_node();
        //     })
        .def("__repr__", &torch::nested_tensor::JITTHPSizeNode::str)
        .def("__len__", &torch::nested_tensor::JITTHPSizeNode::len);

}
} // namespace jit
} // namespace torch

namespace torch {
namespace nested_tensor {

std::vector<py::object> unbind_THPSizeNode(
    SizeNode size_node,
    std::string name);

struct THPSizeNode {
  THPSizeNode(SizeNode size_node, std::string name)
      : _size_node(size_node),
        _name(name),
        _elements(unbind_THPSizeNode(_size_node, _name)) {}
  int64_t len() {
    if (_size_node.is_leaf()) {
      return _size_node.size();
    } else {
      return _size_node.degree();
    }
  }
  std::string str() {
    return SizeNode___str__(_size_node, _name);
  }
  const SizeNode& get_size_node() {
    return _size_node;
  }
  std::string get_name() {
    return _name;
  }
  const std::vector<py::object>& get_elements() {
    return _elements;
  }

 private:
  SizeNode _size_node;
  std::string _name;
  std::vector<py::object> _elements;
};

template <class Result, class F>
static inline Result data_map(
    c10::either<_ListNestedTensor, _BufferNestedTensor>& data,
    F fn) {
  return data.map<Result>(fn, fn);
}

struct THPNestedTensor {
  THPNestedTensor() = delete;
  THPNestedTensor(_BufferNestedTensor data) : _data(data) {}
  THPNestedTensor(_ListNestedTensor data) : _data(data) {}
  at::Tensor get_buffer() {
    return _data.right().get_buffer();
  }
  int64_t element_size() {
    return data_map<int64_t>(
        _data, [](auto data) { return data.element_size(); });
  }
  pybind11::object getDtype();
  pybind11::object getLayout();
  pybind11::object getDevice();
  bool requires_grad() {
    return data_map<bool>(
        _data, [](auto data) { return data.requires_grad(); });
  }
  c10::either<_ListNestedTensor, _BufferNestedTensor> data() {
    return _data;
  }
  THPSizeNode nested_size() {
    return THPSizeNode(
        data_map<SizeNode>(_data, [](auto data) { return data.nested_size(); }),
        "NestedSize");
  }
  THPSizeNode nested_stride() {
    return THPSizeNode(
        data_map<SizeNode>(
            _data, [](auto data) { return data.nested_stride(); }),
        "NestedStride");
  }
  THPNestedTensor requires_grad_(pybind11::bool_ requires_grad) {
    return THPNestedTensor(
        data_map<THPNestedTensor>(_data, [&requires_grad](auto data) {
          return data.requires_grad_(requires_grad);
        }));
  }
  THPNestedTensor grad() {
    return data_map<THPNestedTensor>(
        _data, [](auto data) { return THPNestedTensor(data.grad()); });
  }
  THPNestedTensor detach() {
    return data_map<THPNestedTensor>(
        _data, [](auto data) { return THPNestedTensor(data.detach()); });
  }
  THPNestedTensor pin_memory() {
    return data_map<THPNestedTensor>(
        _data, [](auto data) { return THPNestedTensor(data.pin_memory()); });
  }
  std::string str() {
    return data_map<std::string>(_data, [](auto data) {
      return TensorNode___str__(data.get_structure());
    });
  }
  int64_t len() {
    return data_map<int64_t>(_data, [](auto data) { return data.__len__(); });
  }
  bool is_pinned() {
    return data_map<bool>(_data, [](auto data) { return data.is_pinned(); });
  }
  int64_t nested_dim() {
    return data_map<int64_t>(
        _data, [](auto data) { return data.nested_dim(); });
  }
  int64_t dim() {
    return data_map<int64_t>(_data, [](auto data) { return data.dim(); });
  }
  int64_t numel() {
    return data_map<int64_t>(_data, [](auto data) { return data.numel(); });
  }
  at::Tensor to_tensor() {
    return data_map<at::Tensor>(
        _data, [](auto data) { return data.to_tensor(); });
  }
  bool is_contiguous() {
    return data_map<bool>(
        _data, [](auto data) { return data.is_contiguous(); });
  }
  TensorNode get_structure() {
    return data_map<TensorNode>(
        _data, [](auto data) { return data.get_structure(); });
  }

 private:
  c10::either<_ListNestedTensor, _BufferNestedTensor> _data;
};

} // namespace nested_tensor
} // namespace torch
