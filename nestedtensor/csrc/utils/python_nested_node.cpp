#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/script.h>

namespace torch {
namespace nested_tensor {
namespace {

template <typename T>
struct THPNestedNode : torch::CustomClassHolder {
  THPNestedNode(NestedNode<T> size_node, std::string name)
      : _size_node(size_node), _name(name) {}
  int64_t len() {
    return _size_node.degree();
  }
  std::string str() {
    return NestedNode___str__(
        _size_node, _name, [](c10::IValue payload, const std::string& tabs) {
          std::stringstream ss;
          ss << tabs << payload;
          return ss.str();
        });
  }
  const NestedNode<T>& get_node() const {
    return _size_node;
  }
  pybind11::object unbind() {
    std::vector<pybind11::object> result;
    for (const auto& child : _size_node.unbind()) {
      if (child.height() == 0) {
        result.push_back(wrap_nested_node(child));
      } else {
        result.push_back(pybind11::cast(THPNestedNode<T>(child, _name)));
      }
    }
    return pybind11::cast(result);
  }

 private:
  NestedNode<T> _size_node;
  std::string _name;
};

using THPSizeNode = THPNestedNode<c10::List<int64_t>>;

static auto nestedtensor =
    torch::class_<THPSizeNode>("nestedtensor", "SizeNode")
    .def("__str__", &THPSizeNode::str)
    ;

c10::intrusive_ptr<THPSizeNode> get_nested_size(at::Tensor self, c10::optional<int64_t> index_) {
    auto nt = at::get_nested_tensor(self);
    return c10::make_intrusive<THPSizeNode>(THPSizeNode(nt.nested_size(), "SizeNode"));
}

c10::intrusive_ptr<THPSizeNode> get_nested_stride(at::Tensor self, c10::optional<int64_t> index_) {
    auto nt = at::get_nested_tensor(self);
    return c10::make_intrusive<THPSizeNode>(THPSizeNode(nt.nested_stride(), "SizeNode"));
}

static auto registry =
    torch::RegisterOperators()
        .op("nestedtensor::nested_size", &get_nested_size)
        .op("nestedtensor::nested_stride", &get_nested_stride);

}
} // namespace nested_tensor
} // namespace torch
