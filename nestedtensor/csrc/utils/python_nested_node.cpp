#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/script.h>

namespace torch {
namespace nested_tensor {
namespace {

// TODO: Need special NestedSize object
// TODO: Check if this is enough to make it jit-able

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
  std::vector<c10::intrusive_ptr<THPNestedNode<T>>> unbind() {
    std::vector<c10::intrusive_ptr<THPNestedNode<T>>> result;
    for (const auto& child : _size_node.unbind()) {
      result.push_back(
          c10::make_intrusive<THPNestedNode<T>>(
            THPNestedNode<T>(child, _name)));
    }
    return result;
  }
  T get_payload() {
    if(! _size_node.is_leaf()) {
      throw std::runtime_error("Needs to be a leaf!");
    }
    return _size_node.payload();
  }

 private:
  NestedNode<T> _size_node;
  std::string _name;
};

using THPSizeNode = THPNestedNode<c10::List<int64_t>>;

bool eq_size_node(
    c10::intrusive_ptr<THPSizeNode> a_, 
    c10::intrusive_ptr<THPSizeNode> b_
    ) {
  SizeNode a = a_->get_node();
  SizeNode b = b_->get_node();
  if (!shape_matches(a, b)) {
    return false;
  }
  auto fn = [](c10::List<int64_t> a, c10::List<int64_t> b) {
    for (size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  };
  return all<decltype(fn)>(std::move(fn), a, b);
}

bool ne_size_node(
    c10::intrusive_ptr<THPSizeNode> a_, 
    c10::intrusive_ptr<THPSizeNode> b_
    ) {
    return !eq_size_node(a_, b_);
}

static auto nestedtensor =
    torch::class_<THPSizeNode>("nestedtensor", "SizeNode")
    .def("__str__", &THPSizeNode::str)
    .def("__len__", &THPSizeNode::len)
    .def("unbind", &THPSizeNode::unbind)
    .def("get_payload", &THPSizeNode::get_payload)
    .def("__eq__", &eq_size_node)
    .def("__ne__", &ne_size_node)
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
