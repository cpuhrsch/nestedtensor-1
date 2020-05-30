#include <nestedtensor/csrc/creation.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <nestedtensor/csrc/utils/python_nested_node.h>

namespace torch {
namespace nested_tensor {
namespace {

static auto nestedtensor =
    torch::class_<SizeNode>("nestedtensor", "SizeNode");

c10::intrusive_ptr<SizeNode> get_nested_size(at::Tensor self, c10::optional<int64_t> index_) {
    auto nt = at::get_nested_tensor(self);
    return c10::make_intrusive<SizeNode>(nt.nested_size());
}

c10::intrusive_ptr<SizeNode> get_nested_stride(at::Tensor self, c10::optional<int64_t> index_) {
    auto nt = at::get_nested_tensor(self);
    return c10::make_intrusive<SizeNode>(nt.nested_stride());
}

std::string get_size_node_str(c10::intrusive_ptr<SizeNode> _size_node) {
    return NestedNode___str__(
        *_size_node, "SizeNode", [](c10::List<int64_t> payload, const std::string& tabs) {
          std::stringstream ss;
          ss << tabs << payload.vec();
          return ss.str();
        });
}

static auto registry =
    torch::RegisterOperators()
        .op("nestedtensor::nested_size", &get_nested_size)
        .op("nestedtensor::nested_stride", &get_nested_stride)
        .op("nestedtensor::get_size_node_str", &get_size_node_str);

}
} // namespace nested_tensor
} // namespace torch
