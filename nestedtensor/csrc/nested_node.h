#pragma once
#include <torch/csrc/jit/pybind_utils.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch {
namespace nested_tensor {

// NOTE: For comparisons please use the map and reduce
// functions to define what you mean by equal, etc. on your own
// There can be ambiguity in the depth of comparison and
// even in the value (should it construct a new tree or
// return a single value).
template <typename T>
struct NestedNode {
  NestedNode() : _payload(c10::nullopt) {}
  NestedNode(std::vector<NestedNode<T>>&& children)
      : _children(std::move(children)), _payload(c10::nullopt) {}
  NestedNode(T payload) : _payload(payload) {}
  inline bool is_leaf() const {
    return _children.size() == 0;
  }
  inline c10::optional<T> payload() const {
    return _payload;
  }
  inline NestedNode<T> children(size_t i) const {
    return _children[i];
  }
  inline const NestedNode<T>* children_data(size_t i) const {
    return _children.data() + i;
  }
  inline size_t degree() const {
    return _children.size();
  }
  inline int64_t height() const {
    const NestedNode<T>* start_structure = this;
    int64_t height = 0;
    while (!start_structure->is_leaf()) {
      height++;
      start_structure = start_structure->children_data(0);
    }
    return height;
  }

 private:
  const std::vector<NestedNode<T>> _children;
  // TODO: Make this const?
  // _VariableNode _variable_node;
  c10::optional<T> _payload;
};

using TensorNode = NestedNode<at::Tensor>;
using IValueNode = NestedNode<c10::IValue>;

// This is a C++ representation of a nested list of torch.Sizes
//
// It can never be a list of just numbers, because torch.Size
// is always a list and NestedTensors represent lists of torch.Tensors
//
// Noteworthy cases:
//
// This is an empty list of lists if we construct
// nested_tensor([])
// which is of nested_dim 1, dim 1 and tensor_dim 0
//
// This is a list of empty lists if we construct e.g.
// nested_tensor([torch.tensor(0), torch.tensor(1), ...])
// which is of nested_dim 1, dim 1 and tensor_dim 0
//
// This is a list of list of numbers if we construct e.g.
// nested_tensor([torch.tensor([1]), torch.tensor([2]), ...])
// which is of nested_dim 1, dim 2 and tensor_dim 1
//
// That means, if the list is not empty it is either a list of
// lists of numbers or a list of empty lists.

using SizeNode = NestedNode<c10::List<int64_t>>;
using IntegerNode = NestedNode<int64_t>;

inline std::vector<std::string> split_str(
    std::string s,
    std::string delimiter) {
  std::vector<std::string> result;
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    result.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  result.push_back(s);
  return result;
}

template <typename T, typename F>
std::string NestedNode___str__(
    const NestedNode<T>& nested_node,
    const std::string name,
    F payload_to_str,
    const std::string& tabs = "") {
  std::stringstream result;
  auto tabs_ = tabs + "\t";
  // result << "nested_tensor([";
  result << name << "([";
  if (nested_node.is_leaf()) {
    result << payload_to_str(*nested_node.payload(), tabs_);
  } else {
    for (size_t i = 0; i < nested_node.degree(); i++) {
      if (i > 0) {
        result << ",";
      }
      result << "\n" << tabs_;
      result << NestedNode___str__<T, F>(
          nested_node.children(i), name, payload_to_str, tabs_);
    }
  }
  result << std::endl;
  result << tabs << "])";
  return result.str();
}

c10::optional<c10::IValue> py_obj_to_ivalue(py::object py_obj);

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride);

int64_t size_node_memory(SizeNode nested_size, SizeNode nested_stride);

template <typename A, typename B = py::object>
B wrap_nested_node(NestedNode<A> nested_node) {
  if (nested_node.is_leaf()) {
    return B(py::cast(torch::jit::toPyObject(*nested_node.payload())));
  } else {
    std::vector<B> result;
    for (size_t i = 0; i < nested_node.degree(); i++) {
      result.push_back(wrap_nested_node(nested_node.children(i)));
    }
    return B(py::cast(result));
  }
}

at::Tensor NestedNode_to_tensor(const NestedNode<at::Tensor>& nested_node);

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node);

bool _verify_variables(
    const torch::autograd::Variable& first_variable,
    const TensorNode nested_node);

template <typename A>
inline c10::optional<A> get_first_leaf(NestedNode<A> nested_node) {
  const NestedNode<A>* start = &nested_node;
  while (!start->is_leaf()) {
    start = start->children_data(0);
  }
  return start->payload();
}

template <class F, class A, class TypeList>
class _map;

template <class F, class A, class... Args>
class _map<F, A, c10::guts::typelist::typelist<Args...>> {
 public:
  // NOTE: We must move F to avoid copying objects if it is a lambda with
  // captures.
  static NestedNode<A> function(
      F&& fn,
      const NestedNode<Args>&... nested_node) {
    auto first_node = std::get<0>(std::forward_as_tuple(nested_node...));
    if (first_node.is_leaf()) {
      return NestedNode<A>(std::forward<F>(fn)(*nested_node.payload()...));
    } else {
      std::vector<NestedNode<A>> result;
      for (size_t i = 0; i < first_node.degree(); i++) {
        result.emplace_back(
            function(std::forward<F>(fn), nested_node.children(i)...));
      }
      return NestedNode<A>(std::move(result));
    }
  };
};

// NOTE: Assuming all NestedNodes have same shape.
// TODO: Add check
// TODO: Add static assert to verify lambda arguments match nested_node types
// TODO: Do we want broadcasting?
template <class F, class... B>
static inline NestedNode<
    typename c10::guts::infer_function_traits<F>::type::return_type>
map(F&& fn, const NestedNode<B>&... nested_node) {
  return _map<
      F,
      typename c10::guts::infer_function_traits<F>::type::return_type,
      typename c10::guts::infer_function_traits<F>::type::parameter_types>::
      function(std::move(fn), nested_node...);
}

template <typename A>
inline c10::List<A> flatten(NestedNode<A> nested_node) {
  assert(nested_node.degree() > 0);
  c10::List<A> result;
  for (size_t i = 0; i < nested_node.degree(); i++) {
    if (nested_node.height() == 1) {
      result.push_back(*nested_node.children(i).payload());
    } else {
      c10::List<A> tmp = flatten<A>(nested_node.children(i));
      result.append(std::move(tmp));
    }
  }
  return result;
}

template <class R, class A>
inline std::pair<int64_t, NestedNode<R>> _unflatten(
    const NestedNode<A>& structure,
    const c10::List<R>& content,
    int64_t index) {
  if (structure.is_leaf()) {
    return std::pair<int64_t, NestedNode<R>>(
        index + 1, NestedNode<R>(content[index]));
  } else {
    std::vector<NestedNode<R>> result;
    for (size_t i = 0; i < structure.degree(); i++) {
      auto result_i = _unflatten<R, A>(structure.children(i), content, index);
      index = std::get<0>(result_i);
      result.push_back(std::get<1>(result_i));
    }
    return std::pair<int64_t, NestedNode<R>>(index, NestedNode<R>(std::move(result)));
  }
}

// NOTE: structure is only used as a shape guidance and its content doesn't
// matter. This function uses structure and content to create a new NestedNode
// with the same shape as structure and content distributed in-order
template <class R, class A>
inline NestedNode<R> unflatten(NestedNode<A> structure, c10::List<R> content) {
  auto _result = _unflatten<R, A>(structure, content, 0);
  return std::get<1>(_result);
}

// TODO: Assuming all NestedNodes have same shape.
template <typename F, typename A, typename... B>
inline A reduce(NestedNode<B>... nested_node, F fn, A ident) {
  A result = ident;
  auto first_node = std::get<0>(std::forward_as_tuple(nested_node...));
  if (first_node.is_leaf()) {
    result = fn(*nested_node.payload()..., result);
  } else {
    for (size_t i = 0; i < first_node.degree(); i++) {
      result = reduce<F, A, B...>(nested_node.children(i)..., fn, result);
    }
  }
  return result;
}

// TODO: Assuming all NestedNodes have same shape.
template <class F, class... A>
inline void apply(F&& fn, NestedNode<A>... nested_node) {
  auto first_node = std::get<0>(std::forward_as_tuple(nested_node...));
  if (first_node.is_leaf()) {
    std::forward<F>(fn)(*nested_node.payload()...);
  } else {
    for (size_t i = 0; i < first_node.degree(); i++) {
      apply<F, A...>(std::forward<F>(fn), nested_node.children(i)...);
    }
  }
}

} // namespace nested_tensor
} // namespace torch
