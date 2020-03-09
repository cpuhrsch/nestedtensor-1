#include <nested_tensor.h>

namespace torch {
namespace nested_tensor {

int64_t num_memory(c10::List<int64_t> size, c10::List<int64_t> stride) {
  // 0-dim Tensors have torch.Size of .size() 0, but carry 1 memory.
  // Empty 1-dim Tensors (torch.tensor([])) have torch.Size of .size() 1,
  // but carry 0 memory.
  if (size.size() == 0) {
    return 1;
  }
  return size[0] * stride[0];
}

std::vector<c10::optional<int64_t>> construct_size(const SizeNode& size_node) {
  if (size_node.is_leaf()) {
    std::vector<c10::optional<int64_t>> result;
    for (const auto& size : size_node.payload()) {
      result.push_back(size);
    }
    return result;
  }
  std::vector<c10::optional<int64_t>> result;
  result.push_back(size_node.degree());

  if (size_node.degree() > 0) {
    for (const auto& size : construct_size(size_node.children(0))) {
      result.push_back(size);
    }
    for (size_t i = 1; i < size_node.degree(); i++) {
      auto size_node_i = construct_size(size_node.children(i));
      for (size_t j = 1; j < result.size(); j++) {
        if (result[j] && ((*result[j]) != size_node_i[j - 1])) {
          result[j] = c10::nullopt;
        }
      }
    }
  }

  return result;
}

std::vector<c10::optional<int64_t>> NestedTensor::size() {
  return construct_size(_nested_size);
}

c10::List<int64_t> _cont_stride(c10::List<int64_t> size) {
  std::vector<int64_t> stride(size.size());
  int64_t p = 1;
  size_t p_i = size.size();
  for (size_t i = 0; i < size.size(); i++) {
    p_i--;
    stride[p_i] = p;
    p *= size[p_i];
  }
  return c10::List<int64_t>(stride);
}

TensorNode build_structure(
    const at::Tensor& buffer,
    const SizeNode& nested_size,
    const SizeNode& nested_stride) {
  c10::List<int64_t> split_sizes = flatten(
      map([](c10::List<int64_t> a,
             c10::List<int64_t> b) { return num_memory(a, b); },
          nested_size,
          nested_stride));
  std::vector<int64_t> nonzero_split_sizes;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      nonzero_split_sizes.push_back(split_sizes[i]);
    }
  }
  std::vector<at::Tensor> buffers_;
  if (nonzero_split_sizes.size() > 0) {
    buffers_ =
        at::split_with_sizes(buffer, c10::IntArrayRef(nonzero_split_sizes), 0);
  }
  std::vector<at::Tensor> buffers;
  int64_t index = 0;
  for (size_t i = 0; i < split_sizes.size(); i++) {
    if (split_sizes[i] > 0) {
      buffers.push_back(buffers_[index]);
      index++;
    } else {
      buffers.push_back(at::empty({}, buffer.options()));
    }
  }
  TensorNode tmp = unflatten(nested_size, c10::List<at::Tensor>(buffers));
  return map(
      [](at::Tensor buffer,
         c10::List<int64_t> size,
         c10::List<int64_t> stride) {
        return at::as_strided(
            buffer,
            c10::IntArrayRef(size.vec()),
            c10::IntArrayRef(stride.vec()));
      },
      tmp,
      nested_size,
      nested_stride);
}

TensorNode build_structure(
    const at::Tensor& buffer,
    const SizeNode& nested_size) {
  SizeNode nested_stride = map(
      [](c10::List<int64_t> size) { return _cont_stride(size); }, nested_size);
  return build_structure(buffer, nested_size, nested_stride);
}

SizeNode infer_nested_size(const TensorNode& _structure) {
  return map(
      [](at::Tensor tensor) { return c10::List<int64_t>(tensor.sizes()); },
      _structure);
}

NestedTensor NestedTensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }
  TensorNode flat_structure =
      map([](at::Tensor tensor) { return tensor.reshape({-1}); }, _structure);
  auto tensors = flatten(flat_structure).vec();
  if (tensors.size() == 0) {
    return NestedTensor(at::ones({0}), _nested_size);
  }
  return NestedTensor(at::cat(tensors, 0), _nested_size);
}

at::Tensor _to_tensor(TensorNode node) {
  // TODO: Recursive stacking is expensive.
  if (node.is_leaf()) {
    return node.payload();
  }
  if (node.degree() == 0) {
    return at::empty({0});
  }
  std::vector<at::Tensor> flat;
  for (auto child : node.unbind()) {
    flat.push_back(_to_tensor(child));
  }
  return stack(flat);
}

at::Tensor NestedTensor::to_tensor() {
  // TODO: Not necessarily a view because of stack and reshape.
  std::vector<int64_t> new_size;
  for (const auto& si : size()) {
    if (!si) {
      // TODO: This assumes we'll extend to_tensor to also work with int64_t at
      // this level.
      throw std::out_of_range(
          "to_tensor()/to_tensor(0) only works if there is no None in size().");
    }
    new_size.push_back(*si);
  }
  if (is_contiguous()) {
    return (*_buffer).reshape(at::IntArrayRef(new_size));
  }
  return _to_tensor(_structure);
}

TensorNode _unbind_tensors(TensorNode structure) {
  std::vector<TensorNode> result_nodes;
  if (structure.is_leaf()) {
    for (at::Tensor tensor : structure.payload().unbind()) {
      result_nodes.emplace_back(TensorNode(std::move(tensor)));
    }
  } else {
    for (TensorNode child : structure.unbind()) {
      result_nodes.emplace_back(_unbind_tensors(child));
    }
  }
  return TensorNode(std::move(result_nodes));
}

NestedTensor NestedTensor::to_nested_tensor(c10::optional<int64_t> dim__) {
  int64_t dim_ = 0;
  if (dim__) {
    dim_ = *dim__;
  }
  int64_t dim = at::maybe_wrap_dim(dim_, this->dim());
  // if dim < nested_dim() the NestedTensor is already nested
  // up to the given dimension.
  if (dim >= nested_dim()) {
    TensorNode unbound = _unbind_tensors(_structure);
    for (int64_t i = 0; i < (dim - nested_dim()); i++) {
      unbound = _unbind_tensors(unbound);
    }
    return NestedTensor(std::move(unbound));
  }
  return *this;
}

std::vector<int64_t> max_size(SizeNode nested_size) {
    if (nested_size.degree() == 0) {
        return std::vector<int64_t>(1, 0);
    }
    if (nested_size.is_leaf()) {
        return nested_size.payload().vec();
    }
    std::vector<int64_t> result;
    result.push_back(nested_size.degree());
    for (int64_t s : max_size(nested_size.children(0))) {
        result.push_back(s);
    }
    for (size_t i = 1; i < nested_size.degree(); i++) {
        std::vector<int64_t> max_size_i = max_size(nested_size.children(i));
        for (size_t j = 1; j < result.size(); j++) {
            result[j] = max_size_i[j - 1] > result[j] ? max_size_i[j - 1] : result[j];
        }
    }
    return result;
}

TensorNode empty(TensorNode input) {
    return map([](at::Tensor data) { 
            return torch::empty_like(data);
            },
            input);
}

// If a TensorNode has degree 0 in the context of a NestedTensor with other
// TensorNodes that are not ancestors (e.g. siblings), then to maintain
// a valid NestedTensor any new children must be of dimension 0.
// An empty NestedTensor has dimension 0, so to add a Tensor of any
// different dimensionality it will influence the global dimensionality
// constraint.
// That means, when we return a tensor mask in this case we can confidently
// return an empty data scalar and bool scale of value False.
TensorNode make_full(const at::Tensor& first_variable, TensorNode input, std::vector<int64_t> size, int64_t level = 0) {
    if (input.is_leaf()) {
        // std::cout << "payload: " << input.payload();
        auto payload_size = input.payload().sizes();
        std::vector<int64_t> target_size;
        for (size_t i_ = 0; i_ < payload_size.size(); i_++) {
            int64_t i = payload_size.size() - 1 - i_;
            int64_t pad_amount = size[i + level] - payload_size[i];
            TORCH_CHECK(pad_amount >= 0, "Tensor size(", i, ") ", payload_size[i], " larger than target size ", size[i + level], ".");
            target_size.push_back(0);
            target_size.push_back(pad_amount);
        }
        at::Tensor result_tensor = constant_pad_nd(input.payload(), IntArrayRef(target_size), 0);
        // std::cout << " = ";
        // std::cout << "result_tensor: " << result_tensor;
        return TensorNode(std::move(result_tensor));
    }
    TORCH_CHECK(input.degree() <= size[level], "Given input is wider than requested size.");
    std::vector<TensorNode> result;
    if (input.degree() == 0) {
        std::cout << "JJJ" << std::endl;
        result.push_back(TensorNode(first_variable.new_empty({0})));
    } else {
        std::vector<TensorNode> unbound = input.unbind();
        for (size_t i = 0; i < unbound.size(); i++) {
            result.push_back(make_full(first_variable, unbound[i], size, level + 1));
        }
    }
    for (size_t i = 0; i < (size[level] - result.size()); i++) {
        result.push_back(empty(result[0]));
    }
    return TensorNode(std::move(result));
}

std::pair<at::Tensor, at::Tensor>
NestedTensor::to_tensor_mask(c10::optional<int64_t> mask_dim) {
    std::vector<int64_t> ms =  max_size(nested_size());
    std::cout << "max_size(nested_size()): ";
    for (auto s : ms) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
    TensorNode full_node = make_full(_first_variable, _structure, ms);
    return std::pair<at::Tensor, at::Tensor> (
            _to_tensor(make_full(_first_variable, _structure, ms)),
            torch::zeros({}));
}

NestedTensor::NestedTensor(TensorNode&& structure)
    : _structure(structure),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(infer_nested_size(_structure)) {}

NestedTensor::NestedTensor(at::Tensor&& buffer, SizeNode nested_size)
    : _buffer(buffer),
      _structure(build_structure(*_buffer, nested_size)),
      _first_variable(
          get_first_leaf(_structure) ? *get_first_leaf(_structure)
                                     : at::ones({})),
      _nested_size(nested_size) {}

} // namespace nested_tensor
} // namespace torch
