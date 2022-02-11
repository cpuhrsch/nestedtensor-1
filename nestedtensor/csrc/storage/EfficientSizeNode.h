#pragma once
#include <nestedtensor/csrc/storage/common.h>

namespace torch {
namespace nested_tensor {

namespace impl {
inline at::Tensor stack_sizes(SizeNode size_node) {
  TORCH_CHECK(size_node.height() == 1, "stack_sizes: Expected height equals 1.");
  if (size_node.degree() == 0) {
    return torch::zeros({}, torch::kInt64);
  }
  std::vector<SizeNode> unbound_size_node = size_node.unbind();
  std::vector<int64_t> result_sizes_vector;
  for(int64_t i = 0; i < unbound_size_node.size(); i++) {
    std::vector<int64_t> sizes = unbound_size_node[i].payload();
    if(i == 0) {
      result_sizes_vector.reserve(size_node.degree() * sizes.size());
    }
    for (size_t j = 0; j < sizes.size(); j++) {
      result_sizes_vector.push_back(sizes[j]);
    }
  }
  return torch::tensor(result_sizes_vector, torch::kInt64).reshape({static_cast<int64_t>(size_node.degree()), -1});
}

inline std::vector<c10::optional<int64_t>> construct_efficient_size(
    int64_t out,
    std::vector<int64_t> sizes) {
//    const at::Tensor& sizes) {
  std::vector<c10::optional<int64_t>> result;
  result.push_back(out);
  size_t nested_dim = result.size();
  int64_t size_0 = out;
  int64_t size_1 = sizes.size() / out;
  int64_t* sizes_ptr = sizes.data();
  result.resize(nested_dim + size_1);
  for (int64_t i = 0; i < size_1; i++) {
    result[nested_dim + i] = sizes_ptr[i];
  }
  for (int64_t j = 0; j < size_1; j++) {
    for (int64_t i = 0; i < size_0; i++) {
      if (result[nested_dim + j] &&
          (result[nested_dim + j] != sizes_ptr[i * size_1 + j])) {
        result[nested_dim + j] = c10::nullopt;
      }
    }
  }
  return result;
}

inline std::vector<int64_t> _tensor_to_vec(at::Tensor tensor) {
  tensor = tensor.contiguous();
  auto tensor_data = tensor.data_ptr<int64_t>();
  return std::vector<int64_t>(tensor_data, tensor_data + tensor.numel());
}

} // namespace impl

struct EfficientSizeNode {
  explicit EfficientSizeNode(const SizeNode& size_node)
      : _structure(size_node.degree()),
        _sizes(impl::_tensor_to_vec(impl::stack_sizes(size_node))),
        _opt_sizes(impl::construct_efficient_size(_structure, _sizes))
  {}

  explicit EfficientSizeNode(
      int64_t structure,
      const at::Tensor& sizes)
      : _structure(structure),
        _sizes(impl::_tensor_to_vec(sizes)),
        _opt_sizes(impl::construct_efficient_size(_structure, _sizes))
  {
    TORCH_CHECK(sizes.dim() == 2, "Expected sizes to be dim 1.");
  }

  explicit EfficientSizeNode(
      int64_t structure,
      std::vector<int64_t> sizes)
      : _structure(structure),
        _sizes(sizes),
        _opt_sizes(impl::construct_efficient_size(_structure, _sizes))
  {}

  SizeNode to_size_node() const {
    std::vector<std::vector<int64_t>> _tmp_sizes;
    at::Tensor sizes_t = sizes();
    if (sizes_t.dim() > 0) {
      _tmp_sizes.resize(sizes_t.size(0));
      int64_t* _sizes_ptr = sizes_t.data_ptr<int64_t>();
      for (int64_t i = 0; i < sizes_t.size(0); i++) {
        _tmp_sizes[i].resize(sizes_t.size(1));
        for (int64_t j = 0; j < sizes_t.size(1); j++) {
          _tmp_sizes[i][j] = _sizes_ptr[i * sizes_t.size(1) + j];
        }
      }
    }
    std::vector<SizeNode> _tmp_size_nodes;
    for (int64_t i = 0; i < _structure; i++) {
      _tmp_size_nodes.push_back(SizeNode(std::move(_tmp_sizes[i])));
    }
    return SizeNode(std::move(_tmp_size_nodes));
  }
  int64_t height() const {
    return 1;
  }
  int64_t degree() const {
    if (sizes().dim() == 0) {
      return 0;
    }
    return sizes().size(0);
  }
  int64_t dim() const {
    return sizes().dim() > 0 ? 1 + sizes().size(1) : 1;
  }
  const std::vector<c10::optional<int64_t>>& opt_sizes() const {
    return _opt_sizes;
  }
  void refresh_opt_sizes() {
    _opt_sizes = impl::construct_efficient_size(_structure, _sizes);
  }
  const at::Tensor sizes() const {
    auto result = torch::tensor(_sizes);
    return result.reshape({_structure, -1});
  }
  const int64_t structure() const {
    return _structure;
  }
  EfficientSizeNode clone() const {
    return EfficientSizeNode(_structure, _sizes);
  }
  int64_t numel() const {
    at::Tensor sizes_t = sizes();
    if (sizes_t.dim() == 0 && _structure > 0) {
      return _structure;
    }
    if (sizes_t.dim() > 0) {
      if (sizes_t.numel() == 0) {
        return 0;
      }
      Tensor nt_sizes = at::native::narrow(
          sizes_t, 1 /* dim */, 0 /* start */, 1 /* length */);
      for (int64_t i = 1; i < sizes_t.size(1); i++) {
        Tensor tmp = at::native::narrow(
            sizes_t, 1 /* dim */, i /* start */, 1 /* length */);
        nt_sizes = nt_sizes * tmp;
      }
      return nt_sizes.sum().item<int64_t>();
    }
    return 0;
  }

 private:
  int64_t _structure;
  std::vector<int64_t> _sizes;
  bool _opt_sizes_set = false;
  std::vector<c10::optional<int64_t>> _opt_sizes;
};

inline bool efficient_size_structure_matches(
    const EfficientSizeNode& size_node0,
    const EfficientSizeNode& size_node1) {
  return size_node0.structure() == size_node1.structure();
}

inline bool efficient_size_matches(
    const EfficientSizeNode& size_node0,
    const EfficientSizeNode& size_node1) {
  if (!efficient_size_structure_matches(size_node0, size_node1)) {
    return false;
  }
  at::Tensor sizes0 = size_node0.sizes();
  at::Tensor sizes1 = size_node1.sizes();
  return at::equal(sizes0, sizes1);
}

template <class F>
inline EfficientSizeNode map_efficient_size(
    F&& fn,
    const EfficientSizeNode& size_node) {
  at::Tensor sizes = size_node.sizes().clone();
  if (sizes.dim() == 0) {
    return EfficientSizeNode(size_node.structure(), sizes);
  }
  int64_t* sizes_ptr = sizes.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes.size(0); i++) {
    fn(sizes_ptr + i * sizes.size(1), sizes.size(1));
  }
  return EfficientSizeNode(size_node.structure(), sizes);
}

template <class F>
inline EfficientSizeNode map_efficient_size(
    F&& fn,
    const EfficientSizeNode& size_node0,
    const EfficientSizeNode& size_node1) {
  TORCH_CHECK(
      efficient_size_structure_matches(size_node0, size_node1),
      "map_efficient_size: Length doesn't match.");
  at::Tensor sizes0 = size_node0.sizes().clone();
  at::Tensor sizes1 = size_node1.sizes().clone();
  TORCH_CHECK(sizes0.dim() == sizes1.dim(), "Sizes need to match in dim.");
  if (sizes0.dim() == 0) {
    return EfficientSizeNode(size_node0.structure(), sizes0);
  }
  TORCH_CHECK(sizes0.size(0) == sizes1.size(0), "Sizes need to match in size(0).");
  TORCH_CHECK(sizes0.size(1) == sizes1.size(1), "Sizes need to match in size(1).");
  int64_t* sizes_ptr0 = sizes0.data_ptr<int64_t>();
  int64_t* sizes_ptr1 = sizes1.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes0.size(0); i++) {
    fn(sizes_ptr0 + i * sizes0.size(1), sizes_ptr1 + i * sizes1.size(1), sizes0.size(1));
  }
  return EfficientSizeNode(size_node0.structure(), sizes0);
}

template <class F>
inline void apply_efficient_size(
    F&& fn,
    EfficientSizeNode& size_node0,
    EfficientSizeNode& size_node1) {
  at::Tensor sizes0 = size_node0.sizes();
  at::Tensor sizes1 = size_node1.sizes();
  int64_t* sizes0_ptr = sizes0.data_ptr<int64_t>();
  int64_t* sizes1_ptr = sizes1.data_ptr<int64_t>();
  TORCH_CHECK(
      efficient_size_structure_matches(size_node0, size_node1),
      "apply_efficient_size: Length doesn't match.");
  for (int64_t i = 0; i < sizes0.size(0); i++) {
    fn(sizes0_ptr + i * sizes0.size(1),
       sizes0.size(1),
       sizes1_ptr + i * sizes1.size(1),
       sizes1.size(1));
  }
  size_node0.refresh_opt_sizes();
  size_node1.refresh_opt_sizes();
}

} // namespace nested_tensor
} // namespace torch
