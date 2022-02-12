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
    std::vector<int64_t> sizes_data,
    int64_t sizes_size_0,
    int64_t sizes_size_1,
    int64_t sizes_dim) {
  std::vector<c10::optional<int64_t>> result;
  result.push_back(out);
  size_t nested_dim = result.size();
  if (sizes_dim > 0) {
    int64_t size_0 = sizes_size_0;
    int64_t size_1 = sizes_size_1;
    int64_t* sizes_ptr = sizes_data.data();
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

  explicit EfficientSizeNode(
      int64_t structure,
      std::vector<int64_t> sizes_data,
      int64_t sizes_size_0,
      int64_t sizes_size_1,
      int64_t sizes_dim)
      : _structure(structure),
        _sizes_data(sizes_data),
        _sizes_size_0(sizes_size_0),
        _sizes_size_1(sizes_size_1),
        _sizes_dim(sizes_dim),
        _opt_sizes(impl::construct_efficient_size(_structure,
                                                  _sizes_data,
                                                  _sizes_size_0,
                                                  _sizes_size_1,
                                                  _sizes_dim))
  {}

  explicit EfficientSizeNode(const SizeNode& size_node)
      : EfficientSizeNode(size_node.degree(),
                          impl::stack_sizes(size_node)) 
  {}

  explicit EfficientSizeNode(
      int64_t structure,
      const at::Tensor& sizes)
      : EfficientSizeNode(structure,
                          impl::_tensor_to_vec(sizes),
                          sizes.dim() > 0 ? sizes.size(0) : 0,
                          sizes.dim() > 0 ? sizes.size(1) : 0,
                          sizes.dim())
  {
    TORCH_CHECK(sizes.dim() == 2 || sizes.dim() == 0, "Expected sizes to be dim 2 or dim 0.");
  }


  SizeNode to_size_node() const {
    std::vector<std::vector<int64_t>> _tmp_sizes;
    at::Tensor _sizes = sizes();
    if (_sizes_dim > 0) {
      _tmp_sizes.resize(_sizes_size_0);
      int64_t* _sizes_ptr = _sizes.data_ptr<int64_t>();
      for (int64_t i = 0; i < _sizes_size_0; i++) {
        _tmp_sizes[i].resize(_sizes_size_1);
        for (int64_t j = 0; j < _sizes_size_1; j++) {
          _tmp_sizes[i][j] = _sizes_ptr[i * _sizes_size_1 + j];
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
    if (_sizes_dim == 0) {
      return 0;
    }
    return _sizes_size_0;
  }
  int64_t dim() const {
    return _sizes_dim > 0 ? 1 + _sizes_size_1 : 1;
  }
  const std::vector<c10::optional<int64_t>>& opt_sizes() const {
    return _opt_sizes;
  }
  const at::Tensor sizes() const {
    if (_sizes_dim == 0 && _sizes_size_0 == 0 && _sizes_size_1 == 0) {
      return torch::zeros({}, torch::kInt64);
    }
    auto result = torch::tensor(_sizes_data);
    return result.reshape({_sizes_size_0, _sizes_size_1});
  }
  const int64_t structure() const {
    return _structure;
  }
  EfficientSizeNode clone() const {
    std::vector<int64_t> new_vector_sizes;
    for (size_t i = 0; i < _sizes_data.size(); i++) {
      new_vector_sizes.push_back(_sizes_data[i]);
    }
    return EfficientSizeNode(_structure, new_vector_sizes, _sizes_size_0, _sizes_size_1, _sizes_dim);
  }
  int64_t numel() const {
    if (_sizes_dim == 0 && _structure > 0) {
      return _structure;
    }
    if (_sizes_dim > 0) {
      if (_sizes_size_0 * _sizes_size_1 == 0) {
        return 0;
      }
      at::Tensor _sizes = sizes();
      Tensor nt_sizes = at::native::narrow(
          _sizes, 1 /* dim */, 0 /* start */, 1 /* length */);
      for (int64_t i = 1; i < _sizes.size(1); i++) {
        Tensor tmp = at::native::narrow(
            _sizes, 1 /* dim */, i /* start */, 1 /* length */);
        nt_sizes = nt_sizes * tmp;
      }
      return nt_sizes.sum().item<int64_t>();
    }
    return 0;
  }

 private:
  int64_t _structure;
  std::vector<int64_t> _sizes_data;
  int64_t _sizes_size_0;
  int64_t _sizes_size_1;
  int64_t _sizes_dim;
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
inline std::tuple<
EfficientSizeNode,
EfficientSizeNode>
  map_efficient_size_stride(
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
    return std::make_tuple(EfficientSizeNode(size_node0.structure(), sizes0),
                           EfficientSizeNode(size_node0.structure(), sizes0));
  }
  TORCH_CHECK(sizes0.size(0) == sizes1.size(0), "Sizes need to match in size(0).");
  TORCH_CHECK(sizes0.size(1) == sizes1.size(1), "Sizes need to match in size(1).");
  int64_t* sizes0_ptr = sizes0.data_ptr<int64_t>();
  int64_t* sizes1_ptr = sizes1.data_ptr<int64_t>();
  for (int64_t i = 0; i < sizes0.size(0); i++) {
    fn(sizes0_ptr + i * sizes0.size(1),
       sizes0.size(1),
       sizes1_ptr + i * sizes1.size(1),
       sizes1.size(1));
  }
  return std::make_tuple(
      EfficientSizeNode(size_node0.structure(), sizes0),
      EfficientSizeNode(size_node1.structure(), sizes1));
}

} // namespace nested_tensor
} // namespace torch
