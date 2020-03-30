#include <python_args.h>
#include <reductions.h>
#include <torch/torch.h>

namespace torch {
namespace nested_tensor {

// TODO: Support DimnameList
template <class F>
void add_reduction(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name,
    F& fn) {
  auto tmp_fn = [&fn](
                    THPNestedTensor self,
                    std::vector<int64_t> dim,
                    bool keepdim,
                    c10::optional<py::object> dtype,
                    c10::optional<THPNestedTensor> out) -> THPNestedTensor {
    return THPNestedTensor(NestedTensor(TensorNode(at::ones({}))));
  };

  // sum(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, ScalarType?
  // dtype=None, Tensor out=None)
  m.def(
      name.c_str(),
      [&tmp_fn](
          THPNestedTensor self,
          int64_t dim,
          bool keepdim,
          c10::optional<py::object> dtype,
          c10::optional<THPNestedTensor> out) -> THPNestedTensor {
        return tmp_fn(self, {dim}, keepdim, dtype, out);
      },
      py::arg("self"),
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr,
      py::arg("out") = nullptr);
  m.def(
      name.c_str(),
      tmp_fn,
      py::arg("self"),
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr,
      py::arg("out") = nullptr);

  // sum(IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)
  c.def(
      name.c_str(),
      [&tmp_fn](
          THPNestedTensor self,
          int64_t dim,
          bool keepdim,
          c10::optional<py::object> dtype) -> THPNestedTensor {
        return tmp_fn(self, {dim}, keepdim, dtype, c10::nullopt);
      },
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr);
  c.def(
      name.c_str(),
      [&tmp_fn](
          THPNestedTensor self,
          std::vector<int64_t> dim,
          bool keepdim,
          c10::optional<py::object> dtype) -> THPNestedTensor {
        return tmp_fn(self, dim, keepdim, dtype, c10::nullopt);
      },
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr);
}

template <class F>
void add_full_reduction(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c,
    std::string name,
    F& fn) {
  auto tmp_fn =
      [&fn](THPNestedTensor self, c10::optional<ST> dtype) -> at::Tensor {
    TensorNode result_node =
        map([&fn, &dtype](Tensor data) { return fn(data, dtype); },
            self.data().get_structure());
    // Will be a list of 0-dim Tensors
    at::Tensor values = stack(flatten(result_node).vec());
    return fn(values, dtype);
  };
  // sum(Tensor input, *, ScalarType? dtype=None)
  m.def(name.c_str(), tmp_fn, py::arg("self"), py::arg("dtype") = nullptr);
  // sum(*, ScalarType? dtype=None)
  c.def(name.c_str(), tmp_fn, py::arg("dtype") = nullptr);
}

// complete reductions
//        'mean',
//        'prod',
//        'sum',

void add_reductions_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  auto fn = [](Tensor& result, c10::optional<ST> dtype) {
    if (dtype) {
      return at::native::sum(result, (*dtype).val);
    } else {
      return at::native::sum(result, c10::nullopt);
    }
  };
  add_full_reduction(m, c, "sum", fn);
  add_reduction(m, c, "sum", fn);
}

} // namespace nested_tensor
} // namespace torch
