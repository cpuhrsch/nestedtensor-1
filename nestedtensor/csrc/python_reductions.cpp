#include <python_args.h>
#include <python_reductions.h>
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
      py::arg("dim") = nullptr,
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr,
      py::arg("out") = nullptr);
  m.def(
      name.c_str(),
      tmp_fn,
      py::arg("self"),
      py::arg("dim") = nullptr,
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr,
      py::arg("out") = nullptr);

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

// complete reductions
//        'mean',
//        'prod',
//        'sum',

void add_reductions_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {
  auto tmp_fn = [](THPNestedTensor self,
                   c10::optional<std::vector<int64_t>> dim,
                   bool keepdim,
                   c10::optional<ST> dtype) -> py::object {
    if (!dim) {
      if (dtype) {
        return py::cast(torch::nested_tensor::sum(self.data(), dtype->val));
      }
      return py::cast(torch::nested_tensor::sum(self.data(), c10::nullopt));
    }
    if (dtype) {
      return py::cast(
          torch::nested_tensor::sum(self.data(), *dim, keepdim, dtype->val));
    }
    return py::cast(
        torch::nested_tensor::sum(self.data(), *dim, keepdim, c10::nullopt));
  };
  // sum(*, ScalarType? dtype=None)
  // sum(IntArrayRef[1] dim, bool keepdim=False, *, ScalarType? dtype=None)
  c.def(
      "sum",
      tmp_fn,
      py::arg("dim") = nullptr,
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr);
  c.def(
      "sum",
      [&tmp_fn](
          THPNestedTensor self,
          c10::optional<int64_t> dim,
          bool keepdim,
          c10::optional<ST> dtype) -> py::object {
        if (dim) {
          auto dim_arg = std::vector<int64_t>(1);
          dim_arg[0] = *dim;
          return tmp_fn(self.data(), dim_arg, keepdim, dtype);
        }
        return tmp_fn(self.data(), c10::nullopt, keepdim, dtype);
      },
      py::arg("dim") = nullptr,
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr);
}

} // namespace nested_tensor
} // namespace torch
