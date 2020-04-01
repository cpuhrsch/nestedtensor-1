#include <python_args.h>
#include <python_reductions.h>
#include <reductions.h>
#include <torch/torch.h>

namespace torch {
namespace nested_tensor {

// complete reductions
//        'mean',
//        'prod',
//        'sum',

// TODO: Support DimnameList
void add_reductions_functions(
    pybind11::module m,
    pybind11::class_<torch::nested_tensor::THPNestedTensor> c) {

  c.def(
      "sum",
      [](THPNestedTensor self, c10::optional<ST> dtype) -> at::Tensor {
        return dtype ? sum(self.data(), dtype->val)
                     : sum(self.data(), c10::nullopt);
      },
      py::arg("dtype") = nullptr);

  auto py_sum = [](THPNestedTensor self,
                   std::vector<int64_t> dim,
                   bool keepdim,
                   c10::optional<ST> dtype) -> THPNestedTensor {
    return dtype
        ? THPNestedTensor(sum(self.data(), dim, keepdim, dtype->val))
        : THPNestedTensor(sum(self.data(), dim, keepdim, c10::nullopt));
  };
  c.def(
      "sum",
      py_sum,
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr);

  c.def(
      "sum",
      [&py_sum](THPNestedTensor self,
         int64_t dim,
         bool keepdim,
         c10::optional<ST> dtype) -> THPNestedTensor {
        return py_sum(self.data(), {dim}, keepdim, dtype);
      },
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr);

  m.def(
      "sum",
      [](THPNestedTensor self, c10::optional<ST> dtype) -> at::Tensor {
        return dtype ? sum(self.data(), dtype->val)
                     : sum(self.data(), c10::nullopt);
      },
      py::arg("self"),
      py::arg("dtype") = nullptr);

  // sum(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, ScalarType?
  // dtype=None, Tensor out=None)
  auto py_sum_out = [](THPNestedTensor self,
                       std::vector<int64_t> dim,
                       bool keepdim,
                       c10::optional<ST> dtype,
                       c10::optional<THPNestedTensor> out) -> THPNestedTensor {
    if (out) {
      if (dtype) {
        return THPNestedTensor(
            sum_out(out->data(), self.data(), dim, keepdim, dtype->val));
      }
      return THPNestedTensor(
          sum_out(out->data(), self.data(), dim, keepdim, c10::nullopt));
    }
    return dtype
        ? THPNestedTensor(sum(self.data(), dim, keepdim, dtype->val))
        : THPNestedTensor(sum(self.data(), dim, keepdim, c10::nullopt));
  };

  m.def(
      "sum",
      py_sum_out,
      py::arg("self"),
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr,
      py::arg("out") = nullptr);

  m.def(
      "sum",
      [&py_sum_out](
          THPNestedTensor self,
          int64_t dim,
          bool keepdim,
          c10::optional<ST> dtype,
          c10::optional<THPNestedTensor> out) -> THPNestedTensor {
        return py_sum_out(self, {dim}, keepdim, dtype, out);
      },
      py::arg("self"),
      py::arg("dim"),
      py::arg("keepdim") = false,
      py::arg("dtype") = nullptr,
      py::arg("out") = nullptr);
}

} // namespace nested_tensor
} // namespace torch
