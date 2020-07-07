#include <ATen/core/op_registration/op_registration.h>
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <nestedtensor/csrc/utils/nested_node_functions.h>
#include <torch/library.h>
#include <nestedtensor/csrc/cuda/multi_tensor_apply.cuh>

namespace at {

Tensor NestedTensor_cos_cuda(const Tensor& self) {
  std::cout << "CUDAA232AA" << std::endl;
  if (self.device() == DeviceType::CUDA) {
    return self.clone();
  }
}

}
