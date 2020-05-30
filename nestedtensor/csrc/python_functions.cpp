#include <nestedtensor/csrc/python_functions.h>	
#include <nestedtensor/csrc/nested_tensor_impl.h>
#include <torch/extension.h>	

using namespace torch::nn;	
namespace F = torch::nn::functional;

namespace torch {
namespace nested_tensor {

NestedTensor cross_entropy(NestedTensor& input,
                           NestedTensor& target,
                           c10::optional<at::Tensor>& weight,
                           bool size_average, // TODO: use
                           int64_t ignore_index,
                           bool reduce, // TODO: use
                           std::string reduction) {
  TensorNode input_structure = input.get_structure();
  TensorNode target_structure = target.get_structure();
  F::CrossEntropyFuncOptions::reduction_t redct;
  if (reduction == "mean" || reduction == "none") {
      redct = torch::kMean;
  } else if (reduction == "sum") {
      redct = torch::kSum;
  } else {
      throw std::runtime_error("Unexpected mode for reduction: " + reduction);
  }

  auto options = F::CrossEntropyFuncOptions().reduction(redct);
  if (ignore_index) {
      options = options.ignore_index(ignore_index);
  }

  TensorNode res = map([&, options] (at::Tensor input_tensor, at::Tensor target_tensor){
      return F::cross_entropy(input_tensor.unsqueeze(0), target_tensor.unsqueeze(0), options).squeeze(0);
  }, input_structure, target_structure);

  return NestedTensor(std::move(res));
}

NestedTensor interpolate(NestedTensor& input,
                         c10::optional<std::vector<std::vector<int64_t>>> size,
                         c10::optional<at::ArrayRef<double>> scale_factor,
                         c10::optional<std::string> mode,
                         c10::optional<bool> align_corners) {
    F::InterpolateFuncOptions::mode_t int_mode;
    if (mode.value() == "nearest" || mode.value() == "none") {
        int_mode = torch::kNearest;
    } else if (mode.value()== "trilinear") {
        int_mode = torch::kTrilinear;
    } else if (mode.value()== "linear") {
        int_mode = torch::kLinear;
    } else if (mode.value()== "bicubic") {
        int_mode = torch::kBicubic;
    } else if (mode.value()== "area") {
        int_mode = torch::kArea;
    } else if (mode.value()== "bilinear") {
        int_mode = torch::kBilinear;
    }  else {
        throw std::runtime_error("Unexpected mode for interpolate: " + mode.value());
    }

    TensorNode input_structure = input.get_structure();
    auto options = F::InterpolateFuncOptions().mode(int_mode);
    if (align_corners.has_value()) {
      options.align_corners() = align_corners.value();
    }

    // Either scale factor or size can be passed
    if (scale_factor.has_value()) {
      options = options.scale_factor(scale_factor.value().vec());
      TensorNode res = map(
        [&options](at::Tensor input_tensor) {
          return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
        },
        input_structure);
      return NestedTensor(std::move(res));
    }

    // Get input leaves count
    auto fn = [](at::Tensor leaf, int64_t input) {
      return input + 1;
    };
    auto leaves_count = reduce<decltype(fn), int64_t, at::Tensor>(input.get_structure(), fn, 0);

    if (size.has_value()) {
      // There can be either 1 size for all tensor or an individual size value per tensor
      if (size.value().size() != 1 && size.value().size() != leaves_count) {
        throw std::runtime_error( "Interpolate has to take either 1 size tuple or same amount as leaves in Nested Tensor.");
      }

      if (size.value().size() == 1) {
        TensorNode res = map(
          [&options, &size](at::Tensor input_tensor) {
            options = options.size(size.value()[0]);
            return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
          },
          input_structure);
        return NestedTensor(std::move(res));
      } else {
        int size_i = 0;
        TensorNode res = map(
            [&options, &size_i, &size](at::Tensor input_tensor) {
              options = options.size(size.value()[size_i]);
              size_i++;
              return F::interpolate(input_tensor.unsqueeze(0), options).squeeze(0);
            },
            input_structure);
        return NestedTensor(std::move(res));
      }
    }

    throw std::runtime_error("Either size or scale_factor should be defined.");
}

namespace {

static auto registry =
    torch::RegisterOperators()
    .op(
        "nestedtensor::cross_entropy",
        [](at::Tensor input_,
           at::Tensor target_,
           c10::optional<at::Tensor> weight,
           bool size_average, // TODO: use
           int64_t ignore_index,
           bool reduce, // TODO: use
           std::string reduction) {
          auto input = get_nested_tensor_impl(input_)->_data;
          auto target = get_nested_tensor_impl(target_)->_data;
          return at::detail::make_tensor<NestedTensorImpl>(cross_entropy(
              input,
              target,
              weight,
              size_average,
              ignore_index,
              reduce,
              reduction));
        })
    .op(
      "nestedtensor::interpolate",
      [](at::Tensor input_,
         c10::optional<std::vector<std::vector<int64_t>>> size,
         c10::optional<std::vector<double>> scale_factor,
         std::string mode, // nearest
         bool align_corners, // false
         bool recompute_scale_factor // false
         ) {
        auto input = get_nested_tensor_impl(input_)->_data;
        if (scale_factor.has_value() && size.has_value()) {
          throw std::runtime_error(
              "only one of size or scale_factor should be defined");
        }

        if (size.has_value()) {
          return at::detail::make_tensor<NestedTensorImpl>(interpolate(
              input, size.value(), c10::nullopt, mode, align_corners));
        }

        if (scale_factor.has_value()) {
          return at::detail::make_tensor<NestedTensorImpl>(interpolate(
              input,
              c10::nullopt,
              scale_factor.value(),
              mode,
              align_corners));
        }

        throw "Either size or scale factor have to be passed.";
      });

}
} // namespace nested_tensor
}
