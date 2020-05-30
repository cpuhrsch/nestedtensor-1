#include <torch/script.h>

namespace torch {
namespace nested_tensor {
namespace {

struct SentencePiece : torch::CustomClassHolder {
private:
  std::string _mah_name;

public:
  explicit SentencePiece(const std::string &content) : _mah_name(content) {
  }

  std::string get_mah_name() {
    return _mah_name;
  }
};

// Registers our custom class with torch.
static auto sentencepiece =
    torch::class_<SentencePiece>("nestedtensor", "SentencePiece")
    .def("get_mah_name", &SentencePiece::get_mah_name);

c10::intrusive_ptr<SentencePiece> make_mah_model(std::string mah_name) {
  return c10::make_intrusive<SentencePiece>(mah_name);
}

static auto registry =
    torch::RegisterOperators()
        .op("nestedtensor::make_mah_model", &make_mah_model);
}
}
}
