#include <torch/extension.h>

struct ST {
  c10::ScalarType val;
};

namespace pybind11 {
namespace detail {
template <>
struct type_caster<ST> {
 public:
  /**
   * This macro establishes the name 'inty' in
   * function signatures and declares a local variable
   * 'value' of type inty
   */
  PYBIND11_TYPE_CASTER(ST, _("ScalarType"));

  /**
   * Conversion part 1 (Python->C++): convert a PyObject into a inty
   * instance or return false upon failure. The second argument
   * indicates whether implicit conversions should be applied.
   */
  bool load(handle obj, bool) {
    /* Extract PyObject from handle */
    if (THPDtype_Check(obj.ptr())) {
      auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());
      value.val = dtype->scalar_type;
      return true;
    }
    if (obj.ptr() == (PyObject*)(&PyFloat_Type)) {
      value.val = c10::ScalarType::Float;
      return true;
    }
    if (obj.ptr() == (PyObject*)(&PyBool_Type)) {
      value.val = c10::ScalarType::Bool;
      return true;
    }
    if (obj.ptr() == (PyObject*)(&PyLong_Type)) {
      value.val = c10::ScalarType::Long;
      return true;
    }
    return false;
  }

  /**
   * Conversion part 2 (C++ -> Python): convert an inty instance into
   * a Python object. The second and third arguments are used to
   * indicate the return value policy and parent object (for
   * ``return_value_policy::reference_internal``) and are generally
   * ignored by implicit casters.
   */
  static handle cast(
      ST src,
      return_value_policy /* policy */,
      handle /* parent */) {
    return py::handle(reinterpret_cast<PyObject*>(torch::getDtype(src.val)));
  }
};
} // namespace detail
} // namespace pybind11
